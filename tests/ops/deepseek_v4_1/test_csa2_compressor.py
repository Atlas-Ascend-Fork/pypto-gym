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
"""CSA2 compressor fwd+bwd entry test (self-contained golden)."""
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.testing import assert_allclose
import pytest
import torch
import torch_npu

_p = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _p)
while _p != os.path.dirname(_p):
    if os.path.isdir(os.path.join(_p, "src")):
        sys.path.insert(0, os.path.join(_p, "src"))
        sys.path.insert(0, os.path.join(_p, "src", "pypto_gym", "ops", "pypto_tensor"))
        break
    _p = os.path.dirname(_p)

try:
    from deepseek_v4_1.csa2_compressor import csa2_compressor, CSA2Compressor  # noqa: E402
except ImportError:
    from csa2_compressor import csa2_compressor, CSA2Compressor

logger = logging.getLogger(__name__)
EPS = 1e-20
OUT_ERR_RATIO = 1e-3
ATOL, RTOL = 1e-4, 7.8125e-3


def rms_norm_golden(x: torch.Tensor, eps: float, weight: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight * x).to(dtype)


@dataclass
class GoldenCompressInputsCSA2:
    x: Any
    wkv: Any
    wgate: Any
    weight: Any
    kv_state: Any          # (bsz, ratio, head_dim) fp32, 就地更新
    score_state: Any       # (bsz, ratio, head_dim) fp32, -inf 初值
    start_pos: Any         # int
    ratio: Any
    eps: Any = 1e-20


def golden_compress_csa2(inputs: GoldenCompressInputsCSA2):
    """源码 Compressor.forward 的 golden (向量化, 逐行对应)。

    prefill(start_pos==0): 完整组 unflatten 后 softmax(dim=2) 池化, 余数 split 一次写 state
    decode (start_pos!=0): 单 token 写 state[start_pos % ratio], 组满时对 state 池化
    无压缩时返回 None, 否则返回 pre-RoPE latent。
    """
    x = inputs.x
    wkv = inputs.wkv
    wgate = inputs.wgate
    weight = inputs.weight
    kv_state = inputs.kv_state
    score_state = inputs.score_state
    start_pos = inputs.start_pos
    ratio = inputs.ratio
    eps = inputs.eps

    bsz, seqlen, _ = x.size()
    dtype = x.dtype
    assert ratio in (1, 2), "ratio must be 1 or 2"

    if ratio == 1:
        # 源码: return self.norm(self.wkv(x)) —— bf16 投影(不转 fp32), 无 gate/state,
        # 每 token 一个 latent, 忽略 start_pos
        kv = torch.matmul(x, wkv.transpose(-2, -1))          # bf16 -> bf16
        return rms_norm_golden(kv, eps, weight)

    x = x.float()
    # wkv/wgate 统一 bf16: 与 kernel (wrapper 归一 bf16) 逐位对齐
    wkv = wkv.to(torch.bfloat16).float()
    wgate = wgate.to(torch.bfloat16).float()
    kv = torch.matmul(x, wkv.transpose(-2, -1))       # (bsz, seqlen, d)
    score = torch.matmul(x, wgate.transpose(-2, -1))

    if start_pos == 0:
        # ---- prefill: 池化完整组, 尾部余数存 state ----
        should_compress = seqlen >= ratio
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        if remainder:
            kv, kv_state[:bsz, :remainder] = kv.split([cutoff, remainder], dim=1)
            score, score_state[:bsz, :remainder] = score.split([cutoff, remainder], dim=1)
        kv = kv.unflatten(1, (-1, ratio))                    # (bsz, n_groups, ratio, d)
        score = score.unflatten(1, (-1, ratio))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)          # (bsz, n_groups, d)
    else:
        # ---- decode: 每步 1 个 token, 填槽位, 组满时池化 ----
        assert seqlen == 1, "decode branch requires exactly 1 token per step"
        should_compress = (start_pos + 1) % ratio == 0
        slot = start_pos % ratio
        kv_state[:bsz, slot] = kv.squeeze(1)
        score_state[:bsz, slot] = score.squeeze(1)
        if should_compress:
            kv = (kv_state[:bsz] * score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)

    if not should_compress:
        return None
    # 源码: return self.norm(kv.to(dtype)) —— 池化结果先落回 bf16 再 RMSNorm (pre-RoPE)
    return rms_norm_golden(kv.to(dtype), eps, weight)


def gen_inputs_csa2(
    bsz: int,
    seq: int,
    h: int,
    d: int,
    ratio: int,
    device: str,
):
    """生成输入: x / wkv / wgate / weight / kv_state(0) / score_state(-inf)

    wkv/wgate 统一 bf16 (ratio1/2 同规格); ratio == 1 时源码没有 wgate/kv_state/
    score_state, 对应位置返回 None。
    """
    torch.manual_seed(42)
    x = torch.rand((bsz, seq, h), dtype=torch.bfloat16, device=device)
    weight = torch.rand((d,), dtype=torch.float32, device=device) + 0.5
    if ratio == 1:
        wkv = torch.rand((d, h), dtype=torch.bfloat16, device=device)
        return (x, wkv, None, weight, None, None)

    # 保持 fp32 抽取再量化 bf16: 与统一 bf16 前的测试数据同值 (rand 的 dtype 会改变
    # 随机流), 使对比只反映 dtype 契约而非数据实现差异
    wkv = torch.rand((d, h), dtype=torch.float32, device=device).to(torch.bfloat16)
    wgate = torch.rand((d, h), dtype=torch.float32, device=device).to(torch.bfloat16)
    kv_state = torch.zeros((bsz, ratio, d), dtype=torch.float32, device=device)
    score_state = torch.full((bsz, ratio, d), float("-inf"), dtype=torch.float32, device=device)
    return (x, wkv, wgate, weight, kv_state, score_state)


def _assert_tensor(name, got, ref, rtol, atol, max_error_ratio=None):
    got_np = got.cpu().float().numpy()
    ref_np = ref.cpu().float().numpy()
    finite = np.isfinite(got_np) & np.isfinite(ref_np)
    if finite.any():
        diff = np.abs(got_np[finite] - ref_np[finite])
        ref_abs_mean = np.abs(ref_np[finite]).mean()
    else:
        diff = np.zeros(1)
        ref_abs_mean = 0.0
    logger.info("  [%s] shape=%s max_abs=%.6e mean_abs=%.6e |ref|mean=%.6f",
                name, tuple(got.shape), diff.max(), diff.mean(), ref_abs_mean)
    if max_error_ratio is None:
        assert_allclose(got_np, ref_np, rtol=rtol, atol=atol)
        return
    # "紧 rtol + 超阈元素比例阈值" (与 test_CSA2_compressor_fwd_gen.py /
    # test_CSA2_compressor_entry_gen.py 同一口径, 参考 common_utils/compare.py):
    # kernel 与 golden 的 fp32 前像存在累加顺序级差异, 偶发翻转 bf16(latent) 的舍入
    # 边界 → 个别元素 1-2 ulp 差; 该翻转是逐元素概率事件, 与规模/形状无关.
    with np.errstate(invalid="ignore"):
        diff_all = np.abs(got_np - ref_np)
    tol = atol + rtol * np.abs(ref_np)
    bad = finite & (diff_all > tol)
    bad |= ~finite & ~((got_np == ref_np) | (np.isnan(got_np) & np.isnan(ref_np)))
    n_bad = int(bad.sum())
    n_thr = round(max_error_ratio * ref_np.size)
    if n_bad:
        logger.info("  [%s] 越限元素 %s/%s (阈值 %s)", name, n_bad, ref_np.size, n_thr)
    assert n_bad <= n_thr, (
        f"{name}: 越限元素 {n_bad} > 阈值 {n_thr} "
        f"(max_error_ratio={max_error_ratio}x{ref_np.size}), max_diff={diff.max():.3e}")



def _compressor_prep():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch_npu.npu.config.allow_internal_format = True
    return f"npu:{device_id}"


def rms_norm_golden_backward(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float):
    """dy: dL/dy; x: RMSNorm 输入; 返回 (dx, dweight)

    dx (dlatent/dkv) 是反向内部量, 保持 fp32 返回 —— kernel 侧 (rms_norm_backward_2d)
    全程 fp32, 量化点在尾段 cube matmul 操作数 (见 golden_compress_csa2_backward 尾段
    注释)。此处若按前向输出 dtype 回 cast bf16, dlatent 幅值大时 (decode 随机 state
    → rms 小 → dlatent ~O(1)) 会引入 ~2^-9 相对差而 kernel 无此量化。ratio==1 分支
    在调用点显式量化 (对齐 kernel 的 dkv→bf16 cast 点)。
    """
    x_fp = x.float()
    w_fp = weight.float()
    hidden_dim = x_fp.shape[-1]
    var = x_fp.square().mean(-1, keepdim=True)
    rms = torch.rsqrt(var + eps)
    dy_fp = dy.float()

    d_y = dy_fp * w_fp
    term1 = d_y * rms
    sum_term = torch.sum(d_y * x_fp, dim=-1, keepdim=True)
    term2 = x_fp * sum_term * (rms ** 3) / hidden_dim
    dx_fp = term1 - term2
    dweight = torch.sum(dy_fp * x_fp * rms, dim=list(range(dy_fp.dim() - 1)))
    return dx_fp, dweight.to(weight.dtype)


def softmax_golden(x: torch.Tensor, dim: int):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def softmax_golden_backward(dy: torch.Tensor, y: torch.Tensor, dim: int):
    """y = softmax(x, dim); dy = dL/dy -> dL/dx"""
    d = dy * y
    sum_d = torch.sum(d, dim=dim, keepdim=True)
    return y * (dy - sum_d)


def golden_compress_csa2_backward(inputs: GoldenCompressInputsCSA2, d_out: torch.Tensor):
    """CSA2 compressor 解析反向 (与前向三分支一一对应).

    d_out: dL/dout, 形状与前向输出一致
    返回 (dx, dwkv, dwgate, dweight); ratio == 1 时 dwgate 为 None

    反向 cube matmul 操作数量化约定 (对齐 kernel bf16-MAC 形态, 见函数尾段注释):
    dx / dwkv / dwgate 的操作数在 matmul 边界量化 bf16, fp32 累加, dx 单次最终
    舍入 —— 与 kernel 的声明语义一致 (比率 1 分支天然对齐, 比率 2 显式镜像)。
    """
    x = inputs.x
    wkv = inputs.wkv
    wgate = inputs.wgate
    weight = inputs.weight
    kv_state = inputs.kv_state
    score_state = inputs.score_state
    start_pos = inputs.start_pos
    ratio = inputs.ratio
    eps = inputs.eps

    bsz, seqlen, h = x.shape
    d = weight.shape[0]
    dtype = x.dtype
    assert ratio >= 1, "ratio must be >= 1"

    if ratio == 1:
        # 前向: out = norm(wkv(x)) —— 投影结果本身即 norm 输入(bf16 路径下为 bf16)
        kv = torch.matmul(x, wkv.transpose(-2, -1))
        dkv, dweight = rms_norm_golden_backward(d_out, kv, weight, eps)
        # kernel 侧 dkv 在 cube matmul 边界量化 bf16 (Phase C dx 与 D 段 scratch 同点)
        dkv = dkv.to(dtype)
        dx = torch.matmul(dkv, wkv.float())
        dwkv = torch.matmul(dkv.reshape(-1, d).transpose(-2, -1), x.float().reshape(-1, h))
        return dx.to(dtype), dwkv.to(wkv.dtype), None, dweight.to(weight.dtype)

    x_fp32 = x.float()
    # kv/score 重算对齐 kernel gb 段 bf16-MAC 语义: wkv/wgate 统一 bf16 (测试直接生成
    # bf16 权重, .to(bf16) 为 no-op), x 原生 bf16 (float() 精确); bf16×bf16 乘积在 fp32
    # 精确, 仅累加顺序与 kernel tile 分组不同 (容差内)
    wkv_b = wkv.to(torch.bfloat16).float()
    wgate_b = wgate.to(torch.bfloat16).float()
    kv = torch.matmul(x_fp32, wkv_b.transpose(-2, -1))           # (bsz, seqlen, d)
    score = torch.matmul(x_fp32, wgate_b.transpose(-2, -1))

    dkv = torch.zeros_like(kv)
    dscore = torch.zeros_like(score)

    if start_pos == 0:
        # ---- prefill: 完整组可微; 余数 token 无输出 → 梯度 0 ----
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        if cutoff > 0:
            n_groups = cutoff // ratio
            gkv = kv[:, :cutoff].reshape(bsz, n_groups, ratio, d)
            gsc = score[:, :cutoff].reshape(bsz, n_groups, ratio, d)
            alpha = gsc.softmax(dim=2, dtype=torch.float32)
            latent = (gkv * alpha).sum(dim=2)                   # (bsz, n_groups, d)
            dlatent, dweight = rms_norm_golden_backward(d_out, latent.to(dtype), weight, eps)
            dlatent4 = dlatent.unsqueeze(2)
            d_alpha = dlatent4 * gkv
            d_gkv = dlatent4 * alpha
            d_gsc = softmax_golden_backward(d_alpha, alpha, dim=2)
            dkv[:, :cutoff] = d_gkv.reshape(bsz, cutoff, d)
            dscore[:, :cutoff] = d_gsc.reshape(bsz, cutoff, d)
        else:
            dweight = torch.zeros(d, dtype=weight.dtype, device=weight.device)
    else:
        # ---- decode: latent 由 state 池化, 仅当前 token 写入的槽位可微 ----
        assert seqlen == 1, "decode branch requires exactly 1 token per step"
        slot = start_pos % ratio
        alpha = score_state[:bsz].softmax(dim=1, dtype=torch.float32)   # (bsz, ratio, d)
        latent = (kv_state[:bsz] * alpha).sum(dim=1, keepdim=True)      # (bsz, 1, d)
        dlatent, dweight = rms_norm_golden_backward(d_out, latent.to(dtype), weight, eps)
        d_alpha = dlatent * kv_state[:bsz]
        d_kv_slot = dlatent[:, 0] * alpha[:, slot]                      # (bsz, d)
        d_sc_slot = softmax_golden_backward(d_alpha, alpha, dim=1)[:, slot]
        dkv[:, 0] = d_kv_slot
        dscore[:, 0] = d_sc_slot

    # ---- 反向 cube matmul 操作数量化约定 (对齐 kernel bf16-MAC 形态) ----
    # kernel 的 dx / dwkv / dwgate cube matmul 要求 A/B 同 dtype (bf16), fp32 累加:
    #   dx 段: bf16(d_kv) @ bf16(wkv) + bf16(d_sc) @ bf16(wgate) (单次最终舍入)
    #   dw 段: bf16(d_kv)/bf16(d_sc) @ x(原生 bf16, 值精确) fp32 累加
    #   gb 段: kv/score 重算用单次量化 bf16 权值 (见函数头部, 同一口径)
    # golden 在相同量化点镜像: 操作数先量化 bf16 再以 fp32 精确矩阵乘 (乘积精确 +
    # fp32 累加, 仅累加顺序与 kernel tile 分组不同)。wkv/wgate 的量化值与 kernel
    # wrapper 的 wkv.to(bf16) 逐位一致 (同一 torch 调用), 对比仍能捕获 kernel 的
    # 结构/逻辑错误; 度量的是 "kernel 是否正确实现既定 bf16 语义", 不惩罚声明为
    # 语义的操作数量化。ratio==1 分支天然已对齐 (dkv 由 rms_norm_golden_backward
    # 直接返回 bf16, wkv 原生 bf16, x.float() 为精确值)。
    dkv_b = dkv.to(dtype).float()
    dscore_b = dscore.to(dtype).float()
    dx = torch.matmul(dkv_b, wkv_b) + torch.matmul(dscore_b, wgate_b)
    # dwkv/dwgate 需对 batch 与 seq 同时求和 → 先展平 (batch*seq, ·)
    dwkv = torch.matmul(dkv_b.reshape(-1, d).transpose(-2, -1), x_fp32.reshape(-1, h))
    dwgate = torch.matmul(dscore_b.reshape(-1, d).transpose(-2, -1), x_fp32.reshape(-1, h))
    return (dx.to(dtype), dwkv.to(wkv.dtype), dwgate.to(wgate.dtype), dweight.to(weight.dtype))


def _make_inputs(x, wkv, wgate, weight, kv_state, score_state, start_pos, ratio, eps):
    return GoldenCompressInputsCSA2(
        x=x, wkv=wkv, wgate=wgate, weight=weight,
        kv_state=kv_state, score_state=score_state,
        start_pos=start_pos, ratio=ratio, eps=eps)


def _assert_bwd(name, got, ref, rtol, atol, max_error_ratio=None):
    """max_error_ratio=None: 逐元素全量校验; =r: 允许 ≤ round(r*numel) 个元素超阈
    (参考 common_utils/compare.py, test_sparse_flash_attention_quant.py 同款)。"""
    got_np = got.cpu().float().numpy()
    ref_np = ref.cpu().float().numpy()
    diff = np.abs(got_np - ref_np)
    logger.info("  [%s] shape=%s max_abs=%.6e mean_abs=%.6e |ref|mean=%.6f",
                name, tuple(got.shape), diff.max(), diff.mean(), np.abs(ref_np).mean())
    if max_error_ratio is None:
        assert_allclose(got_np, ref_np, rtol=rtol, atol=atol)
        return
    tol = atol + rtol * np.abs(ref_np)
    bad = diff > tol
    n_bad, n_thr = int(bad.sum()), round(max_error_ratio * ref_np.size)
    if n_bad:
        logger.info("  [%s] 越限元素 %s/%s (阈值 %s), 前 10 个:",
                    name, n_bad, ref_np.size, n_thr)
        for i in np.argwhere(bad)[:10]:
            i = tuple(int(v) for v in i)
            logger.info("    %s: got=%.8f ref=%.8f diff=%.3e thr=%.3e",
                        i, got_np[i], ref_np[i], diff[i], tol[i])
    assert n_bad <= n_thr, (
        f"{name}: 越限元素 {n_bad} > 阈值 {n_thr} "
        f"(max_error_ratio={max_error_ratio}x{ref_np.size}), max_diff={diff.max():.3e}")

def _assert_bwd_all(dx, gdx, dwkv, gdwkv, dwgate, gdwgate, dweight, gdweight,
                    atol=1e-3, atol_dw=None, atol_dx=None):
    """精度基线 (dtype 键控): bf16 (dx / dwkv) atol=1e-4 rtol=0.0078125;
    ratio2 dwkv/dwgate (bf16): rtol=0.0078125 + atol 按 n_tok 档。

    文档化豁免 (累积类, 物理上不满足基线, 机制见各注释):
      * dx (bf16): 稀疏翻转类 → 基线 + 越限比例阈值 1e-3 (参考 common_utils/compare.py);
        小 h 冒烟维度 (h=64, softmax alpha 未饱和敏感性, 越限率 0.5% 超比例阈值) 另经
        atol_dx 放宽
      * ratio2 dwkv/dwgate (bf16): 主误差为 fp32 累加重结合随 n_tok 增长 (128K 实测
        越限 1.5%/12%) → atol 按 n_tok 档放宽 (调用方传 atol: 1e-3 / >=65536 取 4e-3);
        bf16 末次舍入另叠 ≤1 ulp, 由 rtol=0.0078125 覆盖
      * dweight (fp32): bf16(latent) 翻转累积随 token 数增长 → atol_dw 按 n_tok 档
        放宽 (1e-2 / 3e-2 / 6e-2; 最低档覆盖单次翻转事件 ~4-8e-3 落在小 |ref| 元素)
    """
    if atol_dw is None:
        atol_dw = 1e-2
    if atol_dx is None:
        atol_dx = 1e-4
    _assert_bwd("dx", dx, gdx, rtol=0.0078125, atol=atol_dx, max_error_ratio=1e-3)
    if gdwgate is None:
        # ratio1: dwkv 为 bf16 输出 → bf16 基线 (实测 flat 通过, 128K worst_ratio 0.98)
        _assert_bwd("dwkv", dwkv, gdwkv, rtol=0.0078125, atol=1e-4)
    else:
        # ratio2: dwkv/dwgate 对外 dtype 随 wkv/wgate 变成 bf16 (输出分辨率 1 ulp =
        # 0.78125%), 故 rtol 必须取 bf16 ulp —— 合法的 1-ulp 差会超原 fp32 口径 0.005
        # (decode 实测 max_abs 0.03125 / rel 0.0056)。atol 沿袭原 n_tok 档 (累加顺序差)。
        _assert_bwd("dwkv", dwkv, gdwkv, rtol=0.0078125, atol=atol)
        _assert_bwd("dwgate", dwgate, gdwgate, rtol=0.0078125, atol=atol)
    # dweight = Σ dy·x·rms, x=bf16(pooled latent): 量化边界翻转随 token 数累积
    _assert_bwd("dweight", dweight, gdweight, rtol=0.005, atol=atol_dw)


def _run_entry_case(ra, bsz, seq, h, d, start_pos=0, use_module=True, atol_dx=None):
    """公共入口 forward + backward vs golden (前向三分支共用)."""
    device = _compressor_prep()
    x, wkv, wgate, weight, kv_state, score_state = gen_inputs_csa2(
        bsz, seq, h, d, ra, device)
    sp = torch.tensor([start_pos], dtype=torch.int32, device=device)

    # golden 用独立 state 副本, 避免被 kernel 原地更新污染
    g_kv = kv_state.clone() if kv_state is not None else None
    g_sc = score_state.clone() if score_state is not None else None

    for t in (x, wkv, wgate, weight):
        if t is not None:
            t.requires_grad_(True)

    if use_module:
        model = CSA2Compressor(ratio=ra, eps=EPS)
        out = model(x, wkv, wgate, weight, kv_state, score_state, sp)
    else:
        out = csa2_compressor(
            x, wkv, wgate, weight, kv_state, score_state, sp, ratio=ra, eps=EPS)
    torch.npu.synchronize()

    logger.info("=== entry ratio=%s bsz=%s seq=%s d=%s start_pos=%s ===",
                ra, bsz, seq, d, start_pos)

    # ---- 前向 vs golden (golden 前向同时推进其 state 副本) ----
    g_inputs = _make_inputs(
        x.detach(), wkv.detach(),
        wgate.detach() if wgate is not None else None, weight.detach(),
        g_kv, g_sc, start_pos, ra, EPS)
    ref_out = golden_compress_csa2(g_inputs)
    if ref_out is not None:
        _assert_tensor("out", out.detach(), ref_out, rtol=RTOL, atol=ATOL,
                       max_error_ratio=OUT_ERR_RATIO)

    # ---- 反向 vs golden ----
    grad_out = torch.randn(out.shape, dtype=torch.bfloat16, device=device)
    out.backward(grad_out)
    torch.npu.synchronize()

    gdx, gdwkv, gdwgate, gdweight = golden_compress_csa2_backward(g_inputs, grad_out)
    _assert_bwd_all(
        x.grad, gdx, wkv.grad, gdwkv,
        wgate.grad if wgate is not None else None, gdwgate,
        weight.grad, gdweight, atol_dx=atol_dx)
    logger.info("  PASS")


@pytest.mark.soc("910")
def test_csa2_entry_ratio2_prefill():
    """ratio == 2 prefill: seq % ratio == 0, 完整组可微"""
    _run_entry_case(ra=2, bsz=64, seq=2, h=5120, d=512, start_pos=0)


if __name__ == "__main__":
    test_csa2_entry_ratio2_prefill()
    logger.info("PASS")

