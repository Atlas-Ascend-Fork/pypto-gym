# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Golden reference for block_attn_res_update_rms_norm.

Reference arithmetic and input distributions are aligned with the AscendC
ops-transformer spec.py: FTZ on div/sqrt/exp_sub boundaries, single-rounding
FMA, explicit BF16 RNE, and the e2e input intervals.
"""

import logging
from typing import NamedTuple

import numpy as np
import torch

_LOG = logging.getLogger(__name__)

_FP32_TINY = float(np.finfo(np.float32).tiny)


# ---------------- helpers copied from the AscendC spec.py (aligned) ----------------
def _round_to_bfloat16_float32(value):
    value = np.ascontiguousarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    nan_mask = np.isnan(value)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded_bits = (bits + rounding_bias) & np.uint32(0xFFFF0000)
    rounded_bits = np.where(
        nan_mask,
        (bits & np.uint32(0xFFFF0000)) | np.uint32(0x00010000),
        rounded_bits,
    ).astype(np.uint32, copy=False)
    return rounded_bits.view(np.float32)


def _ftz_float32(value):
    value = np.asarray(value, dtype=np.float32)
    signed_zero = np.copysign(np.float32(0.0), value)
    return np.where(np.abs(value) < _FP32_TINY, signed_zero, value).astype(np.float32, copy=False)


def _div_ftz_float32(lhs, rhs):
    lhs = _ftz_float32(lhs)
    rhs = _ftz_float32(rhs)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        quotient = np.divide(lhs, rhs)
    return _ftz_float32(quotient)


def _sqrt_ftz_float32(value):
    value = _ftz_float32(value)
    with np.errstate(invalid="ignore", under="ignore"):
        result = np.sqrt(value)
    return _ftz_float32(result)


def _exp_sub_ftz_float32(lhs, rhs):
    lhs = _ftz_float32(lhs)
    rhs = _ftz_float32(rhs)
    difference = _ftz_float32(lhs - rhs)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.exp(difference).astype(np.float32, copy=False)
    return _ftz_float32(result)


def _fma_float32(dst, src0, src1):
    dst = np.asarray(dst, dtype=np.float32)
    src0 = np.asarray(src0, dtype=np.float32)
    src1 = np.asarray(src1, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = dst.astype(np.float64) * src0.astype(np.float64) + src1.astype(np.float64)
    return result.astype(np.float32, copy=False)


# ---------------- golden ----------------
def block_attn_res_update_rms_norm_golden(
    partial_block: torch.Tensor,
    delta: torch.Tensor,
    pseudo_query: torch.Tensor,
    numerator: torch.Tensor,
    logit_max: torch.Tensor,
    exp_sum: torch.Tensor,
    gamma: torch.Tensor,
    score_eps: float = 1e-6,
    norm_eps: float = 1e-6,
):
    """Returns (p_updated_partial, y). partial_block is NOT mutated here.

    Update part mirrors the AscendC numpy golden (FTZ/FMA/BF16-RNE boundaries);
    the RMSNorm epilogue consumes the rounded h in FP32 like the AscendC spec.
    """
    partial = np.asarray(partial_block.cpu(), dtype=np.float32)
    delta_f = delta.cpu().to(torch.float32).numpy()
    query = np.asarray(pseudo_query.cpu(), dtype=np.float32)
    num = np.asarray(numerator.cpu(), dtype=np.float32)
    lmax = np.asarray(logit_max.cpu(), dtype=np.float32)
    esum = np.asarray(exp_sum.cpu(), dtype=np.float32)
    gam = gamma.cpu().to(torch.float32).numpy()

    partial_out = np.add(partial, delta_f).astype(np.float32, copy=False)

    square_sum = np.sum(partial_out * partial_out, axis=-1, dtype=np.float32)
    dot_sum = np.sum(partial_out * query, axis=-1, dtype=np.float32)
    inv_d = np.float32(1.0 / partial_out.shape[-1])
    rms = _sqrt_ftz_float32(square_sum * inv_d + np.float32(score_eps))
    score = _div_ftz_float32(dot_sum, rms)

    current_max = np.maximum(lmax, score)
    alpha = _exp_sub_ftz_float32(lmax, current_max)
    beta = _exp_sub_ftz_float32(score, current_max)
    denominator = _fma_float32(esum, alpha, beta)
    inv_denominator = _div_ftz_float32(np.ones_like(denominator), denominator)
    alpha = _ftz_float32(alpha * inv_denominator)
    beta = _ftz_float32(beta * inv_denominator)

    partial_scaled = partial_out * beta[:, None]
    h_fp32 = _fma_float32(num, alpha[:, None], partial_scaled)
    h = _round_to_bfloat16_float32(h_fp32)

    # RMSNorm epilogue on the rounded h (AscendC spec's _assert_results form)
    h32 = torch.from_numpy(np.ascontiguousarray(h))
    rstd = torch.rsqrt((h32 * h32).mean(-1, keepdim=True) + norm_eps)
    y = (h32 * rstd * torch.from_numpy(gam)).to(torch.bfloat16)

    p_out = torch.from_numpy(np.ascontiguousarray(partial_out))
    return p_out, y


# ---------------- inputs (e2e intervals from the AscendC spec) ----------------
class Inputs(NamedTuple):
    """Named input bundle: keeps the helper's seven tensors explicit."""

    partial_block: torch.Tensor
    delta: torch.Tensor
    pseudo_query: torch.Tensor
    numerator: torch.Tensor
    logit_max: torch.Tensor
    exp_sum: torch.Tensor
    gamma: torch.Tensor


def make_inputs(tokens, dim=7168, seed=0, device="cpu", cancellation=False, ftz=False,
                recip_ftz=False):
    torch.manual_seed(seed + 1000003)
    if cancellation:
        half = dim // 2
        x = torch.empty((tokens, half), device=device, dtype=torch.float32).uniform_(-1.0, 1.0)
        noise = torch.empty((tokens, dim), device=device, dtype=torch.float32).uniform_(-1e-3, 1e-3)
        partial = torch.cat([x, -x], dim=-1) + noise
        delta = torch.zeros(tokens, dim, device=device, dtype=torch.bfloat16)
        query = torch.ones(dim, device=device, dtype=torch.float32) * 0.25
    else:
        partial = torch.empty((tokens, dim), device=device, dtype=torch.float32).uniform_(-1.0, 1.0)
        delta = torch.empty((tokens, dim), device=device, dtype=torch.float32).uniform_(-0.25, 0.25).to(torch.bfloat16)
        query = torch.empty(dim, device=device, dtype=torch.float32).uniform_(-0.25, 0.25)
    numerator = torch.empty((tokens, dim), device=device, dtype=torch.float32).uniform_(-1.0, 1.0)
    if ftz:
        logit_max = torch.full((tokens,), 100.0, device=device, dtype=torch.float32)
        exp_sum = torch.empty((tokens,), device=device, dtype=torch.float32).uniform_(1.0, 2.0)
    elif recip_ftz:
        logit_max = torch.full((tokens,), 100.0, device=device, dtype=torch.float32)
        exp_sum = torch.full((tokens,), 1e30, device=device, dtype=torch.float32)
    else:
        logit_max = torch.empty((tokens,), device=device, dtype=torch.float32).uniform_(-0.5, 0.5)
        exp_sum = torch.empty((tokens,), device=device, dtype=torch.float32).uniform_(1.0, 2.0)
    gamma = (torch.randn(dim, device=device) * 0.5 + 0.5).to(torch.bfloat16)
    gamma[0] = 0
    return Inputs(partial, delta, query, numerator, logit_max, exp_sum, gamma)


def block_attn_res_update_rms_norm_golden_f64(
    partial_block, delta, pseudo_query, numerator, logit_max, exp_sum, gamma,
    score_eps=1e-6, norm_eps=1e-6,
):
    """Order-free float64 oracle (bf16 roundings of h and y preserved).

    Used as the reference for the ill-conditioned cancellation regression,
    where the FP32 np.sum-ordered golden itself exceeds the 1e-3 gate
    (measured: 0.68e-3 .. 1.45e-3 mismatch vs this oracle across seeds).
    """
    partial = partial_block.cpu().double()
    p = partial + delta.cpu().double()
    q = pseudo_query.cpu().double()
    dot = (p * q).sum(-1)
    ssq = (p * p).sum(-1)
    score = dot / (ssq / p.shape[-1] + score_eps).sqrt()
    m = torch.maximum(logit_max.cpu().double(), score)
    a = (logit_max.cpu().double() - m).exp()
    b = (score - m).exp()
    inv = 1.0 / (exp_sum.cpu().double() * a + b)
    alpha, beta = a * inv, b * inv
    h = (numerator.cpu().double() * alpha[:, None] + p * beta[:, None]).to(torch.bfloat16).double()
    rstd = 1.0 / ((h * h).sum(-1) / h.shape[-1] + norm_eps).sqrt()
    y = (h * rstd[:, None] * gamma.cpu().double()).to(torch.bfloat16)
    return p.float(), y


def _validate():
    """Self-check without an NPU: the FP32 reference must satisfy its own contract
    against the independent float64 oracle, and p must equal partial + fp32(delta)."""
    for t in (1, 3, 8):
        args = make_inputs(t, seed=7, device="cpu")
        p_out, y = block_attn_res_update_rms_norm_golden(*args)
        _, y64 = block_attn_res_update_rms_norm_golden_f64(*args)
        if tuple(y.shape) != (t, args[0].shape[1]) or y.dtype != torch.bfloat16:
            _LOG.error("T=%d FAIL shape/dtype %s %s", t, tuple(y.shape), y.dtype)
            return False
        if tuple(p_out.shape) != (t, args[0].shape[1]) or p_out.dtype != torch.float32:
            _LOG.error("T=%d FAIL p shape/dtype %s %s", t, tuple(p_out.shape), p_out.dtype)
            return False
        expect_p = (args[0] + args[1].float()).float()
        if not torch.equal(p_out, expect_p):
            _LOG.error("T=%d FAIL p != partial + fp32(delta)", t)
            return False
        a = y.float().numpy()
        b = y64.float().numpy()
        bad = int((~np.isclose(a, b, rtol=1e-3, atol=0.0)).sum())
        ratio = bad / a.size
        if not (np.isfinite(a).all() and ratio <= 1e-3):
            _LOG.error("T=%d FAIL vs f64 oracle: ratio=%.3e bad=%d", t, ratio, bad)
            return False
        _LOG.info("T=%d PASS shape=%s dtype=%s p=exact vs-oracle-mismatch=%.1e",
                  t, tuple(y.shape), y.dtype, ratio)
    _LOG.info("golden self-check: PASS")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(0 if _validate() else 1)
