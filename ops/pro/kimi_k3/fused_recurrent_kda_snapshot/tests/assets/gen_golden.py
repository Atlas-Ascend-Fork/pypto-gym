#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reference for fused_recurrent_kda_snapshot, with CPU and NPU modes.

Per (batch, head) token recurrence over a `[Dv, Dk]` state with per-token
snapshots written back into the state pool, Q/K row normalization, gate/beta
activations, and BSND / BNSD / TND packed-QKV storage.

The math has exactly one implementation (torch, normalization and activations
outside the token loop, batch/head vectorized). NumPy inputs run it on CPU in
FP64 and get NumPy back; torch inputs run it on their device in FP32. CPU mode
is the independent precision reference and NPU mode is the performance
baseline.
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

import numpy as np

_LOG = logging.getLogger(__name__)

SUPPORTED_HEAD_DIM = 128
MAX_VALUE_HEADS = 96


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bf16_round(x):
    """Round an FP32 array to the nearest BF16-representable value (RNE)."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    lsb = (u >> 16) & np.uint32(1)
    rounded = u + np.uint32(0x7FFF) + lsb
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _qkv_bsnd(rng, b, s, n, d, dtype):
    p = n * d
    mixedqkv = bf16_round(rng.standard_normal((b, s, 3 * p)).astype(dtype))
    g = bf16_round(rng.standard_normal((b, s, n, d)).astype(dtype))
    beta = bf16_round(rng.standard_normal((b, s, n, 1)).astype(dtype))
    return mixedqkv, g, beta, b * s, None


def _qkv_bnsd(rng, b, s, n, d, dtype):
    p = n * d
    mixedqkv = bf16_round(rng.standard_normal((b, 3 * n, s, d)).astype(dtype))
    g = bf16_round(rng.standard_normal((b, n, s, d)).astype(dtype))
    beta = bf16_round(rng.standard_normal((b, n, s, 1)).astype(dtype))
    return mixedqkv, g, beta, b * s, None


def _qkv_tnd(rng, b, s, n, d, cu_seqlens, dtype):
    p = n * d
    seqlens = cu_seqlens if cu_seqlens is not None else [s] * b
    cu = np.zeros(b + 1, dtype=np.int32)
    for i, length in enumerate(seqlens):
        cu[i + 1] = cu[i] + length
    token_count = int(cu[-1])
    mixedqkv = bf16_round(rng.standard_normal((token_count, 3 * p)).astype(dtype))
    g = bf16_round(rng.standard_normal((token_count, n, d)).astype(dtype))
    beta = bf16_round(rng.standard_normal((token_count, n, 1)).astype(dtype))
    return mixedqkv, g, beta, token_count, cu


def _qkv_inputs(rng, layout, b, s, n, d, cu_seqlens, dtype):
    if layout == "BSND":
        return _qkv_bsnd(rng, b, s, n, d, dtype)
    if layout == "BNSD":
        return _qkv_bnsd(rng, b, s, n, d, dtype)
    if layout == "TND":
        return _qkv_tnd(rng, b, s, n, d, cu_seqlens, dtype)
    raise ValueError(layout)


def _make_state(rng, pool_size, n, d, state_dtype):
    state = rng.standard_normal((pool_size, n, d, d)).astype(np.float64) * 0.5
    if state_dtype == "bf16":
        return bf16_round(state.astype(np.float32)).astype(np.float64)
    return state.astype(np.float32).astype(np.float64)


def make_inputs(
    batch=2,
    seq=8,
    heads=6,
    dim=SUPPORTED_HEAD_DIM,
    seed=0,
    layout="BSND",
    state_dtype=np.float32,
    pool=None,
    state_indices=None,
    accepted=None,
    cu_seqlens=None,
    scale=None,
    lower_bound=-5.0,
    dtype=np.float32,
    device=None,
):
    rng = np.random.default_rng(seed)
    mixedqkv, g, beta, token_count, cu_out = _qkv_inputs(
        rng, layout, batch, seq, heads, dim, cu_seqlens, dtype)
    pool_size = token_count if pool is None else pool
    state = _make_state(rng, pool_size, heads, dim, state_dtype)
    a_log = rng.standard_normal(heads).astype(np.float32) * 0.5
    dt_bias = rng.standard_normal((heads, dim)).astype(np.float32) * 0.1
    if state_indices is None:
        state_indices = np.arange(token_count, dtype=np.int32)
    if accepted is None:
        accepted = np.ones(batch, dtype=np.int32)
    inp = {
        "mixedqkv": mixedqkv,
        "state": state,
        "beta": beta,
        "g": g,
        "scale": scale,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "lower_bound": lower_bound,
        "layout": layout,
        "cu_seqlens": cu_out,
        "ssm_state_indices": np.asarray(state_indices, dtype=np.int32),
        "num_accepted_tokens": np.asarray(accepted, dtype=np.int32),
        "batch": batch,
        "seq": seq,
        "heads": heads,
        "dim": dim,
        "token_count": token_count,
    }
    if device is None:
        return inp
    return _to_torch(inp, device, state_dtype)


def _layout_views(inp):
    """Q/K/V, gate and beta views, with dense layouts normalized to [B, S, N, D]."""
    layout = inp["layout"]
    b, s, n, d = inp["batch"], inp["seq"], inp["heads"], inp["dim"]
    if layout == "BSND":
        qkv = inp["mixedqkv"].reshape(b, s, 3, n, d)
        q_raw, k_raw, v_raw = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        g_view = inp["g"].reshape(b, s, n, d)
        beta_view = inp["beta"].reshape(b, s, n, 1)
    elif layout == "BNSD":
        qkv = inp["mixedqkv"].reshape(b, 3, n, s, d)
        q_raw = qkv[:, 0].transpose(1, 2)
        k_raw = qkv[:, 1].transpose(1, 2)
        v_raw = qkv[:, 2].transpose(1, 2)
        g_view = inp["g"].reshape(b, n, s, d).transpose(1, 2)
        beta_view = inp["beta"].reshape(b, n, s, 1).transpose(1, 2)
    else:
        qkv = inp["mixedqkv"].reshape(-1, 3, n, d)
        q_raw, k_raw, v_raw = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        g_view = inp["g"].reshape(-1, n, d)
        beta_view = inp["beta"].reshape(-1, n, 1)
    return q_raw, k_raw, v_raw, g_view, beta_view


# ---------------------------------------------------------------------------
# Reference: one torch implementation, CPU or NPU by input device
# ---------------------------------------------------------------------------

def _to_torch(inp, device, state_dtype=np.float32):
    """Convert the NumPy input dict to torch tensors on ``device``."""
    import torch
    dev = torch.device(device)
    on_cpu = dev.type == "cpu"
    out = dict(inp)

    def as_tensor(name, bf16=False):
        value = torch.from_numpy(np.ascontiguousarray(inp[name])).to(dev)
        if bf16 and not on_cpu:
            value = value.to(torch.bfloat16)
        return value

    out["mixedqkv"] = as_tensor("mixedqkv", bf16=True)
    out["g"] = as_tensor("g", bf16=True)
    out["beta"] = as_tensor("beta", bf16=True)
    state = torch.from_numpy(inp["state"])
    if on_cpu:
        state = state.clone()          # keep the caller's NumPy input untouched
    else:
        state = torch.from_numpy(inp["state"].astype(np.float32))
        if state_dtype == "bf16":
            state = state.to(torch.bfloat16)
    out["state"] = state.to(dev)
    out["A_log"] = as_tensor("A_log")
    out["dt_bias"] = as_tensor("dt_bias")
    out["ssm_state_indices"] = torch.from_numpy(
        np.ascontiguousarray(inp["ssm_state_indices"])).to(torch.int32).to(dev)
    out["num_accepted_tokens"] = torch.from_numpy(
        np.ascontiguousarray(inp["num_accepted_tokens"])).to(torch.int32).to(dev)
    if inp["layout"] == "TND":
        out["cu_seqlens"] = torch.from_numpy(
            np.ascontiguousarray(inp["cu_seqlens"])).to(torch.int32).to(dev)
    else:
        out["cu_seqlens"] = None
    return out


def reference(inp):
    """Return (updated_state, out); NumPy inputs run on CPU, torch inputs on device."""
    torch_in = type(inp["state"]).__module__.split(".")[0] == "torch"
    if not torch_in:
        inp = _to_torch(inp, "cpu")
    state, out = _reference_impl(inp)
    if torch_in:
        return state, out
    return state.numpy(), out.numpy()


def _reference_impl(inp):
    import torch
    on_cpu = inp["state"].device.type == "cpu"
    calc_dtype = torch.float64 if on_cpu else torch.float32
    out_dtype = calc_dtype if on_cpu else torch.bfloat16
    if inp["layout"] == "TND":
        return _reference_impl_tnd(inp, calc_dtype, out_dtype)
    return _reference_impl_dense(inp, calc_dtype, out_dtype)


class DenseViews(NamedTuple):
    layout: str
    device: object
    batch: int
    seq: int
    heads: int
    out: object
    q_raw: object
    k_raw: object
    v_raw: object
    g_view: object
    beta_view: object


def _dense_views(inp, out_dtype):
    import torch
    layout = inp["layout"]
    device = inp["mixedqkv"].device
    q_raw, k_raw, v_raw, g_view, beta_view = _layout_views(inp)
    if layout == "BSND":
        batch, seq, _ = inp["mixedqkv"].shape
        heads = inp["g"].shape[2]
        out = torch.empty((batch, seq, heads, inp["dim"]), dtype=out_dtype,
                          device=device)
    else:
        batch, _, seq, _ = inp["mixedqkv"].shape
        heads = inp["mixedqkv"].shape[1] // 3
        out = torch.empty((batch, heads, seq, inp["dim"]), dtype=out_dtype,
                          device=device)
    return DenseViews(layout, device, batch, seq, heads, out,
                      q_raw, k_raw, v_raw, g_view, beta_view)


def _reference_impl_dense(inp, calc_dtype, out_dtype):
    import torch
    views = _dense_views(inp, out_dtype)
    layout, device, batch, seq, heads, out = views[:6]
    q_raw, k_raw, v_raw, g_view, beta_view = views[6:]

    dim = inp["dim"]
    alpha = torch.exp(inp["A_log"].to(calc_dtype))
    q_norm = q_raw.to(calc_dtype)
    q_norm = q_norm / torch.sqrt((q_norm * q_norm).sum(dim=-1, keepdim=True) + 1e-6)
    k_norm = k_raw.to(calc_dtype)
    k_norm = k_norm / torch.sqrt((k_norm * k_norm).sum(dim=-1, keepdim=True) + 1e-6)
    beta_v = torch.sigmoid(beta_view.to(calc_dtype))
    gate = float(inp["lower_bound"]) * torch.sigmoid(
        alpha.view(1, 1, heads, 1)
        * (g_view.to(calc_dtype) + inp["dt_bias"].to(calc_dtype).view(1, 1, heads, dim)))
    decay = torch.exp(gate)
    scale = inp["scale"]
    scale_value = 1.0 / math.sqrt(dim) if scale is None else float(scale)

    state = inp["state"]
    indices = inp["ssm_state_indices"].to(torch.int64)
    accepted = inp["num_accepted_tokens"].to(torch.int64)
    base = torch.arange(batch, device=device, dtype=torch.int64) * seq
    init_token = base + (accepted - 1).clamp(min=0)
    init_slot = indices[init_token]
    work = state[init_slot].to(calc_dtype)
    tokens = base.view(batch, 1) + torch.arange(seq, device=device).view(1, seq)

    for t in range(seq):
        dec_t = decay[:, t]
        k_t = k_norm[:, t]
        q_t = q_norm[:, t]
        v_t = v_raw[:, t].to(calc_dtype)
        beta_t = beta_v[:, t]
        state_bar = work * dec_t.unsqueeze(-2)
        u = beta_t * (v_t - (state_bar @ k_t.unsqueeze(-1)).squeeze(-1))
        work = state_bar + u.unsqueeze(-1) * k_t.unsqueeze(-2)
        o = scale_value * (work @ q_t.unsqueeze(-1)).squeeze(-1)
        state.index_copy_(0, indices[tokens[:, t]], work.to(state.dtype))
        if layout == "BSND":
            out[:, t] = o.to(out_dtype)
        else:
            out[:, :, t] = o.to(out_dtype)
    return state, out


def _reference_impl_tnd(inp, calc_dtype, out_dtype):
    import torch
    device = inp["mixedqkv"].device
    q_raw, k_raw, v_raw, g_view, beta_view = _layout_views(inp)
    token_count, heads, dim = inp["g"].shape
    cu = inp["cu_seqlens"].to(torch.int64).tolist()
    idx_host = inp["ssm_state_indices"].to(torch.int64).tolist()
    acc_host = inp["num_accepted_tokens"].to(torch.int64).tolist()
    alpha = torch.exp(inp["A_log"].to(calc_dtype))
    scale = inp["scale"]
    scale_value = 1.0 / math.sqrt(dim) if scale is None else float(scale)
    state = inp["state"]
    out = torch.empty((token_count, heads, dim), dtype=out_dtype, device=device)

    for bi in range(len(cu) - 1):
        start, end = int(cu[bi]), int(cu[bi + 1])
        init_token = start + max(int(acc_host[bi]) - 1, 0)
        for h in range(heads):
            work = state[idx_host[init_token], h].to(calc_dtype).clone()
            for token in range(start, end):
                q_vec = q_raw[token, h].to(calc_dtype)
                k_vec = k_raw[token, h].to(calc_dtype)
                v_vec = v_raw[token, h].to(calc_dtype)
                q = q_vec / torch.sqrt((q_vec * q_vec).sum() + 1e-6)
                k = k_vec / torch.sqrt((k_vec * k_vec).sum() + 1e-6)
                beta_v = torch.sigmoid(beta_view[token, h, 0].to(calc_dtype))
                gate = float(inp["lower_bound"]) * torch.sigmoid(
                    alpha[h] * (g_view[token, h].to(calc_dtype)
                                + inp["dt_bias"][h].to(calc_dtype)))
                decay = torch.exp(gate)
                state_bar = work * decay.unsqueeze(0)
                u = beta_v * (v_vec - (state_bar @ k))
                work = state_bar + torch.outer(u, k)
                o = scale_value * (work @ q)
                state[idx_host[token], h] = work.to(state.dtype)
                out[token, h] = o.to(out_dtype)
    return state, out


# ===== numerical comparison criteria =====


def check_precision(actual, expected, rtol=5e-3, atol=5e-3, floor=1.0):
    """Compare two arrays; `floor` keeps tiny expected values on the absolute scale."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        return False, f"shape mismatch: {actual.shape} vs {expected.shape}"
    if actual.size == 0:
        return True, "empty"
    diff = np.abs(actual - expected)
    bound = atol + rtol * np.abs(expected)
    bad = diff > bound
    n_bad = int(bad.sum())
    worst = float((diff / np.maximum(np.abs(expected), floor)).max())
    if n_bad == 0:
        return True, f"max_rel={worst:.3e} over {actual.size} elements"
    idx = np.argwhere(bad)[:5]
    detail = ", ".join(
        f"[{','.join(str(v) for v in i)}] got={actual[tuple(i)]:.6g} want={expected[tuple(i)]:.6g}"
        for i in idx)
    return False, f"{n_bad}/{actual.size} outside atol={atol} rtol={rtol}: {detail}"


# ===== self-check: NPU mode against CPU mode =====

def _validate(device=None):
    import torch
    if device is None:
        if not torch.npu.is_available() or torch.npu.device_count() == 0:
            _LOG.info("NPU is unavailable; CPU reference self-check only")
            return
        device = torch.device("npu:0")
    _LOG.info("=" * 60)
    _LOG.info("fused_recurrent_kda_snapshot gen_golden 自检（NPU vs CPU）")
    _LOG.info("Device: %s", device)
    _LOG.info("=" * 60)
    cases = (
        ("BSND", dict(batch=2, seq=8, heads=6, seed=3)),
        ("BNSD", dict(batch=2, seq=5, heads=4, seed=11)),
        ("TND", dict(batch=3, seq=5, heads=3, seed=17,
                     cu_seqlens=[4, 5, 1])),
    )
    for layout, kwargs in cases:
        cpu = make_inputs(layout=layout, **kwargs)
        expect_state, expect_out = reference(cpu)
        npu = make_inputs(layout=layout, device=device, **kwargs)
        got_state, got_out = reference(npu)
        got_state = got_state.cpu().float().numpy()
        got_out = got_out.cpu().float().numpy()
        state_err = abs(got_state.astype("float64") - expect_state).max() / max(
            abs(expect_state).max(), 1.0)
        out_err = abs(got_out.astype("float64") - expect_out).max() / max(
            abs(expect_out).max(), 1.0)
        _LOG.info("  [%s] state rel %.3e  out rel %.3e", layout, state_err,
                  out_err)
        if state_err >= 5e-3 or out_err >= 5e-3:
            raise AssertionError(f"{layout} mismatch")
    _LOG.info("自检完成")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _validate()
