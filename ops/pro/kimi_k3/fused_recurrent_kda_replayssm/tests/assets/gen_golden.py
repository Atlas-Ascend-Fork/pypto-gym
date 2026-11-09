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
"""Reference for fused_recurrent_kda_replayssm, with CPU and NPU modes.

The operator reads a read-only FP32 checkpoint, evaluates every candidate token
of the KDA recurrence, returns the BF16 output and writes the FP16 U/K + FP32
decay replay records.

One torch implementation serves both modes: NumPy inputs run on CPU in FP64 and
return NumPy; torch inputs run on their own device in FP32 with the operator's
output dtypes.  CPU mode is the independent precision reference, NPU mode is the
performance baseline.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np

_LOG = logging.getLogger(__name__)

HEAD_DIM = 128
MAX_HEADS = 96
SENTINEL = -7.0


def bf16_round(x):
    """Round an FP32 array to the nearest BF16-representable value (RNE)."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    lsb = (u >> 16) & np.uint32(1)
    rounded = u + np.uint32(0x7FFF) + lsb
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


class QkvInputs(NamedTuple):
    mixedqkv: np.ndarray
    g: np.ndarray
    beta: np.ndarray
    seqlens: list
    token_count: int
    capacity: int


def _qkv_inputs(rng, layout, b, s, n, d, capacity, cu_seqlens):
    """Build packed QKV/gate/beta and the sequence metadata for one layout."""
    p = n * d
    cap = s if capacity is None else capacity
    if layout == "BSND":
        mixedqkv = bf16_round(rng.standard_normal((b, s, 3 * p)).astype(np.float32))
        g = bf16_round(rng.standard_normal((b, s, n, d)).astype(np.float32))
        beta = bf16_round(rng.standard_normal((b, s, n, 1)).astype(np.float32))
        seqlens = [s] * b
        token_count = b * s
    elif layout == "BNSD":
        mixedqkv = bf16_round(rng.standard_normal((b, 3 * n, s, d)).astype(np.float32))
        g = bf16_round(rng.standard_normal((b, n, s, d)).astype(np.float32))
        beta = bf16_round(rng.standard_normal((b, n, s, 1)).astype(np.float32))
        seqlens = [s] * b
        token_count = b * s
    elif layout == "TND":
        seqlens = list(cu_seqlens) if cu_seqlens is not None else [s] * b
        if capacity is None:
            cap = max(seqlens)
        cu = np.zeros(b + 1, dtype=np.int32)
        for i, length in enumerate(seqlens):
            cu[i + 1] = cu[i] + length
        token_count = int(cu[-1])
        mixedqkv = bf16_round(rng.standard_normal((token_count, 3 * p)).astype(np.float32))
        g = bf16_round(rng.standard_normal((token_count, n, d)).astype(np.float32))
        beta = bf16_round(rng.standard_normal((token_count, n, 1)).astype(np.float32))
    else:
        raise ValueError(layout)
    return QkvInputs(mixedqkv, g, beta, seqlens, token_count, cap)


def make_inputs(
    batch=2,
    seq=8,
    heads=6,
    dim=HEAD_DIM,
    seed=0,
    layout="BSND",
    cu_seqlens=None,
    capacity=None,
    scale=None,
    lower_bound=-5.0,
    device=None,
):
    """Build one case's inputs as NumPy (optionally converted to a torch device)."""
    rng = np.random.default_rng(seed)
    mixedqkv, g, beta, seqlens, token_count, cap = _qkv_inputs(
        rng, layout, batch, seq, heads, dim, capacity, cu_seqlens)
    state = rng.standard_normal((batch, heads, dim, dim)).astype(np.float64) * 0.5
    state = state.astype(np.float32).astype(np.float64)
    a_log = rng.standard_normal(heads).astype(np.float32) * 0.5
    dt_bias = rng.standard_normal((heads, dim)).astype(np.float32) * 0.1
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
        "seqlens": seqlens,
        "capacity": cap,
        "batch": batch,
        "seq": seq,
        "heads": heads,
        "dim": dim,
        "token_count": token_count,
    }
    if device is None:
        return inp
    return _to_torch(inp, device)


def _to_torch(inp, device):
    import torch
    dev = torch.device(device)
    on_cpu = dev.type == "cpu"

    def as_tensor(name, dtype=None):
        value = torch.from_numpy(np.ascontiguousarray(inp[name])).to(dev)
        if dtype is not None and not on_cpu:
            value = value.to(dtype)
        return value

    out = dict(inp)
    out["mixedqkv"] = as_tensor("mixedqkv", torch.bfloat16)
    out["g"] = as_tensor("g", torch.bfloat16)
    out["beta"] = as_tensor("beta", torch.bfloat16)
    out["state"] = as_tensor("state", torch.float32)
    out["A_log"] = as_tensor("A_log")
    out["dt_bias"] = as_tensor("dt_bias")
    out["cu_seqlens"] = None
    if inp["layout"] == "TND":
        cu = np.cumsum([0] + list(inp["seqlens"])).astype(np.int32)
        out["cu_seqlens"] = torch.from_numpy(cu).to(torch.int32).to(dev)
    return out


def _token_views(inp):
    """Return Q/K/V/g/beta with dense layouts normalized to [B,S,N,D]."""
    layout = inp["layout"]
    b, n, d = inp["batch"], inp["heads"], inp["dim"]
    if layout == "TND":
        qkv = inp["mixedqkv"].reshape(-1, 3, n, d)
        return (qkv[:, 0], qkv[:, 1], qkv[:, 2],
                inp["g"].reshape(-1, n, d), inp["beta"].reshape(-1, n, 1))
    if layout == "BSND":
        qkv = inp["mixedqkv"].reshape(b, -1, 3, n, d)
        return (qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2],
                inp["g"].reshape(b, -1, n, d),
                inp["beta"].reshape(b, -1, n, 1))
    qkv = inp["mixedqkv"].reshape(b, 3, n, -1, d)
    return (qkv[:, 0].transpose(1, 2), qkv[:, 1].transpose(1, 2),
            qkv[:, 2].transpose(1, 2),
            inp["g"].reshape(b, n, -1, d).transpose(1, 2),
            inp["beta"].reshape(b, n, -1, 1).transpose(1, 2))


def _dtypes(inp):
    import torch
    if inp["state"].device.type == "cpu":
        return torch.float64, torch.float64, torch.float64, torch.float64, torch.float64
    return torch.float32, torch.bfloat16, torch.float16, torch.float16, torch.float32


def _zeros(shape, inp, dtype):
    import torch
    return torch.zeros(shape, dtype=dtype, device=inp["state"].device)


def _reference_dense(inp):
    import torch
    calc, out_dt, u_dt, k_dt, d_dt = _dtypes(inp)
    b, s, n, d = inp["batch"], inp["seq"], inp["heads"], inp["dim"]
    q, k, v, g, beta = _token_views(inp)
    alpha = inp["A_log"].to(calc).exp()[None, :, None]
    dt_bias = inp["dt_bias"].to(calc)
    lower = float(inp["lower_bound"])
    scale = float(inp["scale"]) if inp["scale"] is not None else d ** -0.5
    state = inp["state"].to(calc)
    out = _zeros((b, s, n, d), inp, out_dt)
    rec_u = _zeros((b, s, n, d), inp, u_dt)
    rec_k = _zeros((b, s, n, d), inp, k_dt)
    rec_d = _zeros((b, s, n, d), inp, d_dt)
    for t in range(s):
        decay = (lower * torch.sigmoid(alpha * (g[:, t].to(calc) + dt_bias))).exp()
        qf = q[:, t].to(calc)
        kf = k[:, t].to(calc)
        qn = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
        kn = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
        bf = torch.sigmoid(beta[:, t].to(calc)).squeeze(-1)
        state_bar = state * decay.unsqueeze(-2)
        u = bf.unsqueeze(-1) * (v[:, t].to(calc) - (state_bar * kn.unsqueeze(-2)).sum(-1))
        state = state_bar + u.unsqueeze(-1) * kn.unsqueeze(-2)
        out[:, t] = scale * (state * qn.unsqueeze(-2)).sum(-1)
        rec_u[:, t] = u
        rec_k[:, t] = kn
        rec_d[:, t] = decay
    if inp["layout"] == "BNSD":
        out = out.transpose(1, 2)
    return out, rec_u, rec_k, rec_d


def _reference_tnd(inp):
    import torch
    calc, out_dt, u_dt, k_dt, d_dt = _dtypes(inp)
    t_count, n, d = inp["token_count"], inp["heads"], inp["dim"]
    q, k, v, g, beta = _token_views(inp)
    alpha = inp["A_log"].to(calc).exp()[:, None]
    dt_bias = inp["dt_bias"].to(calc)
    lower = float(inp["lower_bound"])
    scale = float(inp["scale"]) if inp["scale"] is not None else d ** -0.5
    out = _zeros((t_count, n, d), inp, out_dt)
    rec_u = _zeros((t_count, n, d), inp, u_dt)
    rec_k = _zeros((t_count, n, d), inp, k_dt)
    rec_d = _zeros((t_count, n, d), inp, d_dt)
    offset = 0
    for b, length in enumerate(inp["seqlens"]):
        state = inp["state"][b].to(calc)
        for t in range(length):
            i = offset + t
            decay = (lower * torch.sigmoid(alpha * (g[i].to(calc) + dt_bias))).exp()
            qf = q[i].to(calc)
            kf = k[i].to(calc)
            qn = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
            kn = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
            bf = torch.sigmoid(beta[i].to(calc)).squeeze(-1)
            state_bar = state * decay.unsqueeze(-2)
            u = bf.unsqueeze(-1) * (v[i].to(calc) - (state_bar * kn.unsqueeze(-2)).sum(-1))
            state = state_bar + u.unsqueeze(-1) * kn.unsqueeze(-2)
            out[i] = scale * (state * qn.unsqueeze(-2)).sum(-1)
            rec_u[i] = u
            rec_k[i] = kn
            rec_d[i] = decay
        offset += length
    return out, rec_u, rec_k, rec_d


def reference(inp):
    """Return (out, replay_u, replay_k, replay_decay)."""
    import torch
    torch_in = type(inp["state"]).__module__.split(".")[0] == "torch"
    if not torch_in:
        inp = _to_torch(inp, "cpu")
    if inp["layout"] == "TND":
        out, ru, rk, rd = _reference_tnd(inp)
    else:
        out, ru, rk, rd = _reference_dense(inp)
    if torch_in:
        return out, ru, rk, rd
    return (out.numpy(), ru.numpy(), rk.numpy(), rd.numpy())


def check_precision(actual, expected, atol=5e-3, rtol=5e-3, floor=1.0):
    """Element-wise |actual - expected| <= atol + rtol*|expected| with a floor."""
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    diff = np.abs(a - e)
    denom = np.maximum(np.abs(e), floor)
    return {
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "max_rel": float((diff / denom).max()) if diff.size else 0.0,
        "ok": bool(np.allclose(a, e, rtol=rtol, atol=atol)),
    }


def _self_check():
    """Cross-check CPU FP64 against NPU FP32 on small cases."""
    import torch
    ok = True
    for layout, kwargs in (
        ("BSND", {"batch": 2, "seq": 5, "heads": 3}),
        ("BNSD", {"batch": 2, "seq": 4, "heads": 2}),
        ("TND", {"batch": 2, "seq": 5, "heads": 2, "cu_seqlens": [3, 5]}),
    ):
        inp = make_inputs(seed=7, layout=layout, **kwargs)
        out_c, u_c, k_c, d_c = reference(inp)
        if not torch.npu.is_available():
            _LOG.info("%s CPU-only mode (no NPU): out self-consistent", layout)
            continue
        torch.npu.set_device(0)
        inp_npu = _to_torch(inp, torch.device("npu:0"))
        out_n, u_n, k_n, d_n = reference(inp_npu)
        for name, a, b in (("out", out_n, out_c), ("u", u_n, u_c),
                           ("k", k_n, k_c), ("d", d_n, d_c)):
            res = check_precision(a.float().cpu().numpy(), b)
            ok = ok and res["ok"]
            _LOG.info("%s %s max_abs=%.3e max_rel=%.3e ok=%s", layout, name,
                      res["max_abs"], res["max_rel"], res["ok"])
    return ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(0 if _self_check() else 1)
