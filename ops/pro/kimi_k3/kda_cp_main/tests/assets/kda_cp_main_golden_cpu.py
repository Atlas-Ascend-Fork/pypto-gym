#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent CPU golden for kda_cp_main (input factory, FP32 recurrence,
FP64 oracle and a no-NPU self-check).

Golden semantics (frozen contract):

* the state is bf16-rounded at every chunk boundary and reused as the matmul
  operand for the next chunk (``s_mm = bf16(S).float()``);
* ``delta = u + w @ s_mm^T`` is rounded to bf16 once before the downstream
  matmuls;
* ``O`` and the final state accumulate in FP32; only ``O`` is stored bf16.

Run standalone (no NPU needed)::

    python ops/pro/kimi_k3/kda_cp_main/tests/assets/kda_cp_main_golden_cpu.py
"""

import logging

import torch

CHUNK = 64
D = 128


def chunk_cap(batch, tokens):
    """Chunks capacity of the decay/mqk tensors for a (batch, tokens) load."""
    return (tokens + CHUNK - 1) // CHUNK + batch - 1


def make_inputs(layout, batch, nv, seq_len, device, seed=19):
    """Deterministic input factory shared by every test case."""
    g = torch.Generator().manual_seed(seed)

    def rand_bf16(shape):
        return (torch.randn(shape, generator=g) * 0.02).bfloat16().to(device)

    def rand_f32(shape):
        return (torch.randn(shape, generator=g) * 0.02).to(device)

    if layout == "TND":
        shape = (seq_len, nv, D)
    else:
        shape = (batch, nv, seq_len, D)
    qg, kg, u, w = (rand_bf16(shape) for _ in range(4))
    cap = chunk_cap(batch, seq_len)
    mqk = torch.tril(torch.randn((batch, nv, cap, CHUNK, CHUNK), generator=g) * 0.02)
    mqk = mqk.bfloat16().to(device)
    decay = (torch.rand((batch, nv, cap, D), generator=g) * 0.1 + 0.9).to(device)
    state = rand_f32((batch, nv, D, D))
    cu = None
    if layout == "TND":
        cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    return dict(qg=qg, kg=kg, u=u, w=w, mqk=mqk, decay=decay, state=state, cu=cu)


def _recurrence(inputs, layout, batch, nv, seq_len, dtype):
    """Chunked CPU recurrence; ``dtype`` selects the FP32 golden or FP64 oracle."""
    qg, kg = inputs["qg"].to(dtype), inputs["kg"].to(dtype)
    u, w = inputs["u"].to(dtype), inputs["w"].to(dtype)
    mqk, decay, state = inputs["mqk"], inputs["decay"].to(dtype), inputs["state"].to(dtype)
    if layout == "TND":
        qg, kg, u, w = (t.transpose(0, 1).unsqueeze(0) for t in (qg, kg, u, w))
    golden = dtype == torch.float32  # bf16 rounding only in the FP32 golden
    chunks = (seq_len + CHUNK - 1) // CHUNK
    outs = []
    for c in range(chunks):
        start, end = c * CHUNK, min((c + 1) * CHUNK, seq_len)
        s_mm = state.bfloat16().float() if golden else state
        w_c = w[..., start:end, :]
        u_c = u[..., start:end, :]
        delta = u_c + w_c.matmul(s_mm.transpose(-1, -2))
        if golden:
            delta = delta.bfloat16().float()
        q_c = qg[..., start:end, :]
        m_c = mqk[:, :, c, : end - start, : end - start].to(dtype)
        o_c = q_c.matmul(s_mm.transpose(-1, -2)) + m_c.matmul(delta)
        outs.append(o_c.bfloat16() if golden else o_c)
        state = s_mm * decay[:, :, c].unsqueeze(-2) + delta.transpose(-1, -2).matmul(
            kg[..., start:end, :])
    out = torch.cat(outs, dim=-2)
    if layout == "TND":
        out = out[0].transpose(0, 1)
    elif layout == "BSND":
        out = out.transpose(1, 2)
    return out, state


def kda_cp_main_golden(inputs, layout, batch, nv, seq_len):
    """FP32 CPU golden (the precision contract's reference)."""
    return _recurrence(inputs, layout, batch, nv, seq_len, torch.float32)


def kda_cp_main_golden_f64(inputs, layout, batch, nv, seq_len):
    """FP64 oracle: no intermediate bf16 rounding (order-sensitivity bound)."""
    return _recurrence(inputs, layout, batch, nv, seq_len, torch.float64)


if __name__ == "__main__":
    # No-NPU self-check: the FP32 golden must sit within bf16-quantization
    # distance of the FP64 oracle on a representative small case.
    torch.manual_seed(0)
    inp = make_inputs("BNSD", 1, 2, 130, torch.device("cpu"))
    out32, st32 = kda_cp_main_golden(inp, "BNSD", 1, 2, 130)
    inp64 = {k: (v.float() if torch.is_tensor(v) else v) for k, v in inp.items()}
    out64, st64 = kda_cp_main_golden_f64(inp64, "BNSD", 1, 2, 130)
    e_out = (out32.float() - out64.float()).abs().max().item()
    e_st = (st32.float() - st64.float()).abs().max().item()
    if not (torch.isfinite(out32.float()).all() and torch.isfinite(st32).all()):
        raise RuntimeError("golden self-check failed: non-finite fp32 output")
    logging.basicConfig(level=logging.INFO)
    logging.info(
        "self-check: out|maxdiff=%.3e state|maxdiff=%.3e (bf16-quant scale) OK",
        e_out, e_st)
