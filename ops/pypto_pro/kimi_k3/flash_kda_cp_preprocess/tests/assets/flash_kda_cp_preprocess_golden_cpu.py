#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You should not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""PyPTO-Pro flash_kda_cp_preprocess CPU golden reference (higher precision).

与 NPU golden（flash_kda_cp_preprocess_golden.py）数学逻辑完全一致，在 CPU 上执行：

  - 不 import torch_npu，不做设备迁移；输入张量必须在 CPU 上。
  - 全部计算 FP32；两个合同规定的舍入点保留（L2 归一化结果 to(bf16) 回存；
    qg/kg/u/w/mqk 输出 bf16 舍入）——输出 dtype 由合同固定（BF16/FP32 混合），
    不适用「一律返回 FP32」的模板默认：bf16 舍入是合同语义的一部分，也是
    flash_kda_cp_main 消费方读取的口径。此为对模板规则的有意偏离，已在此披露。
  - 与 NPU golden 的唯一差异是设备与（无）TILE_FWK_DEVICE_ID。

数学语义与判据来源：独立 FP32 递推参考（eisum + linalg.inv 口径）。
"""

import logging
from typing import NamedTuple

import torch
import torch.nn.functional as F


class FlashKDACPGolden(NamedTuple):
    """golden 8 路输出（NamedTuple 兼容 tuple 解包）。"""
    transition: torch.Tensor
    state_ext: torch.Tensor
    decay: torch.Tensor
    qg: torch.Tensor
    kg: torch.Tensor
    u: torch.Tensor
    w: torch.Tensor
    mqk: torch.Tensor


class _P0Inputs(NamedTuple):
    """P0 用例输入包（构造与 kwargs 一体）。"""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    metadata: torch.Tensor
    kwargs: dict


class _IndependentOutputs(NamedTuple):
    """独立对照实现的 8 路输出（NamedTuple 兼容 zip 解包）。"""
    transition: torch.Tensor
    state_ext: torch.Tensor
    decay: torch.Tensor
    qg: torch.Tensor
    kg: torch.Tensor
    u: torch.Tensor
    w: torch.Tensor
    mqk: torch.Tensor


CHUNK = 64
HEAD_DIM = 128
L2_EPS = 1e-6


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    """L2 norm over the last dim in FP32 (`_normalize` 口径)。"""
    return x * torch.rsqrt(x.float().square().sum(dim=-1, keepdim=True) + L2_EPS)


def _strict_lower_inverse(t: torch.Tensor) -> torch.Tensor:
    """(I + T)^(-1) for strictly lower triangular T (nilpotent, T^n = 0)."""
    n = t.shape[-1]
    eye = torch.eye(n, dtype=t.dtype, device=t.device)
    inv = eye - t
    power = t
    exponent = 1
    while exponent < n - 1:
        power = power @ power
        inv = inv @ (eye + power)
        exponent *= 2
    return inv


def _tril_mask(n: int, diagonal: int, device) -> torch.Tensor:
    return torch.tril(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=diagonal)


def _chunk_math(qc, kc, vc, gc, bc, scale):
    """One 64-token chunk, batched over BH (identical to the NPU golden)."""
    bh, chunk, dim = kc.shape
    cumulative = gc.cumsum(dim=-2)
    pair_decay = (cumulative.unsqueeze(-2) - cumulative.unsqueeze(-3)).exp()
    kk = torch.einsum("bid,bijd,bjd->bij", kc, pair_decay, kc)
    strict = (kk * bc.unsqueeze(-1)).masked_fill(~_tril_mask(chunk, -1, kc.device), 0)
    inverse = _strict_lower_inverse(strict)
    inverse_beta = inverse * bc.unsqueeze(-2)
    k_decayed = cumulative.exp() * kc
    corrected_key = inverse_beta @ k_decayed
    corrected_value = inverse_beta @ vc
    last_decay = cumulative[:, -1, :]
    restored_key = (last_decay.unsqueeze(-2) - cumulative).exp() * kc
    q_decayed = qc * cumulative.exp() * scale
    chunk_transition = (
        torch.diag_embed(last_decay.exp())
        - restored_key.transpose(-1, -2) @ corrected_key
    )
    chunk_state = restored_key.transpose(-1, -2) @ corrected_value
    mqk = torch.einsum("bid,bijd,bjd->bij", qc * scale, pair_decay, kc)
    mqk = mqk.masked_fill(~_tril_mask(chunk, 0, kc.device), 0)
    return {
        "qg": q_decayed,
        "kg": restored_key,
        "u": corrected_value,
        "w": -corrected_key,
        "decay": last_decay.exp(),
        "mqk": mqk,
        "transition": chunk_transition,
        "state": chunk_state,
    }


def _chunk_recurrence(qc, kc, vc, gc, bc, scale, nv, chunks, seq_len):
    """块循环 + 递推装配（alloc、rolling transition/state、strip pad）。"""
    dim = qc.shape[-1]
    device = qc.device
    eye = torch.eye(dim, dtype=torch.float32, device=device)
    transition = eye.expand(nv, dim, dim).clone()
    state_ext = torch.zeros(nv, dim, dim, dtype=torch.float32, device=device)
    qg_out = torch.zeros(nv, chunks, CHUNK, dim, dtype=torch.float32, device=device)
    kg_out = torch.zeros_like(qg_out)
    u_out = torch.zeros_like(qg_out)
    w_out = torch.zeros_like(qg_out)
    decay_out = torch.zeros(nv, chunks, dim, dtype=torch.float32, device=device)
    mqk_out = torch.zeros(nv, chunks, CHUNK, CHUNK, dtype=torch.float32, device=device)

    for c in range(chunks):
        result = _chunk_math(
            qc[:, c], kc[:, c], vc[:, c], gc[:, c], bc[:, c], scale
        )
        transition = result["transition"] @ transition
        state_ext = result["transition"] @ state_ext + result["state"]
        qg_out[:, c] = result["qg"]
        kg_out[:, c] = result["kg"]
        u_out[:, c] = result["u"]
        w_out[:, c] = result["w"]
        decay_out[:, c] = result["decay"]
        mqk_out[:, c] = result["mqk"]

    def strip_pad(t):
        return t.reshape(nv, -1, dim)[:, :seq_len]

    return {
        "transition": transition.transpose(-1, -2),
        "state_ext": state_ext.transpose(-1, -2),
        "decay": decay_out,
        "qg": strip_pad(qg_out),
        "kg": strip_pad(kg_out),
        "u": strip_pad(u_out),
        "w": strip_pad(w_out),
        "mqk": mqk_out,
        "chunks": chunks,
    }


def _sequence_outputs(q_seq, k_seq, v_seq, g_seq, beta_seq, a_log, dt_bias, scale, lower_bound):
    """Per logical sequence: math over chunks (identical to the NPU golden)."""
    nqk, seq_len, dim = q_seq.shape
    nv = v_seq.shape[0]
    rep = nv // nqk
    q_rep = q_seq.repeat_interleave(rep, dim=0).float()
    k_rep = k_seq.repeat_interleave(rep, dim=0).float()
    gate = lower_bound * torch.sigmoid(
        a_log.view(nv, 1, 1).exp() * (g_seq.float() + dt_bias.view(nv, 1, dim))
    )
    beta = torch.sigmoid(beta_seq.float())
    k_n = _normalize_rows(k_rep).to(k_seq.dtype).float()
    q_n = _normalize_rows(q_rep).to(q_seq.dtype).float()

    pad = (-seq_len) % CHUNK
    chunks = (seq_len + CHUNK - 1) // CHUNK

    kc = F.pad(k_n, (0, 0, 0, pad)).reshape(nv, chunks, CHUNK, dim)
    vc = F.pad(v_seq.float(), (0, 0, 0, pad)).reshape(nv, chunks, CHUNK, dim)
    gc = F.pad(gate, (0, 0, 0, pad)).reshape(nv, chunks, CHUNK, dim)
    qc = F.pad(q_n, (0, 0, 0, pad)).reshape(nv, chunks, CHUNK, dim)
    bc = F.pad(beta, (0, pad)).reshape(nv, chunks, CHUNK)

    return _chunk_recurrence(qc, kc, vc, gc, bc, scale, nv, chunks, seq_len)


def _logical_views(q, k, v, g, beta, layout_qkv, cu_seqlens):
    """Split physical inputs into per-sequence logical [N, S, D] tensors."""
    if layout_qkv == "BNSD":
        batch = q.shape[0]
        seqs = [
            (q[b], k[b], v[b], g[b], beta[b], q.shape[2])
            for b in range(batch)
        ]
    elif layout_qkv == "BSND":
        batch = q.shape[0]
        seqs = [
            (q[b].transpose(0, 1), k[b].transpose(0, 1), v[b].transpose(0, 1),
             g[b].transpose(0, 1), beta[b].transpose(0, 1), q.shape[1])
            for b in range(batch)
        ]
    elif layout_qkv == "TND":
        if cu_seqlens is None:
            boundaries = [0, q.shape[0]]
        else:
            boundaries = [int(x) for x in cu_seqlens.cpu().tolist()]
        seqs = []
        for b in range(len(boundaries) - 1):
            start, stop = boundaries[b], boundaries[b + 1]
            seqs.append((
                q[start:stop].transpose(0, 1),
                k[start:stop].transpose(0, 1),
                v[start:stop].transpose(0, 1),
                g[start:stop].transpose(0, 1),
                beta[start:stop].transpose(0, 1),
                stop - start,
            ))
    else:
        raise ValueError(f"layout_qkv must be BNSD, BSND, or TND, got {layout_qkv!r}")
    return seqs


def flash_kda_cp_preprocess_golden_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    metadata: torch.Tensor,
    *,
    scale: float = 0.08838834764831843,
    lower_bound: float = -0.1,
    layout_qkv: str = "BNSD",
    cu_seqlens=None,
):
    """CPU FP32 chunk-recurrence reference for the eight public outputs.

    Signature and math identical to flash_kda_cp_preprocess_golden; all
    tensors must live on CPU. metadata is existence/dtype-checked only.
    Output dtypes follow the contract (FP32 summary/decay, BF16 chunk
    tensors) — the BF16 casts are contract semantics, not a precision
    downgrade of this reference.
    """
    if metadata is not None and metadata.dtype != torch.int32:
        raise ValueError("metadata must be int32")
    seqs = _logical_views(q, k, v, g, beta, layout_qkv, cu_seqlens)
    batch = len(seqs)
    if layout_qkv == "BSND":
        nv = v.shape[2]
    else:
        nv = v.shape[1]
    if layout_qkv == "TND":
        total_tokens = q.shape[0]
        capacity = (total_tokens + CHUNK - 1) // CHUNK + batch - 1
    else:
        seq_len = seqs[0][5]
        capacity = (seq_len + CHUNK - 1) // CHUNK + batch - 1

    transition = torch.empty(batch, nv, HEAD_DIM, HEAD_DIM, dtype=torch.float32)
    state_ext = torch.empty_like(transition)
    decay = torch.empty(batch, nv, capacity, HEAD_DIM, dtype=torch.float32)
    mqk = torch.empty(batch, nv, capacity, CHUNK, CHUNK, dtype=torch.bfloat16)
    qg_slots, kg_slots, u_slots, w_slots = [], [], [], []

    for b, (q_seq, k_seq, v_seq, g_seq, beta_seq, seq_len) in enumerate(seqs):
        result = _sequence_outputs(
            q_seq, k_seq, v_seq, g_seq, beta_seq, a_log, dt_bias, scale, lower_bound
        )
        transition[b] = result["transition"]
        state_ext[b] = result["state_ext"]
        chunks = result["chunks"]
        decay[b, :, :chunks] = result["decay"]
        mqk[b, :, :chunks] = result["mqk"].to(torch.bfloat16)
        qg_slots.append(result["qg"].to(torch.bfloat16))
        kg_slots.append(result["kg"].to(torch.bfloat16))
        u_slots.append(result["u"].to(torch.bfloat16))
        w_slots.append(result["w"].to(torch.bfloat16))

    if layout_qkv == "TND":
        qg = torch.cat([t.transpose(0, 1) for t in qg_slots], dim=0)
        kg = torch.cat([t.transpose(0, 1) for t in kg_slots], dim=0)
        u = torch.cat([t.transpose(0, 1) for t in u_slots], dim=0)
        w = torch.cat([t.transpose(0, 1) for t in w_slots], dim=0)
    else:
        qg = torch.stack(qg_slots, dim=0)
        kg = torch.stack(kg_slots, dim=0)
        u = torch.stack(u_slots, dim=0)
        w = torch.stack(w_slots, dim=0)

    return FlashKDACPGolden(transition, state_ext, decay, qg, kg, u, w, mqk)


# ==========================================
# 独立对照实现（einsum + linalg.inv）
# ==========================================

def _repeat_gqa_norm(q, k, g, beta, a_log, dt_bias, lower_bound):
    """GQA 头扩展 + 门控激活 + L2 归一化（独立对照口径）。"""
    heads = g.shape[1]
    key_heads = k.shape[1]
    k = k.repeat_interleave(heads // key_heads, dim=1)
    q = q.repeat_interleave(heads // key_heads, dim=1)
    dim = q.shape[-1]
    gate = lower_bound * torch.sigmoid(
        a_log.view(1, heads, 1, 1).exp() * (g.float() + dt_bias.view(1, heads, 1, dim))
    )
    beta = torch.sigmoid(beta.float())
    normalized_k = _normalize_rows(k).to(k.dtype).float()
    normalized_q = _normalize_rows(q).to(q.dtype).float()
    return {
        "q": normalized_q,
        "k": normalized_k,
        "gate": gate,
        "beta": beta,
    }


def _inv_chunk_terms(key, value, query, cumulative, beta_chunk, causal, identity):
    """独立对照的单块核心数学（einsum + linalg.inv 口径）。"""
    key_pair_decay = (cumulative.unsqueeze(-2) - cumulative.unsqueeze(-3)).exp()
    kk = torch.einsum("bhid,bhijd,bhjd->bhij", key, key_pair_decay, key)
    strict = (kk * beta_chunk.unsqueeze(-1)).masked_fill(~causal, 0)
    inverse = torch.linalg.inv(identity + strict)
    inverse_beta = inverse * beta_chunk.unsqueeze(-2)
    corrected_key = inverse_beta @ (cumulative.exp() * key)
    corrected_value = inverse_beta @ value
    last_decay = cumulative[:, :, -1]
    restored_key = (last_decay.unsqueeze(-2) - cumulative).exp() * key
    chunk_transition = torch.diag_embed(last_decay.exp()) - restored_key.transpose(-1, -2) @ corrected_key
    chunk_state = restored_key.transpose(-1, -2) @ corrected_value
    return {
        "pair_decay": key_pair_decay,
        "corrected_key": corrected_key,
        "corrected_value": corrected_value,
        "restored_key": restored_key,
        "last_decay_exp": last_decay.exp(),
        "cum_exp": cumulative.exp(),
        "transition": chunk_transition,
        "state": chunk_state,
    }


def _independent_reference(q, k, v, g, beta, a_log, dt_bias, seq_len, scale, lower_bound):
    """Independent FP32 reference (einsum + linalg.inv)."""
    batch, heads, _, dim = v.shape
    prepared = _repeat_gqa_norm(q, k, g, beta, a_log, dt_bias, lower_bound)
    normalized_k = prepared["k"]
    normalized_q = prepared["q"]
    gate = prepared["gate"]
    padded_len = ((seq_len + 63) // 64) * 64
    pad_rows = padded_len - seq_len
    key_chunks = F.pad(normalized_k, (0, 0, 0, pad_rows)).reshape(batch, heads, -1, 64, dim)
    value_chunks = F.pad(v.float(), (0, 0, 0, pad_rows)).reshape(batch, heads, -1, 64, dim)
    gate_chunks = F.pad(gate, (0, 0, 0, pad_rows)).reshape(batch, heads, -1, 64, dim)
    beta_chunks = F.pad(prepared["beta"], (0, pad_rows)).reshape(batch, heads, -1, 64)
    query_chunks = F.pad(normalized_q, (0, 0, 0, pad_rows)).reshape(batch, heads, -1, 64, dim)
    causal = torch.tril(torch.ones(64, 64, dtype=torch.bool), diagonal=-1)
    mqk_causal = torch.tril(torch.ones(64, 64, dtype=torch.bool))
    identity = torch.eye(64)
    transition = torch.eye(dim).expand(batch, heads, dim, dim).clone()
    state_ext = torch.zeros(batch, heads, dim, dim)
    qg_chunks, kg_chunks, u_chunks, w_chunks, mqk_chunks, decay_chunks = [], [], [], [], [], []
    for chunk_idx in range(key_chunks.shape[2]):
        key = key_chunks[:, :, chunk_idx]
        value = value_chunks[:, :, chunk_idx]
        query = query_chunks[:, :, chunk_idx]
        cumulative = gate_chunks[:, :, chunk_idx].cumsum(dim=-2)
        terms = _inv_chunk_terms(key, value, query, cumulative,
                                 beta_chunks[:, :, chunk_idx], causal, identity)
        state_ext = terms["transition"] @ state_ext + terms["state"]
        transition = terms["transition"] @ transition
        qg_chunks.append((query * terms["cum_exp"] * scale).to(torch.bfloat16))
        kg_chunks.append(terms["restored_key"].to(torch.bfloat16))
        u_chunks.append(terms["corrected_value"].to(torch.bfloat16))
        w_chunks.append((-terms["corrected_key"]).to(torch.bfloat16))
        decay_chunks.append(terms["last_decay_exp"])
        mqk = torch.einsum("bhid,bhijd,bhjd->bhij", query * scale,
                           terms["pair_decay"], key)
        mqk_chunks.append(mqk.masked_fill(~mqk_causal, 0).to(torch.bfloat16))

    def flatten(chunks):
        return torch.stack(chunks, dim=2).reshape(batch, heads, padded_len, dim)[:, :, :seq_len]

    return _IndependentOutputs(
        transition.transpose(-1, -2),
        state_ext.transpose(-1, -2),
        torch.stack(decay_chunks, dim=2),
        flatten(qg_chunks),
        flatten(kg_chunks),
        flatten(u_chunks),
        flatten(w_chunks),
        torch.stack(mqk_chunks, dim=2),
    )


def _assert_close(observed, reference, name, atol=5e-3, rtol=5e-3):
    if observed.shape != reference.shape:
        raise ValueError(f"shape mismatch: {name} {observed.shape} != {reference.shape}")
    error = (observed.float() - reference.float()).abs()
    tolerance = atol + rtol * reference.float().abs()
    normalized = (error / tolerance).flatten().quantile(0.999).item()
    if normalized > 1.0:
        raise ValueError(f"precision contract violated: {name} 99.9pct={normalized:.4f}")
    return normalized


# ==========================================
# 验证
# ==========================================

# CPU 验证只跑小形状 P0 与泛化用例；S=8192 的 perf P0 在 CPU 上无独立价值
# （其数学与 S=65 多 chunk 递推完全同构），跳过以控制验证时长。
_CPU_MAX_TOKENS = 4096


def _make_p0_inputs(seq_len, n_v, n_kv, layout):
    """构造 P0 用例输入（布局感知）与调用 kwargs。"""
    gen = torch.Generator().manual_seed(53)
    if layout == "TND":
        q = torch.randn(seq_len, n_kv, 128, generator=gen).bfloat16()
        k = torch.randn(seq_len, n_kv, 128, generator=gen).bfloat16()
        v = torch.randn(seq_len, n_v, 128, generator=gen).bfloat16()
        g = torch.randn(seq_len, n_v, 128, generator=gen).bfloat16()
        beta = torch.randn(seq_len, n_v, generator=gen).bfloat16()
        cu = torch.tensor([0, seq_len], dtype=torch.int32)
    else:
        if layout == "BSND":
            q = torch.randn(1, seq_len, n_kv, 128, generator=gen).bfloat16()
            k = torch.randn(1, seq_len, n_kv, 128, generator=gen).bfloat16()
            v = torch.randn(1, seq_len, n_v, 128, generator=gen).bfloat16()
            g = torch.randn(1, seq_len, n_v, 128, generator=gen).bfloat16()
            beta = torch.randn(1, seq_len, n_v, generator=gen).bfloat16()
        else:
            q = torch.randn(1, n_kv, seq_len, 128, generator=gen).bfloat16()
            k = torch.randn(1, n_kv, seq_len, 128, generator=gen).bfloat16()
            v = torch.randn(1, n_v, seq_len, 128, generator=gen).bfloat16()
            g = torch.randn(1, n_v, seq_len, 128, generator=gen).bfloat16()
            beta = torch.randn(1, n_v, seq_len, generator=gen).bfloat16()
        cu = None
    a_log = torch.linspace(-1.0, 0.5, n_v, dtype=torch.float32)
    dt_bias = torch.randn(n_v, 128, generator=gen)
    metadata = torch.zeros(10 + 143 * ((seq_len + 63) // 64), dtype=torch.int32)
    kwargs = {"scale": 128 ** -0.5, "lower_bound": -0.1, "layout_qkv": layout}
    if cu is not None:
        kwargs["cu_seqlens"] = cu
    return _P0Inputs(q, k, v, g, beta, a_log, dt_bias, metadata, kwargs)


def _check_p0_case(name, inputs, expected_shapes, expected_dtypes):
    """执行单条 P0 用例并逐输出校验 shape/dtype/finite。"""
    want_shapes = expected_shapes.get(name)
    if want_shapes is None:
        raise KeyError(f"unknown case: {name}")
    outs = flash_kda_cp_preprocess_golden_cpu(
        inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta, inputs.a_log,
        inputs.dt_bias, inputs.metadata, **inputs.kwargs)
    logging.info(f"\n[case: {name}]")
    for i, (o, want_shape, want_dtype) in enumerate(
            zip(outs, want_shapes, expected_dtypes)):
        ok_shape = tuple(o.shape) == want_shape
        ok_dtype = o.dtype == want_dtype
        ok_finite = torch.isfinite(o.float()).all().item()
        logging.info(f"  out[{i}] shape={tuple(o.shape)} dtype={o.dtype} finite={ok_finite} "
              f"... {'PASS' if (ok_shape and ok_dtype and ok_finite) else 'FAIL'}")
        if not (ok_shape and ok_dtype and ok_finite):
            raise ValueError(f"golden output check failed: case={name} out[{i}] "
                            f"shape={ok_shape} dtype={ok_dtype} finite={ok_finite}")


def _validate_p0_cases(p0_small, expected_shapes, expected_dtypes):
    """小形状合同 P0：golden 输出与期望 shape/dtype 逐一对照。"""
    for name, seq_len, n_v, n_kv, layout in p0_small:
        inputs = _make_p0_inputs(seq_len, n_v, n_kv, layout)
        _check_p0_case(name, inputs, expected_shapes, expected_dtypes)


def _validate_cross_check():
    """独立对照（einsum + linalg.inv）。"""
    logging.info("\n[cross-check vs independent reference (linalg.inv)]")
    for name, seq_len, n_v, n_kv in [
        ("bnsd_minimal_full", 64, 1, 1),
        ("bnsd_whitebox", 17, 3, 1),
        ("bnsd_two_chunks", 65, 3, 1),
        ("bnsd_even_chunks", 128, 2, 2),
    ]:
        gen = torch.Generator().manual_seed(53)
        q = torch.randn(1, n_kv, seq_len, 128, generator=gen).bfloat16()
        k = torch.randn(1, n_kv, seq_len, 128, generator=gen).bfloat16()
        v = torch.randn(1, n_v, seq_len, 128, generator=gen).bfloat16()
        g = torch.randn(v.shape, generator=gen).bfloat16()
        beta = torch.randn(1, n_v, seq_len, generator=gen).bfloat16()
        a_log = torch.linspace(-1.0, 0.5, n_v, dtype=torch.float32)
        dt_bias = torch.randn(n_v, 128, generator=gen)
        metadata = torch.zeros(10 + 143 * ((seq_len + 63) // 64), dtype=torch.int32)
        observed = flash_kda_cp_preprocess_golden_cpu(
            q, k, v, g, beta, a_log, dt_bias, metadata,
            scale=128 ** -0.5, lower_bound=-0.1, layout_qkv="BNSD")
        reference = _independent_reference(
            q, k, v, g, beta, a_log, dt_bias, seq_len, 128 ** -0.5, -0.1)
        scores = []
        for out_name, o, r in zip(
                ("transition", "state_ext", "decay", "qg", "kg", "u", "w", "mqk"),
                observed, reference):
            scores.append(_assert_close(o, r, f"{name}.{out_name}"))
        logging.info(f"  {name}: max 99.9pct normalized error = {max(scores):.4f} PASS")


def _check_dense_general():
    """泛化：S=1 / B=2 容量 / lower_bound=-5.0 极值。"""

    def run_dense(batch, n_kv, n_v, seq_len, lower_bound=-0.1, seed=99):
        gen = torch.Generator().manual_seed(seed)
        q = torch.randn(batch, n_kv, seq_len, 128, generator=gen).bfloat16()
        k = torch.randn(batch, n_kv, seq_len, 128, generator=gen).bfloat16()
        v = torch.randn(batch, n_v, seq_len, 128, generator=gen).bfloat16()
        g = torch.randn(v.shape, generator=gen).bfloat16()
        beta = torch.randn(batch, n_v, seq_len, generator=gen).bfloat16()
        a_log = torch.linspace(-1.0, 0.5, n_v, dtype=torch.float32)
        dt_bias = torch.randn(n_v, 128, generator=gen)
        metadata = torch.zeros(10 + 143 * batch * ((seq_len + 63) // 64),
                               dtype=torch.int32)
        return flash_kda_cp_preprocess_golden_cpu(
            q, k, v, g, beta, a_log, dt_bias, metadata,
            scale=128 ** -0.5, lower_bound=lower_bound, layout_qkv="BNSD")

    outs = run_dense(1, 1, 1, 1)
    if tuple(outs[2].shape) != (1, 1, 1, 128) or not torch.isfinite(outs[0]).all():
        raise ValueError("S=1 single token check failed")
    logging.info("  S=1 single token: PASS")
    outs = run_dense(2, 1, 2, 100, seed=101)
    if tuple(outs[2].shape) != (2, 2, 3, 128):
        raise ValueError("B=2 capacity check failed")
    logging.info("  B=2 capacity C=L+B-1: PASS")
    outs = run_dense(1, 1, 1, 128, lower_bound=-5.0, seed=102)
    if not (torch.isfinite(outs[0]).all() and torch.isfinite(outs[7]).all()):
        raise ValueError("lower_bound=-5.0 extreme check failed")
    logging.info("  lower_bound=-5.0 extreme: PASS")


def _check_packed_tnd():
    """泛化：packed TND (17,65) 与逐序列 BNSD 的一致性。"""
    gen = torch.Generator().manual_seed(53)
    first_q = torch.randn(1, 1, 17, 128, generator=gen).bfloat16()
    first_k = torch.randn(1, 1, 17, 128, generator=gen).bfloat16()
    first_v = torch.randn(1, 3, 17, 128, generator=gen).bfloat16()
    first_g = torch.randn(first_v.shape, generator=gen).bfloat16()
    first_beta = torch.randn(1, 3, 17, generator=gen).bfloat16()
    gen2 = torch.Generator().manual_seed(58)
    second_q = torch.randn(1, 1, 65, 128, generator=gen2).bfloat16()
    second_k = torch.randn(1, 1, 65, 128, generator=gen2).bfloat16()
    second_v = torch.randn(1, 3, 65, 128, generator=gen2).bfloat16()
    second_g = torch.randn(second_v.shape, generator=gen2).bfloat16()
    second_beta = torch.randn(1, 3, 65, generator=gen2).bfloat16()
    a_log = torch.linspace(-1.0, 0.5, 3, dtype=torch.float32)
    dt_bias = torch.randn(3, 128, generator=torch.Generator().manual_seed(53))

    def pack(tensors):
        transposed = []
        for t in tensors:
            transposed.append(t[0].transpose(0, 1))
        return torch.cat(transposed, dim=0).contiguous()

    q_p = pack((first_q, second_q))
    k_p = pack((first_k, second_k))
    v_p = pack((first_v, second_v))
    g_p = pack((first_g, second_g))
    b_p = pack((first_beta, second_beta))
    cu = torch.tensor([0, 17, 82], dtype=torch.int32)
    metadata = torch.zeros(10 + 143 * 3, dtype=torch.int32)
    packed = flash_kda_cp_preprocess_golden_cpu(
        q_p, k_p, v_p, g_p, b_p, a_log, dt_bias, metadata,
        scale=128 ** -0.5, lower_bound=-0.1, layout_qkv="TND", cu_seqlens=cu)
    if tuple(packed[2].shape) != (2, 3, 3, 128):
        raise ValueError("packed TND decay capacity check failed")
    for seq, (q_s, k_s, v_s, g_s, b_s) in enumerate(
            ((first_q, first_k, first_v, first_g, first_beta),
             (second_q, second_k, second_v, second_g, second_beta))):
        single = flash_kda_cp_preprocess_golden_cpu(
            q_s, k_s, v_s, g_s, b_s, a_log, dt_bias, metadata,
            scale=128 ** -0.5, lower_bound=-0.1, layout_qkv="BNSD")
        chunks = (q_s.shape[2] + 63) // 64
        for out_name, o, r in zip(
                ("transition", "state_ext", "decay", "mqk"),
                (packed[0][seq:seq + 1], packed[1][seq:seq + 1],
                 packed[2][seq:seq + 1, :, :chunks], packed[7][seq:seq + 1, :, :chunks]),
                (single[0], single[1], single[2], single[7])):
            _assert_close(o, r, f"packed.seq{seq}.{out_name}")
    logging.info("  packed TND (17,65): PASS")


def _validate():
    """Run the small contract P0 cases plus cross/generalization checks on CPU."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("=" * 60)
    logging.info("flash_kda_cp_preprocess_golden_cpu 验证报告")
    logging.info("=" * 60)

    p0_small = [
        ("bnsd_minimal_full", 64, 1, 1, "BNSD"),
        ("bnsd_whitebox", 17, 3, 1, "BNSD"),
        ("bsnd_whitebox", 49, 5, 1, "BSND"),
        ("tnd_whitebox", 577, 96, 3, "TND"),
    ]
    expected_shapes = {
        "bnsd_minimal_full": (
            (1, 1, 128, 128), (1, 1, 128, 128), (1, 1, 1, 128),
            (1, 1, 64, 128), (1, 1, 64, 128), (1, 1, 64, 128), (1, 1, 64, 128),
            (1, 1, 1, 64, 64)),
        "bnsd_whitebox": (
            (1, 3, 128, 128), (1, 3, 128, 128), (1, 3, 1, 128),
            (1, 3, 17, 128), (1, 3, 17, 128), (1, 3, 17, 128), (1, 3, 17, 128),
            (1, 3, 1, 64, 64)),
        "bsnd_whitebox": (
            (1, 5, 128, 128), (1, 5, 128, 128), (1, 5, 1, 128),
            (1, 5, 49, 128), (1, 5, 49, 128), (1, 5, 49, 128), (1, 5, 49, 128),
            (1, 5, 1, 64, 64)),
        "tnd_whitebox": (
            (1, 96, 128, 128), (1, 96, 128, 128), (1, 96, 10, 128),
            (577, 96, 128), (577, 96, 128), (577, 96, 128), (577, 96, 128),
            (1, 96, 10, 64, 64)),
    }
    expected_dtypes = (
        torch.float32, torch.float32, torch.float32,
        torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.bfloat16,
        torch.bfloat16,
    )

    _validate_p0_cases(p0_small, expected_shapes, expected_dtypes)
    _validate_cross_check()
    logging.info("\n[generalization]")
    _check_dense_general()
    _check_packed_tnd()

    logging.info("\n" + "=" * 60)
    logging.info("validation complete")
    logging.info("=" * 60)


if __name__ == "__main__":
    _validate()
