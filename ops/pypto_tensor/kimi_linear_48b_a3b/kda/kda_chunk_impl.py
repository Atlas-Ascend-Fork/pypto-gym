# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""kda_chunk — chunked Kimi Delta Attention (KDA) kernel for Kimi-Linear-48B-A3B.

Per-channel-gate chunked linear attention (the KDA prefill path), validated
against the golden ``_naive_recurrent_kda`` (state_v_first, ``[B,H,V,K]``). Drop-in
PyPTO replacement for the upstream ``fla.ops.kda.chunk_kda`` entry point used by
``KimiDeltaAttention`` during prefill (``mode == 'chunk'``).

TILING — NESTED 64-outer / 16-inner (the current shipped layout):
    The host->kernel loop steps a 64-row OUTER tile. All elementwise / row-local
    vector pre-compute over those 64 rows (cumulative gate, exp(G), exp(-G), qd,
    kd, kg_pos, beta*v, beta*kg, state-update key) runs at
    ``set_vec_tile_shapes(64, 128)`` -> ~4x fewer vector-pipe launches than a flat
    16-row tile. The intra-chunk pairwise term and the serial state-coupled
    matmuls stay at M=16 inside a compile-time ``range(4)`` (NSUB) loop, because
    the pairwise matmuls are block-diagonal and the state matmuls read the
    serially-updated state per sub-block.

NUMERICAL-STABILITY INVARIANT (non-negotiable):
    qd=q*exp(G), kd=k*exp(-G) uses exp(-G). The real KDA gate is a per-channel
    log-gate g in [-5, 0]. Over a 16-step window the cumulative gate range is at
    most 16*5 = 80, so exp(80) ~ 5.5e34 < 3.4e38 (fp32-safe). A 64-step window
    would give exp(320) = inf (NaN).

    THE 64-OUTER TILE NEVER WIDENS THAT WINDOW. The cumulative gate is computed
    BLOCK-DIAGONALLY: ``tril_le64`` is a [64,64] block-diagonal lower-triangular
    matrix (4 independent 16x16 blocks, zero off-block), so G[row] is the cumsum
    of g WITHIN its own 16-block only and every exp argument stays bounded by 80,
    exactly as a flat chunk_size=16. Matches the recurrent golden to < 1e-7 even
    for the strongest gate g=-5.

Per 16-subchunk c, for one (b,h), with state S=[V,K]  (n = SUB = 16):
    g_cum  = cumsum_n(g)                          # [n,K]  (block-local), all <= 0
    qd is q*exp(g_cum); kd is k*exp(-g_cum); kg_pos is k*exp(g_cum)
    a_qk: scaled qd·kdᵀ masked by tril_le         # [n,n]
    a_kk: beta * (kg_pos·kdᵀ) masked by tril_lt   # [n,n] strict-lower, nilpotent
    t_inv: inverse of (I + a_kk)                  # via nilpotent doubling
    u and w: apply t_inv to (beta*v) and (beta*exp(g_cum)*k)
    v_new is u - w·stateᵀ                         # [n,V]
    o is scale*(qd·stateᵀ) + a_qk·v_new           # [n,V]
    g_last is g_cum[n-1]; kg is exp(g_last - g_cum)*k  # [n,K] (g_last - g_cum <= 0, safe)
    state: exp(g_last)*state + contraction of v_newᵀ with kg  # [V,K]

Tile geometry:
  * Outer tile OUT = 64 rows; inner subchunk SUB = 16 rows; head dim K = V = 128.
  * Triangular inverse via nilpotent doubling (4 doublings: 2^4 = 16).

Scope (caller-enforced via wrapper):
  * head_k_dim K = 128, head_v_dim V = 128.
  * ``use_qk_l2norm_in_kernel = True`` (L2 norm done on host before packing).

Out-of-scope shapes raise ``NotImplementedError``. The modeling-layer hook falls
back to the upstream / torch chunk function in that case.
"""
__all__ = ["kda_chunk_wrapper"]

import os
import sys

import pypto
import torch
import torch_npu  # noqa: F401  required for NPU device init
from torch._dynamo import allow_in_graph

# Same-dir sibling import: works both as a package (.kda.kda_chunk_impl) and when
# the kda dir is put on sys.path directly (the op tests import it that way).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _device_guard import _bound_device_ok  # noqa: E402


# Inner subchunk size (SUB) and outer tile (OUT). SUB bounds the cumulative-gate
# range to SUB*5 so exp(-G) stays in fp32 range; OUT=4*SUB batches the safe
# row-local vector work without widening that gate window (block-diagonal tril).
SUB = 16
OUT = 64
NSUB = OUT // SUB  # 4


# ---------------------------------------------------------------------------
# Triangular inverse via nilpotent doubling.
#   For a strict-lower nilpotent m whose sixteenth power vanishes, the inverse
#   of (I minus m) equals the product over j of (I plus m raised to the power
#   two-to-the-j). With a sub-block of 16, the sixteenth power is zero, so four
#   doublings (which reach the power of 16) suffice. We want the inverse of
#   (I plus a_kk), so we pass m as the negation of a_kk.
# ---------------------------------------------------------------------------

def inverse_nilpotent(m, eye16):
    """m, eye16: 16-by-16; returns the inverse of (I minus m). With SUB equal to
    16 this needs four nilpotent doublings.
    """
    # matmul tile m16 k16 n16
    pypto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])
    pypto.set_vec_tile_shapes(16, 16)
    acc = eye16 + m                                          # I + m
    p = m
    for _ in range(3):
        p = pypto.matmul(p, p, pypto.DT_FP32)                # m^{2^j}
        acc = pypto.matmul(acc, eye16 + p, pypto.DT_FP32)    # acc *= (I + m^{2^j})
    return acc


# ---------------------------------------------------------------------------
# Per-64-row outer tile: widened vector pre-compute + 4 serial 16-subchunks.
# ---------------------------------------------------------------------------

def pypto_kda_outer64(query, key, value, gate, beta, state, tril_le64, tril_le16, tril_lt16, eye16, scale):
    """query,key,gate: [64,K]; value: [64,V]; beta: [64,1]; state: [V,K].
    Returns o [64,V], state_new [V,K]. Vector pre-compute batched at 64 rows;
    cube/state work done per 16-sub-block with serial state carry.
    """
    # ---- widened (64-row) vector pre-compute ----
    pypto.set_vec_tile_shapes(64, 128)
    # block-diagonal cumulative gate: [64,64]@[64,128]->[64,128]
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [128, 128])
    g_cum = pypto.matmul(tril_le64, gate, pypto.DT_FP32)     # [64,K] per-block cumsum (<=0)
    pypto.set_vec_tile_shapes(64, 128)
    exp_g = pypto.exp(g_cum)                                 # (0,1]
    exp_neg_g = pypto.exp(g_cum * (-1.0))                    # bounded by exp(SUB*5)
    qd = query * exp_g                                       # [64,K]
    kd = key * exp_neg_g                                     # [64,K]
    kg_pos = key * exp_g                                     # [64,K]
    bv = value * beta                                        # [64,V]
    bgk = kg_pos * beta                                      # [64,K]

    o_blocks = []
    cur_state = state                                       # [V,K] serial carry
    for c in range(NSUB):
        s0 = c * SUB
        qd_b = qd[s0:s0 + SUB, :]
        kd_b = kd[s0:s0 + SUB, :]
        kgp_b = kg_pos[s0:s0 + SUB, :]
        bv_b = bv[s0:s0 + SUB, :]
        bgk_b = bgk[s0:s0 + SUB, :]
        beta_b = beta[s0:s0 + SUB, :]
        gcum_b = g_cum[s0:s0 + SUB, :]
        key_b = key[s0:s0 + SUB, :]

        # a_qk: scaled qd·kdᵀ masked by tril_le ; [16,128]@[128,16]->[16,16]
        pypto.set_cube_tile_shapes([16, 16], [128, 128], [16, 16])
        aqk = pypto.matmul(qd_b, kd_b, pypto.DT_FP32, b_trans=True)
        pypto.set_vec_tile_shapes(16, 16)
        aqk = aqk * scale * tril_le16

        pypto.set_cube_tile_shapes([16, 16], [128, 128], [16, 16])
        akk = pypto.matmul(kgp_b, kd_b, pypto.DT_FP32, b_trans=True)
        pypto.set_vec_tile_shapes(16, 16)
        akk = (akk * beta_b) * tril_lt16                    # strict-lower
        t_inv = inverse_nilpotent(akk * (-1.0), eye16)      # inverse of (I + a_kk)

        # u and w: apply the triangular inverse to beta*v and beta*exp(G)*k ; [16,16]@[16,128]->[16,128]
        pypto.set_cube_tile_shapes([16, 16], [16, 16], [128, 128])
        u = pypto.matmul(t_inv, bv_b, pypto.DT_FP32)
        w = pypto.matmul(t_inv, bgk_b, pypto.DT_FP32)

        # w_s: w times stateᵀ (state-coupled) ; [16,128]@[128,128]->[16,128]
        pypto.set_cube_tile_shapes([16, 16], [128, 128], [128, 128])
        w_s = pypto.matmul(w, cur_state, pypto.DT_FP32, b_trans=True)
        pypto.set_vec_tile_shapes(16, 128)
        v_new = u - w_s                                     # [16,V]

        # o_inter: scaled qd times stateᵀ (state-coupled) ; [16,128]@[128,128]->[16,128]
        pypto.set_cube_tile_shapes([16, 16], [128, 128], [128, 128])
        o_inter = pypto.matmul(qd_b, cur_state, pypto.DT_FP32, b_trans=True)
        pypto.set_vec_tile_shapes(16, 128)
        o_inter = o_inter * scale
        # o_intra: a_qk times v_new ; [16,16]@[16,128]->[16,128]
        pypto.set_cube_tile_shapes([16, 16], [16, 16], [128, 128])
        o_intra = pypto.matmul(aqk, v_new, pypto.DT_FP32)
        pypto.set_vec_tile_shapes(16, 128)
        o_b = o_inter + o_intra                             # [16,V]
        o_blocks.append(o_b)

        # state update for this sub-block (serial carry)
        g_last = gcum_b[SUB - 1:SUB, :]                     # [1,K]
        exp_g_last = pypto.exp(g_last)                      # [1,K]
        kg = key_b * pypto.exp(g_last - gcum_b)             # [16,K] (arg<=0)
        # upd: v_newᵀ contracted with kg ; [128,16]@[16,128]->[128,128]
        pypto.set_cube_tile_shapes([128, 128], [16, 16], [128, 128])
        upd = pypto.matmul(v_new, kg, pypto.DT_FP32, a_trans=True)
        pypto.set_vec_tile_shapes(128, 128)
        cur_state = cur_state * exp_g_last + upd            # [V,K]

    o = pypto.concat(o_blocks, dim=0)                       # [64,V]
    return o, cur_state


# ---------------------------------------------------------------------------
# Layer I — kernel impl (loops over BH and 64-row outer tiles).
# ---------------------------------------------------------------------------

def _kda_chunk_kernel_impl(query, key, value, gate, beta, states,
                           tril_le64, tril_le16, tril_lt16, eye16,
                           core_attn_out, last_state_data):
    """query,key,gate: [BH,T,K]; value: [BH,T,V]; beta: [BH,T,1];
    states/last_state_data: [BH,V,K]; core_attn_out: [BH,T,V].
    T is padded to a multiple of OUT=64 by the host wrapper."""
    bh = states.shape[0]
    t = query.shape[1]
    d = query.shape[2]
    scale = float(d) ** -0.5

    for bh_idx in pypto.loop(bh, name="LOOP_BH", idx_name="bh_idx", parallel=True):
        pypto.set_vec_tile_shapes(128, 128)
        last_state = states[bh_idx]                          # [V,K] (loop-carried, in-place)
        for s_idx in pypto.loop(0, t, OUT, name="LOOP_S", idx_name="s_idx",
                                submit_before_loop=True):
            q_v = pypto.view(query, [1, OUT, 128], [bh_idx, s_idx, 0]).reshape([OUT, 128])
            k_v = pypto.view(key, [1, OUT, 128], [bh_idx, s_idx, 0]).reshape([OUT, 128])
            v_v = pypto.view(value, [1, OUT, 128], [bh_idx, s_idx, 0]).reshape([OUT, 128])
            g_v = pypto.view(gate, [1, OUT, 128], [bh_idx, s_idx, 0]).reshape([OUT, 128])
            b_v = pypto.view(beta, [1, OUT, 1], [bh_idx, s_idx, 0]).reshape([OUT, 1])

            o, cur_state = pypto_kda_outer64(q_v, k_v, v_v, g_v, b_v, last_state,
                tril_le64, tril_le16, tril_lt16, eye16, scale)

            o_blk = o.reshape([1, OUT, 128])
            pypto.assemble(o_blk, [bh_idx, s_idx, 0], core_attn_out)
            last_state[:] = cur_state                         # in-place loop carry

        ls_blk = last_state.reshape([1, 128, 128])
        pypto.assemble(ls_blk, [bh_idx, 0, 0], last_state_data)


# ---------------------------------------------------------------------------
# Layer J — JIT entry (module-level dynamic shapes).
# ---------------------------------------------------------------------------

_BH = pypto.DYNAMIC
_T = pypto.DYNAMIC
_QKG = [_BH, _T, 128]
_V = [_BH, _T, 128]
_BETA = [_BH, _T, 1]
_STATE = [_BH, 128, 128]
_OUT = [_BH, _T, 128]
_LL64 = [OUT, OUT]
_LL16 = [SUB, SUB]


@pypto.frontend.jit
def kda_chunk_npu(
    query: pypto.Tensor(_QKG, pypto.DT_FP32),
    key: pypto.Tensor(_QKG, pypto.DT_FP32),
    value: pypto.Tensor(_V, pypto.DT_FP32),
    gate: pypto.Tensor(_QKG, pypto.DT_FP32),
    beta: pypto.Tensor(_BETA, pypto.DT_FP32),
    states: pypto.Tensor(_STATE, pypto.DT_FP32),
    tril_le64: pypto.Tensor(_LL64, pypto.DT_FP32),
    tril_le16: pypto.Tensor(_LL16, pypto.DT_FP32),
    tril_lt16: pypto.Tensor(_LL16, pypto.DT_FP32),
    eye16: pypto.Tensor(_LL16, pypto.DT_FP32),
    core_attn_out: pypto.Tensor(_OUT, pypto.DT_FP32),
    last_state_data: pypto.Tensor(_STATE, pypto.DT_FP32),
):
    _kda_chunk_kernel_impl(query, key, value, gate, beta, states,
                           tril_le64, tril_le16, tril_lt16, eye16,
                           core_attn_out, last_state_data)


# ---------------------------------------------------------------------------
# Layer K — host wrapper (FLA call signature, @allow_in_graph).
# ---------------------------------------------------------------------------

_K = 128
_V_DIM = 128


def _build_tril_eye(device):
    """Build the four constant mask/identity tensors the kernel needs:
    block-diagonal 64-by-64 lower-tri (4 independent 16-by-16 blocks, zero
    off-block) so the cumulative-gate window stays within SUB which is 16 steps,
    plus the 16-by-16 lower-tri (inclusive), strict-lower, and identity.
    Returns (tril_le64, tril_le16, tril_lt16, eye16)."""
    idx64 = torch.arange(OUT, device=device)
    blk = (idx64[:, None] // SUB) == (idx64[None, :] // SUB)
    tril_le64 = ((idx64[:, None] >= idx64[None, :]) & blk).float()
    idx16 = torch.arange(SUB, device=device)
    tril_le16 = (idx16[:, None] >= idx16[None, :]).float()
    tril_lt16 = (idx16[:, None] > idx16[None, :]).float()
    eye16 = torch.eye(SUB, dtype=torch.float32, device=device)
    return tril_le64, tril_le16, tril_lt16, eye16


@allow_in_graph
def kda_chunk_wrapper(q, k, v, g, beta, initial_state=None,
                      output_final_state=True, use_qk_l2norm_in_kernel=True,
                      cu_seqlens=None, scale=None, **kwargs):
    """FLA-layout chunk/prefill KDA wrapper for Kimi-Linear-48B-A3B.

    Mirrors the upstream ``fla.ops.kda.chunk_kda`` signature so the modeling
    layer can swap it in when ``USE_PTO_KDA`` is set. Nested 64-outer / 16-inner
    tiling (inner subchunk SUB=16 keeps the exp window fp32-safe).

    q,k,v,g: [B,T,H,128]; beta: [B,T,H]; initial_state: [B,H,V,K] fp32 or None.
    Returns ``(out [B,T,H,V] same dtype as q, final_state [B,H,V,K] fp32)``.

    Constraints (raises ``NotImplementedError`` otherwise):
      * head_k_dim K = 128, head_v_dim V = 128
      * ``use_qk_l2norm_in_kernel`` is True
    """
    in_dtype = q.dtype
    batch, seq_len, num_heads, head_k_dim = q.shape
    head_v_dim = v.shape[-1]

    if (head_k_dim != _K) or (head_v_dim != _V_DIM) or (not use_qk_l2norm_in_kernel):
        raise NotImplementedError(
            "kda_chunk_wrapper requires head_k_dim=128, head_v_dim=128, "
            "use_qk_l2norm_in_kernel=True; "
            f"got K={head_k_dim}, V={head_v_dim}, use_qk_l2norm_in_kernel={use_qk_l2norm_in_kernel}"
        )

    device = q.device
    if not _bound_device_ok(device):
        raise NotImplementedError(
            "PyPTO KDA binds to a single NPU per process; tensor is on a "
            "different NPU -> torch fallback (for an NPU-sharded model, only "
            "the bound device's KDA layers run on PyPTO)")
    bh = batch * num_heads

    # L2 norm in kernel (match golden exactly), done on host before packing.
    qf = q.float()
    kf = k.float()
    qf = qf / (qf.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)
    kf = kf / (kf.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)

    pad = (OUT - seq_len % OUT) % OUT
    seq_len_pad = seq_len + pad

    def pack(x, lastdim):
        x = x.permute(0, 2, 1, 3).contiguous().reshape(bh, seq_len, lastdim)
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        return x.contiguous()

    q2 = pack(qf, head_k_dim)
    k2 = pack(kf, head_k_dim)
    v2 = pack(v.float(), head_v_dim)
    g2 = pack(g.float(), head_k_dim)
    b2 = beta.float().permute(0, 2, 1).contiguous().reshape(bh, seq_len, 1)
    if pad:
        b2 = torch.nn.functional.pad(b2, (0, 0, 0, pad))
    b2 = b2.contiguous()

    if initial_state is None:
        st = torch.zeros(bh, head_v_dim, head_k_dim, dtype=torch.float32, device=device)
    else:
        st = initial_state.float().reshape(bh, head_v_dim, head_k_dim).contiguous()

    tril_le64, tril_le16, tril_lt16, eye16 = _build_tril_eye(device)

    out2 = torch.zeros(bh, seq_len_pad, head_v_dim, dtype=torch.float32, device=device)
    last_state = torch.zeros(bh, head_v_dim, head_k_dim, dtype=torch.float32, device=device)

    kda_chunk_npu(q2, k2, v2, g2, b2, st,
                  tril_le64, tril_le16, tril_lt16, eye16,
                  out2, last_state)
    # No explicit synchronize: the kernel enqueues on the current stream, so the
    # reads below (and the model's downstream ops) are stream-ordered after it.
    # Dropping the device-wide sync is required for torch.compile/aclgraph capture
    # and avoids serializing every KDA layer. Verified: op tests bit-identical and
    # aclgraph capture+replay bit-exact without it.
    out = out2[:, :seq_len].reshape(batch, num_heads, seq_len, head_v_dim).permute(0, 2, 1, 3).contiguous().to(in_dtype)
    final_state = last_state.reshape(batch, num_heads, head_v_dim, head_k_dim)
    return out, (final_state if output_final_state else None)


# ---------------------------------------------------------------------------
# Layer L — torch.library registration (torch.compile / aclgraph capture).
# ---------------------------------------------------------------------------
# `@allow_in_graph` above lets dynamo treat the wrapper as an opaque call, but
# graph capture (torchair / aclgraph) additionally needs a Meta ("fake") impl so
# the tracer can infer output shapes/dtypes WITHOUT launching the NPU kernel.
# Registering the op as `pypto::kda_chunk_kimi` (Meta + NPU keys) provides that;
# the model reaches it via `kda_chunk_pypto` only when aclgraph is enabled, so
# the default eager dispatch path (kda_chunk_wrapper) is unchanged. Mirrors the
# sibling phi_3_mini_4k_instruct/rms_norm registration. output_final_state is
# fixed True here (the cache always needs the state); cu_seqlens must be None.
pyptolib = torch.library.Library("pypto", "FRAGMENT")  # type: ignore[arg-type]

if not hasattr(torch.ops.pypto, "kda_chunk_kimi"):
    pyptolib.define(
        "kda_chunk_kimi(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
        "Tensor? initial_state, float scale, bool use_qk_l2norm_in_kernel) "
        "-> (Tensor, Tensor)")

    @torch.library.impl(pyptolib, "kda_chunk_kimi", "Meta")  # type: ignore[arg-type]
    def _kda_chunk_kimi_meta(q, k, v, g, beta, initial_state, scale,
                             use_qk_l2norm_in_kernel):
        batch, seq_len, num_heads, head_k_dim = q.shape
        head_v_dim = v.shape[-1]
        out = torch.empty((batch, seq_len, num_heads, head_v_dim),
                          dtype=q.dtype, device=q.device)
        final_state = torch.empty((batch, num_heads, head_v_dim, head_k_dim),
                                  dtype=torch.float32, device=q.device)
        return out, final_state

    @torch.library.impl(pyptolib, "kda_chunk_kimi", "NPU")  # type: ignore[arg-type]
    def _kda_chunk_kimi_npu(q, k, v, g, beta, initial_state, scale,
                            use_qk_l2norm_in_kernel):
        return kda_chunk_wrapper(
            q, k, v, g, beta, initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, scale=scale)


def kda_chunk_pypto(q, k, v, g, beta, initial_state=None, scale=None,
                    use_qk_l2norm_in_kernel=True):
    """aclgraph-capturable entry: routes through ``torch.ops.pypto.kda_chunk_kimi``.

    Same result as ``kda_chunk_wrapper`` (always returns ``(out, final_state)``),
    but as a registered custom op it can be captured by torch.compile / aclgraph.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return torch.ops.pypto.kda_chunk_kimi(
        q, k, v, g, beta, initial_state, scale, use_qk_l2norm_in_kernel)
