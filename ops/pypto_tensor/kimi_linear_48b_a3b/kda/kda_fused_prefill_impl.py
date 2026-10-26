# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""kda_fused_prefill — fused chunked Kimi Delta Attention (KDA) prefill kernel.

PyPTO kernel replaces the whole per-layer KDA attention core between
the projections and ``o_proj``:
    conv1d(k=4)+silu (q,k,v) -> l2norm -> gate transform -> beta sigmoid
    -> WY-chunked gated delta-rule -> o_norm (RMSNorm * sigmoid gate)
    -> bf16 output

CORE MATH (per 128-token chunk, per (b,h), state S = [K,V]):
    gcum  = tril_incl @ g                       # [128,K] per-channel cumsum
    A     = tril-decay(k)  /  A_qk = tril(q)    # 16-row pivot bands (see below)
    a_inv = (I + beta*A_strict)^-1 * beta       # 8x16 hierarchical block inverse
    w,u   = a_inv @ (k*exp(gcum)), a_inv @ v    # WY/UT transform
    vi    = u - w @ S                           # chunk-local correction
    oc    = (q*exp(gcum)) @ S + A_qk @ vi       # output
    S'    = S * exp(gcum[last]) + kdec^T @ vi   # state update (no dense KxK
                                                #  transition is materialized)
"""
__all__ = ["kda_fused_prefill", "KdaPrefillInputs", "KdaPrefillWeights"]

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pypto
import torch
import torch_npu  # noqa: F401  required for NPU device init
from torch._dynamo import allow_in_graph

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device_guard import _bound_device_ok  # noqa: E402

OUT = 128          # WY chunk length (static)
_BC = 16           # intra-chunk decay pivot band (exp window <= 15*5 = 75)
_NC = OUT // _BC   # 8 bands per chunk
_DECAY_CAP = 80.0  # clamp col-factor exponent below fp32 exp overflow (~88.7)
_L2_EPS = 1e-6


# ---------------------------------------------------------------------------
# Host-side argument groups (public API)
# ---------------------------------------------------------------------------

class KdaPrefillInputs(NamedTuple):
    """Per-call fused-prefill input tensors."""
    q: torch.Tensor      # [B,T,H,128] pre-conv projection (bf16)
    k: torch.Tensor
    v: torch.Tensor
    g_raw: torch.Tensor  # [B,T,H*128] f_b_proj output (any float dtype)
    g_gate: torch.Tensor  # [B,T,H*128] g_b_proj output
    b_raw: torch.Tensor  # [B,T,H] b_proj output


class KdaPrefillWeights(NamedTuple):
    """Per-layer weight-only parameters plus config scalars."""
    dt_bias: torch.Tensor   # [H*128] fp32 parameter
    a_log: torch.Tensor     # [1,1,H,1] or [H] fp32 parameter (log gate-decay)
    o_norm_w: torch.Tensor  # [128] fp32 o_norm weight
    conv_wq: torch.Tensor   # [H*128, 4] fp32 depthwise conv weight
    conv_wk: torch.Tensor   # [H*128, 4] fp32 depthwise conv weight
    conv_wv: torch.Tensor   # [H*128, 4] fp32 depthwise conv weight
    lower_bound: float = -5.0
    rms_eps: float = 1e-5


class _PackedPrefill(NamedTuple):
    """Host-packed kernel operands plus the dims the unpack step needs."""
    x_q: torch.Tensor      # [BH, 3+Tpad, 128] fp32 pre-conv
    x_k: torch.Tensor
    x_v: torch.Tensor
    cw_q: torch.Tensor     # [BH, 4, 128] fp32 per-tap conv weights
    cw_k: torch.Tensor
    cw_v: torch.Tensor
    g2: torch.Tensor       # [BH, Tpad, 128] fp32 gate-raw
    gg2: torch.Tensor      # [BH, Tpad, 128] fp32 output gate
    b2: torch.Tensor       # [BH, Tpad, 1] fp32 beta-raw
    a_scl: torch.Tensor    # [BH, 1, 1] fp32 exp(a_log)
    a_bias: torch.Tensor   # [BH, 1, 128] fp32 exp(a_log)*dt_bias
    o_w: torch.Tensor      # [1, 128] fp32 o_norm weight
    batch: int
    heads: int
    seq_len: int
    pad: int
    head_k_dim: int
    head_v_dim: int
    device: torch.device
    in_dtype: torch.dtype


# ---------------------------------------------------------------------------
# Traced-side operand groups
# ---------------------------------------------------------------------------

@dataclass
class _PrefillIO:
    """The JIT entry's device tensors + trace-time scalars."""
    x_q: any
    x_k: any
    x_v: any
    cw_q: any
    cw_k: any
    cw_v: any
    g_raw: any
    g_gate: any
    b_raw: any
    a_scl: any
    a_bias: any
    o_norm_w: any
    states: any
    core_attn_out: any
    last_state_data: any
    lb: float
    rms_eps: float


@dataclass
class _WyConsts:
    """Launch-invariant in-kernel constants (built once per launch)."""
    tril_incl: any
    trils_neg: any
    eye_stack: any
    z16: any
    z32: any
    z64: any


@dataclass
class _HeadParams:
    """Per-(b,h) gate params plus the shared scalars for the chunk compute."""
    a_scl: any
    a_bias: any
    o_w: any
    scale: float
    lb: float
    rms_eps: float


@dataclass
class _ChunkInputs:
    """One 128-token chunk tile set + the state carry entering it."""
    qc: any
    kc: any
    vc: any
    g_raw: any
    b_raw: any
    g_gate: any
    s_carry: any


@dataclass
class _MergeSpec:
    """One hierarchical-inverse merge quadrant: position, size, zero block."""
    x_ofs: int
    y_ofs: int
    m_len: int
    zero_t: any


# ---------------------------------------------------------------------------
# 8x16 block matrix inverse
# ---------------------------------------------------------------------------

def _inv_min_length(attn_dim1, eye, row_num, col_num):
    size = col_num // row_num                                   # 8
    pypto.set_vec_tile_shapes(128, 128)
    cur = attn_dim1[:2, :]                                      # [2,128] rows 0,1
    for i in range(2, row_num, 1):                              # 2..15
        cur = cur + 0.0                                         # land in UB
        row = attn_dim1.view([1, col_num], [i, 0])              # [1,128]
        row_exp = (row.reshape([size, row_num])
                   .view([size, i], [0, 0]).transpose(1, 0)
                   .reshape([size * i, 1]))
        cur_r = cur.reshape([size * i, row_num])                # [8i,16]
        prod_mul = (row_exp * cur_r).reshape([i, col_num])      # [i,128]
        prod = pypto.sum(prod_mul, dim=0, keepdim=True)         # [1,128]
        cur = pypto.concat([cur, row + prod], dim=0)            # grow [i+1,128]
    return cur + eye                                            # [16,128] add diag I


def _inv_merge(attn, inv11, inv22, spec):
    """Merge two inverted diagonal blocks into the [2m,2m] block inverse."""
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    a21 = attn.view([spec.m_len, spec.m_len],
                    [spec.x_ofs + spec.m_len, spec.y_ofs])      # [m,m] lower-left
    a21_inv = pypto.matmul(pypto.matmul(inv22, a21, pypto.DT_FP32), inv11, pypto.DT_FP32)
    pypto.set_vec_tile_shapes(128, 128)
    top = pypto.concat([inv11, spec.zero_t], dim=1)             # [m,2m] inv11 | 0
    bot = pypto.concat([a21_inv, inv22], dim=1)                 # [m,2m] a21_inv | inv22
    return pypto.concat([top, bot], dim=0)                      # [2m,2m]


def _inverse8(neg_a, eye_stack, z16, z32, z64):
    """Hierarchical inverse"""
    pypto.set_vec_tile_shapes(128, 128)
    diag = [neg_a.view([16, 16], [16 * i, 16 * i]) + 0.0 for i in range(8)]
    diag_inv = _inv_min_length(pypto.concat(diag, dim=1), eye_stack, 16, OUT)  # [16,128]
    li = [diag_inv[:, 16 * i:16 * (i + 1)] + 0.0 for i in range(8)]
    l4 = [_inv_merge(neg_a, li[2 * i], li[2 * i + 1],
                     _MergeSpec(32 * i, 32 * i, 16, z16)) for i in range(4)]
    l2 = [_inv_merge(neg_a, l4[2 * i], l4[2 * i + 1],
                     _MergeSpec(64 * i, 64 * i, 32, z32)) for i in range(2)]
    return _inv_merge(neg_a, l2[0], l2[1], _MergeSpec(0, 0, 64, z64))  # [128,128]


# ---------------------------------------------------------------------------
# Intra-chunk decay A (k-side) / A_qk (q-side), 8 pivot bands of 16 rows
# ---------------------------------------------------------------------------

def _intra_chunk_a(q_s, k_s, gcum):
    pypto.set_vec_tile_shapes(_BC, 128)
    pypto.set_cube_tile_shapes([_BC, _BC], [128, 128], [OUT, OUT])
    af_bands = []
    a2_bands = []
    for a in range(_NC):
        r0 = a * _BC
        gp = gcum[r0:r0 + 1, :]                       # [1,K] pivot at band start
        g_band = gcum[r0:r0 + _BC, :]                 # [BC,K]
        row_dec = pypto.exp(g_band - gp)              # exponent <= 0
        kg_a = k_s[r0:r0 + _BC, :] * row_dec
        qg_a = q_s[r0:r0 + _BC, :] * row_dec
        col_dec = pypto.exp(pypto.minimum(gp - gcum, _DECAY_CAP))  # clamp future
        ki_a = k_s * col_dec
        af_bands.append(pypto.matmul(kg_a, ki_a, pypto.DT_FP32, b_trans=True))
        a2_bands.append(pypto.matmul(qg_a, ki_a, pypto.DT_FP32, b_trans=True))
    return pypto.concat(af_bands, dim=0), pypto.concat(a2_bands, dim=0)


# ---------------------------------------------------------------------------
# Per-chunk glue helpers
# ---------------------------------------------------------------------------

def _l2_normalise_rows(x):
    return pypto.div(x, pypto.add(pypto.sqrt(pypto.sum(pypto.mul(x, x), -1, True)), _L2_EPS))


def _sigmoid(x):
    return pypto.reciprocal(pypto.add(pypto.exp(pypto.mul(x, -1.0)), 1.0))


def _conv_silu(x_pad, bh_idx, s_idx, taps):
    """Causal 4-tap depthwise conv + silu for one [OUT,128] chunk slice."""
    pypto.set_vec_tile_shapes(128, 128)
    x3 = pypto.view(x_pad, [1, OUT, 128], [bh_idx, s_idx + 3, 0]).reshape([OUT, 128])
    acc = pypto.mul(x3, taps[3])
    x2 = pypto.view(x_pad, [1, OUT, 128], [bh_idx, s_idx + 2, 0]).reshape([OUT, 128])
    acc = pypto.add(acc, pypto.mul(x2, taps[2]))
    x1 = pypto.view(x_pad, [1, OUT, 128], [bh_idx, s_idx + 1, 0]).reshape([OUT, 128])
    acc = pypto.add(acc, pypto.mul(x1, taps[1]))
    x0 = pypto.view(x_pad, [1, OUT, 128], [bh_idx, s_idx, 0]).reshape([OUT, 128])
    acc = pypto.add(acc, pypto.mul(x0, taps[0]))
    return pypto.mul(acc, _sigmoid(acc))             # silu


def _chunk_compute(chunk, consts, head):
    """One 128-token chunk: WY core + fused gate/beta/o_norm glue."""
    pypto.set_vec_tile_shapes(128, 128)
    q_f = _l2_normalise_rows(chunk.qc)                   # [128,K]
    k_f = _l2_normalise_rows(chunk.kc)                   # [128,K]
    q_s = q_f * head.scale
    k_s = k_f
    # --- fused gate transform: g = lb * sigmoid(a*g_raw + a*bias) ---
    gc = _sigmoid(pypto.add(pypto.mul(head.a_scl, chunk.g_raw), head.a_bias)) * head.lb
    bc = _sigmoid(chunk.b_raw)                            # [128,1] fused beta
    pypto.set_cube_tile_shapes([64, 128], [128, 128], [128, 128])
    pypto.set_vec_tile_shapes(128, 128)
    gcum = pypto.matmul(consts.tril_incl, gc, pypto.DT_FP32)   # [128,K]
    eg = pypto.exp(gcum)
    kg = k_s * eg                                       # cross-chunk state term
    a_full, a2_raw = _intra_chunk_a(q_s, k_s, gcum)     # unmasked [128,128] x2
    pypto.set_vec_tile_shapes(128, 128)
    bc_col = bc.transpose(0, 1)                         # [1,128] col-scale
    neg_a = a_full * bc * consts.trils_neg  # bc [128,1] row-scale; trils_neg carries -1
    a_inv = _inverse8(neg_a, consts.eye_stack, consts.z16, consts.z32, consts.z64)
    a_inv = a_inv * bc_col
    pypto.set_cube_tile_shapes([64, 128], [128, 128], [128, 128])
    # all-fp32 WY/output paths (oc, s_new, vi clean of bf16 rounding)
    w = pypto.matmul(a_inv, kg, pypto.DT_FP32)              # [128,K] fp32
    u = pypto.matmul(a_inv, chunk.vc, pypto.DT_FP32)        # [128,V] fp32
    qg = q_s * eg
    a2 = a2_raw * consts.tril_incl
    vi = u - pypto.matmul(w, chunk.s_carry, pypto.DT_FP32)  # [128,V] fp32
    oc = pypto.matmul(qg, chunk.s_carry, pypto.DT_FP32) \
        + pypto.matmul(a2, vi, pypto.DT_FP32)
    glast = pypto.view(gcum, [1, 128], [OUT - 1, 0])   # [1,K] (g pad=0 -> flat)
    kdec = k_s * pypto.exp(glast - gcum)                # [128,K]
    s_new = chunk.s_carry * pypto.exp(glast).transpose(0, 1) \
        + pypto.matmul(kdec, vi, pypto.DT_FP32, a_trans=True)  # [K,V] fp32
    # --- fused o_norm: RMSNorm * w * sigmoid(gate), bf16 out ---
    var = pypto.sum(pypto.mul(oc, oc), 1, True) * (1.0 / 128.0)
    on = pypto.mul(pypto.mul(oc, pypto.rsqrt(pypto.add(var, head.rms_eps))), head.o_w)
    og = pypto.mul(on, _sigmoid(chunk.g_gate))          # pad rows: on~0 -> og~0
    return pypto.cast(og, pypto.DT_BF16), s_new


# ---------------------------------------------------------------------------
# In-kernel loop-invariant constants, built once per launch
# ---------------------------------------------------------------------------

def _build_consts_inkernel():
    pypto.set_vec_tile_shapes(16, 64)
    t16 = pypto.tril(pypto.ones([16, 16], dtype=pypto.DT_FP32), 0)
    s16 = pypto.tril(pypto.ones([16, 16], dtype=pypto.DT_FP32), -1)
    e16 = t16 - s16                                       # eye 16x16
    pypto.set_vec_tile_shapes(16, 128)
    eye_stack = pypto.concat([e16] * 8, dim=1)            # [16,128] 8 diag I blocks
    pypto.set_vec_tile_shapes(128, 128)
    o128 = pypto.ones([128, 128], dtype=pypto.DT_FP32)
    tril_incl = pypto.tril(o128, 0)                       # [128,128] incl-lower
    trils_neg = pypto.tril(o128, -1) * (-1.0)             # strict-lower, -1 folded
    # zero blocks for the hierarchical merge levels: 16x16, 32x32, 64x64
    z16 = pypto.zeros([16, 16], dtype=pypto.DT_FP32)
    z32 = pypto.zeros([32, 32], dtype=pypto.DT_FP32)
    z64 = pypto.zeros([64, 64], dtype=pypto.DT_FP32)
    return _WyConsts(tril_incl=tril_incl, trils_neg=trils_neg, eye_stack=eye_stack,
                     z16=z16, z32=z32, z64=z64)


# ---------------------------------------------------------------------------
# Kernel impl: BH loop + 128-row WY chunk loop with a [K,V] state carry
# ---------------------------------------------------------------------------

def _kda_fused_prefill_kernel_impl(io):
    """io: _PrefillIO — x_q/x_k/x_v: [BH, 3+T, 128] PRE-conv (3 left-pad
    rows + right-pad %128); cw_*: [BH, 4, 128] per-tap depthwise conv
    weights; g_raw/g_gate: [BH,T,K]; b_raw: [BH,T,1]; a_scl: [BH,1,1];
    a_bias: [BH,1,K]; o_norm_w: [1,K]; states/last_state_data: [BH,V,K];
    core_attn_out: [BH,T,V] bf16 (post-conv+chunk+norm, gated)."""
    bh = io.states.shape[0]
    t = io.x_q.shape[1] - 3                            # padded seq len
    scale = float(io.x_q.shape[2]) ** -0.5
    consts = _build_consts_inkernel()

    for bh_idx in pypto.loop(bh, name="LOOP_BH", idx_name="bh_idx", parallel=True):
        pypto.experimental.set_operation_options(combine_axis=True)
        pypto.set_vec_tile_shapes(128, 128)
        s_seed = io.states[bh_idx]                     # [V,K]
        s_carry = s_seed.transpose(0, 1) + 0.0         # [K,V] internal carry
        # per-(b,h) gate params + conv tap weights (loop-invariant)
        a_scl_h = pypto.reshape(pypto.view(io.a_scl, [1, 1, 1], [bh_idx, 0, 0]), [1, 1])
        a_bias_h = pypto.reshape(pypto.view(io.a_bias, [1, 1, 128], [bh_idx, 0, 0]), [1, 128])
        taps_q = [pypto.reshape(pypto.view(io.cw_q, [1, 1, 128], [bh_idx, j, 0]),
                                [1, 128]) for j in range(4)]
        taps_k = [pypto.reshape(pypto.view(io.cw_k, [1, 1, 128], [bh_idx, j, 0]),
                                [1, 128]) for j in range(4)]
        taps_v = [pypto.reshape(pypto.view(io.cw_v, [1, 1, 128], [bh_idx, j, 0]),
                                [1, 128]) for j in range(4)]
        head = _HeadParams(a_scl=a_scl_h, a_bias=a_bias_h, o_w=io.o_norm_w,
                           scale=scale, lb=io.lb, rms_eps=io.rms_eps)
        for s_idx in pypto.loop(0, t, OUT, name="LOOP_S", idx_name="s_idx",
                                submit_before_loop=False, unroll_list=[4]):
            chunk = _ChunkInputs(
                qc=_conv_silu(io.x_q, bh_idx, s_idx, taps_q),   # [OUT,128] post-conv
                kc=_conv_silu(io.x_k, bh_idx, s_idx, taps_k),
                vc=_conv_silu(io.x_v, bh_idx, s_idx, taps_v),
                g_raw=pypto.view(io.g_raw, [1, OUT, 128], [bh_idx, s_idx, 0]).reshape([OUT, 128]),
                b_raw=pypto.view(io.b_raw, [1, OUT, 1], [bh_idx, s_idx, 0]).reshape([OUT, 1]),
                g_gate=pypto.view(io.g_gate, [1, OUT, 128], [bh_idx, s_idx, 0]).reshape([OUT, 128]),
                s_carry=s_carry)

            og, s_new = _chunk_compute(chunk, consts, head)

            pypto.set_vec_tile_shapes(128, 128)
            o_blk = og.reshape([1, OUT, 128])
            pypto.assemble(o_blk, [bh_idx, s_idx, 0], io.core_attn_out)
            s_carry[:] = s_new
            pypto.set_vec_tile_shapes(128, 128)

        ls_blk = s_carry.transpose(0, 1).reshape([1, 128, 128])   # [K,V]->[V,K] ABI
        pypto.assemble(ls_blk, [bh_idx, 0, 0], io.last_state_data)


# ---------------------------------------------------------------------------
# JIT entry
# ---------------------------------------------------------------------------

_BH = pypto.DYNAMIC
_T = pypto.DYNAMIC
_QKG = [_BH, _T, 128]
_BETA = [_BH, _T, 1]
_STATE = [_BH, 128, 128]
_OUT = [_BH, _T, 128]


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 256,
        "device_sched_mode": 1,
        "launch_sched_aicpu_num": 3,
    },
    pass_options={
        "vec_nbuffer_setting": {-2: 1, -1: 32},
        "cube_l1_reuse_setting": {-1: 32},
        "cube_nbuffer_setting": {-1: 8},
    },
)
def _kda_fused_prefill_npu(
    x_q: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, 128], pypto.DT_FP32),
    x_k: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, 128], pypto.DT_FP32),
    x_v: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, 128], pypto.DT_FP32),
    cw_q: pypto.Tensor([pypto.DYNAMIC, 4, 128], pypto.DT_FP32),
    cw_k: pypto.Tensor([pypto.DYNAMIC, 4, 128], pypto.DT_FP32),
    cw_v: pypto.Tensor([pypto.DYNAMIC, 4, 128], pypto.DT_FP32),
    g_raw: pypto.Tensor(_QKG, pypto.DT_FP32),
    g_gate: pypto.Tensor(_QKG, pypto.DT_FP32),
    b_raw: pypto.Tensor(_BETA, pypto.DT_FP32),
    a_scl: pypto.Tensor([pypto.DYNAMIC, 1, 1], pypto.DT_FP32),
    a_bias: pypto.Tensor([pypto.DYNAMIC, 1, 128], pypto.DT_FP32),
    o_norm_w: pypto.Tensor([1, 128], pypto.DT_FP32),
    states: pypto.Tensor(_STATE, pypto.DT_FP32),
    core_attn_out: pypto.Tensor(_OUT, pypto.DT_BF16),
    last_state_data: pypto.Tensor(_STATE, pypto.DT_FP32),
    lb: float = -5.0,
    rms_eps: float = 1e-5,
):
    io = _PrefillIO(x_q=x_q, x_k=x_k, x_v=x_v, cw_q=cw_q, cw_k=cw_k, cw_v=cw_v,
                    g_raw=g_raw, g_gate=g_gate, b_raw=b_raw, a_scl=a_scl,
                    a_bias=a_bias, o_norm_w=o_norm_w, states=states,
                    core_attn_out=core_attn_out,
                    last_state_data=last_state_data, lb=lb, rms_eps=rms_eps)
    _kda_fused_prefill_kernel_impl(io)


# ---------------------------------------------------------------------------
# Host wrapper (model level)
# ---------------------------------------------------------------------------

_K = 128
_V_DIM = 128


def _pack_prefill(inputs, weights, conv_cache):
    """[B,T,H,D] model layout -> [BH, 3+Tpad, D] fp32 kernel operands.
    Returns the packed operands plus the dims the unpack step needs."""
    q = inputs.q
    batch, seq_len, num_heads, head_k_dim = q.shape
    head_v_dim = inputs.v.shape[-1]
    bh = batch * num_heads
    pad = (OUT - seq_len % OUT) % OUT
    csq, csk, csv = (conv_cache if conv_cache is not None else (None, None, None))

    def pack_conv(x, lastdim, cs):
        # [B,T,H,D] -> [BH, 3+Tpad, D] fp32, left 3 rows from cache (or zeros)
        x = x.permute(0, 2, 1, 3).contiguous().reshape(bh, seq_len, lastdim).float()
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        if cs is not None:
            # repack the per-channel conv history into per-head rows
            csb = (cs.reshape(batch, num_heads, lastdim, 3)
                   .permute(0, 1, 3, 2).contiguous().reshape(bh, 3, lastdim))
            x = torch.cat([csb, x], dim=1)
        else:
            x = torch.nn.functional.pad(x, (0, 0, 3, 0))
        return x

    def pack(x, lastdim):
        x = x.permute(0, 2, 1, 3).contiguous().reshape(bh, seq_len, lastdim)
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        return x

    xq2 = pack_conv(q, head_k_dim, csq)
    xk2 = pack_conv(inputs.k, head_k_dim, csk)
    xv2 = pack_conv(inputs.v, head_v_dim, csv)
    g2 = pack(inputs.g_raw.float().reshape(batch, seq_len, num_heads, head_k_dim), head_k_dim)
    gg2 = pack(inputs.g_gate.float().reshape(batch, seq_len, num_heads, head_v_dim), head_v_dim)
    b2 = inputs.b_raw.float().permute(0, 2, 1).contiguous().reshape(bh, seq_len, 1)
    if pad:
        b2 = torch.nn.functional.pad(b2, (0, 0, 0, pad), value=-100.0)  # sigmoid -> 0

    def pack_cw(w):
        # [H*128, 4] -> [H, 4, 128] -> repeat batch -> [BH, 4, 128] fp32
        w = w.float().reshape(num_heads, 128, 4).permute(0, 2, 1)
        return w.repeat(batch, 1, 1)

    cwq2 = pack_cw(weights.conv_wq)
    cwk2 = pack_cw(weights.conv_wk)
    cwv2 = pack_cw(weights.conv_wv)

    # per-head gate params broadcast over batch: a = exp(a_log) [H]
    a = torch.exp(weights.a_log.float().reshape(num_heads))
    a_scl = a.reshape(num_heads, 1, 1).repeat(batch, 1, 1)
    a_bias = (a.reshape(num_heads, 1, 1)
              * weights.dt_bias.float().reshape(num_heads, 1, head_k_dim)
              ).repeat(batch, 1, 1)
    o_w = weights.o_norm_w.float().reshape(1, head_v_dim)
    if pad:
        # fill the pad rows so the gate transform yields exactly zero
        g_pad = (-100.0 - a_bias) / a_scl                  # [BH,1,128] fp32
        g2[:, seq_len:, :] = g_pad.expand(bh, pad, head_k_dim)

    return _PackedPrefill(
        x_q=xq2, x_k=xk2, x_v=xv2, cw_q=cwq2, cw_k=cwk2, cw_v=cwv2,
        g2=g2, gg2=gg2, b2=b2, a_scl=a_scl, a_bias=a_bias, o_w=o_w,
        batch=batch, heads=num_heads, seq_len=seq_len, pad=pad,
        head_k_dim=head_k_dim, head_v_dim=head_v_dim,
        device=q.device, in_dtype=q.dtype)


def _unpack_prefill(packed, out2, last_state):
    """[BH, Tpad, V] kernel output -> [B,T,H,V] output + [B,H,V,K] state +
    the (cs_q, cs_k, cs_v) conv-history outputs ([B, H*128, 3] format)."""
    batch, heads = packed.batch, packed.heads
    seq_len = packed.seq_len
    out = (out2[:, :seq_len]
           .reshape(batch, heads, seq_len, packed.head_v_dim)
           .permute(0, 2, 1, 3).contiguous())
    if packed.in_dtype != torch.bfloat16:
        out = out.to(packed.in_dtype)
    final_state = last_state.reshape(batch, heads, packed.head_v_dim,
                                     packed.head_k_dim)

    def cs_out(x, lastdim):
        # last 3 VALID pre-conv tokens, [B, H*128, 3] module format
        xc = x[:, seq_len:seq_len + 3, :]
        return (xc.reshape(batch, heads, 3, lastdim)
                .permute(0, 1, 3, 2).contiguous()
                .reshape(batch, heads * lastdim, 3))
    cs_out_t = (cs_out(packed.x_q, packed.head_k_dim),
                cs_out(packed.x_k, packed.head_k_dim),
                cs_out(packed.x_v, packed.head_v_dim))
    return out, final_state, cs_out_t


@allow_in_graph
def kda_fused_prefill(inputs, weights, initial_state=None,
                      output_final_state=True, conv_cache=None, **kwargs):
    """Fused KDA prefill: conv1d+silu + gate transform + beta sigmoid + WY
    chunk + o_norm + output gate, implemented by one fused PyPTO kernel.

    Args:
        inputs: ``KdaPrefillInputs`` — q, k, v ([B,T,H,128] pre-conv
            projections, bf16), g_raw, g_gate ([B,T,H*128] projection
            outputs, any float dtype), b_raw ([B,T,H]).
        weights: ``KdaPrefillWeights`` — per-layer dt_bias ([H*128] fp32),
            a_log ([1,1,H,1] or [H] fp32), o_norm_w ([128] fp32),
            conv_wq/wk/wv ([H*128, 4] depthwise conv weights,
            ``module.weight.reshape(-1, conv_size)``), lower_bound, rms_eps.
        initial_state: [B,H,V,K] fp32 or None.
        conv_cache: optional (cs_q, cs_k, cs_v), each [B, H*128, 3] pre-conv
            history (continuing prefill); None -> zero left-pad.
        **kwargs: ignored (call-site compatibility).

    Returns:
        (o [B,T,H,128] same dtype as inputs.q (fp32 internally),
         final_state [B,H,V,K] fp32 or None,
         conv_cache_out (cs_q, cs_k, cs_v) each [B, H*128, 3]).
    """
    head_k_dim = inputs.q.shape[-1]
    head_v_dim = inputs.v.shape[-1]
    if (head_k_dim != _K) or (head_v_dim != _V_DIM):
        raise NotImplementedError(
            f"kda_fused_prefill requires K=V=128; got "
            f"K={head_k_dim}, V={head_v_dim}")
    if not _bound_device_ok(inputs.q.device):
        raise NotImplementedError(
            "PyPTO KDA binds to a single NPU per process; tensor is on a "
            "different NPU -> torch fallback")

    packed = _pack_prefill(inputs, weights, conv_cache)
    bh = packed.batch * packed.heads
    seq_len_pad = packed.seq_len + packed.pad
    if initial_state is None:
        st = torch.zeros(bh, packed.head_v_dim, packed.head_k_dim,
                         dtype=torch.float32, device=packed.device)
    else:
        st = initial_state.float().reshape(
            bh, packed.head_v_dim, packed.head_k_dim).contiguous()
    out2 = torch.zeros(bh, seq_len_pad, packed.head_v_dim,
                       dtype=torch.bfloat16, device=packed.device)
    last_state = torch.zeros(bh, packed.head_v_dim, packed.head_k_dim,
                             dtype=torch.float32, device=packed.device)

    _kda_fused_prefill_npu(packed.x_q, packed.x_k, packed.x_v,
                          packed.cw_q, packed.cw_k, packed.cw_v,
                          packed.g2, packed.gg2, packed.b2,
                          packed.a_scl, packed.a_bias, packed.o_w,
                          st, out2, last_state,
                          float(weights.lower_bound), float(weights.rms_eps))
    # no explicit synchronize: the kernel is stream-ordered (aclgraph-safe)
    out, final_state, cs_out_t = _unpack_prefill(packed, out2, last_state)
    return out, (final_state if output_final_state else None), cs_out_t
