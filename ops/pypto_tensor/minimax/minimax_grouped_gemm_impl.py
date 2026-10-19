# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Grouped GEMM for the MiniMax MoE text backbones (M2.7 + M3) — Ascend 910B (dav-c220).

One all-experts-in-one-call grouped GEMM shared by both MiniMax variants (and
structurally by ``llada2_moe``): ``pypto.loop`` over experts, ``pypto.loop_unroll``
over tokens, UB-fitting vector tiles. The only per-variant differences are the
expert activation and the (env-overridable) tile defaults, selected by ``activation``:

    "silu"      — M2.7: SiLU-SwiGLU.      VEC_TILE 128 / CUBE_NBUF 2 / VEC_NBUF 2 / L1 2
    "swigluoai" — M3:   clamped GLU.      VEC_TILE 256 / CUBE_NBUF 4 / VEC_NBUF 1 / L1 3

swigluoai (GPT-OSS style), with alpha/limit from the M3 config:
    gate = clamp(gate, max=limit); up = clamp(up, -limit, +limit)
    glu  = gate * sigmoid(alpha * gate); out = (up + 1) * glu

The 910B needs a UB-fitting vector-width cap: the chip has split CUBE/VECTOR cores
with a 192 KB UB, so a FP32 vector tile at the full intermediate/hidden width
overflows UB and fails the ``OoOSchedule`` pass; a capped width ([128,128] FP32 x2buf
= 128 KB) schedules and matches the reference. Every tile knob and the swiglu
alpha/limit are env-overridable (``PYPTO_*``); see ai/910b-hardware.md.

Weight layout (both variants, F.linear convention):
    gate_up_proj [E, 2*I, H], down_proj [E, H, I]
``convert_minimax_weights`` transposes them to the kernel's direct-matmul layout
    w13_flat [E*H, 2*I], w2_flat [E*I, H].
"""

import os
from typing import NamedTuple

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph

import pypto


class MoeDims(NamedTuple):
    """MoE shape + activation passed to the grouped-GEMM kernel.

    ``activation`` selects the expert activation / tile-default variant: ``"silu"``
    (MiniMax M2.7, default) or ``"swigluoai"`` (MiniMax-M3).
    """
    num_experts: int
    hidden_size: int
    intermediate_size: int
    activation: str = "silu"


ND = pypto.TileOpFormat.TILEOP_ND

# swigluoai parameters (MiniMax-M3 config: swiglu_alpha / swiglu_limit); env-overridable.
_SWIGLU_ALPHA = float(os.environ.get("PYPTO_SWIGLU_ALPHA", "1.702"))
_SWIGLU_LIMIT = float(os.environ.get("PYPTO_SWIGLU_LIMIT", "7.0"))

# Per-variant tile defaults — each tuned on the 910B for its expert dims (M2.7 H=3072/I=1536;
# M3 H=6144/I=3072). Every value is env-overridable (shared PYPTO_* names; only one variant
# runs per process), so the kernel can be re-tuned without editing source.
_VARIANT_DEFAULTS = {
    "silu": {"vec_tile": 128, "cube_nbuf": 2, "vec_nbuf": 2, "l1_reuse": 2},
    "swigluoai": {"vec_tile": 256, "cube_nbuf": 4, "vec_nbuf": 1, "l1_reuse": 3},
}
_UNROLL = [int(x) for x in os.environ.get("PYPTO_UNROLL", "1,2,4,8,16,32,64").split(",")]
_MM1_K = int(os.environ.get("PYPTO_MM1_K", "128"))
_MM1_N = int(os.environ.get("PYPTO_MM1_N", "256"))
_MM2_K = int(os.environ.get("PYPTO_MM2_K", "128"))
_MM2_N = int(os.environ.get("PYPTO_MM2_N", "256"))


def convert_minimax_weights(gate_up_proj, down_proj):
    """Convert MiniMax expert weights to grouped GEMM kernel format.

    Parameters
    ----------
    gate_up_proj : [E, 2*I, H] BF16 — gate||up weights (F.linear fmt, gate then up)
    down_proj    : [E, H,   I] BF16 — down weights (F.linear fmt)

    Returns
    -------
    w13_flat : [E*H, 2*I] BF16 — flattened gate||up (direct matmul fmt)
    w2_flat  : [E*I, H]   BF16 — flattened down (direct matmul fmt)
    """
    num_experts = gate_up_proj.shape[0]
    two_intermediate = gate_up_proj.shape[1]
    hidden_size = gate_up_proj.shape[2]
    intermediate_size = down_proj.shape[2]

    w13_flat = gate_up_proj.transpose(1, 2).reshape(num_experts * hidden_size, two_intermediate).contiguous()
    w2_flat = down_proj.transpose(1, 2).reshape(num_experts * intermediate_size, hidden_size).contiguous()
    return w13_flat, w2_flat


def _swiglu_silu(gate_up):
    """SiLU-SwiGLU (M2.7): ``SiLU(gate) * up``."""
    half = gate_up.shape[1] // 2
    gate = pypto.view(gate_up, [gate_up.shape[0], half], [0, 0])
    up = pypto.view(gate_up, [gate_up.shape[0], half], [0, half])
    neg = pypto.mul(gate, -1.0)
    exp_neg = pypto.exp(neg)
    denom = pypto.add(exp_neg, 1.0)
    silu = pypto.div(gate, denom)
    return pypto.mul(silu, up)


def _swiglu_oai(gate_up):
    """swigluoai clamped GLU (M3): ``(clamp(up) + 1) * (g * sigmoid(alpha*g))``."""
    half = gate_up.shape[1] // 2
    gate = pypto.view(gate_up, [gate_up.shape[0], half], [0, 0])
    up = pypto.view(gate_up, [gate_up.shape[0], half], [0, half])
    gate_c = pypto.clip(gate, -float("inf"), _SWIGLU_LIMIT)
    up_c = pypto.clip(up, -_SWIGLU_LIMIT, _SWIGLU_LIMIT)
    glu = pypto.mul(gate_c, pypto.sigmoid(pypto.mul(gate_c, _SWIGLU_ALPHA)))
    return pypto.mul(pypto.add(up_c, 1.0), glu)


_ACTIVATIONS = {"silu": _swiglu_silu, "swigluoai": _swiglu_oai}


def _build_kernel(activation):
    """Build the jitted grouped-GEMM kernel for one activation variant.

    The kernel body is written once here; the activation function and the tile
    pass-options are closed over per variant, so M2.7 and M3 share a single body.
    """
    act_fn = _ACTIVATIONS[activation]
    d = _VARIANT_DEFAULTS[activation]
    vt = int(os.environ.get("PYPTO_VEC_TILE", d["vec_tile"]))
    cube_nbuf = int(os.environ.get("PYPTO_CUBE_NBUFFER", d["cube_nbuf"]))
    vec_nbuf = int(os.environ.get("PYPTO_VEC_NBUFFER", d["vec_nbuf"]))
    l1_reuse = int(os.environ.get("PYPTO_L1_REUSE", d["l1_reuse"]))

    @pypto.frontend.jit(
        runtime_options={"device_sched_mode": 1, "stitch_function_max_num": 64,
                         # expert_cumsum read on host for per-expert slicing (required under CANN 9.1.0)
                         "ready_on_host_tensors": ["expert_cumsum"]},
        pass_options={
            "cube_l1_reuse_setting": {-1: l1_reuse},
            "cube_nbuffer_setting": {-1: cube_nbuf},
            "vec_nbuffer_setting": {-1: vec_nbuf},
        },
    )
    def _kernel(
        sorted_tokens: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16, format=ND),
        w13_flat: pypto.Tensor([], pypto.DT_BF16, format=ND),
        w2_flat: pypto.Tensor([], pypto.DT_BF16, format=ND),
        expert_cumsum: pypto.Tensor([], pypto.DT_INT32, format=ND),
        result: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16, format=ND),
    ):
        # Shapes derived from the tensors: sorted_tokens [N, H], w13_flat [E*H, 2I], expert_cumsum [E+1].
        num_experts = expert_cumsum.shape[0] - 1
        hidden_size = sorted_tokens.shape[1]
        two_i = w13_flat.shape[1]
        intermediate_size = two_i // 2
        pypto.experimental.set_operation_options(combine_axis=True)

        for e_idx in pypto.loop(0, num_experts, 1, name="EXPERT_LOOP", idx_name="e_idx"):
            e_start = expert_cumsum[e_idx]
            e_end = expert_cumsum[e_idx + 1]
            n_e = e_end - e_start

            for tok_idx, tile_batch in pypto.loop_unroll(
                0, n_e, 1, name="LOOP_TOKEN", idx_name="tok_idx", unroll_list=_UNROLL,
            ):
                w13_e = pypto.view(w13_flat, [hidden_size, two_i], [e_idx * hidden_size, 0])
                w2_e = pypto.view(w2_flat, [intermediate_size, hidden_size], [e_idx * intermediate_size, 0])
                tile_x = pypto.view(sorted_tokens, [tile_batch, hidden_size], [e_start + tok_idx, 0])

                # mm1 -> gate_up [tile_batch, 2*I] FP32
                pypto.set_cube_tile_shapes([tile_batch, tile_batch], [_MM1_K, _MM1_K * 2], [_MM1_N, _MM1_N], True)
                gate_up = pypto.matmul(tile_x, w13_e, pypto.DT_FP32)

                # activation in FP32 -> BF16, vector tile width capped to UB-fitting `vt`
                pypto.set_vec_tile_shapes(min(tile_batch, vt), vt)
                sw = pypto.cast(act_fn(gate_up), pypto.DT_BF16)

                # mm2 -> down [tile_batch, H] FP32
                pypto.set_cube_tile_shapes([tile_batch, tile_batch], [_MM2_K, _MM2_K * 2], [_MM2_N, _MM2_N], False)
                down = pypto.matmul(sw, w2_e, pypto.DT_FP32)
                pypto.set_vec_tile_shapes(min(tile_batch, vt), vt)
                out = pypto.cast(down, pypto.DT_BF16)

                pypto.assemble(out, [e_start + tok_idx, 0], result)

    return _kernel


# The jit decorator is applied once per variant at import (cheap; TBE compile is deferred to the
# first real call). Only the variant actually invoked by the running model is compiled.
_KERNELS = {name: _build_kernel(name) for name in _ACTIVATIONS}


def _check(sorted_tokens, weights, expert_cumsum, result, dims):
    w13_flat, w2_flat = weights
    num_experts, hidden_size, intermediate_size = dims.num_experts, dims.hidden_size, dims.intermediate_size
    if sorted_tokens.dim() != 2 or sorted_tokens.dtype != torch.bfloat16:
        raise ValueError("sorted_tokens must be 2-D bfloat16")
    if w13_flat.dim() != 2 or w13_flat.dtype != torch.bfloat16:
        raise ValueError("w13_flat must be 2-D bfloat16")
    if w2_flat.dim() != 2 or w2_flat.dtype != torch.bfloat16:
        raise ValueError("w2_flat must be 2-D bfloat16")
    if expert_cumsum.dim() != 1 or expert_cumsum.dtype != torch.int32:
        raise ValueError("expert_cumsum must be 1-D int32")
    if sorted_tokens.shape[1] != hidden_size:
        raise ValueError(f"sorted_tokens hidden {sorted_tokens.shape[1]} != {hidden_size}")
    if tuple(w13_flat.shape) != (num_experts * hidden_size, 2 * intermediate_size):
        raise ValueError("w13_flat shape mismatch")
    if tuple(w2_flat.shape) != (num_experts * intermediate_size, hidden_size):
        raise ValueError("w2_flat shape mismatch")
    if expert_cumsum.shape[0] != num_experts + 1:
        raise ValueError("expert_cumsum length must be num_experts + 1")
    if result.shape != sorted_tokens.shape:
        raise ValueError("result shape must equal sorted_tokens shape")


@allow_in_graph
def minimax_moe_grouped_gemm(
    sorted_tokens: torch.Tensor,
    weights: "tuple[torch.Tensor, torch.Tensor]",
    expert_cumsum: torch.Tensor,
    result: torch.Tensor,
    dims: "MoeDims",
) -> None:
    """Grouped GEMM for MiniMax MoE (all experts, single kernel call).

    910B-adapted: UB-fitting vector tile (PYPTO_VEC_TILE; default 128 for silu / 256 for swigluoai).
    The expert activation variant is carried by ``dims.activation`` ("silu" / "swigluoai").

    Parameters
    ----------
    sorted_tokens : [N_total, H] BF16 — tokens pre-sorted by expert
    weights : (w13_flat [E*H, 2*I], w2_flat [E*I, H]) BF16 — converted gate||up / down weights
    expert_cumsum : [E+1] INT32 — cumulative token counts per expert
    result : [N_total, H] BF16 — output buffer
    dims : MoeDims(num_experts, hidden_size, intermediate_size, activation)
    """
    if isinstance(sorted_tokens, FakeTensor):
        return
    activation = dims.activation
    if activation not in _KERNELS:
        raise ValueError(f"activation must be one of {sorted(_KERNELS)}, got {activation!r}")
    w13_flat, w2_flat = weights
    _check(sorted_tokens, weights, expert_cumsum, result, dims)
    _KERNELS[activation](sorted_tokens, w13_flat, w2_flat, expert_cumsum, result)
