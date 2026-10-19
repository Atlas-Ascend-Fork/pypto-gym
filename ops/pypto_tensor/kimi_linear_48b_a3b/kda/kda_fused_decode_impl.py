# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full KDA layer fused decode for Kimi-Linear-48B-A3B.

Fuses the entire KimiDeltaAttention layer into one PyPTO operator:
    input projections (q_proj, k_proj, v_proj) → conv1d → gate → recurrent
    → o_norm → output projection (o_proj)

Dimensions:
    B: batch size (always 1 in decode mode)
    H: hidden_size (8192)
    P: proj_dim = num_heads * head_dim = 64 * 128 = 8192
    num_heads: 64, head_dim: 128

Tiling:
    Cube tiles: [16, 16], [128, 128], [128, 128] - legal tiles for decode (M=1 pads to 16)
    Vec tiles: 16 x 128 for vector operations, 32 x 128 for state operations

Inputs (to kernel):
    hidden: [B, H] bf16/fp16
    weights: KdaFusedWeights (packed weights, fp32)
    buffers: KdaFusedBuffers (conv_state, recurrent_state, output, state_out, cs_out for capture-safety)

Outputs:
    output: [B, H] bf16/fp16
    updated buffers (in-place)
"""
__all__ = [
    "KdaFusedWeights",
    "prepare_kda_fused_weights",
    "KdaFusedBuffers",
    "KdaBufferParams",
    "make_fused_buffers",
    "seed_fused_buffers",
    "kda_fused_decode_step",
    "kda_fused_decode",
]

import os
import sys
from dataclasses import dataclass
from typing import NamedTuple

import pypto
import torch
import torch_npu
from torch._dynamo import allow_in_graph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _device_guard import _bound_device_ok  # noqa: E402

CONV_K = 4
CONV_HIST = CONV_K - 1
STATE_TILE_M = 32
VEC_TILE_M = 64
VEC_TILE_DIM = 128


@dataclass
class KdaBufferParams:
    """Parameters for KDA buffer allocation."""
    layer_idx: int
    batch: int
    num_heads: int
    head_dim: int
    hidden_size: int
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class ConvBranchParams:
    b_idx: int
    base: int
    head_dim: int
    fp_in: pypto.DataType


@dataclass(frozen=True)
class KernelParams:
    """Parameters for the KDA layer megakernel."""
    hidden_size: int
    num_heads: int
    num_heads_padded: int
    head_dim: int
    proj_dim: int
    scale: float
    l2_eps: float
    rms_eps: float
    lower_bound: float
    fp_in: pypto.DataType


@dataclass
class BatchContext:
    """Context for processing one (head, batch) pair."""
    x_qkv: any
    g_raw: any
    beta_raw: any
    g_gate: any
    dt_bias: any
    a_log: any
    o_norm_w: any
    state_in: any
    state_out: any
    cs: any
    cw: any
    cs_out: any
    o_intermediate: any
    lower_bound: float
    rms_eps: float
    scale: float
    l2_eps: float
    num_heads: int
    proj_dim: int
    fp_in: pypto.DataType


def _validate_decode_input(hidden, weights):
    """Validate input shapes and device, return (batch, squeeze_dim) or raise."""
    squeeze_dim = False
    if hidden.dim() == 3:
        if hidden.shape[1] != 1:
            raise NotImplementedError(f"hidden must be [B, 1, H] or [B, H], got {hidden.shape}")
        squeeze_dim = True
        hidden = hidden.squeeze(1)
    elif hidden.dim() != 2:
        raise NotImplementedError(f"hidden must be 2D or 3D, got {hidden.dim()}D")
    
    batch, hidden_size = hidden.shape
    if hidden_size != weights.hidden_size:
        raise NotImplementedError(f"hidden_size mismatch: {hidden_size} vs {weights.hidden_size}")
    if not _bound_device_ok(hidden.device):
        raise NotImplementedError(f"tensor not on bound NPU")
    
    return batch, hidden, squeeze_dim


def _prepare_decode_params(weights, dtype):
    """Create KernelParams from weights and dtype."""
    fp_in = pypto.DT_BF16 if dtype == torch.bfloat16 else pypto.DT_FP16
    return KernelParams(
        hidden_size=weights.hidden_size,
        num_heads=weights.num_heads,
        num_heads_padded=weights.num_heads_padded,
        head_dim=weights.head_dim,
        proj_dim=weights.num_heads * weights.head_dim,
        scale=1.0 / (weights.head_dim ** 0.5),
        l2_eps=1e-6,
        rms_eps=weights.rms_eps,
        lower_bound=weights.lower_bound,
        fp_in=fp_in,
    )


@dataclass
class DecodeIO:
    """I/O tensors for decode kernel."""
    hidden: torch.Tensor
    output: torch.Tensor
    state_out: torch.Tensor
    cs_out: torch.Tensor


def _call_decode_kernel(kernel, weights, buffers, io):
    """Execute the fused decode kernel."""
    kernel(
        io.hidden,
        weights.qkv_w,
        buffers.conv_state,
        weights.conv_w,
        weights.f_a_w,
        weights.f_b_w,
        weights.dt_bias,
        weights.a_log,
        weights.b_w,
        weights.g_a_w,
        weights.g_b_w,
        weights.o_norm_w,
        weights.o_proj_w,
        io.output,
        buffers.state,
        io.state_out,
        io.cs_out,
    )


_kernel_cache = {}


def _conv_branch(x_qkv, cs, cw, cs_out, params):
    """Four-tap causal depthwise conv + silu for one (branch, head) slice.

    Taps are oldest-first: cs[b,0] is t-3, cs[b,2] is t-1, current token is tap 3.
    ShortConvolution caches PRE-conv inputs, so the new history is
    (tap1, tap2, x_raw).
    """
    b_idx, base, head_dim, fp_in = params.b_idx, params.base, params.head_dim, params.fp_in
    x_raw = pypto.view(x_qkv, [1, head_dim], [b_idx, base])
    acc = pypto.mul(pypto.cast(x_raw, pypto.DT_FP32),
                    pypto.view(cw, [1, head_dim], [CONV_HIST, base]))
    hist = []
    for tap in range(CONV_HIST):
        s_raw = pypto.reshape(pypto.view(cs, [1, 1, head_dim], [b_idx, tap, base]), [1, head_dim])
        hist.append(s_raw)
        acc = pypto.add(acc, pypto.mul(pypto.cast(s_raw, pypto.DT_FP32),
                                       pypto.view(cw, [1, head_dim], [tap, base])))
    neg_acc = pypto.mul(acc, -1.0)
    exp_neg_acc = pypto.exp(neg_acc)
    sigmoid_acc = pypto.reciprocal(pypto.add(exp_neg_acc, 1.0))
    y = pypto.mul(acc, sigmoid_acc)

    new_hist = [hist[1], hist[2], pypto.cast(x_raw, fp_in)]
    for tap in range(CONV_HIST):
        pypto.assemble(pypto.reshape(new_hist[tap], [1, 1, head_dim]),
                       [b_idx, tap, base], cs_out)
    return y


def _l2_normalise(x, l2_eps):
    """x / (sqrt(sum(x^2)) + eps) -- the exact form the torch goldens use."""
    return pypto.div(x, pypto.add(pypto.sqrt(pypto.sum(pypto.mul(x, x), -1, True)),
                                  l2_eps))


def _sigmoid_gate(arg, lower_bound):
    """sigmoid(arg) * lower_bound -- gate activation."""
    neg_arg = pypto.mul(arg, -1.0)
    exp_neg_arg = pypto.exp(neg_arg)
    sigmoid_arg = pypto.reciprocal(pypto.add(exp_neg_arg, 1.0))
    return pypto.mul(sigmoid_arg, lower_bound)


def _sigmoid_beta(arg):
    """sigmoid(arg) -- beta activation."""
    neg_arg = pypto.mul(arg, -1.0)
    exp_neg_arg = pypto.exp(neg_arg)
    return pypto.reciprocal(pypto.add(exp_neg_arg, 1.0))


def _sigmoid_output_gate(arg):
    """sigmoid(arg) -- output gate activation."""
    neg_arg = pypto.mul(arg, -1.0)
    exp_neg_arg = pypto.exp(neg_arg)
    return pypto.reciprocal(pypto.add(exp_neg_arg, 1.0))


def _process_batch_item(ctx, h_idx, b_idx):
    """Conv1d + Gate + Recurrent + o_norm for one (head, batch) pair."""
    h_off = h_idx * VEC_TILE_DIM
    pypto.set_vec_tile_shapes(VEC_TILE_M, VEC_TILE_DIM)
    
    q_c = _conv_branch(ctx.x_qkv, ctx.cs, ctx.cw, ctx.cs_out,
                       ConvBranchParams(b_idx, 0 * ctx.proj_dim + h_off, VEC_TILE_DIM, ctx.fp_in))
    k_c = _conv_branch(ctx.x_qkv, ctx.cs, ctx.cw, ctx.cs_out,
                       ConvBranchParams(b_idx, 1 * ctx.proj_dim + h_off, VEC_TILE_DIM, ctx.fp_in))
    v_c = _conv_branch(ctx.x_qkv, ctx.cs, ctx.cw, ctx.cs_out,
                       ConvBranchParams(b_idx, 2 * ctx.proj_dim + h_off, VEC_TILE_DIM, ctx.fp_in))
    
    q_n = _l2_normalise(q_c, ctx.l2_eps)
    k_n = _l2_normalise(k_c, ctx.l2_eps)
    
    g_r = pypto.view(ctx.g_raw, [1, VEC_TILE_DIM], [b_idx, h_off])
    dtb = pypto.view(ctx.dt_bias, [1, VEC_TILE_DIM], [0, h_off])
    a_h = pypto.exp(pypto.view(ctx.a_log, [1, 1], [0, h_idx]))
    g_arg = pypto.mul(pypto.add(g_r, dtb), a_h)
    g_log = _sigmoid_gate(g_arg, ctx.lower_bound)
    
    beta_arg = pypto.view(ctx.beta_raw, [1, 1], [b_idx, h_idx])
    beta = _sigmoid_beta(beta_arg)
    
    bh = b_idx * ctx.num_heads + h_idx
    pypto.set_vec_tile_shapes(STATE_TILE_M, VEC_TILE_DIM)
    state = pypto.reshape(pypto.view(ctx.state_in, [1, VEC_TILE_DIM, VEC_TILE_DIM], [bh, 0, 0]),
                          [VEC_TILE_DIM, VEC_TILE_DIM])
    state_gated = pypto.mul(state, pypto.exp(g_log))
    pred = pypto.reshape(pypto.sum(pypto.mul(k_n, state_gated), -1, True), [1, VEC_TILE_DIM])
    delta = pypto.mul(beta, pypto.sub(v_c, pred))
    state_new = pypto.add(state_gated, pypto.mul(pypto.reshape(delta, [VEC_TILE_DIM, 1]), k_n))
    o = pypto.mul(pypto.reshape(pypto.sum(pypto.mul(q_n, state_new), -1, True), [1, VEC_TILE_DIM]), ctx.scale)
    pypto.assemble(pypto.reshape(state_new, [1, VEC_TILE_DIM, VEC_TILE_DIM]), [bh, 0, 0], ctx.state_out)
    
    pypto.set_vec_tile_shapes(VEC_TILE_M, VEC_TILE_DIM)
    var = pypto.mul(pypto.sum(pypto.mul(o, o), -1, True), 1.0 / float(VEC_TILE_DIM))
    o_n = pypto.mul(pypto.mul(o, pypto.rsqrt(pypto.add(var, ctx.rms_eps))), ctx.o_norm_w)
    g_gate_arg = pypto.view(ctx.g_gate, [1, VEC_TILE_DIM], [b_idx, h_off])
    sigmoid_g_gate = _sigmoid_output_gate(g_gate_arg)
    o_g = pypto.mul(o_n, sigmoid_g_gate)
    pypto.assemble(pypto.cast(o_g, ctx.fp_in), [b_idx, h_off], ctx.o_intermediate)


def _create_kernel(params):
    """Create the JIT-compiled KDA layer megakernel."""
    hidden_size, num_heads, proj_dim, fp_in = (
        params.hidden_size, params.num_heads, params.proj_dim, params.fp_in)
    scale, l2_eps, rms_eps, lower_bound = (
        params.scale, params.l2_eps, params.rms_eps, params.lower_bound)

    batch_dim, batch_heads = 1, num_heads

    @pypto.frontend.jit(
        runtime_options={"stitch_function_max_num": 256, "device_sched_mode": 2},
        pass_options={"vec_nbuffer_setting": {"DEFAULT": 4}},
    )
    def kda_layer_fused_npu(
        hidden: pypto.Tensor([1, hidden_size], fp_in),
        qkv_w: pypto.Tensor([3 * proj_dim, hidden_size], fp_in),
        cs: pypto.Tensor([1, CONV_HIST, 3 * proj_dim], fp_in),
        cw: pypto.Tensor([CONV_K, 3 * proj_dim], pypto.DT_FP32),
        f_a_w: pypto.Tensor([VEC_TILE_DIM, hidden_size], fp_in),
        f_b_w: pypto.Tensor([proj_dim, VEC_TILE_DIM], fp_in),
        dt_bias: pypto.Tensor([1, proj_dim], pypto.DT_FP32),
        a_log: pypto.Tensor([1, num_heads], pypto.DT_FP32),
        b_w: pypto.Tensor([num_heads, hidden_size], fp_in),
        g_a_w: pypto.Tensor([VEC_TILE_DIM, hidden_size], fp_in),
        g_b_w: pypto.Tensor([proj_dim, VEC_TILE_DIM], fp_in),
        o_norm_w: pypto.Tensor([1, VEC_TILE_DIM], pypto.DT_FP32),
        o_proj_w: pypto.Tensor([hidden_size, proj_dim], fp_in),
        output: pypto.Tensor([1, hidden_size], fp_in),
        state_in: pypto.Tensor([num_heads, VEC_TILE_DIM, VEC_TILE_DIM], pypto.DT_FP32),
        state_out: pypto.Tensor([num_heads, VEC_TILE_DIM, VEC_TILE_DIM], pypto.DT_FP32),
        cs_out: pypto.Tensor([1, CONV_HIST, 3 * proj_dim], fp_in),
    ):
        n_batch = hidden.shape[0]

        pypto.set_vec_tile_shapes(VEC_TILE_M, VEC_TILE_DIM)
        
        pypto.set_cube_tile_shapes([16, 16], [128, 128], [128, 512])
        x_qkv = pypto.matmul(hidden, qkv_w, pypto.DT_FP32, b_trans=True)
        x_qkv = pypto.cast(x_qkv, fp_in)
        
        pypto.set_cube_tile_shapes([16, 16], [128, 128], [128, 256])
        f_a = pypto.matmul(hidden, f_a_w, fp_in, b_trans=True)
        g_raw = pypto.matmul(f_a, f_b_w, pypto.DT_FP32, b_trans=True)
        
        pypto.set_cube_tile_shapes([16, 16], [128, 128], [128, 128])
        beta_raw = pypto.matmul(hidden, b_w, pypto.DT_FP32, b_trans=True)
        
        pypto.set_cube_tile_shapes([16, 16], [128, 128], [128, 256])
        g_a = pypto.matmul(hidden, g_a_w, fp_in, b_trans=True)
        g_gate = pypto.matmul(g_a, g_b_w, pypto.DT_FP32, b_trans=True)

        pypto.set_vec_tile_shapes(VEC_TILE_M, VEC_TILE_DIM)
        o_intermediate = pypto.tensor([1, proj_dim], fp_in, "o_intermediate")
        
        ctx = BatchContext(
            x_qkv=x_qkv, g_raw=g_raw, beta_raw=beta_raw, g_gate=g_gate,
            dt_bias=dt_bias, a_log=a_log, o_norm_w=o_norm_w,
            state_in=state_in, state_out=state_out,
            cs=cs, cw=cw, cs_out=cs_out, o_intermediate=o_intermediate,
            lower_bound=lower_bound, rms_eps=rms_eps, scale=scale, l2_eps=l2_eps,
            num_heads=num_heads, proj_dim=proj_dim, fp_in=fp_in
        )

        for h_idx in pypto.loop(num_heads, name="LOOP_HEAD", idx_name="h_idx", parallel=True):
            for b_idx in pypto.loop(n_batch, name="LOOP_BATCH", idx_name="b_idx", unroll=True):
                _process_batch_item(ctx, h_idx, b_idx)

        pypto.set_cube_tile_shapes([16, 16], [128, 128], [128, 256])
        output[:] = pypto.cast(
            pypto.matmul(o_intermediate, o_proj_w, pypto.DT_FP32, b_trans=True),
            fp_in
        )

    return kda_layer_fused_npu


def _get_kernel(params):
    """Get or create the JIT-compiled KDA layer megakernel."""
    key = (params.hidden_size, params.num_heads, params.num_heads_padded, params.head_dim, params.fp_in,
           params.scale, params.l2_eps, params.rms_eps, params.lower_bound)
    if key not in _kernel_cache:
        _kernel_cache[key] = _create_kernel(params)
    return _kernel_cache[key]


# ---------------------------------------------------------------------------
# weights and buffers
# ---------------------------------------------------------------------------

class KdaFusedWeights(NamedTuple):
    """All weights for a KDA layer, packed for the fused decode."""
    qkv_w: torch.Tensor       # [3P, H] bf16, fused q|k|v projection weights
    conv_w: torch.Tensor      # [K, 3P] fp32, tap-major
    f_a_w: torch.Tensor       # [D, H] bf16
    f_b_w: torch.Tensor       # [P, D] bf16
    dt_bias: torch.Tensor     # [1, P] fp32
    a_log: torch.Tensor       # [1, num_heads] fp32
    b_w: torch.Tensor         # [num_heads_padded, H] bf16 (padded to 16 for PyPTO)
    g_a_w: torch.Tensor       # [D, H] bf16
    g_b_w: torch.Tensor       # [P, D] bf16
    o_norm_w: torch.Tensor    # [1, D] fp32
    o_proj_w: torch.Tensor    # [H, P] bf16
    hidden_size: int
    num_heads: int
    num_heads_padded: int  # Padded to multiple of 16 for PyPTO matmul
    head_dim: int
    rms_eps: float
    lower_bound: float


def prepare_kda_fused_weights(layer, lower_bound: float = -5.0) -> KdaFusedWeights:
    """Pack a KimiDeltaAttention module's weights for the megakernel.

    Cube matmul weights are bf16 (cube accumulates in fp32 via out_dtype).
    Vector op weights (conv_w, dt_bias, a_log, o_norm_w) are fp32.
    """
    cached = getattr(layer, "_kda_layer_weights", None)
    if cached is not None:
        return cached

    hidden_size = layer.hidden_size
    num_heads, head_dim = layer.num_heads, layer.head_dim
    proj_dim = num_heads * head_dim
    dt = layer.q_proj.weight.dtype

    if layer.q_proj.weight.shape[0] != proj_dim:
        raise ValueError(
            f"projection_size mismatch: q_proj gives {layer.q_proj.weight.shape[0]}, "
            f"num_heads*head_dim = {proj_dim}; hidden_size is {layer.hidden_size}")
    if layer.conv_size != CONV_K:
        raise NotImplementedError(f"conv kernel {layer.conv_size} != {CONV_K}")

    conv_w = torch.cat([
        layer.q_conv1d.weight.reshape(proj_dim, CONV_K),
        layer.k_conv1d.weight.reshape(proj_dim, CONV_K),
        layer.v_conv1d.weight.reshape(proj_dim, CONV_K),
    ], dim=0).t().contiguous().float()

    qkv_w = torch.cat([layer.q_proj.weight, layer.k_proj.weight,
                       layer.v_proj.weight], dim=0).detach().to(dt).contiguous()

    num_heads_padded = (num_heads + 15) // 16 * 16
    b_w_original = layer.b_proj.weight.detach().to(dt).contiguous()
    if num_heads_padded != num_heads:
        b_w = torch.zeros(num_heads_padded, hidden_size, dtype=dt, device=b_w_original.device)
        b_w[:num_heads] = b_w_original
    else:
        b_w = b_w_original

    packed = KdaFusedWeights(
        qkv_w=qkv_w,
        conv_w=conv_w,
        f_a_w=layer.f_a_proj.weight.detach().to(dt).contiguous(),
        f_b_w=layer.f_b_proj.weight.detach().to(dt).contiguous(),
        dt_bias=layer.dt_bias.detach().float().reshape(1, proj_dim).contiguous(),
        a_log=layer.A_log.detach().float().reshape(1, num_heads).contiguous(),
        b_w=b_w,
        g_a_w=layer.g_a_proj.weight.detach().to(dt).contiguous(),
        g_b_w=layer.g_b_proj.weight.detach().to(dt).contiguous(),
        o_norm_w=layer.o_norm.weight.detach().float().reshape(1, head_dim).contiguous(),
        o_proj_w=layer.o_proj.weight.detach().to(dt).contiguous(),
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_heads_padded=num_heads_padded,
        head_dim=head_dim,
        rms_eps=float(getattr(layer.o_norm, "eps", 1e-5)),
        lower_bound=float(lower_bound),
    )
    setattr(layer, "_kda_layer_weights", packed)
    return packed


class KdaFusedBuffers(NamedTuple):
    """Capture-safe buffers for the megakernel (all tensors pre-allocated)."""
    cs: torch.Tensor           # [B, CONV_HIST, 3P] conv history
    state: torch.Tensor        # [B*H, D, D] recurrent state
    output: torch.Tensor       # [B, H] output buffer
    state_out: torch.Tensor    # [B*H, D, D] recurrent state output
    cs_out: torch.Tensor       # [B, CONV_HIST, 3P] conv history output

    @property
    def conv_state(self):
        """Alias for compatibility with seed_from_prefill."""
        return self.cs


def make_fused_buffers(params: "KdaBufferParams") -> KdaFusedBuffers:
    """Allocate capture-safe buffers for one KDA layer's megakernel."""
    batch, num_heads, head_dim = params.batch, params.num_heads, params.head_dim
    dtype, device, hidden_size = params.dtype, params.device, params.hidden_size
    
    proj_dim = num_heads * head_dim
    cs = torch.zeros(batch, CONV_HIST, 3 * proj_dim, dtype=dtype, device=device)
    state = torch.zeros(batch * num_heads, head_dim, head_dim,
                        dtype=torch.float32, device=device)
    output = torch.zeros(batch, hidden_size, dtype=dtype, device=device)
    state_out = torch.zeros_like(state)
    cs_out = torch.zeros_like(cs)
    return KdaFusedBuffers(cs, state, output, state_out, cs_out)


def seed_fused_buffers(buf: KdaFusedBuffers,
                       conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                       recurrent_state: torch.Tensor) -> None:
    """Copy prefill state into buffers (in-place)."""
    # Pack conv state: (3x [B, P, K-1]) -> [B, K-1, 3P]
    packed = torch.cat(list(conv_state), dim=1).transpose(1, 2).contiguous()
    buf.cs.copy_(packed)
    buf.state.copy_(recurrent_state.reshape(buf.state.shape).float())


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def kda_fused_decode_step(
    hidden: torch.Tensor,
    weights: KdaFusedWeights,
    buffers: KdaFusedBuffers,
) -> torch.Tensor:
    """Graph-capturable per-token KDA layer step (no allocation, no exceptions).

    Args:
        hidden: [batch, hidden_size] or [batch, 1, hidden_size] input tensor
        weights: Packed weights from prepare_kda_fused_weights
        buffers: Persistent buffers from make_fused_buffers

    Returns:
        output: [batch, hidden_size] or [batch, 1, hidden_size] tensor

    Raises:
        NotImplementedError: If device is not the bound NPU or shapes are wrong.
    """
    batch, hidden, squeeze_dim = _validate_decode_input(hidden, weights)
    params = _prepare_decode_params(weights, hidden.dtype)
    kernel = _get_kernel(params)
    
    output = buffers.output
    state_out = buffers.state_out
    cs_out = buffers.cs_out
    
    io = DecodeIO(hidden, output, state_out, cs_out)
    _call_decode_kernel(kernel, weights, buffers, io)
    
    buffers.state.copy_(io.state_out)
    buffers.conv_state.copy_(io.cs_out)
    
    if squeeze_dim:
        io.output = io.output.unsqueeze(1)
    
    return io.output


@allow_in_graph
def kda_fused_decode(
    hidden: torch.Tensor,
    weights: KdaFusedWeights,
    buffers: KdaFusedBuffers,
) -> torch.Tensor:
    """Eager-mode entry point with guards."""
    return kda_fused_decode_step(hidden, weights, buffers)
