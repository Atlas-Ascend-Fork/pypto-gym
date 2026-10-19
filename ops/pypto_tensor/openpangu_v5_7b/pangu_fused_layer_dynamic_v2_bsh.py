# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pangu-7B Fused Layer with Dynamic KV Cache Length (BSH KV Cache).

Architecture (all in single ``@pypto.frontend.jit`` kernel):

1. Input RMSNorm: ``add(hidden_states, residual) -> RMSNorm -> * input_ln_weight``
2. Attention: QKV proj (+bias) -> RoPE -> KV cache update -> GQA Attention -> O proj (+bias)
3. Post-attention RMSNorm: ``add(attn_proj, new_residual1) -> RMSNorm -> * post_ln_weight``
4. FFN: gate/up proj -> SwiGLU -> down proj

KV cache layout: BSH ``[kv_len, kv_size]`` where ``kv_size = num_kv_heads * head_dim``.
"""

__all__ = [
    "DynamicFusedLayerConfigV2BSH",
    "PanguFusedLayerV2BSHModule",
    "pangu_fused_layer_v2_bsh_graph",
]

from dataclasses import dataclass
from typing import Tuple

import torch
import pypto
from torch._dynamo import allow_in_graph
from torch._subclasses.fake_tensor import FakeTensor

NUM_ATTENTION_HEADS = 32
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 12800
NUM_HEADS_PER_GROUP = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS
F_NEGA_1 = -1.0


@dataclass
class DynamicFusedLayerConfigV2BSH:
    """Configuration for the fused layer operator (BSH KV cache)."""

    hidden_size: int = HIDDEN_SIZE
    intermediate_size: int = INTERMEDIATE_SIZE
    num_attention_heads: int = NUM_ATTENTION_HEADS
    num_key_value_heads: int = NUM_KEY_VALUE_HEADS
    head_dim: int = HEAD_DIM
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 32768
    rope_theta: float = 16000000.0
    dtype: pypto.DataType = pypto.DT_BF16

    @property
    def qkv_size(self) -> int:
        return (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def num_heads_per_group(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def scale(self) -> float:
        return 1.0 / (self.head_dim ** 0.5)


def _pypto_rms_norm(x, residual, gamma, eps, hidden_size):
    """RMSNorm using PyPTO ops: ``add(x, residual) -> RMSNorm -> * gamma``."""
    pypto.set_vec_tile_shapes(1, 4096)
    x_fp32 = pypto.cast(x, pypto.DT_FP32)
    residual_fp32 = pypto.cast(residual, pypto.DT_FP32)
    gamma_fp32 = pypto.cast(gamma, pypto.DT_FP32)

    new_residual = pypto.add(x_fp32, residual_fp32)
    squared = new_residual * new_residual
    mean_sq = pypto.sum(squared, dim=-1, keepdim=True)
    mean_sq = mean_sq / hidden_size

    rms = pypto.sqrt(mean_sq + eps)
    normalized = new_residual / rms
    result = normalized * gamma_fp32

    pypto.set_vec_tile_shapes(1, 4096)
    result_bf16 = pypto.cast(result, pypto.DT_BF16)
    result_residual = pypto.cast(new_residual, pypto.DT_BF16)
    return result_bf16, result_residual


def create_dynamic_fused_layer_kernel_v2_bsh(config: DynamicFusedLayerConfigV2BSH):
    """Create the JIT-compiled fused-layer kernel.

    The attention part follows the IFA pattern:
    - Outer loop over ``num_kv_heads``
    - Inner loop over kv sequence tiles (``s_tile``)
    - 2D matmul inside loops with ``valid_shape`` on scores
    - Online softmax accumulation per kv_head
    """
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_heads_per_group = config.num_heads_per_group
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    qkv_size = config.qkv_size
    kv_size = config.kv_size
    scale = config.scale
    rms_norm_eps = config.rms_norm_eps
    dtype = config.dtype

    g_tile = num_heads_per_group
    s_tile = 512

    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0},
        runtime_options={
            "device_sched_mode": 2,
            "stitch_function_max_num": 128,
            "ready_on_host_tensors": ["actual_kv_len"],
        },
        pass_options={
            "vec_nbuffer_setting": {-2: 1, -1: 8},
            "cube_l1_reuse_setting": {-1: 2},
        },
    )
    def pangu_fused_layer_dynamic_v2_bsh_kernel(
        hidden_states: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype),
        residual: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype),
        cos: pypto.Tensor([pypto.STATIC, pypto.STATIC], dtype),
        sin: pypto.Tensor([pypto.STATIC, pypto.STATIC], dtype),
        actual_kv_len: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT64),
        key_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], dtype),
        value_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], dtype),
        input_ln_weight: pypto.Tensor([pypto.STATIC], dtype),
        post_ln_weight: pypto.Tensor([pypto.STATIC], dtype),
        qkv_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC], dtype),
        qkv_bias: pypto.Tensor([pypto.STATIC], dtype),
        o_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC], dtype),
        o_bias: pypto.Tensor([pypto.STATIC], dtype),
        gate_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC], dtype),
        up_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC], dtype),
        down_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC], dtype),
        output: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype),
        residual_out: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype),
    ):
        pypto.experimental.set_operation_options(combine_axis=True)

        # --- Extract dynamic scalars ---
        actual_kv_len_val = actual_kv_len[0] + 1
        prev_len = actual_kv_len_val - 1

        # --- Extract cos/sin for current position ---
        pypto.set_vec_tile_shapes(1, head_dim)
        cos_cur = pypto.view(cos, [1, head_dim], [prev_len, 0])
        sin_cur = pypto.view(sin, [1, head_dim], [prev_len, 0])
        cos_fp32 = pypto.cast(cos_cur, pypto.DT_FP32)
        sin_fp32 = pypto.cast(sin_cur, pypto.DT_FP32)

        # --- Step 1: Input RMSNorm ---
        pypto.set_vec_tile_shapes(hidden_size)
        input_ln_w = pypto.reshape(input_ln_weight, [1, hidden_size], inplace=True)
        post_ln_w = pypto.reshape(post_ln_weight, [1, hidden_size], inplace=True)
        pypto.set_vec_tile_shapes(1, 1, hidden_size)
        hidden_2d = pypto.reshape(hidden_states, [1, hidden_size], inplace=True)
        residual_2d = pypto.reshape(residual, [1, hidden_size], inplace=True)

        hidden_normed, new_residual1 = _pypto_rms_norm(hidden_2d, residual_2d, input_ln_w, rms_norm_eps, hidden_size)

        # --- Step 2: QKV Projection (+bias) ---
        pypto.set_cube_tile_shapes([16, 16], [64, 256], [256, 256])
        pypto.set_vec_tile_shapes(1, hidden_size)
        qkv_fp32 = pypto.matmul(hidden_normed, qkv_weight, out_dtype=pypto.DT_FP32, b_trans=True)
        qkv_bias_fp32 = pypto.cast(qkv_bias, pypto.DT_FP32)
        qkv_bias_2d = pypto.reshape(qkv_bias_fp32, [1, qkv_size])
        qkv_fp32 = pypto.add(qkv_fp32, qkv_bias_2d)

        # --- Step 3: Split Q, K, V ---
        pypto.set_vec_tile_shapes(1, qkv_size)
        q_flat = pypto.view(qkv_fp32, [1, num_heads * head_dim], [0, 0])
        k_flat = pypto.view(qkv_fp32, [1, kv_size], [0, num_heads * head_dim])
        v_flat = pypto.view(qkv_fp32, [1, kv_size], [0, num_heads * head_dim + kv_size])

        pypto.set_vec_tile_shapes(num_heads, head_dim)
        q_fp32 = pypto.reshape(q_flat, [num_heads, head_dim])
        pypto.set_vec_tile_shapes(num_kv_heads, head_dim)
        k_fp32 = pypto.reshape(k_flat, [num_kv_heads, head_dim])

        # --- Step 4: RoPE Q (GPT-NeoX rotate_half, no deinterleave) ---
        pypto.set_vec_tile_shapes(num_heads, head_dim // 2)
        q1 = pypto.view(q_fp32, [num_heads, head_dim // 2], [0, 0])
        q2 = pypto.view(q_fp32, [num_heads, head_dim // 2], [0, head_dim // 2])
        q2_neg = pypto.mul(q2, -1.0)
        q_rot = pypto.concat([q2_neg, q1], dim=-1)
        pypto.set_vec_tile_shapes(num_heads, head_dim)
        q_embed_fp32 = pypto.add(pypto.mul(q_fp32, cos_fp32), pypto.mul(q_rot, sin_fp32))
        q_embed = pypto.cast(q_embed_fp32, dtype)

        # --- Step 5: RoPE K ---
        pypto.set_vec_tile_shapes(num_kv_heads, head_dim // 2)
        k1 = pypto.view(k_fp32, [num_kv_heads, head_dim // 2], [0, 0])
        k2 = pypto.view(k_fp32, [num_kv_heads, head_dim // 2], [0, head_dim // 2])
        k2_neg = pypto.mul(k2, -1.0)
        k_rot = pypto.concat([k2_neg, k1], dim=-1)
        pypto.set_vec_tile_shapes(num_kv_heads, head_dim)
        k_embed_fp32 = pypto.add(pypto.mul(k_fp32, cos_fp32), pypto.mul(k_rot, sin_fp32))
        k_embed = pypto.cast(k_embed_fp32, dtype)

        pypto.set_vec_tile_shapes(1, kv_size)
        v_bf16 = pypto.cast(v_flat, dtype)

        # --- Step 6: Update KV cache at position prev_len ---
        pypto.set_vec_tile_shapes(num_kv_heads, head_dim)
        k_embed_2d = pypto.reshape(k_embed, [num_kv_heads, head_dim])
        k_embed_flat = pypto.reshape(k_embed_2d, [1, kv_size])
        pypto.set_vec_tile_shapes(1, kv_size)
        pypto.assemble(k_embed_flat, [prev_len, 0], key_cache)

        v_cur = pypto.reshape(v_bf16, [1, kv_size])
        pypto.set_vec_tile_shapes(1, kv_size)
        pypto.assemble(v_cur, [prev_len, 0], value_cache)

        # --- Step 7: GQA Attention (IFA pattern, online softmax) ---
        pypto.set_vec_tile_shapes(num_heads, head_dim)
        q_2d = pypto.reshape(q_embed, [num_heads, head_dim])

        attn_out_acc = pypto.tensor([num_heads, head_dim], pypto.DT_FP32, "attn_out_acc")

        for n_idx in pypto.loop(num_kv_heads, name="LOOP_n", idx_name="n_idx"):
            pypto.set_vec_tile_shapes(g_tile, head_dim)
            q_tile = pypto.view(q_2d, [g_tile, head_dim], [n_idx * g_tile, 0])

            out_update = pypto.tensor([g_tile, head_dim], pypto.DT_FP32, "out_update")
            sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
            max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")

            num_kv_tiles = (actual_kv_len_val + s_tile - 1) // s_tile
            for s_idx in pypto.loop(num_kv_tiles, name="LOOP_s", idx_name="s_idx", unroll_list=[4, 2, 1]):
                cur_s_start = s_idx * s_tile
                cur_s_end = (cur_s_start + s_tile).min(actual_kv_len_val)
                cur_s_len = cur_s_end - cur_s_start

                pypto.set_vec_tile_shapes(s_tile, head_dim)
                k_tile = pypto.view(key_cache, [s_tile, head_dim],
                                    [cur_s_start, n_idx * head_dim],
                                    valid_shape=[cur_s_len, head_dim])

                # Q @ K^T
                pypto.set_cube_tile_shapes([16, 16], [128, 128], [256, 256])
                scores = pypto.matmul(q_tile, k_tile, pypto.DT_FP32, b_trans=True)

                pypto.set_vec_tile_shapes(g_tile, s_tile)
                scores = pypto.view(scores, [g_tile, s_tile], [0, 0],
                                    valid_shape=[g_tile, cur_s_len])

                scores_scaled = pypto.mul(scores, scale)
                tile_max = pypto.amax(scores_scaled, dim=-1, keepdim=True)
                score_diff = pypto.sub(scores_scaled, tile_max)
                tile_probs = pypto.exp(score_diff)
                tile_probs_bf16 = pypto.cast(tile_probs, dtype)
                sum_local = pypto.sum(tile_probs, dim=-1, keepdim=True)

                # P @ V
                pypto.set_vec_tile_shapes(s_tile, head_dim)
                v_tile = pypto.view(value_cache, [s_tile, head_dim],
                                    [cur_s_start, n_idx * head_dim],
                                    valid_shape=[cur_s_len, head_dim])
                pypto.set_cube_tile_shapes([16, 16], [128, 128], [128, 128])
                attn_partial = pypto.matmul(tile_probs_bf16, v_tile, pypto.DT_FP32)

                # Online softmax update
                if pypto.is_loop_begin(s_idx):
                    pypto.set_vec_tile_shapes(g_tile, head_dim)
                    out_update[:] = attn_partial
                    pypto.set_vec_tile_shapes(g_tile, 1)
                    sum_update[:] = sum_local
                    max_update[:] = tile_max
                else:
                    pypto.set_vec_tile_shapes(g_tile, 1)
                    max_new = pypto.maximum(max_update, tile_max)
                    max_diff = pypto.sub(max_update, max_new)
                    update_mul = pypto.exp(max_diff)
                    sum_update[:] = sum_update * update_mul + sum_local
                    pypto.set_vec_tile_shapes(g_tile, head_dim)
                    out_update[:] = out_update * update_mul + attn_partial
                    pypto.set_vec_tile_shapes(g_tile, 1)
                    max_update[:] = max_new

            pypto.set_vec_tile_shapes(g_tile, head_dim)
            attn_out = pypto.div(out_update, sum_update)
            pypto.assemble(attn_out, [n_idx * g_tile, 0], attn_out_acc)

        # --- Step 8: Output projection (+bias) ---
        pypto.set_vec_tile_shapes(num_heads, head_dim)
        attn_output_bf16 = pypto.cast(attn_out_acc, dtype)
        attn_flat = pypto.reshape(attn_output_bf16, [1, num_heads * head_dim])
        pypto.set_cube_tile_shapes([16, 16], [32, 512], [192, 192])
        attn_proj = pypto.matmul(attn_flat, o_weight, dtype, b_trans=True)
        o_bias_2d = pypto.reshape(o_bias, [1, hidden_size])
        attn_proj = pypto.add(attn_proj, o_bias_2d)

        # --- Step 9: Post-attention RMSNorm ---
        hidden_normed2, new_residual2 = _pypto_rms_norm(attn_proj, new_residual1, post_ln_w, rms_norm_eps, hidden_size)

        # --- Step 10: FFN (SwiGLU) ---
        pypto.set_cube_tile_shapes([16, 16], [32, 4096, 256], [288, 288])
        gate = pypto.matmul(hidden_normed2, gate_weight, out_dtype=dtype, b_trans=True)
        up = pypto.matmul(hidden_normed2, up_weight, out_dtype=dtype, b_trans=True)

        pypto.set_vec_tile_shapes(1, intermediate_size)
        neg_gate = pypto.mul(gate, F_NEGA_1)
        exp_neg = pypto.exp(neg_gate)
        ones = pypto.full(exp_neg.shape, 1.0, exp_neg.dtype, valid_shape=exp_neg.shape)
        sigmoid_gate = pypto.div(ones, pypto.add(exp_neg, ones))
        swish = pypto.mul(gate, sigmoid_gate)
        activated = pypto.mul(swish, up)

        pypto.set_cube_tile_shapes([16, 16], [32, 512], [192, 192])
        ffn_result = pypto.matmul(activated, down_weight, out_dtype=dtype, b_trans=True)

        # --- Write outputs ---
        pypto.set_vec_tile_shapes(1, 1, hidden_size)
        ffn_result_3d = pypto.reshape(ffn_result, [1, 1, hidden_size])
        new_residual2_3d = pypto.reshape(new_residual2, [1, 1, hidden_size])
        pypto.assemble(ffn_result_3d, [0, 0, 0], output)
        pypto.assemble(new_residual2_3d, [0, 0, 0], residual_out)

    return pangu_fused_layer_dynamic_v2_bsh_kernel


# ---------------------------------------------------------------------------
# torch.ops.pypto custom op registration
# ---------------------------------------------------------------------------

_FUSED_LAYER_OP_SIG = (
    "pangu_fused_layer_v2_bsh("
    "Tensor hidden_states, Tensor residual, Tensor cos, Tensor sin, "
    "Tensor key_cache, Tensor value_cache, "
    "Tensor input_ln_weight, Tensor post_ln_weight, "
    "Tensor qkv_weight, Tensor qkv_bias, Tensor o_weight, Tensor o_bias, "
    "Tensor gate_weight, Tensor up_weight, Tensor down_weight, "
    "Tensor actual_kv_len"
    ") -> (Tensor, Tensor)"
)

_FUSED_LAYER_PARAMS = (
    "hidden_states", "residual", "cos", "sin",
    "key_cache", "value_cache",
    "input_ln_weight", "post_ln_weight",
    "qkv_weight", "qkv_bias", "o_weight", "o_bias",
    "gate_weight", "up_weight", "down_weight",
    "actual_kv_len",
)


def _fused_layer_call(args):
    """Invoke the JIT kernel with positional args (in kernel-signature order)."""
    config = DynamicFusedLayerConfigV2BSH()
    device = args[0].device
    dtype = torch.bfloat16
    output = torch.empty(1, 1, config.hidden_size, dtype=dtype, device=device)
    new_residual = torch.empty(1, 1, config.hidden_size, dtype=dtype, device=device)
    kernel = _get_cached_kernel()
    # Reorder: graph-order (actual_kv_len last) -> kernel-order (actual_kv_len 5th)
    kernel(
        args[0], args[1], args[2], args[3],         # hidden, residual, cos, sin
        args[15],                                    # actual_kv_len (last in graph order)
        args[4], args[5],                            # key_cache, value_cache
        args[6], args[7],                            # input_ln, post_ln
        args[8], args[9], args[10], args[11],        # qkv_w, qkv_b, o_w, o_b
        args[12], args[13], args[14],                # gate, up, down
        output, new_residual,
    )
    return output, new_residual


_cached_kernel = None


def _get_cached_kernel():
    global _cached_kernel
    if _cached_kernel is None:
        _cached_kernel = create_dynamic_fused_layer_kernel_v2_bsh(DynamicFusedLayerConfigV2BSH())
    return _cached_kernel


@allow_in_graph
def npu_pangu_fused_layer_v2_bsh(*args) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused layer op with BSH KV cache, compatible with acl graph.

    Output tensors are created inside and returned, which is required for
    proper acl graph capture and replay.
    """
    if isinstance(args[0], FakeTensor):
        raise TypeError("hidden_states is FakeTensor")
    return _fused_layer_call(args)


_pyptolib = torch.library.Library("pypto", "FRAGMENT")
_pyptolib.define(_FUSED_LAYER_OP_SIG)


@torch.library.impl(_pyptolib, "pangu_fused_layer_v2_bsh", "Meta")
def _pangu_fused_layer_v2_bsh_meta(*args):
    """Shape inference for acl graph tracing."""
    shape = args[0].shape
    dt = args[0].dtype
    dev = args[0].device
    return (
        torch.empty(shape, dtype=dt, device=dev),
        torch.empty(shape, dtype=dt, device=dev),
    )


@torch.library.impl(_pyptolib, "pangu_fused_layer_v2_bsh", "PrivateUse1")
def _pangu_fused_layer_v2_bsh_npu(*args):
    """NPU implementation delegates to the @allow_in_graph function."""
    return npu_pangu_fused_layer_v2_bsh(*args)


def pangu_fused_layer_v2_bsh_graph(*args):
    """Graph-compatible entry point that calls the registered custom op."""
    return torch.ops.pypto.pangu_fused_layer_v2_bsh(*args)


class PanguFusedLayerV2BSHModule(torch.nn.Module):
    """torch.nn.Module wrapper, compatible with acl graph.

    Usage::

        model = PanguFusedLayerV2BSHModule(config)
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            output, new_residual = model(
                hidden_states, residual, cos, sin,
                key_cache, value_cache,
                input_ln_weight, post_ln_weight,
                qkv_weight, qkv_bias, o_weight, o_bias,
                gate_weight, up_weight, down_weight,
                actual_kv_len,
            )
        g.replay()
    """

    def __init__(self, config: DynamicFusedLayerConfigV2BSH = None):
        super().__init__()
        self.config = config or DynamicFusedLayerConfigV2BSH()
        self.hidden_size = self.config.hidden_size

    def forward(self, *args) -> Tuple[torch.Tensor, torch.Tensor]:
        return pangu_fused_layer_v2_bsh_graph(*args)
