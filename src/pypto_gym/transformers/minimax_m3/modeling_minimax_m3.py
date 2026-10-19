# coding=utf-8
# Copyright 2025 MiniMax AI and the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTICE: This file was modified by Huawei Technologies Co., Ltd. in 2026 to
# leverage pypto technology for fusing the MoE expert FFN via a grouped GEMM kernel.
# -----------------------------------------------------------------------------------------------------------
"""In-repo PyTorch MiniMax-M3 text-backbone model, with the MoE expert FFN on the PyPTO kernel.

MiniMax-M3 ships as a vision-language model whose **text backbone** is a Mixtral-style
sparse-MoE decoder. The official HF checkpoint bundles no ``transformers`` text modeling
code (it is served via vLLM / sglang), and ``transformers`` has no native ``minimax_m3_vl``
architecture, so — exactly like ``minimax_m27`` — this module defines the text backbone
**in-repo** (no ``trust_remote_code``) and grafts the routed MoE expert FFN onto the fused
PyPTO grouped-GEMM kernel. Structure verified against the real MiniMax-M3 checkpoint
(``config.json`` text_config + the real weight names).

Deltas vs ``minimax_m27`` (all from the M3 config / weights):
  * **swigluoai** activation (clamped GLU, alpha=1.702 limit=7.0) in the dense MLP, the
    shared expert, and the routed experts (and inside the kernel) — not SiLU-SwiGLU.
  * **shared expert** (``n_shared_experts``) added to the routed output, plus a
    ``routed_scaling_factor`` on the routed sum.
  * **leading dense layers** (``first_k_dense_replace`` = 3): layers 0-2 are a plain MLP
    (``mlp.{gate,up,down}_proj``), layers 3+ are the sparse MoE block.
  * **Gemma-style** ``(1 + weight)`` RMSNorm and **per-head** qk-norm.
  * **partial-rotary** RoPE (rotary_dim 64 of head_dim 128).

Scope: the MiniMax Sparse Attention (MSA) lightning indexer (``self_attn.index_*``) IS now
modelled (see ``MiniMaxM3Attention``): on sparse layers (``sparse_attention_freq[i]==1``, i.e.
the MoE layers ``>= first_k_dense_replace``) a pure-selection indexer scores Q against K with a
small ``index_n_heads``-head dot product, max-pools per-key scores into ``index_block_size`` key
blocks, and keeps per query the top-``index_topk_blocks`` blocks (+ ``index_local_block`` local).
The selection folds into the attention mask, so for ``num_key_blocks <= topk`` (short/medium
contexts) every block is kept and MSA is numerically identical to dense GQA (verified: max abs
diff 0); for long contexts it is block-sparse (the trained long-context speed path). The indexer
weights (``self_attn.index_{q,k}_{proj,norm}``) load directly. The MTP heads are still not
modelled (only ``num_hidden_layers`` decoder layers are built); any MTP/extra tensors are dropped
by ``load_state_dict(strict=False)`` (no MTP weights ship in the current checkpoint).

Only the routed expert FFN is fused (``USE_PTO_GROUPED_GEMM``, default False); routing, the
shared expert, norms and attention stay on the host. The PyPTO patch layer is unchanged from
the prior skeleton; weights are FP8 (block-dequant to BF16) or BF16 — see ``_expert_weight``.
"""

from __future__ import annotations

__all__ = [
    "MiniMaxM3RMSNorm", "MiniMaxM3MLP", "MiniMaxM3Expert", "MiniMaxM3Experts",
    "MiniMaxM3SparseMoeBlock", "MiniMaxM3Attention", "MiniMaxM3DecoderLayer",
    "MiniMaxM3Model", "MiniMaxM3ForCausalLM", "MiniMaxM3PreTrainedModel",
    "MiniMaxM3PyptoExperts", "patch_minimax_m3_moe", "is_minimax_m3_moe", "patch_moe",
    "_dequant_fp8_block", "build_model", "load_streaming_state_dict",
    "attach_expert_fp8", "materialize_meta_buffers", "load_model",
]

import gc
import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

from ops.pypto_tensor import minimax as _pto_ops           # read USE_PTO_* dynamically
from ops.pypto_tensor.minimax import (
    grouped_gemm, MoeDims,
)
from .configuration_minimax_m3 import MiniMaxM3Config


# ============================ model definition (in-repo) ============================

def _swiglu_oai(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """swigluoai (GPT-OSS clamped GLU): ``(clamp(up) + 1) * (g * sigmoid(alpha*g))``.

    gate is clamped to ``max=limit``; up is clamped to ``[-limit, +limit]``. Matches the
    kernel's ``_swiglu_oai`` and the precision-test golden.
    """
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return (up + 1.0) * (gate * torch.sigmoid(alpha * gate))


class MiniMaxM3RMSNorm(nn.Module):
    """RMSNorm over the last dim. ``use_gemma_norm`` -> ``(1 + weight)`` scale (Gemma-style)."""

    def __init__(self, hidden_size, eps=1e-6, use_gemma_norm=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size) if use_gemma_norm else torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.use_gemma_norm = use_gemma_norm

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        weight = (1.0 + self.weight.float()) if self.use_gemma_norm else self.weight.float()
        return (weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, gemma={self.use_gemma_norm}"


class MiniMaxM3MLP(nn.Module):
    """Dense / shared-expert MLP with swigluoai. Names match the checkpoint (gate/up/down_proj)."""

    def __init__(self, config: MiniMaxM3Config, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.alpha = config.swiglu_alpha
        self.limit = config.swiglu_limit

    def forward(self, hidden_states):
        return self.down_proj(_swiglu_oai(self.gate_proj(hidden_states), self.up_proj(hidden_states),
                                          self.alpha, self.limit))


class MiniMaxM3Expert(nn.Module):
    """A single routed expert (eager fallback when the kernel is off). w1=gate, w3=up, w2=down."""

    def __init__(self, config: MiniMaxM3Config):
        super().__init__()
        ffn_dim, hidden_dim = config.intermediate_size, config.hidden_size
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)   # gate
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)   # down
        self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)   # up
        self.alpha = config.swiglu_alpha
        self.limit = config.swiglu_limit

    def forward(self, hidden_states):
        return self.w2(_swiglu_oai(self.w1(hidden_states), self.w3(hidden_states), self.alpha, self.limit))


class MiniMaxM3Experts(nn.ModuleList):
    """ModuleList of routed experts. Eager forward (replaced by the PyPTO kernel when enabled)."""

    def __init__(self, config: MiniMaxM3Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        for _ in range(self.num_experts):
            self.append(MiniMaxM3Expert(config))

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        """hidden_states [N, H]; top_k_index/top_k_weights [N, top_k] -> weighted sum [N, H]."""
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            current_hidden_states = self[expert_idx](current_state) * top_k_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states


class MiniMaxM3SparseMoeBlock(nn.Module):
    """Router (sigmoid + bias) + routed experts (scaled) + shared expert."""

    def __init__(self, config: MiniMaxM3Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = MiniMaxM3Experts(config)
        self.register_buffer("e_score_correction_bias", torch.zeros(config.num_local_experts))
        if config.n_shared_experts and config.n_shared_experts > 0:
            shared_dim = config.shared_intermediate_size * config.n_shared_experts
            self.shared_experts = MiniMaxM3MLP(config, shared_dim)
        else:
            self.shared_experts = None

    def route_tokens_to_experts(self, router_logits):
        routing_weights = torch.nn.functional.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        if self.norm_topk_prob:
            top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_index, top_k_weights

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # M3 routes in FP32 (gate.weight + e_score_correction_bias are fp32); cast so the gate
        # matmul dtypes match and the small inter-expert score gaps survive.
        router_logits = self.gate(hidden_states.float())
        top_k_index, top_k_weights = self.route_tokens_to_experts(router_logits)
        routed = self.experts(hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype))
        routed = routed * self.routed_scaling_factor
        if self.shared_experts is not None:
            routed = routed + self.shared_experts(hidden_states)
        return routed.reshape(batch_size, sequence_length, hidden_dim), router_logits


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
    scaling = kwargs.pop("scaling", None)
    dropout = kwargs.pop("dropout", 0.0)
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Partial rotary: cos/sin have width ``rotary_dim`` (< head_dim); the tail passes through."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


class MiniMaxM3PagedSparseCache:
    """Single-sequence (B=1) paged KV + indexer-K cache for MSA block-sparse DECODE via
    ``npu_fused_infer_attention_score``. Each layer holds a contiguous paged tensor
    ``[max_blocks, page, num_kv_heads*head_dim]`` (logical block i == physical block i) plus the
    RoPE'd indexer keys ``[max_ctx, index_dim]``. The block_table from the lightning indexer selects
    which physical blocks the paged op attends, so decode cost is O(topk*page) instead of O(ctx).

    Holds *RoPE-applied* K and idx_K (the paged op does not rotate). Use is opt-in: attach it as
    ``past_key_values`` and set ``attn.use_msa_sparse = True`` on the sparse layers.
    """

    def __init__(self, config, max_ctx, device, dtype):
        sa = config.sparse_attention_config
        self.page = sa.get("sparse_block_size", 128)
        self.idx_dim = sa.get("sparse_index_dim", 128)
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.nkv = config.num_key_value_heads
        self.kv_hidden = self.nkv * self.head_dim
        n_layers = config.num_hidden_layers
        nb = -(-max_ctx // self.page)
        self.max_blocks = nb

        def z(*s):
            return torch.zeros(*s, device=device, dtype=dtype)
        self.k = [z(nb, self.page, self.kv_hidden) for _ in range(n_layers)]
        self.v = [z(nb, self.page, self.kv_hidden) for _ in range(n_layers)]
        self.idx_k = [z(nb * self.page, self.idx_dim) for _ in range(n_layers)]
        self.length = [0] * n_layers

    def append(self, layer_idx, k_new, v_new, idx_k_new):
        """Append one step's K/V/idx_k to the paged cache and return the post-append sequence length.

        k_new/v_new: [1, nkv, S, D] (RoPE'd); idx_k_new: [1, 1, S, idx_dim] (RoPE'd). Writes through
        the contiguous flat view so block boundaries are handled automatically.
        """
        s = k_new.shape[2]
        start = self.length[layer_idx]
        kf = self.k[layer_idx].view(-1, self.kv_hidden)
        vf = self.v[layer_idx].view(-1, self.kv_hidden)
        kf[start:start + s] = k_new.transpose(1, 2).reshape(s, self.kv_hidden)
        vf[start:start + s] = v_new.transpose(1, 2).reshape(s, self.kv_hidden)
        self.idx_k[layer_idx][start:start + s] = idx_k_new.reshape(s, self.idx_dim)
        self.length[layer_idx] = start + s
        return self.length[layer_idx]


@dataclass
class _QKV:
    """Bundle of the three projected attention states (post-RoPE)."""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


class MiniMaxM3Attention(nn.Module):
    """GQA with per-head qk-norm, partial rotary, and MiniMax sparse-attention selection."""

    def __init__(self, config: MiniMaxM3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:   # per-head RMSNorm over head_dim
            self.q_norm = MiniMaxM3RMSNorm(self.head_dim, eps=config.rms_norm_eps,
                                           use_gemma_norm=config.use_gemma_norm)
            self.k_norm = MiniMaxM3RMSNorm(self.head_dim, eps=config.rms_norm_eps,
                                           use_gemma_norm=config.use_gemma_norm)

        # MiniMax Sparse Attention (MSA) — "lightning indexer": a pure *selection* branch (no value
        # proj, no residual) that scores each query against every key with a small index_n_heads-head
        # dot product, max-pools per-key scores into key-blocks, and keeps per query the top-k blocks
        # (+ the local block, always visible). Active only on sparse layers (sparse_attention_freq==1,
        # i.e. the MoE layers >= first_k_dense_replace). Param names match the checkpoint
        # (self_attn.index_{q,k}_{proj,norm}) so they load directly. When num_key_blocks <= topk every
        # block is kept -> the block mask == the plain causal mask -> MSA is numerically identical to
        # dense GQA (which is why short/medium contexts were safely approximated by dense GQA before).
        sa = getattr(config, "sparse_attention_config", None) or {}
        freq = sa.get("sparse_attention_freq")
        self.is_sparse = bool(sa.get("use_sparse_attention")) and (
            bool(freq[layer_idx]) if freq else layer_idx >= config.first_k_dense_replace)
        if self.is_sparse:
            self.index_head_dim = sa.get("sparse_index_dim", self.head_dim)
            self.index_n_heads = sa.get("sparse_num_index_heads", 4)
            self.index_block_size = sa.get("sparse_block_size", 128)
            self.index_topk_blocks = sa.get("sparse_topk_blocks", 16)
            self.index_local_blocks = sa.get("sparse_local_block", 1)
            self.index_q_proj = nn.Linear(config.hidden_size, self.index_n_heads * self.index_head_dim, bias=False)
            self.index_k_proj = nn.Linear(config.hidden_size, self.index_head_dim, bias=False)
            self.index_q_norm = MiniMaxM3RMSNorm(self.index_head_dim, eps=config.rms_norm_eps,
                                                 use_gemma_norm=config.use_gemma_norm)
            self.index_k_norm = MiniMaxM3RMSNorm(self.index_head_dim, eps=config.rms_norm_eps,
                                                 use_gemma_norm=config.use_gemma_norm)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if self.use_qk_norm:                 # per-head: norm over the head_dim axis
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # MSA block-sparse fast path (opt-in: set ``attn.use_msa_sparse=True`` + attach a
        # MiniMaxM3PagedSparseCache). DECODE attends only the indexer-selected key blocks. By default
        # this uses the native paged kernel as a speed target/oracle; set ``attn.use_pypto_msa=True``
        # as well to exercise the PyPTO indexer + PyPTO sparse-decode kernels. PREFILL populates the
        # cache and falls through to the dense MSA-mask path below. Default off -> existing behaviour
        # (HF Cache + eager) is unchanged.
        use_paged = (getattr(self, "is_sparse", False) and getattr(self, "use_msa_sparse", False)
                     and isinstance(past_key_values, MiniMaxM3PagedSparseCache))
        if use_paged:
            paged_out = self._try_msa_paged_decode(
                hidden_states, position_embeddings, input_shape,
                _QKV(query_states, key_states, value_states), past_key_values)
            if paged_out is not None:
                return paged_out
            # PREFILL: cache populated; key/value_states are the full prompt (no past) -> dense MSA mask
        elif past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # MSA: select top-k key blocks per query and fold the selection into the attention mask.
        # (When num_key_blocks <= topk this keeps every block -> mask == causal -> == dense GQA.)
        if getattr(self, "is_sparse", False):
            mask_kwargs = {**kwargs, "cache_position": cache_position}
            attention_mask = self._msa_attention_mask(
                hidden_states, position_embeddings, attention_mask,
                _QKV(query_states, key_states, value_states), mask_kwargs)

        attention_interface: Callable = eager_attention_forward
        attn_impl = getattr(self.config, "_attn_implementation", "eager")
        if attn_impl != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]

        attn_output, _ = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)

    # ----- MSA lightning indexer (block-sparse key selection) -----
    def _msa_index_qk(self, hidden_states, position_embeddings):
        """Project + qk-norm + partial-RoPE the lightning-indexer Q/K. Shared by the prefill mask path
        and the decode block-sparse path. idx_q: [B, nidx, S, hdim], idx_k: [B, 1, S, hdim] (MQA)."""
        bsz, q_len, _ = hidden_states.shape
        hdim, nh = self.index_head_dim, self.index_n_heads
        idx_q = self.index_q_norm(self.index_q_proj(hidden_states).view(bsz, q_len, nh, hdim)).transpose(1, 2)
        idx_k = self.index_k_norm(self.index_k_proj(hidden_states).view(bsz, q_len, 1, hdim)).transpose(1, 2)
        cos, sin = position_embeddings                       # partial rotary (width rotary_dim)
        return apply_rotary_pos_emb(idx_q, idx_k, cos, sin)

    def _msa_block_indices(self, hidden_states, position_embeddings, position_ids):
        """Port of the MiniMax-M3 lightning indexer. Returns per-query top-k key-block indices
        [B, S_q, topk]; future/empty slots are -1. Pure selection (no value path)."""
        bsz, q_len, _ = hidden_states.shape
        bs, hdim, nh = self.index_block_size, self.index_head_dim, self.index_n_heads
        idx_q, idx_k = self._msa_index_qk(hidden_states, position_embeddings)
        k_len = idx_k.shape[2]
        num_blocks = -(-k_len // bs)                         # ceil-div
        pad = num_blocks * bs - k_len
        # Score in the input dtype (bf16 on NPU): the FP32 cast forces the slow non-cube matmul path
        # on 910B and doubles the idx_k HBM read, while the block top-k *selection* is robust to bf16
        # rounding (validated IDENTICAL to fp32). CPU (fp32 inputs) stays fp32, so tests are unaffected.
        scores = torch.matmul(idx_q, idx_k.transpose(-1, -2))                   # [B, nh, Sq, Sk]
        k_pos = torch.arange(k_len, device=hidden_states.device)
        future = k_pos[None, None, None, :] > position_ids[:, None, :, None]    # causal
        scores = scores.masked_fill(future, float("-inf"))
        if pad:
            scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
        scores = scores.view(bsz, nh, q_len, num_blocks, bs)
        block_scores = scores.amax(dim=-1).amax(dim=1)       # max over block tokens, then over heads
        q_block = position_ids // bs
        if self.index_local_blocks > 0:                      # local blocks always visible (score=+inf)
            local = torch.arange(self.index_local_blocks, device=hidden_states.device)
            local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min=0)
            block_scores.scatter_(-1, local_idx, float("inf"))
        topk = min(self.index_topk_blocks, num_blocks)
        topk_scores, topk_indices = block_scores.topk(topk, dim=-1)
        return topk_indices.masked_fill(topk_scores == float("-inf"), -1)

    def _msa_decode_block_table(self, idx_q, idx_k_cache, length):
        """DECODE (single query) block selection -> int32 block_table [1, n_sel] for paged fia.

        Optimized vs the naive prefill port (validated numerics-identical to fp32 + to scatter+topk):
          * BF16 score, nidx query heads collapsed into the GEMM M-axis so the single MQA idx_k is
            read ONCE (a batched matmul re-reads idx_k per head);
          * the local blocks are deterministically the last ``local`` blocks -> sliced OUT of the topk
            candidates (a view) and appended last, so the current (partial) block is always block_table's
            last entry (matches the paged actual_seq_lengths_kv convention) and we avoid the ~150us
            single-element in-place scatter that forcing them with +inf would cost;
          * no torch.sort (block_table is order-invariant).
        idx_q: [1, nidx, 1, hdim] (RoPE'd); idx_k_cache: [length, hdim] (RoPE'd, contiguous).
        """
        bs, nidx, hdim = self.index_block_size, self.index_n_heads, self.index_head_dim
        nb = -(-length // bs)
        iq = idx_q.reshape(nidx, hdim)
        sc = torch.matmul(iq, idx_k_cache[:length].transpose(0, 1))          # [nidx, length] read once
        pad = nb * bs - length
        if pad:
            sc = torch.nn.functional.pad(sc, (0, pad), value=float("-inf"))
        blk = sc.view(nidx, nb, bs).amax(dim=-1).amax(dim=0)                 # [nb]
        ksel = min(self.index_topk_blocks, nb)
        local = self.index_local_blocks
        if local > 0 and nb > ksel:
            ids = blk[:nb - local].topk(ksel - local, sorted=False).indices.to(torch.int32)
            loc = torch.arange(nb - local, nb, device=ids.device, dtype=torch.int32)
            return torch.cat([ids, loc]).view(1, -1)                        # current block last
        return torch.arange(nb, device=blk.device, dtype=torch.int32).view(1, -1)

    def _msa_block_mask(self, block_indices, attention_mask, key_length, query_states, position_ids):
        """Expand selected block indices into the dense additive mask [B, 1, S_q, S_k] the eager
        attention path expects (0 at allowed (q,k) pairs, min_dtype elsewhere)."""
        dtype, device = query_states.dtype, query_states.device
        bsz, q_len, _ = block_indices.shape
        num_blocks = -(-key_length // self.index_block_size)
        safe = block_indices.masked_fill(block_indices < 0, num_blocks)
        bias = block_indices.new_full((bsz, q_len, num_blocks + 1), float("-inf"), dtype=dtype)
        bias.scatter_(-1, safe, 0.0)
        bias = bias[..., :num_blocks]
        block_keep = (bias == 0.0).repeat_interleave(self.index_block_size, dim=-1)[..., :key_length].unsqueeze(1)
        if attention_mask is not None:
            pad_mask = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
            keep = block_keep & pad_mask
        else:
            k_pos = torch.arange(key_length, device=device)
            future = k_pos[None, None, None, :] > position_ids[:, None, :, None]
            keep = block_keep & ~future
        min_dtype = torch.finfo(dtype).min
        return torch.zeros(keep.shape, dtype=dtype, device=device).masked_fill(~keep, min_dtype)

    def _pypto_msa_decode_output(self, idx_q, query_states, input_shape, past_key_values, length):
        from ops.pypto_tensor.minimax.minimax_m3_msa_indexer_impl import (
            minimax_m3_msa_indexer,
        )
        from ops.pypto_tensor.minimax.minimax_m3_msa_sparse_attention_impl import (
            minimax_m3_msa_sparse_decode,
        )

        page = past_key_values.page
        nb = -(-length // page)
        if nb <= self.index_topk_blocks:
            # Short context: nothing to prune (every block is selected), so the block-sparse
            # decode kernel has no sparsity to exploit and its tiling degenerates — defer to the
            # native paged decode, which handles the dense regime directly.
            return self._native_msa_decode_output(idx_q, query_states, input_shape, past_key_values, length)
        block_ids = minimax_m3_msa_indexer(
            idx_q.reshape(self.index_n_heads, self.index_head_dim),
            past_key_values.idx_k[self.layer_idx][:nb * page],
            nb,
        )
        k_blocks = past_key_values.k[self.layer_idx][:nb].view(
            nb, page, self.num_key_value_heads, self.head_dim).permute(2, 0, 1, 3).unsqueeze(0)
        v_blocks = past_key_values.v[self.layer_idx][:nb].view(
            nb, page, self.num_key_value_heads, self.head_dim).permute(2, 0, 1, 3).unsqueeze(0)
        out = minimax_m3_msa_sparse_decode(
            query_states.squeeze(2), k_blocks.contiguous(), v_blocks.contiguous(), block_ids, length)
        return self.o_proj(out.reshape(*input_shape, -1).contiguous())

    def _native_msa_decode_output(self, idx_q, query_states, input_shape, past_key_values, length):
        import torch_npu

        page = past_key_values.page
        block_table = self._msa_decode_block_table(idx_q, past_key_values.idx_k[self.layer_idx], length)
        nb = -(-length // page)
        fill = length - (nb - 1) * page
        actual_kv = int((block_table.shape[1] - 1) * page + fill)
        q_bsh = query_states.transpose(1, 2).reshape(1, 1, -1).contiguous()
        out = torch_npu.npu_fused_infer_attention_score(
            q_bsh, past_key_values.k[self.layer_idx], past_key_values.v[self.layer_idx],
            block_table=block_table, actual_seq_lengths_kv=[actual_kv],
            num_heads=self.num_attention_heads, num_key_value_heads=self.num_key_value_heads,
            scale=self.scaling, input_layout="BSH", block_size=page)[0]
        return self.o_proj(out.reshape(*input_shape, -1).contiguous())

    def _try_msa_paged_decode(self, hidden_states, position_embeddings, input_shape,
                              qkv, past_key_values):
        idx_q, idx_k = self._msa_index_qk(hidden_states, position_embeddings)
        length = past_key_values.append(self.layer_idx, qkv.k, qkv.v, idx_k)
        if qkv.q.shape[2] != 1:
            return None
        if getattr(self, "use_pypto_msa", False):
            return self._pypto_msa_decode_output(idx_q, qkv.q, input_shape, past_key_values, length)
        return self._native_msa_decode_output(idx_q, qkv.q, input_shape, past_key_values, length)

    def _msa_attention_mask(self, hidden_states, position_embeddings, attention_mask, qkv, kwargs):
        key_states, query_states = qkv.k, qkv.q
        cache_position = kwargs.get("cache_position")
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            k_len = key_states.shape[2]
            base = cache_position if cache_position is not None else torch.arange(
                k_len, device=hidden_states.device)
            position_ids = base[-query_states.shape[2]:].unsqueeze(0)
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        position_ids = position_ids.expand(query_states.shape[0], -1)
        block_indices = self._msa_block_indices(hidden_states, position_embeddings, position_ids)
        return self._msa_block_mask(
            block_indices, attention_mask, key_states.shape[2],
            query_states, position_ids)


class MiniMaxM3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MiniMaxM3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MiniMaxM3Attention(config, layer_idx)
        self.is_moe = layer_idx >= config.first_k_dense_replace
        if self.is_moe:
            self.block_sparse_moe = MiniMaxM3SparseMoeBlock(config)
        else:                                # leading dense layers
            self.mlp = MiniMaxM3MLP(config, config.dense_intermediate_size)
        self.input_layernorm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps,
                                                use_gemma_norm=config.use_gemma_norm)
        self.post_attention_layernorm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps,
                                                         use_gemma_norm=config.use_gemma_norm)

    def forward(self, hidden_states, position_embeddings, **kwargs):
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        past_key_values = kwargs.pop("past_key_values", None)
        cache_position = kwargs.pop("cache_position", None)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states, position_embeddings=position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values,
            cache_position=cache_position, **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            hidden_states, _ = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MiniMaxM3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: MiniMaxM3Config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type)
        if self.rope_init_fn is None:
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            exponent = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
            inv_freq = 1.0 / (config.rope_theta ** exponent)
            self.rope_init_fn = None
            self.attention_scaling = 1.0
        else:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MiniMaxM3PreTrainedModel(PreTrainedModel):
    config_class = MiniMaxM3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxM3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_flash_attn = False
    _can_compile_fullgraph = False


class MiniMaxM3Model(MiniMaxM3PreTrainedModel):
    def __init__(self, config: MiniMaxM3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MiniMaxM3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps,
                                     use_gemma_norm=config.use_gemma_norm)
        self.rotary_emb = MiniMaxM3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                inputs_embeds=None, **kwargs) -> MoeModelOutputWithPast:
        past_key_values = kwargs.pop("past_key_values", None)
        use_cache = kwargs.pop("use_cache", None)
        cache_position = kwargs.pop("cache_position", None)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config, input_embeds=inputs_embeds, attention_mask=attention_mask,
            cache_position=cache_position, past_key_values=past_key_values, position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states, position_embeddings=position_embeddings, attention_mask=causal_mask,
                position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache,
                cache_position=cache_position, **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class MiniMaxM3ForCausalLM(MiniMaxM3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniMaxM3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                inputs_embeds=None, **kwargs) -> MoeCausalLMOutputWithPast:
        past_key_values = kwargs.pop("past_key_values", None)
        use_cache = kwargs.pop("use_cache", None)
        cache_position = kwargs.pop("cache_position", None)
        logits_to_keep = kwargs.pop("logits_to_keep", 0)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=use_cache,
            cache_position=cache_position, **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return MoeCausalLMOutputWithPast(
            loss=None, logits=logits, past_key_values=outputs.past_key_values,
        )


# ============================ PyPTO patch layer ============================
_FP8_BLOCK = 128

# Streaming-path expert-count buckets. The grouped-GEMM kernel is JIT-compiled per
# distinct ``num_experts``; rounding the active count UP to a fixed bucket (padding the
# extra slots with 0-token "dummy" experts the kernel skips) collapses the compile set
# to len(buckets) shapes. Default {32,64,128}; override via PYPTO_MOE_BUCKETS.
_MOE_BUCKETS = sorted(int(x) for x in os.environ.get("PYPTO_MOE_BUCKETS", "32,64,128").split(","))
_MOE_PAD_ROWS = 64


def _bucket_for(n_active: int, num_experts: int) -> int:
    """Smallest configured bucket >= n_active, never exceeding the real expert count."""
    for b in _MOE_BUCKETS:
        if n_active <= b and b <= num_experts:
            return b
    return num_experts


# Module-global streaming weight buffers, shared across all MoE layers (which run
# sequentially) so total buffer memory is ~one layer's worth, not 57x. Keyed by
# (bucket, I, H, device). See MiniMaxM3PyptoExperts.get_stream_buffers.
_STREAM_BUFFERS: dict = {}


def _dequant_fp8_block(weight: torch.Tensor, scale_inv: torch.Tensor, block: int = _FP8_BLOCK) -> torch.Tensor:
    """Dequantize a block-quantized FP8 weight to BF16.

    weight    : [O, I] float8_e4m3fn
    scale_inv : [ceil(O/block), ceil(I/block)] float32 — dequant multipliers
    returns   : [O, I] bfloat16, where W[o,i] = fp8[o,i] * scale_inv[o//block, i//block]
    """
    out_dim, in_dim = weight.shape
    w = weight.to(torch.float32)
    s = scale_inv.repeat_interleave(block, dim=0)[:out_dim].repeat_interleave(block, dim=1)[:, :in_dim]
    return (w * s).to(torch.bfloat16)


def _expert_weight(linear: torch.nn.Module) -> torch.Tensor:
    """Return an expert Linear's weight as BF16, dequantizing block-FP8 if needed."""
    w = linear.weight
    scale_inv = getattr(linear, "weight_scale_inv", None)
    if scale_inv is not None and w.dtype == torch.float8_e4m3fn:
        return _dequant_fp8_block(w, scale_inv)
    return w.to(torch.bfloat16)


class MiniMaxM3PyptoExperts:
    """Replacement ``forward`` for the M3 MoE experts using fused grouped GEMM.

    Bound onto an existing experts ModuleList (so ``self`` is the container of
    experts). Mirrors the original signature:
        forward(hidden_states, top_k_index, top_k_weights) -> [N, H]

    The host MoE block still does routing and applies ``routed_scaling_factor`` and
    the shared expert; this only fuses the routed-expert FFN.
    """

    def __init__(self):
        # The methods below are bound onto a live experts ModuleList by
        # patch_minimax_m3_moe (self is that ModuleList), so this constructor is not
        # invoked at runtime; it declares the instance attributes those methods set.
        self.pypto_streaming = False
        self.pypto_ready = False
        self.pypto_num_experts = 0
        self.pypto_intermediate = 0
        self.pypto_hidden = 0
        self.pypto_w13_flat = None
        self.pypto_w2_flat = None
        # NPUGraph static-route cumsum cache (set by the graph bench / grouped_ffn; see grouped_ffn).
        self.static_cumsum_cache = False
        self.cached_cumsum = None

    @torch.no_grad()
    def ensure_pypto_weights(self):
        """Prepare expert weights for the kernel.

        * **prebuilt** (default): build the full ``[E*H, 2I]`` / ``[E*I, H]`` flat
          caches once, freeing each expert's original Linear weight as we go so the
          block's MoE memory stays ~1x.
        * **streaming**: keep the expert weights as-is (FP8, typically CPU-resident)
          and dequantize only the routed experts per forward (see ``grouped_ffn``).
        """
        if getattr(self, "pypto_ready", False):
            return
        experts = list(self)
        num_experts = len(experts)
        e0 = experts[0]
        intermediate_size, hidden_size = e0.w1.weight.shape          # w1 (gate): [I, H]
        self.pypto_num_experts = num_experts
        self.pypto_intermediate = intermediate_size
        self.pypto_hidden = hidden_size

        if getattr(self, "pypto_streaming", False):
            # keep originals (FP8 on CPU); nothing pre-materialized.
            self.pypto_ready = True
            return

        dev = e0.w1.weight.device
        w13_flat = torch.empty(num_experts * hidden_size, 2 * intermediate_size, dtype=torch.bfloat16, device=dev)
        w2_flat = torch.empty(num_experts * intermediate_size, hidden_size, dtype=torch.bfloat16, device=dev)
        empty = torch.empty(0, dtype=torch.bfloat16, device=dev)
        for e, exp in enumerate(experts):
            w1 = _expert_weight(exp.w1)        # gate [I, H]
            w3 = _expert_weight(exp.w3)        # up   [I, H]
            w2 = _expert_weight(exp.w2)        # down [H, I]
            # gate||up concat then transpose -> [H, 2I]; matches convert_minimax_weights.
            w13_flat[e * hidden_size:(e + 1) * hidden_size] = torch.cat([w1, w3], dim=0).t().contiguous()
            w2_flat[e * intermediate_size:(e + 1) * intermediate_size] = w2.t().contiguous()  # [I, H]
            # release the originals (free ~1x); patched forward uses only the flat caches
            exp.w1.weight.data = empty
            exp.w3.weight.data = empty
            exp.w2.weight.data = empty
        self.pypto_w13_flat = w13_flat
        self.pypto_w2_flat = w2_flat
        self.pypto_ready = True

    @torch.no_grad()
    def fill_active_flat(self, active_ids, w13c, w2c):
        """Dequantize only the routed experts into the first active slots of the buffers.

        FP8->BF16 on the host (910B has no native FP8); remaining slots keep stale data
        but carry 0 routed tokens, so the kernel skips them.
        """
        experts = list(self)
        intermediate_size, hidden_size = self.pypto_intermediate, self.pypto_hidden
        for ci, e in enumerate(active_ids.tolist()):
            exp = experts[e]
            w1 = _expert_weight(exp.w1).to(w13c.device)
            w3 = _expert_weight(exp.w3).to(w13c.device)
            w2 = _expert_weight(exp.w2).to(w13c.device)
            w13c[ci * hidden_size:(ci + 1) * hidden_size] = torch.cat([w1, w3], dim=0).t().contiguous()  # [H,2I]
            w2c[ci * intermediate_size:(ci + 1) * intermediate_size] = w2.t().contiguous()               # [I, H]

    def get_stream_buffers(self, bucket, dev):
        """One persistent (w13, w2) BF16 buffer pair per (bucket, shape, device).

        Shared across **all** layers via a module-global cache: layers run sequentially,
        so a single set of buffers is reused, keeping total weight-buffer memory at ~one
        layer's worth instead of 57x. The kernel sees the same tensor objects, so it does
        not accumulate inputs.
        """
        intermediate_size, hidden_size = self.pypto_intermediate, self.pypto_hidden
        key = (bucket, intermediate_size, hidden_size, str(dev))
        if key not in _STREAM_BUFFERS:
            _STREAM_BUFFERS[key] = (
                torch.empty(bucket * hidden_size, 2 * intermediate_size, dtype=torch.bfloat16, device=dev),
                torch.empty(bucket * intermediate_size, hidden_size, dtype=torch.bfloat16, device=dev))
        return _STREAM_BUFFERS[key]

    @torch.no_grad()
    def grouped_ffn(self, sorted_x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Run the fused expert FFN over expert-sorted tokens. Returns [N, H].

        ``counts`` is the per-expert token count ([E]); ``sorted_x`` is already grouped
        by ascending expert id. In streaming mode we compact to the active experts (and
        pad the count up to a fixed bucket so the kernel compiles a bounded set of shapes).
        """
        num_experts, intermediate_size = self.pypto_num_experts, self.pypto_intermediate
        num_tokens = sorted_x.shape[0]
        hidden_size = sorted_x.shape[1]
        dev = sorted_x.device
        # The grouped-GEMM kernel tiles each expert's dynamic token range with max tile 64.
        # On real decode routing, dynamic offsets can trigger a bounded MTE prefetch/write
        # past the final logical row. Pad the sorted input/output scratch by one tile so
        # codegen overreach lands in valid DDR, then return only real rows.
        sorted_padded = torch.nn.functional.pad(sorted_x, (0, 0, 0, _MOE_PAD_ROWS))
        out = torch.empty_like(sorted_padded)

        def _run(weights, cumsum, dims):
            grouped_gemm(sorted_padded, weights, cumsum, out, dims)

        if getattr(self, "pypto_streaming", False):
            active = counts.nonzero(as_tuple=False).flatten()
            n_active = int(active.numel())
            if n_active == 0:
                return out
            bucket = _bucket_for(n_active, num_experts)
            w13c, w2c = self.get_stream_buffers(bucket, dev)
            self.fill_active_flat(active, w13c, w2c)
            active_counts = counts[active]
            cumsum = torch.zeros(bucket + 1, dtype=torch.int32, device=dev)
            real = torch.cumsum(active_counts, 0).to(torch.int32)
            cumsum[1:n_active + 1] = real
            cumsum[n_active + 1:] = real[-1]
            _run((w13c, w2c), cumsum, MoeDims(bucket, hidden_size, intermediate_size, "swigluoai"))
            return out[:num_tokens]

        # Static-route (NPUGraph) path: build expert_cumsum ONCE outside capture and reuse the
        # persistent tensor on every replay. ready_on_host_tensors reads expert_cumsum on the host;
        # if cumsum is (re)built from counts inside the captured region, that host-read forces a
        # mid-capture device->host sync and the replayed schedule is stale -> 507011. Routing is
        # static here (constant counts), so cache the cumsum once (first/warmup call, pre-capture).
        if getattr(self, "static_cumsum_cache", False):
            cumsum = getattr(self, "cached_cumsum", None)
            if cumsum is None:
                cumsum = torch.zeros(num_experts + 1, dtype=torch.int32, device=dev)
                cumsum[1:] = torch.cumsum(counts, 0).to(torch.int32)
                self.cached_cumsum = cumsum
        else:
            cumsum = torch.zeros(num_experts + 1, dtype=torch.int32, device=dev)
            cumsum[1:] = torch.cumsum(counts, 0).to(torch.int32)
        _run((self.pypto_w13_flat, self.pypto_w2_flat), cumsum,
             MoeDims(num_experts, hidden_size, intermediate_size, "swigluoai"))
        return out[:num_tokens]

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        self.ensure_pypto_weights()
        num_tokens, hidden_size = hidden_states.shape
        top_k = top_k_index.shape[1]
        num_experts = self.pypto_num_experts
        dev = hidden_states.device

        flat_ids = top_k_index.reshape(-1).to(torch.int64)
        sort_perm = flat_ids.argsort()
        sorted_ids = flat_ids[sort_perm]
        tok_src = sort_perm // top_k
        sorted_x = hidden_states[tok_src].contiguous()

        counts = torch.bincount(sorted_ids, minlength=num_experts)

        inv = torch.empty_like(sort_perm)
        inv[sort_perm] = torch.arange(num_tokens * top_k, device=dev)

        out_sorted = self.grouped_ffn(sorted_x, counts)
        return (
            out_sorted[inv].view(num_tokens, top_k, hidden_size)
            .type(top_k_weights.dtype)
            .mul_(top_k_weights.reshape(num_tokens, top_k).unsqueeze(-1))
            .sum(dim=1)
            .type(hidden_states.dtype)
        )


def is_minimax_m3_moe(module) -> bool:
    """Duck-type check for a MiniMax-M3 sparse-MoE block (gate + experts of w1/w2/w3)."""
    experts = getattr(module, "experts", None)
    return (
        experts is not None
        and hasattr(module, "gate")
        and hasattr(experts, "__len__") and len(experts)
        and hasattr(experts[0], "w1") and hasattr(experts[0], "w2") and hasattr(experts[0], "w3")
    )


def patch_minimax_m3_moe(module, streaming: bool = False) -> bool:
    """Route a MiniMax-M3 MoE block's expert FFN through the PyPTO grouped GEMM kernel.

    Accepts either the sparse-MoE block (has ``.experts``) or the experts container
    directly. Replaces the experts' ``forward``. Returns True if patched, False if the
    kernel switch is off or the module is not a recognizable expert container.

    streaming : if True, keep expert weights as-is (FP8, usually CPU-resident) and
        dequantize only the routed experts per forward — HBM peak is ~one layer's active
        experts, so the full model runs on a single die. Default False (prebuilt flats).
    """
    if not getattr(_pto_ops, "USE_PTO_GROUPED_GEMM", False):     # opt-in switch; callers enable it
        return False
    experts = getattr(module, "experts", module)
    if not (hasattr(experts, "__len__") and len(experts) and hasattr(experts[0], "w1")):
        return False
    # Bind the CLASS functions onto the experts container so `self` IS the container
    # (so `list(self)` / self[e] iterate the real experts). Binding an adapter *instance*
    # method would keep self=adapter and break iteration.
    experts.pypto_streaming = bool(streaming)
    experts.pypto_ready = False
    experts.ensure_pypto_weights = MiniMaxM3PyptoExperts.ensure_pypto_weights.__get__(experts)
    experts.get_stream_buffers = MiniMaxM3PyptoExperts.get_stream_buffers.__get__(experts)
    experts.fill_active_flat = MiniMaxM3PyptoExperts.fill_active_flat.__get__(experts)
    experts.grouped_ffn = MiniMaxM3PyptoExperts.grouped_ffn.__get__(experts)
    experts.forward = MiniMaxM3PyptoExperts.forward.__get__(experts)
    return True


def patch_moe(model, streaming: bool = False) -> int:
    """Route every M3 MoE block's expert FFN through the PyPTO kernel. Returns #patched.

    streaming=True keeps experts FP8 on host (dequant per forward); else patch only
    blocks whose experts already live on a die.
    """
    n = 0
    for m in model.modules():
        if not is_minimax_m3_moe(m):
            continue
        if streaming:
            n += int(bool(patch_minimax_m3_moe(m, streaming=True)))
            continue
        try:
            dev = next(m.experts[0].parameters()).device
        except StopIteration:
            dev = None
        if dev is not None and dev.type == "npu":
            n += int(bool(patch_minimax_m3_moe(m)))
    return n


# ============================ checkpoint loading (VL text backbone, FP8/BF16) ============================
# The released MiniMax-M3 checkpoint stores the text backbone under a ``language_model.``
# prefix alongside vision_tower / projector / MSA-indexer (and, on MTP-bearing ckpts, extra)
# weights we do not model. These helpers strip the prefix, skip the out-of-scope tensors
# (vision/projector/index_; any MTP tensors fall through to load_state_dict(strict=False)),
# and (for FP8 checkpoints
# like MiniMax-M3-MXFP8) keep the routed-expert weights FP8 on the host for streaming dequant.

_TEXT_PREFIX = "language_model."
_SKIP_PREFIXES = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")
# MSA lightning-indexer tensors (self_attn.index_{q,k}_{proj,norm}) are now MODELLED (block-sparse
# attention) and load directly into MiniMaxM3Attention — no longer skipped.
_SKIP_SUBSTRS = ()


def _text_key(name: str) -> Optional[str]:
    """Map a checkpoint tensor name to the in-repo model key, or None to drop it."""
    if name.startswith(_SKIP_PREFIXES):
        return None
    if any(s in name for s in _SKIP_SUBSTRS):
        return None
    if name.startswith(_TEXT_PREFIX):
        name = name[len(_TEXT_PREFIX):]
    elif "." in name and name.split(".")[0] not in ("model", "lm_head"):
        return None                              # unknown top-level prefix -> not text backbone
    return name


def build_model(model_path, max_layers=None):
    """Build MiniMaxM3ForCausalLM on meta from the published config (no trust_remote_code)."""
    with open(f"{model_path}/config.json") as cfg_file:
        cj = json.load(cfg_file)
    cfg = MiniMaxM3Config.from_vl_config_dict(cj)
    setattr(cfg, "_attn_implementation", "eager")               # eager attention (no flash_attn on NPU)
    if max_layers is not None:
        cfg.num_hidden_layers = max_layers
    with torch.device("meta"):
        model = MiniMaxM3ForCausalLM(cfg)
    return model, cfg


def load_streaming_state_dict(model_path, max_layers=None):
    """Load the checkpoint for the streaming path.

    Non-expert weights go to a BF16 CPU state_dict; routed-expert weights are kept FP8 on
    CPU as ``{key: (fp8, scale_inv)}`` for just-in-time dequant (BF16 checkpoints store the
    expert directly in ``sd`` instead). Returns (state_dict, expert_fp8). Keys are already
    remapped to the in-repo model (``language_model.`` stripped, out-of-scope tensors dropped).
    """
    from safetensors import safe_open       # lazy: optional 3rd-party dep, not at module load
    with open(f"{model_path}/model.safetensors.index.json") as idx_file:
        idx = json.load(idx_file)["weight_map"]

    def keep_layer(key):
        if max_layers is None or ".layers." not in key:
            return True
        return int(key.split(".layers.")[1].split(".")[0]) < max_layers

    def is_routed_expert_w(key):
        return ".block_sparse_moe.experts." in key and key.endswith(".weight")

    by_shard = {}
    for name, shard in idx.items():
        if name.endswith(".weight_scale_inv"):
            continue
        key = _text_key(name)
        if key is None or not keep_layer(key):
            continue
        by_shard.setdefault(shard, []).append((name, key))

    sd, expert_fp8 = {}, {}
    for shard, names in sorted(by_shard.items()):
        f = safe_open(f"{model_path}/{shard}", "pt")
        present = set(f.keys())
        for name, key in names:
            t = f.get_tensor(name)
            sname = name + "_scale_inv"
            if key.endswith(".e_score_correction_bias") or key.endswith(".block_sparse_moe.gate.weight"):
                # Router bias/gate MUST stay FP32: bias values are ~11-19 with ~1e-3 inter-expert
                # gaps, and bf16's ~3e-2 quantization reorders the sigmoid+bias top-k selection.
                sd[key] = t.to(torch.float32)
            elif is_routed_expert_w(key) and sname in present:
                expert_fp8[key] = (t, f.get_tensor(sname))               # keep FP8 on CPU
            elif t.dtype == torch.float8_e4m3fn and sname in present:
                sd[key] = _dequant_fp8_block(t, f.get_tensor(sname))     # non-expert fp8 -> bf16
            else:
                sd[key] = t.to(torch.bfloat16) if t.is_floating_point() else t
        del f
        gc.collect()
    return sd, expert_fp8


def attach_expert_fp8(model, expert_fp8):
    """Bind kept FP8 routed-expert weights onto the expert Linears as plain CPU attrs.

    They are attached as plain attributes (not Parameters) so ``model.to(npu)`` leaves them
    on host; the streaming MoE forward dequantizes only the routed ones. Returns #attached.
    """
    mods = dict(model.named_modules())
    n = 0
    for key, (w_fp8, scale) in expert_fp8.items():
        lin = mods.get(key[: -len(".weight")])
        if lin is None:
            continue
        if hasattr(lin, "weight"):
            del lin.weight                              # drop the Parameter registration
        lin.weight = w_fp8                              # attach a plain FP8 tensor
        lin.weight_scale_inv = scale
        n += 1
    return n


def _recompute_rope_inv_freq(mod, device):
    """Recompute a rotary ``inv_freq`` buffer via the module's ``rope_init_fn`` (else False)."""
    if not hasattr(mod, "rope_init_fn") or not hasattr(mod, "config"):
        return False
    try:
        inv_freq, scaling = mod.rope_init_fn(mod.config, device)
    except (KeyError, AttributeError, TypeError, RuntimeError):
        return False
    mod.register_buffer("inv_freq", inv_freq.to(device), persistent=False)
    if hasattr(mod, "attention_scaling"):
        mod.attention_scaling = scaling
    return True


def materialize_meta_buffers(model, device):
    """Recompute non-persistent buffers left on meta after load_state_dict(assign=True).

    Rotary inv_freq is rebuilt via its rope_init_fn; every other meta buffer is zero-filled
    (e.g. the ``e_score_correction_bias`` is restored from the checkpoint, not here).
    """
    for mod in model.modules():
        for bn, b in list(mod.named_buffers(recurse=False)):
            if not b.is_meta:
                continue
            if bn == "inv_freq" and _recompute_rope_inv_freq(mod, device):
                continue
            mod.register_buffer(bn, torch.zeros(b.shape, dtype=b.dtype, device=device), persistent=False)


def load_model(model_path, device, use_pypto=True, streaming=True, max_layers=None):
    """Build + load MiniMax-M3 for inference/benchmark on a single die. Returns (model, cfg, n_patched).

    Sequence (mirrors minimax_m27): build on meta -> load non-expert (+ BF16 experts) via
    ``assign=True`` -> attach FP8 routed experts as host tensors -> recompute meta buffers
    (rotary ``inv_freq``) onto ``device`` -> move the rest to ``device`` (the host FP8 expert
    attrs are not Parameters, so ``.to`` leaves them on host) -> patch the MoE blocks.

    FP8 checkpoints (e.g. MiniMax-M3-MXFP8) MUST use ``streaming=True``: the routed experts stay
    FP8 on the host and only the per-layer active experts are dequantized to BF16 each forward, so
    one die holds ~one layer's experts instead of the full ~856 GB BF16. BF16 checkpoints fit the
    streaming or prebuilt path directly (use ``max_layers`` for a quick on-device smoke test).
    """
    if use_pypto:
        _pto_ops.USE_PTO_GROUPED_GEMM = True                 # enable the fused kernel for this run
    model, cfg = build_model(model_path, max_layers=max_layers)
    sd, expert_fp8 = load_streaming_state_dict(model_path, max_layers=max_layers)
    # FP8 routed experts are attached to the host (not Parameters); the prebuilt (non-streaming)
    # path can't see them, so require streaming for an FP8 checkpoint.
    if expert_fp8 and not streaming:
        raise ValueError("FP8 routed experts are host-resident; call load_model(..., streaming=True). "
                         "The prebuilt path cannot reach host-attached experts.")
    # A full BF16 checkpoint puts every routed expert on the die (no host streaming) -> OOM. Fail
    # early with a clear message instead of a mid-load NPU OOM.
    routed_bytes = sum(v.numel() * v.element_size() for k, v in sd.items()
                       if ".block_sparse_moe.experts." in k)
    _die_budget = 55 * 1024 ** 3
    if routed_bytes > _die_budget:
        raise RuntimeError(
            f"Routed experts = {routed_bytes / 1e9:.0f} GB of on-die BF16 weights "
            f"(> ~{_die_budget // 1024 ** 3} GB/die). Use an FP8 checkpoint (e.g. MiniMax-M3-MXFP8 / "
            f"the convert_to_fp8.py output, which streams experts from host) or pass max_layers for a smoke test.")
    model.load_state_dict(sd, strict=False, assign=True)     # missing routed-expert keys -> FP8 below
    attach_expert_fp8(model, expert_fp8)
    materialize_meta_buffers(model, device)                  # recompute meta buffers (rotary inv_freq) -> device
    model.to(device)                                         # FP8 expert attrs aren't Parameters -> stay on host
    model.eval()
    n_patched = patch_moe(model, streaming=streaming) if use_pypto else 0
    return model, cfg, n_patched
