# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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
#
"""PyTorch MiniMax-M2 model defined in-repo, with the MoE expert FFN on the PyPTO kernel."""

__all__ = [
    "MiniMaxM2ForCausalLM", "MiniMaxM2Model", "MiniMaxM2PreTrainedModel",
    "MiniMaxM2ForSequenceClassification", "MiniMaxM2ForTokenClassification",
    "MiniMaxM2ForQuestionAnswering", "MiniMaxM2SparseMoeBlock", "MiniMaxM2Experts",
    "MiniMaxM27PyptoExperts", "patch_minimax_m2_moe", "is_minimax_m2_moe", "_dequant_fp8_block",
    "build_model", "load_streaming_state_dict", "attach_expert_fp8", "materialize_meta_buffers",
    "patch_moe",
]

import gc
import json
import os
from collections.abc import Callable
from typing import Optional, Union
try:                          # Unpack lives in typing_extensions for Python < 3.11
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import OutputRecorder, check_model_inputs
from ops.pypto_tensor import minimax as _pto_ops          # read USE_PTO_* dynamically
from ops.pypto_tensor.minimax import grouped_gemm, MoeDims
from .configuration_minimax_m27 import MiniMaxM27Config as MiniMaxM2Config


class MiniMaxM2MLP(nn.Module):
    def __init__(self, config: MiniMaxM2Config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MiniMaxM2Experts(nn.ModuleList):
    """
    ModuleList of experts.
    """

    def __init__(self, config: MiniMaxM2Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        for _ in range(self.num_experts):
            self.append(MiniMaxM2MLP(config))

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size * sequence_length, hidden_dim)
            selected_experts: (batch_size * sequence_length, top_k)
            routing_weights: (batch_size * sequence_length, top_k)
        Returns:
            (batch_size * sequence_length, hidden_dim)
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            current_hidden_states = self[expert_idx](current_state) * top_k_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states


class MiniMaxM2SparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.jitter_noise = config.router_jitter_noise
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = MiniMaxM2Experts(config)
        self.register_buffer("e_score_correction_bias", torch.zeros(config.num_local_experts))

    def route_tokens_to_experts(self, router_logits):
        routing_weights = torch.nn.functional.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_index, top_k_weights.to(router_logits.dtype)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits = self.gate(hidden_states)
        top_k_index, top_k_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype))
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_logits


@use_kernel_forward_from_hub("RMSNorm")
class MiniMaxM2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MiniMaxM2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    **kwargs: Unpack[TransformersKwargs],
):
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
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, **kwargs):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    unsqueeze_dim = kwargs.get("unsqueeze_dim", 1)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class MiniMaxM2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MiniMaxM2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = MiniMaxM2RMSNorm(self.head_dim * config.num_attention_heads, eps=config.rms_norm_eps)
            self.k_norm = MiniMaxM2RMSNorm(self.head_dim * config.num_key_value_heads, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        cache_position = kwargs.pop("cache_position", None)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.use_qk_norm:  # main diff from Llama
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        key_states = key_states.view(hidden_shape)
        query_states = query_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        attn_impl = getattr(self.config, "_attn_implementation")
        if attn_impl != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MiniMaxM2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MiniMaxM2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MiniMaxM2Attention(config, layer_idx)

        self.block_sparse_moe = MiniMaxM2SparseMoeBlock(config)
        self.input_layernorm = MiniMaxM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniMaxM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        past_key_values = kwargs.pop("past_key_values", None)
        cache_position = kwargs.pop("cache_position", None)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MiniMaxM2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: MiniMaxM2Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class MiniMaxM2PreTrainedModel(PreTrainedModel):
    config: MiniMaxM2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxM2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(MiniMaxM2SparseMoeBlock, index=1),
        "hidden_states": MiniMaxM2DecoderLayer,
        "attentions": MiniMaxM2Attention,
    }


@auto_docstring
class MiniMaxM2Model(MiniMaxM2PreTrainedModel):
    def __init__(self, config: MiniMaxM2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MiniMaxM2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MiniMaxM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MiniMaxM2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
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
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


def _lbl_masked_means(attention_mask, expert_mask, routing_weights, top_k, num_experts):
    """Padding-aware (tokens_per_expert, router_prob_per_expert) for the masked branch.

    Extracted from ``load_balancing_loss_func`` (behavior-identical) to keep that
    function under the size limit. ``compute_device`` and the concatenated-logits row
    count are recovered from ``routing_weights`` (softmax preserves both).
    """
    compute_device = routing_weights.device
    batch_size, sequence_length = attention_mask.shape
    num_hidden_layers = routing_weights.shape[0] // (batch_size * sequence_length)

    # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
    expert_attention_mask = (
        attention_mask[None, :, :, None, None]
        .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
        .reshape(-1, top_k, num_experts)
        .to(compute_device)
    )

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
        expert_attention_mask, dim=0
    )

    # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
    router_per_expert_attention_mask = (
        attention_mask[None, :, :, None]
        .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
        .reshape(-1, num_experts)
        .to(compute_device)
    )

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
        router_per_expert_attention_mask, dim=0
    )
    return tokens_per_expert, router_prob_per_expert


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements
    the loss function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing
    between experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        tokens_per_expert, router_prob_per_expert = _lbl_masked_means(
            attention_mask, expert_mask, routing_weights, top_k, num_experts,
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


@auto_docstring
class MiniMaxM2ForCausalLM(MiniMaxM2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniMaxM2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MiniMaxM2ForCausalLM

        >>> model = MiniMaxM2ForCausalLM.from_pretrained("mistralai/MiniMaxM2-8x7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/MiniMaxM2-8x7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        labels = kwargs.pop("labels", None)
        logits_to_keep = kwargs.pop("logits_to_keep", 0)
        output_router_logits = kwargs.pop("output_router_logits", None)
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # past_key_values / use_cache / cache_position flow through **kwargs to self.model().
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        aux_router_logits = outputs.router_logits if output_router_logits else None
        loss, aux_loss = self._compute_losses(
            logits, labels, aux_router_logits, attention_mask, **kwargs
        )

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def _compute_losses(self, logits, labels, router_logits, attention_mask, **kwargs):
        """Loss path (training only): masked-LM loss + optional router aux loss.

        Returns (loss, aux_loss). ``router_logits`` is the captured router output when the
        aux loss is requested, else ``None`` (the caller passes ``None`` when
        ``output_router_logits`` is False) — so the aux term is gated exactly as before.
        When ``labels is None`` and ``router_logits is None`` both are None, so the
        generation/inference path is unaffected.
        """
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if router_logits is not None:
            aux_loss = load_balancing_loss_func(
                router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # same device
        return loss, aux_loss


class MiniMaxM2ForSequenceClassification(GenericForSequenceClassification, MiniMaxM2PreTrainedModel):
    pass


class MiniMaxM2ForTokenClassification(GenericForTokenClassification, MiniMaxM2PreTrainedModel):
    pass


class MiniMaxM2ForQuestionAnswering(GenericForQuestionAnswering, MiniMaxM2PreTrainedModel):
    pass


# ============================ PyPTO patch layer ============================
_FP8_BLOCK = 128

# Streaming-path expert-count buckets. The grouped-GEMM kernel is JIT-compiled per
# distinct ``num_experts``; if we called it with the raw active-expert count it would
# recompile on almost every forward. Rounding the active count UP to a fixed bucket
# (padding the extra slots with 0-token "dummy" experts the kernel skips) collapses the
# compile set to len(buckets) shapes. Default {32,128,256}; override via PYPTO_MOE_BUCKETS.
_MOE_BUCKETS = sorted(int(x) for x in os.environ.get("PYPTO_MOE_BUCKETS", "32,128,256").split(","))


def _bucket_for(n_active: int, num_experts: int) -> int:
    """Smallest configured bucket >= n_active, never exceeding the real expert count."""
    for b in _MOE_BUCKETS:
        if n_active <= b and b <= num_experts:
            return b
    return num_experts


# Module-global streaming weight buffers, shared across all MoE layers (which run
# sequentially) so total buffer memory is ~one layer's worth, not 62×. Keyed by
# (bucket, I, H, device). See MiniMaxM27PyptoExperts.get_stream_buffers.
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


class MiniMaxM27PyptoExperts:
    """Replacement ``forward`` for ``MiniMaxM2Experts`` using fused grouped GEMM.

    Bound onto an existing ``MiniMaxM2Experts`` instance (so ``self`` is the
    ModuleList of experts). Mirrors the original signature:
        forward(hidden_states, top_k_index, top_k_weights) -> [N, H]
    """

    def __init__(self):
        # The methods below are bound onto a live MiniMaxM2Experts instance by
        # patch_minimax_m2_moe (self is that ModuleList), so this constructor is not
        # invoked at runtime; it declares the instance attributes those methods set.
        self.pypto_streaming = False
        self.pypto_ready = False
        self.pypto_num_experts = 0
        self.pypto_intermediate = 0
        self.pypto_hidden = 0
        self.pypto_w13_flat = None
        self.pypto_w2_flat = None

    @torch.no_grad()
    def ensure_pypto_weights(self):
        """Prepare expert weights for the kernel.

        Two modes (set by ``patch_minimax_m2_moe(..., streaming=...)``):

        * **prebuilt** (default): build the full ``[E*H, 2I]`` / ``[E*I, H]`` flat
          caches once, freeing each expert's original Linear weight as we go so the
          block's MoE memory stays ~1x. Best when the dequantized experts fit memory.
        * **streaming**: keep the expert weights as-is (FP8, typically CPU-resident)
          and dequantize **only the routed experts** per forward (see
          ``grouped_ffn``). HBM peak is ~one layer's *active* experts, not the whole
          model — this is what makes a single-die full-model run feasible (449 GB of
          all-BF16 experts never materializes at once).
        """
        if getattr(self, "pypto_ready", False):
            return
        experts = list(self)
        num_experts = len(experts)
        e0 = experts[0]
        intermediate_size, hidden_size = e0.w1.weight.shape          # w1: [I, H]
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
            w1 = _expert_weight(exp.w1)
            w3 = _expert_weight(exp.w3)
            w2 = _expert_weight(exp.w2)
            w13_flat[e * hidden_size:(e + 1) * hidden_size] = torch.cat([w1, w3], dim=0).t().contiguous()  # [H,2I]
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
        """Dequantize only the routed experts into the first active expert slots of the buffers.

        FP8->BF16 on the host (910B has no native FP8); remaining slots keep stale data but
        carry 0 routed tokens, so the kernel skips them. Dequant cost tracks the true active
        count, not the bucket size.
        """
        experts = list(self)
        intermediate_size, hidden_size = self.pypto_intermediate, self.pypto_hidden
        for ci, e in enumerate(active_ids.tolist()):
            exp = experts[e]
            w1 = _expert_weight(exp.w1).to(w13c.device)
            w3 = _expert_weight(exp.w3).to(w13c.device)
            w2 = _expert_weight(exp.w2).to(w13c.device)
            w13c[ci * hidden_size:(ci + 1) * hidden_size] = torch.cat([w1, w3], dim=0).t().contiguous()  # [H,2I]
            w2c[ci * intermediate_size:(ci + 1) * intermediate_size] = w2.t().contiguous()  # [I, H]

    def get_stream_buffers(self, bucket, dev):
        """Get one persistent (w13, w2) BF16 buffer pair per (bucket, shape, device).

        Shared across **all** layers via a module-global cache: layers run sequentially, so a
        single set of buffers is reused by every layer/step — keeping total weight-buffer
        memory at ~one layer's worth (a few GB) instead of 62×. The kernel sees the same
        tensor objects, so it doesn't accumulate inputs.
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
        pad the count up to a fixed bucket so the kernel compiles a bounded set of shapes)
        so the kernel and the dequant only touch experts that actually received tokens.
        """
        num_tokens, hidden_size = sorted_x.shape
        num_experts, intermediate_size = self.pypto_num_experts, self.pypto_intermediate
        dev = sorted_x.device
        out = torch.empty_like(sorted_x)

        if getattr(self, "pypto_streaming", False):
            # active experts (ascending) — sorted_x blocks already follow this order,
            # so a compact cumsum over active counts partitions it with no remap.
            active = counts.nonzero(as_tuple=False).flatten()
            n_active = int(active.numel())
            if n_active == 0:
                return out
            bucket = _bucket_for(n_active, num_experts)   # 32 / 128 / 256 (fixed compile set)
            # Reuse one persistent flat buffer per bucket and overwrite the active experts
            # in place each call. The grouped-GEMM kernel retains a reference to its weight
            # inputs per call, so handing it a fresh tensor every layer/step would leak (OOM
            # across 62 layers). Same buffer object -> bounded memory, like the prebuilt path.
            w13c, w2c = self.get_stream_buffers(bucket, dev)
            self.fill_active_flat(active, w13c, w2c)
            active_counts = counts[active]
            # cumsum length bucket+1; padded experts get an empty [last,last] range -> skipped
            cumsum = torch.zeros(bucket + 1, dtype=torch.int32, device=dev)
            real = torch.cumsum(active_counts, 0).to(torch.int32)
            cumsum[1:n_active + 1] = real
            cumsum[n_active + 1:] = real[-1]
            grouped_gemm(sorted_x, (w13c, w2c), cumsum, out,
                         MoeDims(bucket, hidden_size, intermediate_size))
            return out

        cumsum = torch.zeros(num_experts + 1, dtype=torch.int32, device=dev)
        cumsum[1:] = torch.cumsum(counts, 0).to(torch.int32)
        grouped_gemm(sorted_x, (self.pypto_w13_flat, self.pypto_w2_flat), cumsum, out,
                     MoeDims(num_experts, hidden_size, intermediate_size))
        return out

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

        out_sorted = self.grouped_ffn(sorted_x, counts)

        inv = torch.empty_like(sort_perm)
        inv[sort_perm] = torch.arange(num_tokens * top_k, device=dev)
        return (
            out_sorted[inv].view(num_tokens, top_k, hidden_size)
            .type(top_k_weights.dtype)
            .mul_(top_k_weights.reshape(num_tokens, top_k).unsqueeze(-1))
            .sum(dim=1)
            .type(hidden_states.dtype)
        )


def patch_minimax_m2_moe(module, streaming: bool = False) -> bool:
    """Route a HF MiniMax M2 MoE block through the PyPTO grouped GEMM kernel.

    Accepts either a ``MiniMaxM2SparseMoeBlock`` (has ``.experts``) or a
    ``MiniMaxM2Experts`` ModuleList directly. Replaces the experts' ``forward``.
    Returns True if patched, False if the kernel switch is off.

    streaming : if True, keep expert weights as-is (FP8, usually CPU-resident) and
        dequantize only the routed experts per forward — HBM peak is ~one layer's
        active experts, so the full 62-layer model runs on a single die without
        materializing all-BF16 experts (449 GB). Default False (prebuilt flats).
    """
    if not getattr(_pto_ops, "USE_PTO_GROUPED_GEMM", False):     # opt-in switch; callers enable it
        return False
    experts = getattr(module, "experts", module)
    if not (hasattr(experts, "__len__") and len(experts) and hasattr(experts[0], "w1")):
        return False
    # Bind the CLASS functions onto the experts ModuleList so `self` IS the ModuleList
    # (so `list(self)` / self[e] iterate the real experts). Binding an adapter *instance*
    # method would keep self=adapter and break iteration.
    experts.pypto_streaming = bool(streaming)
    experts.pypto_ready = False
    experts.ensure_pypto_weights = MiniMaxM27PyptoExperts.ensure_pypto_weights.__get__(experts)
    experts.get_stream_buffers = MiniMaxM27PyptoExperts.get_stream_buffers.__get__(experts)
    experts.fill_active_flat = MiniMaxM27PyptoExperts.fill_active_flat.__get__(experts)
    experts.grouped_ffn = MiniMaxM27PyptoExperts.grouped_ffn.__get__(experts)
    experts.forward = MiniMaxM27PyptoExperts.forward.__get__(experts)
    return True


def is_minimax_m2_moe(module) -> bool:
    """Duck-type check for a MiniMax M2 sparse-MoE block (has gate + experts of w1/w2/w3)."""
    experts = getattr(module, "experts", None)
    return (
        experts is not None
        and hasattr(module, "gate")
        and hasattr(experts, "__len__") and len(experts)
        and hasattr(experts[0], "w1") and hasattr(experts[0], "w2") and hasattr(experts[0], "w3")
    )


# ============================ checkpoint loading (FP8 -> BF16) ============================
# Kept here so the inference / benchmark entry scripts stay thin and import these directly.

def build_model(model_path, max_layers=None):
    """Build MiniMaxM2ForCausalLM on meta from the in-repo config (no trust_remote_code)."""
    with open(f"{model_path}/config.json") as cfg_file:
        cj = json.load(cfg_file)
    drop = ("architectures", "auto_map", "quantization_config", "_pre_quantization_dtype",
            "dtype", "torch_dtype", "transformers_version", "model_type")
    cfg = MiniMaxM2Config(**{k: v for k, v in cj.items() if k not in drop})
    setattr(cfg, "_attn_implementation", "eager")               # eager attention (no flash_attn on NPU)
    if max_layers is not None:
        cfg.num_hidden_layers = max_layers
    with torch.device("meta"):
        model = MiniMaxM2ForCausalLM(cfg)
    return model, cfg


def load_streaming_state_dict(model_path, max_layers=None):
    """Load the checkpoint for the streaming path.

    Non-expert weights go to a BF16 CPU state_dict; expert weights are kept FP8 on CPU
    as {name: (fp8, scale_inv)} for just-in-time dequant.

    Returns (state_dict, expert_fp8).
    """
    from safetensors import safe_open       # lazy: optional 3rd-party dep, not at module load
    with open(f"{model_path}/model.safetensors.index.json") as idx_file:
        idx = json.load(idx_file)["weight_map"]

    def keep(name):
        if max_layers is None or ".layers." not in name:
            return True
        return int(name.split(".layers.")[1].split(".")[0]) < max_layers

    def is_expert_w(name):
        return ".experts." in name and name.endswith(".weight")

    by_shard = {}
    for name, shard in idx.items():
        if name.endswith(".weight_scale_inv"):
            continue
        if keep(name):
            by_shard.setdefault(shard, []).append(name)

    sd, expert_fp8 = {}, {}
    for shard, names in sorted(by_shard.items()):
        f = safe_open(f"{model_path}/{shard}", "pt")
        scaled = set(f.keys())
        for name in names:
            t = f.get_tensor(name)
            sname = name + "_scale_inv"
            if is_expert_w(name) and sname in scaled:
                expert_fp8[name] = (t, f.get_tensor(sname))            # keep FP8 on CPU
            elif t.dtype == torch.float8_e4m3fn and sname in scaled:
                sd[name] = _dequant_fp8_block(t, f.get_tensor(sname))   # non-expert fp8 -> bf16
            else:
                sd[name] = t.to(torch.bfloat16) if t.is_floating_point() else t
        del f
        gc.collect()
    return sd, expert_fp8


def attach_expert_fp8(model, expert_fp8):
    """Bind kept FP8 expert weights onto the expert Linears as plain CPU attrs.

    They are attached as plain attributes (not Parameters) so model.to(npu) leaves them on
    host; the streaming MoE forward dequantizes only the routed ones.
    """
    mods = dict(model.named_modules())
    n = 0
    for wname, (w_fp8, scale) in expert_fp8.items():
        lin = mods.get(wname[: -len(".weight")])
        if lin is None:
            continue
        if hasattr(lin, "weight"):
            del lin.weight                              # drop the Parameter registration
        lin.weight = w_fp8                              # so a plain FP8 tensor can be attached
        lin.weight_scale_inv = scale
        n += 1
    return n


def _recompute_rope_inv_freq(mod, device):
    """Try to recompute a rotary ``inv_freq`` buffer via the module's ``rope_init_fn``.

    Returns True and writes the buffer (plus ``attention_scaling``) on success; returns
    False if the module has no usable rope init or it raised, so the caller zero-fills.
    """
    if not hasattr(mod, "rope_init_fn") or not hasattr(mod, "config"):
        return False
    try:
        inv_freq, scaling = mod.rope_init_fn(mod.config, device)
    except (KeyError, AttributeError, TypeError, RuntimeError):
        # rope_init_fn unavailable/incompatible for this buffer; the caller falls back to
        # the zero-fill (buffer is recomputed on first use).
        return False
    mod.register_buffer("inv_freq", inv_freq.to(device), persistent=False)
    if hasattr(mod, "attention_scaling"):
        mod.attention_scaling = scaling
    return True


def materialize_meta_buffers(model, device):
    """Recompute non-persistent buffers left on meta after load_state_dict(assign=True).

    Rotary inv_freq is rebuilt via its rope_init_fn; every other meta buffer is zero-filled.
    """
    for mod in model.modules():
        for bn, b in list(mod.named_buffers(recurse=False)):
            if not b.is_meta:
                continue
            if bn == "inv_freq" and _recompute_rope_inv_freq(mod, device):
                continue
            mod.register_buffer(bn, torch.zeros(b.shape, dtype=b.dtype, device=device), persistent=False)


def patch_moe(model, streaming=False):
    """Route every MoE block's expert FFN through the PyPTO kernel.

    streaming=True keeps experts FP8 on host (dequant per forward); else patch only blocks
    whose experts live on a die.
    """
    n = 0
    for m in model.modules():
        if not is_minimax_m2_moe(m):
            continue
        if streaming:
            n += int(bool(patch_minimax_m2_moe(m, streaming=True)))
            continue
        try:
            dev = next(m.experts[0].parameters()).device
        except StopIteration:
            dev = None
        if dev is not None and dev.type == "npu":
            n += int(bool(patch_minimax_m2_moe(m)))
    return n
