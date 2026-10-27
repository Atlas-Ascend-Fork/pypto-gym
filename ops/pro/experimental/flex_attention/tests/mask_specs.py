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
"""MaskSpec data classes and built-in attention mask pattern definitions.

Replicates maskgen.core.spec + maskgen.patterns without depending on the
maskgen package. Each pattern is defined in two equivalent forms:
  - a MaskSpec (for _build_spec_mask_info in test_flex_attention.py)
  - a single-function form (Sym DSL, provided for reference)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

Scalars = dict[str, Any]


@dataclass(frozen=True)
class TensorArg:
    name: str
    shape: tuple[str, ...]
    dtype: str = "int32"
    monotonic: bool = False
    nonnegative: bool = True


@dataclass(frozen=True)
class RegionMask:
    name: str
    when: Callable[..., bool]
    visible: Callable[..., bool]


@dataclass(frozen=True)
class MaskSpec:
    tensor_args: list[TensorArg]
    scalar_setup: Callable[..., Scalars]
    regions: list[RegionMask]
    cur_seqlen: str


# ================================================================
#  Common patterns
# ================================================================

def _length_setup(batch_idx: int, seq_lens, **kwargs) -> Scalars:
    return {"seq_hi": int(seq_lens[batch_idx])}


def _when_active(row_idx: int, seq_hi: int, **kwargs) -> bool:
    return row_idx < seq_hi


def _visible_full(row_idx: int, col_idx: int, **kwargs) -> bool:
    return True


def _visible_causal(row_idx: int, col_idx: int, **kwargs) -> bool:
    return col_idx <= row_idx


FULL_SPEC = MaskSpec(
    tensor_args=[TensorArg("seq_lens", ("batch",))],
    scalar_setup=_length_setup,
    regions=[RegionMask(name="full", when=_when_active, visible=_visible_full)],
    cur_seqlen="seq_hi",
)

CAUSAL_SPEC = MaskSpec(
    tensor_args=[TensorArg("seq_lens", ("batch",))],
    scalar_setup=_length_setup,
    regions=[RegionMask(name="causal", when=_when_active, visible=_visible_causal)],
    cur_seqlen="seq_hi",
)


def _prefix_setup(batch_idx: int, seq_lens, prefix_lengths, **kwargs) -> Scalars:
    return {"seq_hi": int(seq_lens[batch_idx]),
            "prefix_hi": int(prefix_lengths[batch_idx])}


def _visible_prefix_lm(row_idx: int, col_idx: int, prefix_hi: int, **kwargs) -> bool:
    return (col_idx < prefix_hi) or (col_idx <= row_idx)


PREFIX_LM_SPEC = MaskSpec(
    tensor_args=[TensorArg("seq_lens", ("batch",)),
                 TensorArg("prefix_lengths", ("batch",))],
    scalar_setup=_prefix_setup,
    regions=[RegionMask(name="prefix-lm", when=_when_active, visible=_visible_prefix_lm)],
    cur_seqlen="seq_hi",
)


def _window_setup(batch_idx: int, seq_lens, window_sizes, **kwargs) -> Scalars:
    return {"seq_hi": int(seq_lens[batch_idx]),
            "window_size": int(window_sizes[batch_idx])}


def _visible_sliding_window(row_idx: int, col_idx: int, window_size: int, **kwargs) -> bool:
    return (row_idx - window_size < col_idx) and (col_idx <= row_idx)


SLIDING_WINDOW_SPEC = MaskSpec(
    tensor_args=[TensorArg("seq_lens", ("batch",)),
                 TensorArg("window_sizes", ("batch",))],
    scalar_setup=_window_setup,
    regions=[RegionMask(name="sliding-window", when=_when_active, visible=_visible_sliding_window)],
    cur_seqlen="seq_hi",
)


def _visible_composed(row_idx: int, col_idx: int, **kwargs) -> bool:
    return (col_idx < 2) or (3 <= col_idx < 5) or (6 <= col_idx <= row_idx)


COMPOSED_SPEC = MaskSpec(
    tensor_args=[TensorArg("seq_lens", ("batch",))],
    scalar_setup=_length_setup,
    regions=[RegionMask(name="composed", when=_when_active, visible=_visible_composed)],
    cur_seqlen="seq_hi",
)


# ================================================================
#  Bidirectional sliding window
# ================================================================

def _when_valid(row_idx: int, seq_hi: int, **kwargs) -> bool:
    return row_idx < seq_hi


def _visible_bidirectional_sw(row_idx: int, col_idx: int, window_size: int, **kwargs) -> bool:
    return (row_idx - window_size <= col_idx) and (col_idx <= row_idx + window_size)


BIDIRECTIONAL_SLIDING_WINDOW_SPEC = MaskSpec(
    tensor_args=[TensorArg("seq_lens", ("batch",)),
                 TensorArg("window_sizes", ("batch",))],
    scalar_setup=_window_setup,
    regions=[RegionMask(name="bidirectional-sw", when=_when_valid, visible=_visible_bidirectional_sw)],
    cur_seqlen="seq_hi",
)


# ================================================================
#  SPU (Context / Label)
# ================================================================

def _spu_setup(batch_idx: int, context_length, labelgroup_size, cu_seqlens, **kwargs) -> Scalars:
    context_hi = int(context_length[batch_idx])
    labelgroup = int(labelgroup_size[batch_idx])
    seq_hi = int(cu_seqlens[batch_idx + 1]) - int(cu_seqlens[batch_idx])
    return {"context_hi": context_hi, "labelgroup": labelgroup, "seq_hi": seq_hi}


def _when_context_fwd(row_idx: int, context_hi: int, **kwargs) -> bool:
    return row_idx < context_hi


def _when_label_fwd(row_idx: int, context_hi: int, seq_hi: int, **kwargs) -> bool:
    return row_idx >= context_hi and row_idx < seq_hi


def _visible_context_fwd(row_idx: int, col_idx: int, context_hi: int, **kwargs) -> bool:
    return col_idx < context_hi


def _visible_label_fwd(row_idx: int, col_idx: int, context_hi: int, labelgroup: int, **kwargs) -> bool:
    label_idx = row_idx - context_hi
    if labelgroup <= 1:
        return (col_idx < context_hi) or (col_idx <= row_idx)
    group_start = ((label_idx // labelgroup) * labelgroup
                   + (labelgroup - 1)
                   * ((label_idx - (label_idx // labelgroup) * labelgroup) // (labelgroup - 1))
                   + context_hi)
    return (col_idx < context_hi) or (group_start <= col_idx and col_idx < row_idx + 1)


SPU_FWD_SPEC = MaskSpec(
    tensor_args=[TensorArg("context_length", ("batch",), nonnegative=True),
                 TensorArg("labelgroup_size", ("batch",), nonnegative=True),
                 TensorArg("cu_seqlens", ("batch+1",), nonnegative=True)],
    scalar_setup=_spu_setup,
    regions=[RegionMask(name="context", when=_when_context_fwd, visible=_visible_context_fwd),
             RegionMask(name="label", when=_when_label_fwd, visible=_visible_label_fwd)],
    cur_seqlen="seq_hi",
)


# ================================================================
#  Four-stage (Context / History+Realtime / Label)
# ================================================================

def _four_stage_setup(batch_idx: int, context_sizes, history_sizes, realtime_sizes,
                      label_nums, label_group_sizes, **kwargs) -> Scalars:
    context_hi = int(context_sizes[batch_idx])
    history_hi = context_hi + int(history_sizes[batch_idx]) + int(realtime_sizes[batch_idx])
    label_group_size = int(label_group_sizes[batch_idx])
    seq_hi = history_hi + int(label_nums[batch_idx]) * label_group_size
    return {"context_hi": context_hi, "history_hi": history_hi,
            "seq_hi": seq_hi, "label_group_size": label_group_size}


def _when_context(row_idx: int, context_hi: int, **kwargs) -> bool:
    return row_idx < context_hi


def _when_history_or_realtime(row_idx: int, context_hi: int, history_hi: int, **kwargs) -> bool:
    return row_idx >= context_hi and row_idx < history_hi


def _when_label(row_idx: int, history_hi: int, seq_hi: int, label_group_size: int, **kwargs) -> bool:
    return label_group_size > 0 and row_idx >= history_hi and row_idx < seq_hi


def _visible_context(row_idx: int, col_idx: int, context_hi: int, history_hi: int, **kwargs) -> bool:
    return (col_idx < context_hi) or (col_idx >= context_hi and col_idx < history_hi)


def _visible_history_or_realtime(row_idx: int, col_idx: int, context_hi: int, **kwargs) -> bool:
    return (col_idx < context_hi) or (col_idx >= context_hi and col_idx <= row_idx)


def _visible_label(row_idx: int, col_idx: int, history_hi: int, label_group_size: int, **kwargs) -> bool:
    group_start = ((row_idx - history_hi) // label_group_size) * label_group_size + history_hi
    return (col_idx < history_hi) or (col_idx >= group_start and col_idx <= row_idx)


FOUR_STAGE_SPEC = MaskSpec(
    tensor_args=[TensorArg("context_sizes", ("batch",), nonnegative=True),
                 TensorArg("history_sizes", ("batch",), nonnegative=True),
                 TensorArg("realtime_sizes", ("batch",), nonnegative=True),
                 TensorArg("label_nums", ("batch",), nonnegative=True),
                 TensorArg("label_group_sizes", ("batch",), nonnegative=True)],
    scalar_setup=_four_stage_setup,
    regions=[RegionMask(name="context", when=_when_context, visible=_visible_context),
             RegionMask(name="history-realtime", when=_when_history_or_realtime, visible=_visible_history_or_realtime),
             RegionMask(name="label", when=_when_label, visible=_visible_label)],
    cur_seqlen="seq_hi",
)


# ================================================================
#  HSTU (Context / History / Label)
# ================================================================

def _hstu_setup(batch_idx: int, context_length, history_length, label_group_size, cu_seqlen, **kwargs) -> Scalars:
    context_hi = int(context_length[batch_idx])
    history_hi = context_hi + int(history_length[batch_idx])
    seq_hi = int(cu_seqlen[batch_idx + 1]) - int(cu_seqlen[batch_idx])
    return {"context_hi": context_hi, "history_hi": history_hi,
            "seq_hi": seq_hi, "label_group": int(label_group_size[batch_idx])}


def _when_hstu_context(row_idx: int, context_hi: int, **kwargs) -> bool:
    return row_idx < context_hi


def _when_hstu_history(row_idx: int, context_hi: int, history_hi: int, **kwargs) -> bool:
    return context_hi <= row_idx < history_hi


def _when_hstu_label(row_idx: int, history_hi: int, seq_hi: int, **kwargs) -> bool:
    return history_hi <= row_idx < seq_hi


def _visible_hstu_context(row_idx: int, col_idx: int, history_hi: int, **kwargs) -> bool:
    return col_idx < history_hi


def _visible_hstu_history(row_idx: int, col_idx: int, **kwargs) -> bool:
    return col_idx <= row_idx


def _visible_hstu_label(row_idx: int, col_idx: int, history_hi: int, label_group: int, **kwargs) -> bool:
    group_start = ((row_idx - history_hi) // label_group) * label_group + history_hi
    return (col_idx < history_hi) or (group_start <= col_idx <= row_idx)


HSTU_FWD_SPEC = MaskSpec(
    tensor_args=[TensorArg("context_length", ("batch",), nonnegative=True),
                 TensorArg("history_length", ("batch",), nonnegative=True),
                 TensorArg("label_group_size", ("batch",), nonnegative=True),
                 TensorArg("cu_seqlen", ("batch+1",), monotonic=True, nonnegative=True)],
    scalar_setup=_hstu_setup,
    regions=[RegionMask(name="context", when=_when_hstu_context, visible=_visible_hstu_context),
             RegionMask(name="history", when=_when_hstu_history, visible=_visible_hstu_history),
             RegionMask(name="label", when=_when_hstu_label, visible=_visible_hstu_label)],
    cur_seqlen="seq_hi",
)


# ================================================================
#  Batchified (group-wise causal / full)
# ================================================================

def _batchified_setup(batch_idx: int, seq_lens, group_sizes, **kwargs) -> Scalars:
    return {"seq_hi": int(seq_lens[batch_idx]), "group_size": int(group_sizes[batch_idx])}


def _when_batchified(row_idx: int, seq_hi: int, group_size: int, **kwargs) -> bool:
    return seq_hi >= 0 and group_size > 0 and row_idx < seq_hi


def _visible_batchified_full(row_idx: int, col_idx: int, group_size: int, seq_hi: int, **kwargs) -> bool:
    group_start = (row_idx // group_size) * group_size
    group_end = min(group_start + group_size, seq_hi)
    return group_start <= col_idx < group_end


def _visible_batchified_causal(row_idx: int, col_idx: int, group_size: int, **kwargs) -> bool:
    group_start = (row_idx // group_size) * group_size
    return group_start <= col_idx <= row_idx


BATCHIFIED_FULL_SPEC = MaskSpec(
    tensor_args=[TensorArg("seq_lens", ("batch",)),
                 TensorArg("group_sizes", ("batch",))],
    scalar_setup=_batchified_setup,
    regions=[RegionMask(name="batchified-full", when=_when_batchified, visible=_visible_batchified_full)],
    cur_seqlen="seq_hi",
)

BATCHIFIED_CAUSAL_SPEC = MaskSpec(
    tensor_args=[TensorArg("seq_lens", ("batch",)),
                 TensorArg("group_sizes", ("batch",))],
    scalar_setup=_batchified_setup,
    regions=[RegionMask(name="batchified-causal", when=_when_batchified, visible=_visible_batchified_causal)],
    cur_seqlen="seq_hi",
)


# ================================================================
#  Block diffusion (noised / clean)
# ================================================================

def _block_diffusion_setup(batch_idx: int, base_seq_lens, block_sizes, **kwargs) -> Scalars:
    base_seq_len = int(base_seq_lens[batch_idx])
    return {"base_seq_len": base_seq_len, "block_size": int(block_sizes[batch_idx]),
            "seq_hi": base_seq_len * 2}


def _when_noised(row_idx: int, base_seq_len: int, block_size: int, **kwargs) -> bool:
    return base_seq_len > 0 and block_size > 0 and row_idx < base_seq_len


def _when_clean(row_idx: int, base_seq_len: int, block_size: int, **kwargs) -> bool:
    return base_seq_len > 0 and block_size > 0 and row_idx >= base_seq_len and row_idx < base_seq_len * 2


def _visible_noised(row_idx: int, col_idx: int, base_seq_len: int, block_size: int, **kwargs) -> bool:
    block_start = ((row_idx % base_seq_len) // block_size) * block_size
    noised_end = min(block_start + block_size, base_seq_len)
    clean_end = min(base_seq_len + block_start, base_seq_len * 2)
    return (block_start <= col_idx < noised_end) or (base_seq_len <= col_idx < clean_end)


def _visible_clean(row_idx: int, col_idx: int, base_seq_len: int, block_size: int, **kwargs) -> bool:
    block_start = ((row_idx - base_seq_len) // block_size) * block_size
    clean_end = min(base_seq_len + block_start + block_size, base_seq_len * 2)
    return base_seq_len <= col_idx < clean_end


BLOCK_DIFFUSION_SPEC = MaskSpec(
    tensor_args=[TensorArg("base_seq_lens", ("batch",)),
                 TensorArg("block_sizes", ("batch",))],
    scalar_setup=_block_diffusion_setup,
    regions=[RegionMask(name="block-diffusion-noised", when=_when_noised, visible=_visible_noised),
             RegionMask(name="block-diffusion-clean", when=_when_clean, visible=_visible_clean)],
    cur_seqlen="seq_hi",
)


# ================================================================
#  JETSPEC training (prefix + sampled)
# ================================================================

def _jetspec_setup(batch_idx: int, prefix_lengths, sampled_lengths, block_sizes, **kwargs) -> Scalars:
    prefix_length = int(prefix_lengths[batch_idx])
    sampled_length = int(sampled_lengths[batch_idx])
    return {"prefix_length": prefix_length, "sampled_length": sampled_length,
            "block_size": int(block_sizes[batch_idx]),
            "seq_hi": prefix_length + sampled_length}


def _when_jetspec(row_idx: int, prefix_length: int, sampled_length: int, block_size: int, **kwargs) -> bool:
    return prefix_length >= 0 and sampled_length >= 0 and block_size > 0 and row_idx < sampled_length


def _visible_jetspec(row_idx: int, col_idx: int, prefix_length: int, sampled_length: int,
                     block_size: int, **kwargs) -> bool:
    q_block = row_idx // block_size
    q_offset = row_idx % block_size
    sampled_start = prefix_length + q_block * block_size
    sampled_end = min(prefix_length + sampled_length, sampled_start + q_offset + 1)
    return (col_idx < prefix_length) or (sampled_start <= col_idx < sampled_end)


JETSPEC_TRAINING_SPEC = MaskSpec(
    tensor_args=[TensorArg("prefix_lengths", ("batch",)),
                 TensorArg("sampled_lengths", ("batch",)),
                 TensorArg("block_sizes", ("batch",))],
    scalar_setup=_jetspec_setup,
    regions=[RegionMask(name="jetspec-training", when=_when_jetspec, visible=_visible_jetspec)],
    cur_seqlen="seq_hi",
)


# ================================================================
#  SVG spatial (prompt + spatial window)
# ================================================================

def _svg_setup(batch_idx: int, seq_lens, prompt_lengths, token_per_frames, frame_widths,
               round_widths, attn_sink_flags, **kwargs) -> Scalars:
    seq_hi = int(seq_lens[batch_idx])
    prompt_length = int(prompt_lengths[batch_idx])
    token_per_frame = int(token_per_frames[batch_idx])
    frame_width = int(frame_widths[batch_idx])
    round_width = int(round_widths[batch_idx])
    attn_sink_flag = int(attn_sink_flags[batch_idx])
    mask_width = (frame_width * token_per_frame // round_width) * round_width if token_per_frame >= 0 and frame_width >= 0 and round_width > 0 else 0
    first_column_end = prompt_length + token_per_frame if attn_sink_flag == 1 else prompt_length
    return {"seq_hi": seq_hi, "prompt_length": prompt_length,
            "first_column_end": first_column_end, "mask_width": mask_width,
            "attn_sink_flag": attn_sink_flag}


def _when_svg_prompt(row_idx: int, seq_hi: int, prompt_length: int, **kwargs) -> bool:
    return row_idx < seq_hi and row_idx < prompt_length


def _when_svg_spatial(row_idx: int, seq_hi: int, prompt_length: int, **kwargs) -> bool:
    return row_idx >= prompt_length and row_idx < seq_hi


def _visible_svg_prompt(row_idx: int, col_idx: int, **kwargs) -> bool:
    return True


def _visible_svg_spatial(row_idx: int, col_idx: int, first_column_end: int, mask_width: int, **kwargs) -> bool:
    return (col_idx < first_column_end) or (row_idx - mask_width < col_idx < row_idx + mask_width)


SVG_SPATIAL_SPEC = MaskSpec(
    tensor_args=[TensorArg("seq_lens", ("batch",)),
                 TensorArg("prompt_lengths", ("batch",)),
                 TensorArg("token_per_frames", ("batch",)),
                 TensorArg("frame_widths", ("batch",)),
                 TensorArg("round_widths", ("batch",)),
                 TensorArg("attn_sink_flags", ("batch",))],
    scalar_setup=_svg_setup,
    regions=[RegionMask(name="svg-prompt", when=_when_svg_prompt, visible=_visible_svg_prompt),
             RegionMask(name="svg-spatial", when=_when_svg_spatial, visible=_visible_svg_spatial)],
    cur_seqlen="seq_hi",
)


# ================================================================
#  REALTIME_VAR — 4-stage with variable realtime offset window
# ================================================================

def _realtime_var_setup(batch_idx, context_sizes, history_sizes, realtime_sizes,
                        label_nums, label_group_sizes, realtime_offsets, **kwargs):
    context_hi = int(context_sizes[batch_idx])
    history_hi = context_hi + int(history_sizes[batch_idx])
    realtime_hi = history_hi + int(realtime_sizes[batch_idx])
    label_group_size = int(label_group_sizes[batch_idx])
    seq_hi = realtime_hi + int(label_nums[batch_idx]) * label_group_size
    return {"context_hi": context_hi, "history_hi": history_hi,
            "realtime_hi": realtime_hi, "seq_hi": seq_hi,
            "label_group_size": label_group_size,
            "realtime_offset": int(realtime_offsets[batch_idx])}


def _when_rt_context(row_idx, context_hi, **kwargs):
    return row_idx < context_hi


def _when_rt_history(row_idx, context_hi, history_hi, **kwargs):
    return context_hi <= row_idx < history_hi


def _when_rt_realtime(row_idx, history_hi, realtime_hi, **kwargs):
    return history_hi <= row_idx < realtime_hi


def _when_rt_label(row_idx, realtime_hi, seq_hi, **kwargs):
    return realtime_hi <= row_idx < seq_hi


def _visible_rt_context(row_idx, col_idx, history_hi, **kwargs):
    return col_idx < history_hi


def _visible_rt_history(row_idx, col_idx, **kwargs):
    return col_idx <= row_idx


def _visible_rt_realtime(row_idx, col_idx, context_hi, realtime_offset, **kwargs):
    lo = max(0, row_idx - realtime_offset)
    return lo <= col_idx <= row_idx


def _visible_rt_label(row_idx, col_idx, **kwargs):
    return True


REALTIME_VAR_SPEC = MaskSpec(
    tensor_args=[TensorArg("context_sizes", ("batch",), nonnegative=True),
                 TensorArg("history_sizes", ("batch",), nonnegative=True),
                 TensorArg("realtime_sizes", ("batch",), nonnegative=True),
                 TensorArg("label_nums", ("batch",), nonnegative=True),
                 TensorArg("label_group_sizes", ("batch",), nonnegative=True),
                 TensorArg("realtime_offsets", ("batch",), nonnegative=True)],
    scalar_setup=_realtime_var_setup,
    regions=[RegionMask(name="context", when=_when_rt_context, visible=_visible_rt_context),
             RegionMask(name="history", when=_when_rt_history, visible=_visible_rt_history),
             RegionMask(name="realtime", when=_when_rt_realtime, visible=_visible_rt_realtime),
             RegionMask(name="label", when=_when_rt_label, visible=_visible_rt_label)],
    cur_seqlen="seq_hi",
)


# ================================================================
#  CALLABLE_MODULO — 4-stage with modulo stripes in realtime left half
# ================================================================

def _modulo_causal_setup(batch_idx, context_sizes, history_sizes, realtime_sizes,
                         label_nums, label_group_sizes, mods, rems, **kwargs):
    context_hi = int(context_sizes[batch_idx])
    history_hi = context_hi + int(history_sizes[batch_idx])
    realtime_hi = history_hi + int(realtime_sizes[batch_idx])
    label_group_size = int(label_group_sizes[batch_idx])
    seq_hi = realtime_hi + int(label_nums[batch_idx]) * label_group_size
    return {"context_hi": context_hi, "history_hi": history_hi,
            "realtime_hi": realtime_hi, "seq_hi": seq_hi,
            "label_group_size": label_group_size,
            "mod": int(mods[batch_idx]), "rem": int(rems[batch_idx])}


def _when_mod_context(row_idx, context_hi, **kwargs):
    return row_idx < context_hi


def _when_mod_history(row_idx, context_hi, history_hi, **kwargs):
    return context_hi <= row_idx < history_hi


def _when_mod_realtime(row_idx, history_hi, realtime_hi, **kwargs):
    return history_hi <= row_idx < realtime_hi


def _when_mod_label(row_idx, realtime_hi, seq_hi, **kwargs):
    return realtime_hi <= row_idx < seq_hi


def _visible_mod_context(row_idx, col_idx, history_hi, **kwargs):
    return col_idx < history_hi


def _visible_mod_history(row_idx, col_idx, **kwargs):
    return col_idx <= row_idx


def _visible_mod_realtime(row_idx, col_idx, history_hi, mod, rem, **kwargs):
    if col_idx < history_hi:
        return col_idx % mod < rem
    return col_idx <= row_idx


def _visible_mod_label(row_idx, col_idx, **kwargs):
    return col_idx <= row_idx


CALLABLE_MODULO_SPEC = MaskSpec(
    tensor_args=[TensorArg("context_sizes", ("batch",), nonnegative=True),
                 TensorArg("history_sizes", ("batch",), nonnegative=True),
                 TensorArg("realtime_sizes", ("batch",), nonnegative=True),
                 TensorArg("label_nums", ("batch",), nonnegative=True),
                 TensorArg("label_group_sizes", ("batch",), nonnegative=True),
                 TensorArg("mods", ("batch",), nonnegative=True),
                 TensorArg("rems", ("batch",), nonnegative=True)],
    scalar_setup=_modulo_causal_setup,
    regions=[RegionMask(name="context", when=_when_mod_context, visible=_visible_mod_context),
             RegionMask(name="history", when=_when_mod_history, visible=_visible_mod_history),
             RegionMask(name="realtime", when=_when_mod_realtime, visible=_visible_mod_realtime),
             RegionMask(name="label", when=_when_mod_label, visible=_visible_mod_label)],
    cur_seqlen="seq_hi",
)


# ================================================================
#  PIECEWISE — row-segmented static mask (callable + interval)
# ================================================================

def _piecewise_setup(batch_idx, seq_lens, row_l2s, col_splits, mods, rems, **kwargs):
    return {"seq_hi": int(seq_lens[batch_idx]),
            "row_l2": int(row_l2s[batch_idx]),
            "col_split": int(col_splits[batch_idx]),
            "mod": int(mods[batch_idx]), "rem": int(rems[batch_idx])}


def _when_piecewise_callable(row_idx, row_l2, **kwargs):
    return row_idx < row_l2


def _when_piecewise_interval(row_idx, row_l2, seq_hi, **kwargs):
    return row_l2 <= row_idx < seq_hi


def _visible_piecewise_callable(row_idx, col_idx, col_split, mod, rem, **kwargs):
    return (col_idx >= col_split) or (col_idx % mod < rem)


def _visible_piecewise_interval(row_idx, col_idx, **kwargs):
    return True


PIECEWISE_SPEC = MaskSpec(
    tensor_args=[TensorArg("seq_lens", ("batch",)),
                 TensorArg("row_l2s", ("batch",), nonnegative=True),
                 TensorArg("col_splits", ("batch",), nonnegative=True),
                 TensorArg("mods", ("batch",), nonnegative=True),
                 TensorArg("rems", ("batch",), nonnegative=True)],
    scalar_setup=_piecewise_setup,
    regions=[RegionMask(name="piecewise-callable", when=_when_piecewise_callable, visible=_visible_piecewise_callable),
             RegionMask(name="piecewise-interval", when=_when_piecewise_interval, visible=_visible_piecewise_interval)],
    cur_seqlen="seq_hi",
)