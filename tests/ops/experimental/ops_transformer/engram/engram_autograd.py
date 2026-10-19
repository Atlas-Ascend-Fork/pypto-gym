#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Engram autograd Function wrapper — connects engram_forward_wrapper + engram_backward_wrapper.

把多头 Engram 的前向 (engram_forward_wrapper) 与反向 (engram_backward_wrapper) 封装成
torch.autograd.Function, 使调用方只需 forward + loss + backward() 即可自动反传。
"""

import os
import sys

# ── sys.path: tests 同目录 + repo root/src (kernel impl) ──
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO_ROOT = _HERE
while not os.path.isdir(os.path.join(_REPO_ROOT, "src")):
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
    if _REPO_ROOT == os.path.dirname(_REPO_ROOT):
        break
_SRC_DIR = _REPO_ROOT
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch
import torch_npu  # noqa: F401

from ops.pypto_tensor.experimental.ops_transformer.engram.engram_forward_impl import (
    engram_forward_wrapper,
)
from ops.pypto_tensor.experimental.ops_transformer.engram.engram_backward_impl import (
    engram_backward_wrapper,
)


class EngramFunc(torch.autograd.Function):
    """Engram 正反向级联封装 (基于 engram_*_wrapper)。

    forward 返回 5 个输出:
      - value_out:    [B, L, M, Hh] BF16  (主输出, 参与 loss)
      - score_back:   [B, L, M, 1]  FP32  (中间激活, 仅用于 backward)
      - key_back:     [B, L, M, Hh] BF16  (中间激活, 仅用于 backward)
      - value_back:   [B, L, Hh]    BF16  (中间激活, 仅用于 backward)
      - gate_back:    [B, L, M, 1]  FP32  (中间激活, 仅用于 backward)

    backward 接收 value_out 的梯度, 结合保存的中间激活计算所有输入梯度。
    注意 engram_backward_wrapper 尾参顺序为 (score, gate, key, value)。
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,       # [B, L, M, Hh] BF16
        embeddings: torch.Tensor,          # [B, L, De]   BF16
        key_proj_weights: torch.Tensor,    # [M, De, Hh]  BF16
        value_proj_weights: torch.Tensor,  # [De, Hh]     BF16
        key_gamma: torch.Tensor,           # [M, Hh]      BF16
        query_gamma: torch.Tensor,         # [M, Hh]      BF16
    ):
        value_out, score_back, key_back, value_back, gate_back = engram_forward_wrapper(
            hidden_states, embeddings,
            key_proj_weights, value_proj_weights,
            key_gamma, query_gamma,
        )
        ctx.save_for_backward(
            hidden_states, embeddings,
            key_proj_weights, value_proj_weights,
            key_gamma, query_gamma,
            key_back, value_back, gate_back, score_back,
        )
        return value_out, score_back, key_back, value_back, gate_back

    @staticmethod
    def backward(ctx, grad_value_out, grad_score_back, grad_key_back, grad_value_back, grad_gate_back):
        # score_back/key_back/value_back/gate_back 是中间激活, loss 不对其求导, 梯度为 None.
        if grad_value_out is None:
            return (None,) * 6

        (
            hidden_states, embeddings,
            key_proj_weights, value_proj_weights,
            key_gamma, query_gamma,
            key_back, value_back, gate_back, score_back,
        ) = ctx.saved_tensors

        # wrapper 尾参顺序: score, gate, key, value (与 engram_backward_golden 一致).
        # forward 已输出 BF16 的 key_back/value_back, 无需额外 cast.
        d_hidden, d_embeddings, d_key_w, d_value_w, d_key_gamma, d_query_gamma = engram_backward_wrapper(
            grad_value_out.contiguous(), hidden_states, embeddings,
            key_proj_weights, value_proj_weights, key_gamma, query_gamma,
            score_back, gate_back, key_back, value_back,
        )
        return d_hidden, d_embeddings, d_key_w, d_value_w, d_key_gamma, d_query_gamma


def engram_autograd(
    hidden_states: torch.Tensor,
    embeddings: torch.Tensor,
    key_proj_weights: torch.Tensor,
    value_proj_weights: torch.Tensor,
    key_gamma: torch.Tensor,
    query_gamma: torch.Tensor,
):
    """Engram 自动求导接口 (推荐使用此函数而非直接调用 Func)."""
    return EngramFunc.apply(
        hidden_states, embeddings,
        key_proj_weights, value_proj_weights,
        key_gamma, query_gamma,
    )

