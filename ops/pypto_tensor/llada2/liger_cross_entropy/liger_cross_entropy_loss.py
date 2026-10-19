#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this files except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""LigerCrossEntropyLoss：公共入口，串起 PyPTO 前向 kernel 与反向 kernel。

对外暴露：
    * ``LigerCrossEntropyFunction`` —— ``torch.autograd.Function``，前向+反向入口；
    * ``liger_cross_entropy``          —— functional 形式（对齐 liger functional API）；
    * ``LigerCrossEntropyLoss``        —— ``nn.Module`` 形式，参数持有在模块上。

依赖边界（除 pip 包外）只有两个 kernel 入口：
    * ``liger_cross_entropy_loss_fwd_impl.liger_cross_entropy_loss_fwd_wrapper`` —— 前向 kernel
    * ``liger_cross_entropy_loss_bwd_impl.liger_cross_entropy_loss_bwd_wrapper`` —— 反向 kernel
"""

from __future__ import annotations

__all__ = [
    "CrossEntropyOutput",
    "LigerCrossEntropyFunction",
    "liger_cross_entropy",
    "LigerCrossEntropyLoss",
]

import os
import sys
from dataclasses import dataclass
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch_npu  # noqa: E402,F401  NPU 设备初始化必需

from liger_cross_entropy_loss_fwd_impl import (  # noqa: E402
    liger_cross_entropy_loss_fwd_wrapper,
)
from liger_cross_entropy_loss_bwd_impl import (  # noqa: E402
    liger_cross_entropy_loss_bwd_wrapper,
)


@dataclass
class CrossEntropyOutput:
    loss: torch.Tensor
    z_loss: Optional[torch.Tensor] = None
    token_accuracy: Optional[torch.Tensor] = None
    predicted_tokens: Optional[torch.Tensor] = None


class LigerCrossEntropyFunction(torch.autograd.Function):
    """前向 kernel + 反向 kernel，对齐 liger-kernel 的 LigerCrossEntropyFunction。"""

    @staticmethod
    def forward(ctx, _input, target, weight=None, ignore_index=-100,
                lse_square_scale=0.0, label_smoothing=0.0, reduction="mean",
                softcap=None, return_z_loss=False, return_token_accuracy=False,
                return_predicted_tokens=False):
        loss, z_loss, token_accuracy, predicted_tokens, saved_input = \
            liger_cross_entropy_loss_fwd_wrapper(
                _input, target, weight, ignore_index, lse_square_scale,
                label_smoothing, reduction, softcap,
                return_z_loss, return_token_accuracy, return_predicted_tokens,
            )
        if _input.requires_grad:
            ctx.save_for_backward(saved_input)
        ctx.needs_input_grad = _input.requires_grad
        return (loss,
                z_loss.detach() if z_loss is not None else None,
                token_accuracy.detach() if token_accuracy is not None else None,
                predicted_tokens.detach() if predicted_tokens is not None else None)

    @staticmethod
    def backward(ctx, grad_loss, grad_z_loss, grad_token_accuracy, grad_predicted_tokens):
        if not ctx.needs_input_grad:
            return (None, None, None, None, None, None,  # noqa: G.FNM.05
                    None, None, None, None, None)
        (saved_input,) = ctx.saved_tensors
        grad_input = liger_cross_entropy_loss_bwd_wrapper(saved_input, grad_loss)
        return (grad_input, None, None, None, None, None,  # noqa: G.FNM.05
                None, None, None, None, None)


def liger_cross_entropy(
    _input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
):
    """functional 入口（对齐 liger functional API）。"""
    loss, z_loss, token_accuracy, predicted_tokens = LigerCrossEntropyFunction.apply(
        _input, target, weight, ignore_index, lse_square_scale,
        label_smoothing, reduction, softcap,
        return_z_loss, return_token_accuracy, return_predicted_tokens,
    )
    if not (return_z_loss or return_token_accuracy or return_predicted_tokens):
        return loss
    return CrossEntropyOutput(
        loss=loss,
        z_loss=z_loss,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )


class LigerCrossEntropyLoss(nn.Module):
    """nn.Module 入口，参数与 liger-kernel 的 LigerCrossEntropyLoss 一致。"""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
    ):
        super().__init__()
        assert (label_smoothing >= 0) and (label_smoothing <= 1), (
            f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}"
        )
        assert reduction in {
            "mean",
            "sum",
            "none",
        }, f"reduction must be one of 'mean', 'sum', or 'none'. Got: {reduction}"
        assert softcap is None or softcap > 0, f"softcap must greater than 0.0 or None. Got: {softcap}"
        self.weight = weight
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.softcap = softcap
        self.return_z_loss = return_z_loss
        self.return_token_accuracy = return_token_accuracy
        self.return_predicted_tokens = return_predicted_tokens

    def forward(self, _input: torch.Tensor, target: torch.Tensor):
        loss, z_loss, token_accuracy, predicted_tokens = LigerCrossEntropyFunction.apply(
            _input,
            target,
            self.weight,
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
            self.softcap,
            self.return_z_loss,
            self.return_token_accuracy,
            self.return_predicted_tokens,
        )
        if not self.return_z_loss and not self.return_token_accuracy and not self.return_predicted_tokens:
            return loss

        return CrossEntropyOutput(
            loss=loss, z_loss=z_loss, token_accuracy=token_accuracy, predicted_tokens=predicted_tokens
        )
