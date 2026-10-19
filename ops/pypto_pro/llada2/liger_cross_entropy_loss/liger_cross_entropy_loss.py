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
"""LigerCrossEntropyLoss: LLaDA2 entry points over the two PyPTO Pro kernels.

Exposes the same four names as the `pypto_tensor/llada2` adapter for this
operator, so the two backends can be swapped without touching a call site:

    * ``LigerCrossEntropyFunction`` -- ``torch.autograd.Function``, forward and
      backward;
    * ``liger_cross_entropy``       -- the functional form, matching the liger
      functional API;
    * ``LigerCrossEntropyLoss``     -- the ``nn.Module`` form, holding the
      options as attributes;
    * ``CrossEntropyOutput``        -- what the two latter return when any of
      the optional outputs is requested.

Everything below the two wrapper calls is autograd plumbing. The memory trick
the operator is built around shows through here: the forward leaves dloss/dx in
the logits buffer and hands it back as ``saved_input``, the context saves that
buffer rather than a second copy, and the backward scales it. That is why the
forward returns five values while the autograd Function returns four.
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
    """The forward and backward kernels, wired into autograd."""

    @staticmethod
    def forward(ctx, _input, target, weight=None, ignore_index=-100,
                lse_square_scale=0.0, label_smoothing=0.0, reduction="mean",
                softcap=None, return_z_loss=False, return_token_accuracy=False,
                return_predicted_tokens=False):
        loss, z_loss, token_accuracy, predicted_tokens, saved_input = liger_cross_entropy_loss_fwd_wrapper(
                _input, target, weight, ignore_index, lse_square_scale,
                label_smoothing, reduction, softcap,
                return_z_loss, return_token_accuracy, return_predicted_tokens,
            )
        # Only the gradient buffer is worth keeping, and only when something
        # will ask for it. It aliases `_input`, so this stores no new memory --
        # which is the whole point of writing the gradient back in place.
        if _input.requires_grad:
            ctx.save_for_backward(saved_input)
        # Deliberately not `ctx.needs_input_grad`: the engine owns that name and
        # fills it with a tuple of bools before forward runs, so assigning to it
        # would shadow the engine's own bookkeeping.
        ctx.input_requires_grad = _input.requires_grad

        # z-loss, accuracy and predicted tokens are diagnostics; nothing
        # differentiates through them. Marking them says so to the engine --
        # detaching alone would not, since every tensor returned from `forward`
        # is re-attached to this Function as an output.
        extras = [t for t in (z_loss, token_accuracy, predicted_tokens)
                  if t is not None]
        if extras:
            ctx.mark_non_differentiable(*extras)
        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    def backward(ctx, grad_loss, grad_z_loss, grad_token_accuracy,
                 grad_predicted_tokens):
        # forward takes eleven arguments, so eleven gradients come back out.
        nones = (None,) * 10
        if not ctx.input_requires_grad:
            return (None,) + nones
        (saved_input,) = ctx.saved_tensors
        grad_input = liger_cross_entropy_loss_bwd_wrapper(saved_input, grad_loss)
        return (grad_input,) + nones


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
    """The functional entry point, matching the liger functional API."""
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
    """The module form; the options live on the module."""

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
        assert softcap is None or softcap > 0, (
            f"softcap must greater than 0.0 or None. Got: {softcap}"
        )
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
        if not (self.return_z_loss or self.return_token_accuracy
                or self.return_predicted_tokens):
            return loss
        return CrossEntropyOutput(
            loss=loss, z_loss=z_loss, token_accuracy=token_accuracy,
            predicted_tokens=predicted_tokens,
        )
