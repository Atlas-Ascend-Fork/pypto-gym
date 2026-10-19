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
"""Pytest harness for the liger_cross_entropy operator, float32.

One file, in four sections:

  * **golden** -- the pure-torch reference, transcribed from the upstream
    implementation. It carries the parts `torch.nn.CrossEntropyLoss` does not
    have: z-loss, token accuracy, first-occurrence predicted tokens, and the
    gradient intermediate the forward leaves in its input buffer.
  * **forward** -- the upstream `(B, T, V)` shape set plus a wide-vocabulary set
    at V = 157184, then one small shape crossed with every semantic switch, all
    against the golden. Plus the argmax-gate case: the same gradient check with
    the metric outputs switched off, at vocabularies wider than one 64-lane
    group.
  * **backward** -- the same shapes, bit-exact; every form `grad_output` arrives
    in and the buffer contract each one implies; and the two composed, checked
    against `F.cross_entropy` autograd.
  * **autograd** -- the `LigerCrossEntropyFunction` layer: `loss.backward()`
    reaching the backward kernel, each reduction selecting the grad_output form
    it should, and the functional and module entry points' return contracts.

Two references are used, deliberately. The golden covers the full upstream
semantics but shares its reading of them with the kernels, so the composed and
autograd sections check against `F.cross_entropy` instead -- it shares no code
with either, and it is what would catch the kernels and the golden misreading
upstream the same way. The price is that it has no z-loss term, so
`lse_square_scale` is checked against the golden only.

Tolerances follow the upstream table. The forward's streamed online softmax
accumulates a row in a different order than the golden's single-shot reduction,
so it is compared under a tolerance; the backward, being a single multiply, is
gated on `torch.equal`.

Run on NPU:
    pytest test_liger_cross_entropy_pro.py -v
or direct:
    python test_liger_cross_entropy_pro.py
"""

import logging
import math
import os
import sys
from typing import Optional, Tuple, Union

import pytest
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_REPO_ROOT = _THIS_DIR
while not os.path.isdir(os.path.join(_REPO_ROOT, "src")):
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
    if _REPO_ROOT == os.path.dirname(_REPO_ROOT):
        break
_SRC_DIR = _REPO_ROOT
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ops.pypto_pro.llada2.liger_cross_entropy_loss import (  # noqa: E501
    CrossEntropyOutput,
    LigerCrossEntropyFunction,
    LigerCrossEntropyLoss,
    liger_cross_entropy,
)
from ops.pypto_pro.llada2.liger_cross_entropy_loss.liger_cross_entropy_loss_bwd_impl import (  # noqa: E501
    liger_cross_entropy_is_identity_grad,
    liger_cross_entropy_loss_bwd_wrapper,
)
from ops.pypto_pro.llada2.liger_cross_entropy_loss.liger_cross_entropy_loss_fwd_impl import (  # noqa: E501
    liger_cross_entropy_loss_fwd_wrapper,
)


def _device():
    import torch_npu  # noqa: F401

    return "npu:{}".format(int(os.environ.get("DEVICE", "0")))


# ═══════════════════════════════════════════════════════════════════
# golden: the pure-torch reference
# ═══════════════════════════════════════════════════════════════════
def liger_cross_entropy_loss_fwd_golden(
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
    """Signature of ``LigerCrossEntropyFunction.forward`` without ``ctx``.

    Returns ``(loss, z_loss, token_accuracy, predicted_tokens, saved_input)``.
    """
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"
    assert isinstance(return_token_accuracy, bool), (
        f"return_token_accuracy must be True or False. Got: {return_token_accuracy}"
    )
    assert isinstance(return_predicted_tokens, bool), (
        f"return_predicted_tokens must be True or False. Got: {return_predicted_tokens}"
    )

    device = _input.device
    bt, v = _input.shape
    n_rows = bt

    target_mask = target != ignore_index
    n_non_ignore = int(target_mask.sum().item())
    assert (target * target_mask).max() < v, f"Target {target.max()} is out of bounds. Expected < {v}"
    assert (target * target_mask).min() >= 0, f"Target {target.min()} is out of bounds. Expected >= 0"

    sum_non_ignore_weight = float(n_non_ignore)
    weight_sum = 0.0
    if weight is not None:
        assert weight.shape[0] == v, f"If given, weight has to be a Tensor of size V. Got: {weight.shape}"
        assert torch.is_floating_point(weight), (
            f"If given, weight has to be a Tensor of floating point dtype. Got: {weight.dtype}"
        )
        sum_non_ignore_weight = float(weight[target.masked_select(target_mask)].sum().item())
        weight_sum = float(weight.sum().item())

    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    has_weight = weight is not None
    has_softcap = softcap is not None
    has_gradients = _input.requires_grad

    x = _input.float()
    w = weight.float() if has_weight else None
    if has_softcap:
        x = softcap * torch.tanh(x / softcap)

    eps_f = label_smoothing / v

    # First pass: per-row max / sum / lse (the closed form of the kernel's
    # online softmax).
    row_max = x.max(dim=-1).values
    exp_x = torch.exp(x - row_max[:, None])
    d = exp_x.sum(dim=-1)
    lse = row_max + torch.log(d)

    arange = torch.arange(n_rows, device=device, dtype=torch.int64)
    ori_x_y = x[arange, target]

    loss_row = lse - ori_x_y
    weight_y = None
    if has_weight:
        weight_y = w[target]
        loss_row = loss_row * weight_y

    if label_smoothing > 0:
        if has_weight:
            scaled_x_sum = (-eps_f * x * w[None, :]).sum(dim=-1)
            smooth_loss = scaled_x_sum + eps_f * lse * weight_sum
        else:
            scaled_x_sum = (-eps_f * x).sum(dim=-1)
            smooth_loss = scaled_x_sum + label_smoothing * lse
        loss_row = loss_row * (1 - label_smoothing) + smooth_loss

    z_loss_row = lse_square_scale * lse * lse

    # Normalization: with a weight the loss uses the summed weight while z-loss
    # always uses the plain non-ignored count.
    if reduction == "mean":
        if has_weight:
            loss_row = loss_row / sum_non_ignore_weight
        else:
            loss_row = loss_row / float(n_non_ignore)
        z_loss_row = z_loss_row / float(n_non_ignore)
    loss_row = loss_row + z_loss_row

    loss_1d = loss_row.to(_input.dtype)
    invalid_rows = ~target_mask
    loss_1d = torch.where(target_mask, loss_1d, torch.zeros((), dtype=_input.dtype, device=device))

    if return_z_loss:
        z_loss_1d = z_loss_row.to(_input.dtype)
        z_loss_1d = torch.where(
            target_mask, z_loss_1d, torch.zeros((), dtype=_input.dtype, device=device)
        )
    else:
        z_loss_1d = None

    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None

    # Argmax metrics: torch's bool argmax returns the first True, which is the
    # same first-occurrence rule the triton kernel gets from tl.min.
    argmax_idx = None
    if return_token_accuracy or return_predicted_tokens:
        argmax_idx = (x == row_max[:, None]).long().argmax(dim=-1)
    if return_token_accuracy:
        token_accuracy_1d = (argmax_idx == target).float()
        token_accuracy_1d = torch.where(
            target_mask, token_accuracy_1d, torch.zeros((), dtype=torch.float32, device=device)
        )
        if reduction == "none":
            token_accuracy = token_accuracy_1d
        else:
            token_accuracy = torch.sum(token_accuracy_1d) / float(n_non_ignore)
    else:
        token_accuracy = None

    if return_predicted_tokens:
        predicted_tokens = torch.full((n_rows,), -1, dtype=torch.int64, device=device)
        if n_non_ignore > 0:
            predicted_tokens[target_mask] = argmax_idx[target_mask]
    else:
        predicted_tokens = None

    # Memory-saving trick: the gradient intermediate replaces the logits in
    # place, through a detached view of the same storage.
    saved_input = _input.detach()

    if has_gradients:
        softmax_p = exp_x / d[:, None]

        if has_weight:
            dloss_ori = (1 - label_smoothing) * softmax_p * weight_y[:, None]
            valid = target_mask
            dloss_ori[arange[valid], target[valid]] = (
                dloss_ori[arange[valid], target[valid]] - (1 - label_smoothing) * weight_y[valid]
            )
            dloss_smooth = eps_f * (-w[None, :] + softmax_p * weight_sum)
            dz_loss = 2 * lse_square_scale * lse[:, None] * softmax_p
            if reduction == "mean":
                dloss_ori = dloss_ori / sum_non_ignore_weight
                dloss_smooth = dloss_smooth / sum_non_ignore_weight
                dz_loss = dz_loss / float(n_non_ignore)
            dx = dloss_ori + dloss_smooth + dz_loss
        else:
            dx = softmax_p + 2 * lse_square_scale * lse[:, None] * softmax_p - eps_f
            valid = target_mask
            dx[arange[valid], target[valid]] = (
                dx[arange[valid], target[valid]] - (1 - label_smoothing)
            )
            if reduction == "mean":
                dx = dx / float(n_non_ignore)

        if has_softcap:
            t = torch.tanh(_input.float() / softcap)
            dx = dx * (1 - t * t)

        dx_cast = dx.to(_input.dtype)
        saved_input[target_mask] = dx_cast[target_mask]
        saved_input[invalid_rows] = 0
    else:
        saved_input[invalid_rows] = 0

    return loss, z_loss, token_accuracy, predicted_tokens, saved_input


def golden_forward(*args, **kwargs) -> Tuple[torch.Tensor, ...]:
    """Alias with the four-tuple return shape of ``LigerCrossEntropyFunction.apply``."""
    loss, z_loss, accuracy, predicted, _ = liger_cross_entropy_loss_fwd_golden(*args, **kwargs)
    return loss, z_loss, accuracy, predicted


# --------------------------------------------------------------------------
# backward: liger_kernel.ops.cross_entropy.cross_entropy_backward
#
# grad_output arrives in one of three forms and upstream treats each
# differently: exactly 1.0 skips the multiply and returns _input untouched;
# a 0-dim tensor (reduction mean/sum) scales every row in place; a [BT]
# vector (reduction none) scales row r by grad_output[r], allocating.
# torch.mul uses opmath_t for the half dtypes, so a bfloat16 or float16
# product is formed in float32 and rounded once -- which this inherits.
# --------------------------------------------------------------------------


def liger_cross_entropy_loss_bwd_golden(
    saved_input: torch.Tensor,
    grad_output: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Signature of ``cross_entropy_backward``; returns the gradient w.r.t. the logits."""
    if not isinstance(grad_output, torch.Tensor):
        grad_output = torch.tensor(
            grad_output, dtype=saved_input.dtype, device=saved_input.device
        )
    if saved_input.ndim != 2:
        raise ValueError(
            "saved_input must be [BT, V], got shape {}".format(tuple(saved_input.shape))
        )
    if grad_output.ndim > 1 or (grad_output.ndim == 1
                                and grad_output.shape[0] != saved_input.shape[0]):
        raise ValueError(
            "grad_output must be a scalar or a [BT] vector, got shape {}".format(
                tuple(grad_output.shape)
            )
        )

    # "If cross entropy is the last layer, grad_output is 1.0. Skip the mul to
    # save time." The comparison is upstream's, verbatim: ``torch.equal`` matches
    # across dtypes on value, so a half or bfloat16 one takes the short circuit
    # too. It is an optimization rather than separate semantics -- multiplying by
    # exactly one in float32 and rounding back is the identity on every finite
    # value, on both infinities, on a signed zero, and on a NaN.
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return saved_input

    if grad_output.ndim > 0:
        return saved_input * grad_output.unsqueeze(dim=1)
    return saved_input * grad_output


# ═══════════════════════════════════════════════════════════════════
# forward
# ═══════════════════════════════════════════════════════════════════

# Upstream's float32 budget is 1e-8; it is loosened here because the kernel's
# streamed online softmax sums a row in a different order than the golden.
ATOL, RTOL = 2e-5, 2e-5


def _close(name, got, ref, atol, rtol, failures):
    if got is None or ref is None:
        if got is not ref:
            failures.append("{}: one side is None".format(name))
        return
    g = got.detach().to(torch.float32).cpu()
    r = ref.detach().to(torch.float32).cpu()
    if g.shape != r.shape:
        failures.append("{}: shape {} != {}".format(name, tuple(g.shape), tuple(r.shape)))
    elif not torch.allclose(g, r, atol=atol, rtol=rtol, equal_nan=True):
        failures.append("{}: max |d| = {:.3e}".format(name, float((g - r).abs().max())))


def run_forward_case(bt, v, ignore_index=-100, n_ignored=0, label_smoothing=0.0,
                     lse_square_scale=0.0, reduction="mean", softcap=None,
                     use_weight=False, scalar=1.0, requires_grad=True, seed=0,
                     return_metrics=True):
    """One case through the golden and the kernel; returns a list of failures.

    `return_metrics` drives `return_token_accuracy` / `return_predicted_tokens`
    together. It is not cosmetic: the kernel runs its first-occurrence argmax
    only when one of them is asked for, so clearing it selects a different code
    path through the gradient pass, and the gradient still has to come out right
    on that path.
    """
    device = _device()
    torch.manual_seed(seed)
    base = (torch.randn(bt, v, dtype=torch.float32) * scalar)
    target = torch.randint(0, v, (bt,), dtype=torch.int64)
    if n_ignored:
        target[torch.randperm(bt)[:min(n_ignored, bt)]] = ignore_index
    weight = (torch.rand(v, dtype=torch.float32) + 0.5) if use_weight else None

    x_ref = base.clone().requires_grad_(requires_grad)
    ref_loss, ref_z, ref_acc, ref_pred, ref_saved = liger_cross_entropy_loss_fwd_golden(
        x_ref, target, weight, ignore_index, lse_square_scale, label_smoothing,
        reduction, softcap, True, True, True)

    x = base.clone().to(device).requires_grad_(requires_grad)
    got = liger_cross_entropy_loss_fwd_wrapper(
        x, target.to(device), None if weight is None else weight.to(device),
        ignore_index, lse_square_scale, label_smoothing, reduction, softcap,
        True, return_metrics, return_metrics)
    torch.npu.synchronize()

    failures = []
    _close("loss", got[0], ref_loss, ATOL, RTOL, failures)
    _close("z_loss", got[1], ref_z, ATOL, RTOL, failures)
    if return_metrics:
        _close("token_accuracy", got[2], ref_acc, 1e-6, 1e-6, failures)
        _close("predicted_tokens", got[3], ref_pred, 0, 0, failures)
    else:
        # Not requested means not returned -- the buffers behind them are left
        # unwritten when the argmax is skipped, so handing them back would be
        # handing back garbage.
        if got[2] is not None:
            failures.append("token_accuracy returned although it was not requested")
        if got[3] is not None:
            failures.append("predicted_tokens returned although it was not requested")
    _close("saved_input", x, ref_saved, ATOL, RTOL, failures)
    # The fifth return value is the buffer the gradient was written into. This
    # kernel writes in place, so it must be the caller's own tensor rather than
    # a copy -- that aliasing is what lets the backward run without the forward
    # having saved a second [BT, V].
    if got[4].data_ptr() != x.data_ptr():
        failures.append("saved_input must alias the logits buffer")
    return failures


# ═══════════════════════════════════════════════════════════════════
# Test case matrices
# ═══════════════════════════════════════════════════════════════════

SHAPES = [
    pytest.param(8192, 32000, id="bt8192_v32000_llama"),
    pytest.param(1269, 32000, id="bt1269_v32000"),
    pytest.param(2048, 32000, id="bt2048_v32000_llama2"),
    pytest.param(256, 512, id="bt256_v512"),
    pytest.param(141, 31, id="bt141_v31"),
    pytest.param(4, 8, id="bt4_v8"),
    pytest.param(63, 41, id="bt63_v41"),
    pytest.param(128, 4096, id="bt128_v4096"),
    pytest.param(391, 157184, id="bt391_v157184"),
    pytest.param(618, 157184, id="bt618_v157184"),
    pytest.param(810, 157184, id="bt810_v157184"),
    pytest.param(1002, 157184, id="bt1002_v157184"),
]

OPTIONS = [
    pytest.param(dict(), id="plain_mean"),
    pytest.param(dict(reduction="sum"), id="sum"),
    pytest.param(dict(reduction="none"), id="none"),
    pytest.param(dict(ignore_index=2, n_ignored=9), id="ignore_index"),
    pytest.param(dict(ignore_index=3, n_ignored=63), id="ignore_all"),
    pytest.param(dict(label_smoothing=0.1), id="label_smoothing"),
    pytest.param(dict(lse_square_scale=1e-4), id="z_loss"),
    pytest.param(dict(softcap=30.0, scalar=10.0), id="softcap"),
    pytest.param(dict(use_weight=True), id="class_weight"),
    pytest.param(dict(use_weight=True, label_smoothing=0.1), id="weight_smoothing"),
    pytest.param(dict(requires_grad=False), id="forward_only"),
    pytest.param(dict(use_weight=True, label_smoothing=0.1, lse_square_scale=1e-4,
                      softcap=30.0, scalar=10.0, ignore_index=1, n_ignored=4),
                 id="all_options"),
]

# Vocabularies wide enough that the argmax scan's column cursor has to advance:
# 4096 is 64 groups inside a single tile, 157184 spans several tiles. The third
# case takes the softcap gate at the same time, which is the one combination of
# the two gates no other test reaches.
METRICS_OFF = [
    pytest.param(128, 4096, dict(), id="bt128_v4096"),
    pytest.param(391, 157184, dict(), id="bt391_v157184"),
    pytest.param(128, 4096, dict(softcap=30.0, scalar=10.0), id="bt128_v4096_softcap"),
]


@pytest.mark.soc("950")
@pytest.mark.parametrize("bt,v", SHAPES)
def test_liger_cross_entropy_forward_shapes(bt, v):
    """Upstream shape coverage plus the wide-vocabulary set."""
    failures = run_forward_case(bt, v, ignore_index=-100 if v > 100 else 2,
                                n_ignored=max(1, bt // 8))
    assert not failures, "; ".join(failures)
    log.info("[PRECISION_PASS] BT=%d V=%d", bt, v)


@pytest.mark.soc("950")
@pytest.mark.parametrize("options", OPTIONS)
def test_liger_cross_entropy_forward_options(options):
    """One small shape crossed with every semantic switch."""
    failures = run_forward_case(63, 41, **options)
    assert not failures, "; ".join(failures)
    log.info("[PRECISION_PASS] %s", options)


@pytest.mark.soc("950")
@pytest.mark.parametrize("bt,v,options", METRICS_OFF)
def test_liger_cross_entropy_forward_without_metrics(bt, v, options):
    """The gradient must survive skipping the argmax, across column groups.

    Every other forward test asks for `token_accuracy` and `predicted_tokens`,
    which keeps the argmax scan switched on. This one switches it off, and it
    deliberately uses vocabularies wider than one 64-lane group so that the
    scan's column cursor has to advance many times -- the cursor is shared with
    the true-class correction, so losing its advance yields a perfectly correct
    loss and a wrong gradient, which only a gradient check at V > 64 can see.
    V = 4096 is one full tile of groups; V = 157184 crosses tile boundaries too.
    """
    failures = run_forward_case(bt, v, n_ignored=max(1, bt // 8),
                                return_metrics=False, **options)
    assert not failures, "; ".join(failures)
    log.info("[PRECISION_PASS] no-metrics BT=%d V=%d %s", bt, v, options)


@pytest.mark.soc("950")
def test_liger_cross_entropy_forward_rejects_non_contiguous():
    """A buffer the launch path would have to copy cannot be written in place.

    Host-side only -- it must fail before it reaches the device, so this one
    needs no NPU.
    """
    logits = torch.zeros(8, 64, dtype=torch.float32)[:, :32]
    assert not logits.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        liger_cross_entropy_loss_fwd_wrapper(logits, torch.zeros(8, dtype=torch.int64))


# ═══════════════════════════════════════════════════════════════════
# backward, and the two composed
# ═══════════════════════════════════════════════════════════════════

CHAIN_ATOL, CHAIN_RTOL = 2e-5, 2e-5


def make_saved(bt, v, seed=0):
    """A stand-in for what the forward leaves in its logits buffer.

    Post-forward values are softmax probabilities minus the one-hot target, so
    they live in (-1, 1) with one large-magnitude entry per row. Reproducing
    that spread matters: a plain normal would never reach the subnormal end.
    """
    torch.manual_seed(seed)
    saved = torch.rand(bt, v, dtype=torch.float32) / float(v)
    saved[torch.arange(bt), torch.randint(0, v, (bt,))] -= 1.0
    return saved


def make_grad(form, bt, seed=1):
    torch.manual_seed(seed)
    return {
        "one": lambda: torch.tensor(1.0),
        "mean": lambda: torch.tensor(1.0 / max(1, bt)),
        "sum": lambda: torch.tensor(2.5),
        "negative": lambda: torch.tensor(-0.75),
        "zero": lambda: torch.tensor(0.0),
        "rows": lambda: torch.randn(bt),
    }[form]()


def run_backward_case(bt, v, form="rows"):
    """One case through the golden and the kernel; returns a list of failures.

    There is no in-place flag: which buffer comes back is decided by the shape
    of `grad_output`, the same way upstream decides it. Exactly 1.0 returns the
    input untouched; a [BT] vector allocates; a 0-dim scalar writes in place.
    This function asserts that contract as well as the numbers.
    """
    device = _device()
    saved = make_saved(bt, v)
    grad = make_grad(form, bt)
    want = liger_cross_entropy_loss_bwd_golden(saved.clone(), grad)

    x = saved.clone().to(device)
    got = liger_cross_entropy_loss_bwd_wrapper(x, grad.to(device))
    torch.npu.synchronize()

    failures = []
    g = got.detach().cpu()
    if g.shape != want.shape:
        failures.append("shape {} != {}".format(tuple(g.shape), tuple(want.shape)))
    elif not torch.equal(g, want):
        diff = (g - want).abs()
        failures.append("not bit-exact: {} of {} differ, max |d| = {}".format(
            int((diff != 0).sum()), diff.numel(), float(diff.max())))

    aliased = got.data_ptr() == x.data_ptr()
    per_row = grad.ndim > 0
    if form == "one":
        if not aliased:
            failures.append("the identity short circuit must return the input itself")
        if not torch.equal(x.detach().cpu(), saved):
            failures.append("the identity short circuit must not touch the buffer")
    elif per_row:
        if aliased:
            failures.append("a per-row grad_output must return a new buffer")
        if not torch.equal(x.detach().cpu(), saved):
            failures.append("a per-row grad_output must leave the input alone")
    else:
        if not aliased:
            failures.append("a scalar grad_output must be applied in place")
    return failures


def run_chain_case(bt, v, reduction="mean", scale=1.0, ignore_index=-100,
                   n_ignored=0, label_smoothing=0.0, use_weight=False):
    """Forward then backward, composed the way an autograd graph would.

    `x.grad` is checked against `F.cross_entropy` autograd, which is the
    independent reference -- it is what would catch a misreading shared between
    the kernels and the golden.
    """
    device = _device()
    torch.manual_seed(0)
    base = torch.randn(bt, v, dtype=torch.float32)
    target = torch.randint(0, v, (bt,), dtype=torch.int64)
    if n_ignored:
        target[torch.randperm(bt)[:min(n_ignored, bt - 1)]] = ignore_index
    weight = (torch.rand(v, dtype=torch.float32) + 0.5) if use_weight else None
    row_w = torch.randn(bt, dtype=torch.float32)

    x = base.clone().to(device).requires_grad_(True)
    loss, _z, _a, _p, saved_input = liger_cross_entropy_loss_fwd_wrapper(
        x, target.to(device), None if weight is None else weight.to(device),
        ignore_index, 0.0, label_smoothing, reduction, None, False, False, False)
    grad_output = row_w.clone() if reduction == "none" else torch.tensor(scale)
    # No short circuit needed at the call site any more: the wrapper owns all
    # three paths, and the identity one hands `saved_input` straight back.
    got_grad = liger_cross_entropy_loss_bwd_wrapper(
        saved_input, grad_output.to(device))
    torch.npu.synchronize()

    x_t = base.clone().requires_grad_(True)
    t_loss = F.cross_entropy(x_t, target, weight=weight, ignore_index=ignore_index,
                             reduction=reduction, label_smoothing=label_smoothing)
    scalar = ((t_loss * row_w).sum() if reduction == "none" else t_loss * scale)
    scalar.backward()

    failures = []
    ref_loss = t_loss if reduction == "none" else t_loss
    if not torch.allclose(loss.detach().to(torch.float32).cpu(),
                          ref_loss.detach().to(torch.float32),
                          atol=CHAIN_ATOL, rtol=CHAIN_RTOL, equal_nan=True):
        failures.append("loss disagrees with F.cross_entropy")
    if not torch.allclose(got_grad.detach().to(torch.float32).cpu(), x_t.grad,
                          atol=CHAIN_ATOL, rtol=CHAIN_RTOL, equal_nan=True):
        failures.append("grad vs F.cross_entropy autograd: max |d| = {:.3e}".format(
            float((got_grad.detach().to(torch.float32).cpu() - x_t.grad).abs().max())))
    return failures


# ═══════════════════════════════════════════════════════════════════
# Test case matrices
# ═══════════════════════════════════════════════════════════════════


GRADS = [
    pytest.param("mean", id="scalar_mean"),
    pytest.param("sum", id="scalar_sum"),
    pytest.param("negative", id="scalar_negative"),
    pytest.param("zero", id="scalar_zero"),
    pytest.param("one", id="identity_short_circuit"),
    pytest.param("rows", id="per_row_vector"),
]

# The chain runs the forward with its metric outputs off, which is the one
# place the suite takes that path. V = 41 is a single 64-lane group, so a
# per-group defect collapses to nothing there; 97 and 200 are 2 and 4 groups.
CHAIN_SHAPES = [
    pytest.param(63, 41, id="bt63_v41"),
    pytest.param(130, 97, id="bt130_v97"),
    pytest.param(7, 200, id="bt7_v200"),
]

CHAIN = [
    pytest.param(dict(), id="mean_identity_grad"),
    pytest.param(dict(scale=3.0), id="mean_scaled"),
    pytest.param(dict(reduction="none"), id="none_per_row"),
    pytest.param(dict(reduction="sum", scale=0.25), id="sum_scaled"),
    pytest.param(dict(scale=2.0, ignore_index=2, n_ignored=9), id="ignore_index"),
    pytest.param(dict(scale=1.5, use_weight=True), id="class_weight"),
    pytest.param(dict(reduction="none", label_smoothing=0.1), id="label_smoothing"),
]


@pytest.mark.soc("950")
@pytest.mark.parametrize("bt,v", SHAPES)
def test_liger_cross_entropy_backward_shapes(bt, v):
    """Upstream shape coverage plus the wide-vocabulary set, bit-exact."""
    for form in ("rows", "sum"):
        failures = run_backward_case(bt, v, form)
        assert not failures, "{}: {}".format(form, "; ".join(failures))
    log.info("[PRECISION_PASS] BT=%d V=%d", bt, v)


@pytest.mark.soc("950")
@pytest.mark.parametrize("form", GRADS)
def test_liger_cross_entropy_backward_grad_forms(form):
    """Every form `grad_output` arrives in, and the aliasing contract."""
    failures = run_backward_case(63, 41, form)
    assert not failures, "; ".join(failures)
    log.info("[PRECISION_PASS] %s", form)


@pytest.mark.soc("950")
def test_liger_cross_entropy_backward_rejects_non_contiguous_scalar():
    """The in-place path cannot run on a buffer the launch path has to copy.

    Only the scalar path promises to write `saved_input` itself, so only it is
    guarded; a per-row grad_output allocates anyway and is unaffected. Host-side
    only, no NPU needed.
    """
    saved = torch.zeros(8, 64, dtype=torch.float32)[:, :32]
    assert not saved.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        liger_cross_entropy_loss_bwd_wrapper(saved, torch.tensor(2.0))


@pytest.mark.soc("950")
@pytest.mark.parametrize("bt,v", CHAIN_SHAPES)
@pytest.mark.parametrize("options", CHAIN)
def test_liger_cross_entropy_forward_backward_chain(bt, v, options):
    """Forward and backward composed, against F.cross_entropy autograd."""
    failures = run_chain_case(bt, v, **options)
    assert not failures, "; ".join(failures)
    log.info("[PRECISION_PASS] chain BT=%d V=%d %s", bt, v, options)


# ═══════════════════════════════════════════════════════════════════
# the autograd layer
# ═══════════════════════════════════════════════════════════════════

# Vocabularies spanning one, two and four groups of 64 lanes, plus one wide
# enough to cross a tile. A defect in the column cursor is invisible at V = 41.
AUTOGRAD_SHAPES = [
    pytest.param(63, 41, id="bt63_v41"),
    pytest.param(130, 97, id="bt130_v97"),
    pytest.param(128, 4096, id="bt128_v4096"),
]

REDUCTIONS = [
    pytest.param("mean", id="mean"),
    pytest.param("sum", id="sum"),
    pytest.param("none", id="none"),
]


def _inputs(bt, v, device, use_weight=False, n_ignored=0, ignore_index=-100):
    torch.manual_seed(0)
    base = torch.randn(bt, v, dtype=torch.float32)
    target = torch.randint(0, v, (bt,), dtype=torch.int64)
    if n_ignored:
        target[torch.randperm(bt)[:min(n_ignored, bt - 1)]] = ignore_index
    weight = (torch.rand(v, dtype=torch.float32) + 0.5) if use_weight else None
    return (base, target, weight,
            base.clone().to(device).requires_grad_(True),
            target.to(device),
            None if weight is None else weight.to(device))


def run_autograd_case(bt, v, reduction="mean", label_smoothing=0.0,
                      use_weight=False, n_ignored=0, ignore_index=-100,
                      lse_square_scale=0.0, scale=1.0):
    """Drive the op through autograd; returns a list of failure strings."""
    device = _device()
    base, target, weight, x, target_d, weight_d = _inputs(
        bt, v, device, use_weight, n_ignored, ignore_index)
    row_w = torch.randn(bt, dtype=torch.float32)

    loss = liger_cross_entropy(
        x, target_d, weight_d, ignore_index, lse_square_scale,
        label_smoothing, reduction, None)
    # "none" leaves a [BT] loss, so autograd needs an explicit gradient -- that
    # is the per-row path. The reduced forms give a 0-dim loss, and scaling it
    # is what produces a 0-dim gradient other than exactly 1.0.
    if reduction == "none":
        loss.backward(gradient=row_w.to(device))
    elif math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=0.0):
        loss.backward()
    else:
        (loss * scale).backward()
    torch.npu.synchronize()

    # `F.cross_entropy` is the reference precisely because it shares no code
    # with the kernels or the golden. The price is that it cannot cover
    # `lse_square_scale`, which adds a term it does not have -- see the note on
    # AUTOGRAD_OPTIONS.
    x_ref = base.clone().requires_grad_(True)
    ref_loss = F.cross_entropy(x_ref, target, weight=weight,
                               ignore_index=ignore_index, reduction=reduction,
                               label_smoothing=label_smoothing)
    if reduction == "none":
        (ref_loss * row_w).sum().backward()
    else:
        (ref_loss * scale).backward()

    failures = []
    if not torch.allclose(loss.detach().to(torch.float32).cpu(),
                          ref_loss.detach(), atol=ATOL, rtol=RTOL,
                          equal_nan=True):
        failures.append("loss disagrees with F.cross_entropy")
    if x.grad is None:
        failures.append("backward left no gradient on the input")
    elif not torch.allclose(x.grad.to(torch.float32).cpu(), x_ref.grad,
                            atol=ATOL, rtol=RTOL, equal_nan=True):
        failures.append("grad vs autograd: max |d| = {:.3e}".format(
            float((x.grad.to(torch.float32).cpu() - x_ref.grad).abs().max())))
    return failures


@pytest.mark.soc("950")
@pytest.mark.parametrize("bt,v", AUTOGRAD_SHAPES)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_liger_cross_entropy_autograd_reductions(bt, v, reduction):
    """Every reduction, driven through `loss.backward()`."""
    failures = run_autograd_case(bt, v, reduction=reduction)
    assert not failures, "; ".join(failures)
    log.info("[PRECISION_PASS] autograd BT=%d V=%d %s", bt, v, reduction)


# `lse_square_scale` is deliberately absent: the reference here is
# `F.cross_entropy`, which has no z-loss term, so a non-zero scale makes both
# the loss and the gradient legitimately differ from it. That option is checked
# against the golden instead, in the forward suite's `z_loss` case.
AUTOGRAD_OPTIONS = [
    pytest.param(dict(scale=3.0), id="mean_scaled"),
    pytest.param(dict(reduction="sum", scale=0.25), id="sum_scaled"),
    pytest.param(dict(label_smoothing=0.1), id="label_smoothing"),
    pytest.param(dict(use_weight=True), id="class_weight"),
    pytest.param(dict(ignore_index=2, n_ignored=9), id="ignore_index"),
    pytest.param(dict(reduction="none", label_smoothing=0.1, use_weight=True),
                 id="none_weighted_smoothed"),
]


@pytest.mark.soc("950")
@pytest.mark.parametrize("options", AUTOGRAD_OPTIONS)
def test_liger_cross_entropy_autograd_options(options):
    """The semantic switches, through autograd rather than the raw wrappers."""
    failures = run_autograd_case(130, 97, **options)
    assert not failures, "; ".join(failures)
    log.info("[PRECISION_PASS] autograd %s", options)


@pytest.mark.soc("950")
def test_liger_cross_entropy_functional_return_shapes():
    """The functional entry returns a tensor, or a CrossEntropyOutput."""
    device = _device()
    _b, _t, _w, x, target_d, _wd = _inputs(63, 41, device)

    plain = liger_cross_entropy(x, target_d)
    assert isinstance(plain, torch.Tensor), "a plain call must return the loss"

    x2 = x.detach().clone().requires_grad_(True)
    out = liger_cross_entropy(x2, target_d, return_z_loss=True,
                              return_token_accuracy=True,
                              return_predicted_tokens=True)
    assert isinstance(out, CrossEntropyOutput)
    assert out.z_loss is not None and out.token_accuracy is not None
    assert out.predicted_tokens is not None
    assert out.predicted_tokens.dtype == torch.int64
    # The diagnostics are marked non-differentiable; nothing backpropagates
    # through them, and the engine has to agree.
    assert not out.z_loss.requires_grad
    assert not out.token_accuracy.requires_grad
    log.info("[PRECISION_PASS] functional return shapes")


@pytest.mark.soc("950")
def test_liger_cross_entropy_module_matches_functional():
    """`LigerCrossEntropyLoss` and the functional form agree, gradient included."""
    device = _device()
    base, target, _w, x, target_d, _wd = _inputs(130, 97, device)

    module = LigerCrossEntropyLoss(reduction="mean")
    loss_mod = module(x, target_d)
    loss_mod.backward()
    torch.npu.synchronize()
    grad_mod = x.grad.detach().cpu().clone()

    x2 = base.clone().to(device).requires_grad_(True)
    loss_fn = liger_cross_entropy(x2, target_d, reduction="mean")
    loss_fn.backward()
    torch.npu.synchronize()

    failures = []
    if not torch.equal(loss_mod.detach().cpu(), loss_fn.detach().cpu()):
        failures.append("module and functional losses differ")
    if not torch.equal(grad_mod, x2.grad.detach().cpu()):
        failures.append("module and functional gradients differ")
    assert not failures, "; ".join(failures)
    log.info("[PRECISION_PASS] module matches functional")


@pytest.mark.soc("950")
def test_liger_cross_entropy_no_grad_input():
    """With `requires_grad` clear the forward still runs and nothing is saved."""
    device = _device()
    _b, _t, _w, _x, target_d, _wd = _inputs(63, 41, device)
    torch.manual_seed(0)
    x = torch.randn(63, 41, dtype=torch.float32, device=device)
    assert not x.requires_grad

    loss = liger_cross_entropy(x, target_d)
    assert not loss.requires_grad, "no input needs a gradient, so neither does the loss"
    log.info("[PRECISION_PASS] no-grad input")


def test_liger_cross_entropy_module_validates_options():
    """The module rejects the option values its contract excludes. No NPU."""
    with pytest.raises(AssertionError, match="label_smoothing"):
        LigerCrossEntropyLoss(label_smoothing=1.5)
    with pytest.raises(AssertionError, match="reduction"):
        LigerCrossEntropyLoss(reduction="average")
    with pytest.raises(AssertionError, match="softcap"):
        LigerCrossEntropyLoss(softcap=0.0)
    log.info("[PRECISION_PASS] module option validation")


# ═══════════════════════════════════════════════════════════════════
# Direct execution entry point (no pytest needed)
# ═══════════════════════════════════════════════════════════════════

def _run(label, fn, *args, **kwargs):
    """One case, reported; returns True when it passed."""
    try:
        failures = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - report and continue
        failures = ["EXCEPTION {}: {}".format(type(exc).__name__, exc)]
    log.info("  %-40s %s", label,
             "PASS" if not failures else "FAIL " + "; ".join(failures))
    return not failures


def main():
    ok = True

    log.info("forward")
    for case in SHAPES:
        bt, v = case.values
        ok &= _run("shape/" + case.id, run_forward_case, bt, v,
                   ignore_index=-100 if v > 100 else 2, n_ignored=max(1, bt // 8))
    for case in OPTIONS:
        options, = case.values
        ok &= _run("options/" + case.id, run_forward_case, 63, 41, **options)
    for case in METRICS_OFF:
        bt, v, options = case.values
        ok &= _run("no_metrics/" + case.id, run_forward_case, bt, v,
                   n_ignored=max(1, bt // 8), return_metrics=False, **options)

    log.info("backward")
    for case in SHAPES:
        bt, v = case.values
        for form in ("rows", "sum"):
            ok &= _run("shape/{}/{}".format(case.id, form),
                       run_backward_case, bt, v, form)
    for case in GRADS:
        form, = case.values
        ok &= _run("grad_form/" + case.id, run_backward_case, 63, 41, form)

    log.info("chain")
    for shape in CHAIN_SHAPES:
        bt, v = shape.values
        for case in CHAIN:
            options, = case.values
            ok &= _run("chain/{}/{}".format(shape.id, case.id),
                       run_chain_case, bt, v, **options)

    log.info("autograd")
    for shape in AUTOGRAD_SHAPES:
        bt, v = shape.values
        for red in REDUCTIONS:
            reduction, = red.values
            ok &= _run("autograd/{}/{}".format(shape.id, red.id),
                       run_autograd_case, bt, v, reduction=reduction)
    for case in AUTOGRAD_OPTIONS:
        options, = case.values
        ok &= _run("autograd/" + case.id, run_autograd_case, 130, 97, **options)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
