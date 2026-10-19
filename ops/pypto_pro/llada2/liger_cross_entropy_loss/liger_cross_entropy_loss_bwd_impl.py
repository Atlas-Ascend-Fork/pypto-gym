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
"""PyPTO-Pro liger_cross_entropy backward kernel implementation.

Backward of the Liger-Kernel fused cross-entropy loss, float32 logits.

The forward already left dloss/dx in `_input`, so the backward is only the chain
rule applied to it:

    dx[r, c] = saved_input[r, c] * grad_output[r]

`grad_output` arrives in one of three forms and upstream treats each
differently: exactly 1.0 skips the multiply and returns `_input` untouched; a
0-dim tensor (`reduction` of "mean" or "sum") scales every row by that one
value, in place; a [BT] vector (`reduction="none"`) scales row r by
`grad_output[r]`.

Materializing the 0-dim case into a [BT] vector on the host would be a broadcast
the kernel should own, so `g` arrives as [1, ng] with ng either 1 or BT and row r
reads element `min(r, ng - 1)` -- which collapses to 0 for every row when ng is
1, and to r when ng is BT, with no condition either way.

Every product is formed in float32 and rounded once, which is what `torch.mul`
does for the half dtypes through its `opmath_t`, so the result is bit-exact
against the torch reference rather than merely close.

The kernel body is machine-generated. Its source of truth is an EasyASC kernel
translated by `easyasc.targets.pypto`; regenerating it is described in the
README beside this file.
"""

import torch

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

# The runner's older PyPTO build leaves gaps in its DT_* pybind
# coverage: some constants (DT_BF16 among them) arrive as bare ints
# that VF-layer kwarg checks reject ("expected ir::DataType, but got
# int"), while others on the same build are real DataType objects
# (add_rms_norm_dynamic_quant exercised FP16/FP32/INT8/INT16/UINT32
# there). Rebuild the broken ones from the enum value AT MODULE
# SCOPE -- the VF parser inlines helper calls and rejects
# isinstance, so the fixups must be plain names by the time a
# decorated body mentions them. A healthy build passes through.
_DTT = type(pl.DT_FP32)


def _dt_fix(v):
    return v if isinstance(v, _DTT) else _DTT(v)


_PL_BF16 = _dt_fix(pl.DT_BF16)
_PL_FP16 = _dt_fix(pl.DT_FP16)
_PL_FP32 = _dt_fix(pl.DT_FP32)
_PL_INT8 = _dt_fix(pl.DT_INT8)
_PL_INT16 = _dt_fix(pl.DT_INT16)
_PL_INT32 = _dt_fix(pl.DT_INT32)
_PL_INT64 = _dt_fix(pl.DT_INT64)
_PL_UINT8 = _dt_fix(pl.DT_UINT8)
_PL_UINT16 = _dt_fix(pl.DT_UINT16)
_PL_UINT32 = _dt_fix(pl.DT_UINT32)
_PL_UINT64 = _dt_fix(pl.DT_UINT64)


@pl.vector_function
def scale_float_vf(src, dst, gtile, valid, groups):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lcb_f_gv = vf.load_align(gtile, 0, dist=pl.LoadDist.BRC_B32)
    for lcb_f_group in pl.range(0, groups, 1):
        lcb_f_active = vf.update_mask((valid - lcb_f_group * 64), dtype=_PL_FP32)
        _m0 = (lcb_f_group * 64)
        lcb_f_value = vf.load_align(src, _m0)
        lcb_f_value = vf.mul(lcb_f_value, lcb_f_gv, lcb_f_active)
        vf.store_align(dst + _m0, lcb_f_value, _tmp_maskreg_0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.jit(auto_mutex=True)
def liger_cross_entropy_loss_bwd_kernel(
    g: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    bt: pl.DT_INT32,
    vv: pl.DT_INT32,
    ng: pl.DT_INT32,
    nunit: pl.DT_INT32,
):
    g_lcb_xin = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 8192], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
        ),
        addrs=0x0,
        mutex_ids=[0, 1],
    )
    g_lcb_xout = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 8192], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
        ),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    g_lcb_gb = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
        ),
        addrs=0x20000,
        mutex_ids=[4, 5],
    )
    with pl.section_vector():
        _k0 = (vv + 8192 - 1)
        _k1 = (_k0 // 8192)
        _k2 = (bt * _k1)
        _k3 = (_k2 + pl.get_block_num() - 1)
        _k4 = (_k3 // pl.get_block_num())
        _k5 = (_k4 * pl.get_block_idx())
        for lcb_t in pl.range(0, (pl.max(0, (pl.min(_k4, (_k2 - _k5))))), 1):
            t1_lcb_gb = g_lcb_gb.next()
            pl.set_validshape(t1_lcb_gb, [1, 1])
            _k6 = (_k5 + lcb_t)
            _k7 = (_k6 // _k1)
            pl.load(t1_lcb_gb, g, [0, (pl.min(_k7, (ng - 1)))])
            t2_lcb_xin = g_lcb_xin.next()
            _k8 = (_k7 * _k1)
            _k9 = (_k6 - _k8)
            _k10 = (_k9 * 8192)
            _k11 = (vv - _k10)
            _k12 = (pl.min(8192, _k11))
            pl.set_validshape(t2_lcb_xin, [1, _k12])
            pl.load(t2_lcb_xin, x, [_k7, _k10])
            t3_lcb_xout = g_lcb_xout.next()
            scale_float_vf(t2_lcb_xin, t3_lcb_xout, t1_lcb_gb, _k12, ((_k12 + 64 - 1) // 64))
            pl.set_validshape(t3_lcb_xout, [1, _k12])
            pl.store(x, t3_lcb_xout, [_k7, _k10])


# ---------------------------------------------------------------------------
# host wrapper
# ---------------------------------------------------------------------------

def liger_cross_entropy_is_identity_grad(grad_output):
    """Upstream's short circuit: a gradient of exactly one leaves the input alone.

    `torch.equal` compares dtype as well as value, which is load bearing: a
    float32 scalar 1.0 against a half-precision saved buffer is *not* equal, so
    such a call falls through to the scaling path instead of silently skipping
    it. That is the upstream semantics, kept deliberately.
    """
    if not isinstance(grad_output, torch.Tensor):
        return float(grad_output) == 1.0
    return bool(torch.equal(grad_output,
                            torch.tensor(1.0, device=grad_output.device)))


def liger_cross_entropy_loss_bwd_wrapper(saved_input, grad_output):
    """Returns the gradient with respect to the logits.

    `saved_input` is the buffer the forward left dloss/dx in. Three paths, the
    same three the upstream triton implementation has, and which of them runs is
    decided by the shape of `grad_output` rather than by a flag:

    * **exactly 1.0** -- cross entropy was the last layer, so the forward
      already produced the finished gradient. Returns `saved_input` itself,
      untouched, with no launch.
    * **a [BT] vector** (`reduction="none"`) -- each row is scaled by its own
      element. Returns a *new* tensor, matching upstream, which reaches this
      case through an allocating `_input * grad_output.unsqueeze(1)`. This
      kernel only writes in place, so the copy is explicit: one extra pass over
      [BT, V] that the in-place path does not pay.
    * **a 0-dim scalar** (`reduction` of "mean" or "sum") -- every row is scaled
      by the same value, in place, and `saved_input` comes back.

    The kernel itself has no branches for any of this: `g` arrives as [1, ng]
    and row r reads element `min(r, ng - 1)`, which collapses to 0 when ng is 1
    and to r when ng is BT.
    """
    if saved_input.ndim != 2:
        raise ValueError(
            "saved_input must be [BT, V], got {}".format(tuple(saved_input.shape)))
    if saved_input.dtype != torch.float32:
        raise ValueError(
            "this kernel is the float32 variant, got {}".format(saved_input.dtype))
    rows, cols = int(saved_input.shape[0]), int(saved_input.shape[1])

    # path 1: the forward already finished the job.
    if liger_cross_entropy_is_identity_grad(grad_output):
        return saved_input

    if not isinstance(grad_output, torch.Tensor):
        grad_output = torch.tensor(grad_output, dtype=saved_input.dtype,
                                   device=saved_input.device)
    if grad_output.ndim > 1:
        raise ValueError(
            "grad_output must be a scalar or a [BT] vector, got {}".format(
                tuple(grad_output.shape)))
    if grad_output.ndim == 1 and grad_output.shape[0] != rows:
        raise ValueError(
            "a per-row grad_output must have {} elements, got {}".format(
                rows, grad_output.shape[0]))

    per_row = grad_output.ndim > 0
    if not per_row and not saved_input.is_contiguous():
        # The scalar path promises to write `saved_input` in place, and that
        # promise cannot be kept on a buffer the launch path has to copy.
        raise ValueError(
            "saved_input must be contiguous; call .contiguous() first")

    # ng is 1 for a 0-dim grad_output and BT for a per-row one. Widening a half
    # gradient to float32 here is exact, and float32 is the dtype the multiply
    # runs in anyway.
    ng = rows if per_row else 1
    g2d = grad_output.detach().to(torch.float32).contiguous().reshape(1, ng)
    # path 2 allocates, path 3 does not -- the upstream contract.
    x2d = saved_input.clone() if per_row else saved_input
    x2d = x2d.detach().contiguous().reshape(rows, cols)

    # Match the row-level task split to the current device's AIV capacity.
    block_dim = min(rows, get_platform_info().vector_core_num)
    liger_cross_entropy_loss_bwd_kernel[None, block_dim](g2d, x2d, rows, cols, ng, 1)
    return x2d.reshape(saved_input.shape)
