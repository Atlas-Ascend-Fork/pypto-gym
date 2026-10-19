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
"""apply_adam_w_v2 PyPTO implementation.

Single-step AdamW update with bias correction.
  m_t    = beta1 * m + (1 - beta1) * grad
  v_t    = beta2 * v + (1 - beta2) * grad * grad
  m_hat  = m_t / (1 - beta1**step)
  v_hat  = v_t / (1 - beta2**step)
  update = m_hat / (sqrt(v_hat) + eps) + weight_decay * weight
  w_new  = weight - lr * update

The implementation provides fp32 and bf16 weight/grad paths. Momentum and
variance tensors are always fp32. General kernels cover dynamic M/K 2D tensors,
while the wrapper selects tile configuration for [7168, K] style network
shapes without duplicating kernel bodies.
"""
from typing import NamedTuple, Tuple
                                                                                      
import pypto
import torch

N_TILE = 1024
M_TILE = 32
VEC_TILE_N = 512
M_TILE_LARGE = 7168
VEC_TILE_M_LARGE = 16
DEFAULT_TILE_CONFIG = [M_TILE, M_TILE, VEC_TILE_N]
LARGE_M_TILE_CONFIG = [M_TILE_LARGE, VEC_TILE_M_LARGE, VEC_TILE_N]
TARGET_M = 7168
TARGET_N_TILE = 4096
TARGET_VEC_TILE_M = 16
TARGET_VEC_TILE_N = 512
TARGET_K_AXIS_TILE_CONFIG = [TARGET_M, TARGET_VEC_TILE_M, TARGET_VEC_TILE_N, TARGET_N_TILE]


class AdamTileMeta(NamedTuple):
    m_dim: int
    n_dim: int
    m_tile: int
    n_tile: int
    m_loops: int
    n_loops: int


def _adam_tile_meta(weight: pypto.Tensor, tile_config: list) -> AdamTileMeta:
    m_dim = weight.shape[0]
    n_dim = weight.shape[1]
    m_tile = tile_config[0]
    pypto.set_vec_tile_shapes(tile_config[1], tile_config[2])
    m_loops = pypto.ceildiv(m_dim, m_tile)
    n_tile = tile_config[3] if len(tile_config) > 3 else N_TILE
    n_loops = pypto.ceildiv(n_dim, n_tile)
    return AdamTileMeta(m_dim, n_dim, m_tile, n_tile, m_loops, n_loops)


def _select_adam_tile_config(weight: torch.Tensor) -> list[int]:
    if weight.dtype == torch.bfloat16 and weight.shape[0] == TARGET_M and weight.shape[1] > 4096:
        return TARGET_K_AXIS_TILE_CONFIG
    if weight.shape[0] >= M_TILE_LARGE and weight.shape[1] <= 4096:
        return LARGE_M_TILE_CONFIG
    return DEFAULT_TILE_CONFIG


# ---------------------------------------------------------------------------
# fp32 path: weight/grad are fp32
# ---------------------------------------------------------------------------
@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
    }
)
def apply_adam_w_v2_kernel_fp32(
    weight: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    grad: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    m: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    v: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    weight_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    m_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    v_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    beta1: float,
    one_m_b1: float,
    beta2: float,
    one_m_b2: float,
    bc1: float,
    bc2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    tile_config: list,
):
    meta = _adam_tile_meta(weight, tile_config)

    for m_idx in pypto.loop(meta.m_loops, name="adamw_m_loop_fp32", idx_name="m_idx", unroll_list=[1]):
        m_offset = m_idx * meta.m_tile
        valid_m = (meta.m_dim - m_offset).min(meta.m_tile)
        for n_idx in pypto.loop(meta.n_loops, name="adamw_n_loop_fp32", idx_name="n_idx", unroll_list=[1]):
            n_offset = n_idx * meta.n_tile
            valid_n = (meta.n_dim - n_offset).min(meta.n_tile)
            valid_shape = [valid_m, valid_n]

            w_tile = pypto.view(weight, [meta.m_tile, meta.n_tile], [m_offset, n_offset], valid_shape=valid_shape)
            g_tile = pypto.view(grad, [meta.m_tile, meta.n_tile], [m_offset, n_offset], valid_shape=valid_shape)
            m_state = pypto.view(m, [meta.m_tile, meta.n_tile], [m_offset, n_offset], valid_shape=valid_shape)
            v_state = pypto.view(v, [meta.m_tile, meta.n_tile], [m_offset, n_offset], valid_shape=valid_shape)

            m_new = pypto.add(pypto.mul(m_state, beta1), pypto.mul(g_tile, one_m_b1))
            grad_sq = pypto.mul(g_tile, g_tile)
            v_new = pypto.add(pypto.mul(v_state, beta2), pypto.mul(grad_sq, one_m_b2))
            m_hat = pypto.div(m_new, bc1, precision_type=pypto.PrecisionType.INTRINSIC)
            v_hat = pypto.div(v_new, bc2, precision_type=pypto.PrecisionType.INTRINSIC)
            denom = pypto.add(pypto.sqrt(v_hat), eps)
            update = pypto.add(pypto.div(m_hat, denom, precision_type=pypto.PrecisionType.INTRINSIC),
                               pypto.mul(w_tile, weight_decay))
            w_new = pypto.sub(w_tile, pypto.mul(update, lr))

            pypto.assemble(w_new, [m_offset, n_offset], weight_out)
            pypto.assemble(m_new, [m_offset, n_offset], m_out)
            pypto.assemble(v_new, [m_offset, n_offset], v_out)


# ---------------------------------------------------------------------------
# bf16 path: weight/grad are bf16; m, v stay fp32; intermediate math fp32.
# ---------------------------------------------------------------------------
@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "device_sched_mode": 1,
    },
    pass_options={"vec_nbuffer_setting": {-1: 4, -2: 1}},
)
def apply_adam_w_v2_kernel_bf16(
    weight: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    grad: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    m: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    v: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    weight_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    m_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    v_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    beta1_bf16: float,
    one_m_beta1_bf16: float,
    beta2_bf16: float,
    one_m_beta2_bf16: float,
    bc1_bf16: float,
    bc2_bf16: float,
    lr_bf16: float,
    weight_decay_bf16: float,
    eps_bf16: float,
    bf16_tile_config: list,
):
    meta = _adam_tile_meta(weight, bf16_tile_config)
    if len(bf16_tile_config) > 3:
        pypto.experimental.set_operation_options(combine_axis=True)
        pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 4})

    for m_idx in pypto.loop(meta.m_loops, name="adamw_m_loop_bf16", idx_name="m_idx", unroll_list=[1]):
        m_offset = m_idx * meta.m_tile
        valid_m = (meta.m_dim - m_offset).min(meta.m_tile)
        for n_idx in pypto.loop(meta.n_loops, name="adamw_n_loop_bf16", idx_name="n_idx", unroll_list=[1]):
            n_offset = n_idx * meta.n_tile
            valid_n = (meta.n_dim - n_offset).min(meta.n_tile)
            valid_shape = [valid_m, valid_n]

            w_tile = pypto.view(weight, [meta.m_tile, meta.n_tile], [m_offset, n_offset], valid_shape=valid_shape)
            g_tile = pypto.view(grad, [meta.m_tile, meta.n_tile], [m_offset, n_offset], valid_shape=valid_shape)
            m_state = pypto.view(m, [meta.m_tile, meta.n_tile], [m_offset, n_offset], valid_shape=valid_shape)
            v_state = pypto.view(v, [meta.m_tile, meta.n_tile], [m_offset, n_offset], valid_shape=valid_shape)

            w_f32 = pypto.cast(w_tile, pypto.DT_FP32)
            g_f32 = pypto.cast(g_tile, pypto.DT_FP32)
            m_new = pypto.add(pypto.mul(m_state, beta1_bf16), pypto.mul(g_f32, one_m_beta1_bf16))
            grad_sq = pypto.mul(g_f32, g_f32)
            v_new = pypto.add(pypto.mul(v_state, beta2_bf16), pypto.mul(grad_sq, one_m_beta2_bf16))
            m_hat = pypto.div(m_new, bc1_bf16, precision_type=pypto.PrecisionType.INTRINSIC)
            v_hat = pypto.div(v_new, bc2_bf16, precision_type=pypto.PrecisionType.INTRINSIC)
            denom = pypto.add(pypto.sqrt(v_hat), eps_bf16)
            update = pypto.add(pypto.div(m_hat, denom, precision_type=pypto.PrecisionType.INTRINSIC),
                               pypto.mul(w_f32, weight_decay_bf16))
            w_new_f32 = pypto.sub(w_f32, pypto.mul(update, lr_bf16))
            w_new_bf16 = pypto.cast(w_new_f32, pypto.DT_BF16)

            pypto.assemble(w_new_bf16, [m_offset, n_offset], weight_out)
            pypto.assemble(m_new, [m_offset, n_offset], m_out)
            pypto.assemble(v_new, [m_offset, n_offset], v_out)


def _parse_adam_args(args: tuple, kwargs: dict) -> tuple[float, float, float, float, float, int]:
    names = ["beta1", "beta2", "lr", "weight_decay", "eps", "step"]
    if len(args) > len(names):
        raise TypeError("too many positional arguments")
    values = dict(zip(names, args))
    for name in names:
        if name in kwargs:
            values[name] = kwargs.pop(name)
    if kwargs:
        raise TypeError(f"unexpected keyword argument(s): {sorted(kwargs)}")
    missing = [name for name in names if name not in values]
    if missing:
        raise TypeError(f"missing required argument(s): {missing}")
    return tuple(values[name] for name in names)


# ---------------------------------------------------------------------------
# Host wrapper: precompute scalars, dispatch by dtype, allocate outputs.
# ---------------------------------------------------------------------------
def apply_adam_w_v2_wrapper(
    weight: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beta1, beta2, lr, weight_decay, eps, step = _parse_adam_args(args, kwargs)

    assert weight.shape == grad.shape == m.shape == v.shape, "shape mismatch"
    assert weight.dim() == 2, "apply_adam_w_v2 currently supports 2D tensors only"
    assert weight.dtype == grad.dtype, "weight/grad dtype must match"
    assert m.dtype == torch.float32 and v.dtype == torch.float32
    assert isinstance(step, int) and step >= 1
    if weight.numel() == 0:
        return torch.empty_like(weight), torch.empty_like(m), torch.empty_like(v)

    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)
    one_m_b1 = 1.0 - beta1
    one_m_b2 = 1.0 - beta2

    weight = weight.contiguous()
    grad = grad.contiguous()
    m = m.contiguous()
    v = v.contiguous()
    weight_out = torch.empty_like(weight)
    m_out = torch.empty_like(m)
    v_out = torch.empty_like(v)

    if weight.dtype == torch.bfloat16:
        kernel = apply_adam_w_v2_kernel_bf16
    elif weight.dtype == torch.float32:
        kernel = apply_adam_w_v2_kernel_fp32
    else:
        raise TypeError(f"unsupported weight dtype {weight.dtype}; expected bf16 or fp32")
    tile_config = _select_adam_tile_config(weight)

    kernel(
        weight, grad, m, v,
        weight_out, m_out, v_out,
        float(beta1), float(one_m_b1),
        float(beta2), float(one_m_b2),
        float(bc1), float(bc2),
        float(lr), float(weight_decay), float(eps),
        list(tile_config),
    )

    return weight_out, m_out, v_out
