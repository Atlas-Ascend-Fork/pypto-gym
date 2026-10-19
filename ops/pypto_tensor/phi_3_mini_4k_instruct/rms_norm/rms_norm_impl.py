# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------



"""PyPTO rms_norm kernel implementation for Phi-3-mini-4k-instruct (D=3072)."""

import pypto
import torch
from torch._dynamo import allow_in_graph
from torch._subclasses.fake_tensor import FakeTensor


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU, "stitch_function_max_num": 256})
def _rms_norm_kernel_d3072(
    x: pypto.Tensor([pypto.DYNAMIC, 3072], pypto.DT_FP16),
    gamma: pypto.Tensor([3072], pypto.DT_FP16),
    out: pypto.Tensor([pypto.DYNAMIC, 3072], pypto.DT_FP16),
):
    D = 3072
    TILE_M = 4
    M = x.shape[0]

    inv_d = 1.0 / D

    pypto.set_vec_tile_shapes(1, D)
    total_steps = (M + TILE_M - 1) // TILE_M
    pypto.set_vec_tile_shapes(TILE_M, D)

    for m_idx in pypto.loop(total_steps, name="m_loop", unroll_list=[4, 2, 1]):
        pypto.set_pass_options(sg_set_scope=1)
        gamma_2d = pypto.reshape(gamma, [1, D])
        gamma_fp32 = pypto.cast(gamma_2d, pypto.DT_FP32)
        m_offset = m_idx * TILE_M
        remaining = M - m_offset
        actual_m = remaining.min(TILE_M)

        x_tile = pypto.view(
            x, [TILE_M, D], [m_offset, 0],
            valid_shape=[actual_m, D],
        )

        x_fp32 = pypto.cast(x_tile, pypto.DT_FP32)
        x2 = pypto.mul(x_fp32, x_fp32)
        mean = pypto.mul(x2, inv_d)
        mean_sum = pypto.sum(mean, -1, keepdim=True)
        eps_add = pypto.add(mean_sum, 1e-5)
        inv_rms = pypto.rsqrt(eps_add)
        norm = pypto.mul(x_fp32, inv_rms)
        weighted = pypto.mul(norm, gamma_fp32)
        result = pypto.cast(weighted, pypto.DT_FP16)

        pypto.assemble(result, [m_offset, 0], out)
        pypto.set_pass_options(sg_set_scope=-1)


def _reshape_to_2d(hidden_states: torch.Tensor) -> torch.Tensor:
    shape = hidden_states.shape
    if hidden_states.dim() == 2:
        return hidden_states.contiguous()
    M = 1
    for s in shape[:-1]:
        M *= s
    D = shape[-1]
    return hidden_states.reshape(M, D).contiguous()


@allow_in_graph
def rms_norm_wrapper(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if isinstance(hidden_states, FakeTensor):
        return torch.empty_like(hidden_states)

    x_2d = _reshape_to_2d(hidden_states)
    out_2d = torch.empty_like(x_2d)
    D = x_2d.shape[-1]

    if D != 3072:
        raise ValueError(f"Unsupported hidden_size: {D}, expected 3072")

    _rms_norm_kernel_d3072(x_2d, weight, out_2d)

    return out_2d.reshape(hidden_states.shape)


def rms_norm_pto(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return rms_norm_wrapper(hidden_states, weight, eps)


pyptolib = torch.library.Library("pypto", "FRAGMENT")  # type: ignore[arg-type]
pyptolib.define("rms_norm_phi3(Tensor hidden_states, Tensor weight) -> Tensor")


@torch.library.impl(pyptolib, "rms_norm_phi3", "Meta")  # type: ignore[arg-type]
def rms_norm_phi3_meta(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(hidden_states)


@torch.library.impl(pyptolib, "rms_norm_phi3", "NPU")  # type: ignore[arg-type]
def rms_norm_phi3_npu(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return rms_norm_wrapper(hidden_states, weight)


def rms_norm_pypto(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.pypto.rms_norm_phi3(hidden_states, weight)
