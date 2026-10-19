# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Grouped GEMM kernel for LLaDA2 MoE: all experts in a single kernel call.

Instead of a per-expert Python dispatch loop (overhead dominates for E=256),
this kernel uses pypto.loop over the expert dimension and pypto.loop_unroll over
the token dimension, processing all experts' SwiGLU FFN in one JIT'd kernel invocation.

Input layout (prepared by the host):
  sorted_tokens  [N_total, H]   BF16  — tokens pre-sorted by expert assignment
  w13_flat       [E*H, 2*I]     BF16  — all experts' gate||up weights, row-concatenated
  w2_flat        [E*I, H]       BF16  — all experts' down weights, row-concatenated
  expert_cumsum  [E+1]          INT32 — cumulative token counts; expert e owns
                                        tokens [cumsum[e], cumsum[e+1])
  result         [N_total, H]   BF16  — output buffer (same shape as sorted_tokens)

Non-tensor params:
  num_experts        (int) — number of experts E
  hidden_size        (int) — H
  intermediate_size  (int) — I
"""

import os
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph

import pypto


ND = pypto.TileOpFormat.TILEOP_ND


def _swiglu_silu(gate_up):
    half = gate_up.shape[1] // 2
    gate = pypto.view(gate_up, [gate_up.shape[0], half], [0, 0])
    up = pypto.view(gate_up, [gate_up.shape[0], half], [0, half])
    neg = pypto.mul(gate, -1.0)
    e = pypto.exp(neg)
    denom = pypto.add(e, 1.0)
    silu = pypto.div(gate, denom)
    return pypto.mul(silu, up)


_CUBE_NBUF = int(os.environ.get("PYPTO_CUBE_NBUFFER", "2"))
_VEC_NBUF = int(os.environ.get("PYPTO_VEC_NBUFFER", "2"))
_MM1_M = int(os.environ.get("PYPTO_MM1_M", "8"))
_MM1_K = int(os.environ.get("PYPTO_MM1_K", "128"))
_MM1_N = int(os.environ.get("PYPTO_MM1_N", "256"))
_MM2_M = int(os.environ.get("PYPTO_MM2_M", "8"))
_MM2_K = int(os.environ.get("PYPTO_MM2_K", "128"))
_MM2_N = int(os.environ.get("PYPTO_MM2_N", "256"))
_VEC_FIRST = int(os.environ.get("PYPTO_VEC_FIRST", "13"))


@pypto.frontend.jit(
    runtime_options={"device_sched_mode": 1,
                     "stitch_function_max_num": 64,
                     # expert_cumsum read on host for per-expert slicing (required under CANN 9.1.0)
                     "ready_on_host_tensors": ["expert_cumsum"]},
    pass_options={
        "cube_l1_reuse_setting": {-1: 2},
        "cube_nbuffer_setting": {-1: _CUBE_NBUF},
        "vec_nbuffer_setting": {-1: _VEC_NBUF},
    },
)
def llada2_moe_grouped_gemm_kernel(
    sorted_tokens: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16, format=ND),
    w13_flat: pypto.Tensor([], pypto.DT_BF16, format=ND),    # [E*H, 2*I]
    w2_flat: pypto.Tensor([], pypto.DT_BF16, format=ND),    # [E*I, H]
    expert_cumsum: pypto.Tensor([], pypto.DT_INT32, format=ND),   # [E+1]
    result: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16, format=ND),
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
):
    two_i = intermediate_size * 2

    pypto.experimental.set_operation_options(combine_axis=True)

    mm1_cube = (_MM1_M, _MM1_K, _MM1_N)
    mm2_cube = (_MM2_M, _MM2_K, _MM2_N)
    vec_first = _VEC_FIRST

    for e_idx in pypto.loop(0, num_experts, 1,
                            name="EXPERT_LOOP", idx_name="e_idx"):
        e_start = expert_cumsum[e_idx]
        e_end = expert_cumsum[e_idx + 1]
        n_e = e_end - e_start

        for tok_idx, tile_batch in pypto.loop_unroll(
            0, n_e, 1,
            name="LOOP_TOKEN",
            idx_name="tok_idx",
            unroll_list=[1, 2, 4, 8, 16, 32, 64],
        ):
            # Slice this expert's weights from the flattened tensors
            # (inside the unroll loop so tile shapes are available)
            pypto.set_cube_tile_shapes(
                [tile_batch, tile_batch],
                [mm1_cube[1], mm1_cube[1] * 2],
                [mm1_cube[2], mm1_cube[2]],
                True,
            )

            w13_e = pypto.view(w13_flat, [hidden_size, two_i],
                               [e_idx * hidden_size, 0])
            w2_e = pypto.view(w2_flat, [intermediate_size, hidden_size],
                              [e_idx * intermediate_size, 0])
            tile_x = pypto.view(
                sorted_tokens,
                [tile_batch, hidden_size],
                [e_start + tok_idx, 0],
            )

            gate_up = pypto.matmul(tile_x, w13_e, pypto.DT_FP32)

            # SwiGLU activation in FP32, cast I-wide result to BF16
            pypto.set_vec_tile_shapes(min(tile_batch, vec_first),
                                       intermediate_size)
            sw_fp32 = _swiglu_silu(gate_up)
            sw = pypto.cast(sw_fp32, pypto.DT_BF16)

            pypto.set_cube_tile_shapes(
                [tile_batch, tile_batch],
                [mm2_cube[1], mm2_cube[1] * 2],
                [mm2_cube[2], mm2_cube[2]],
                False,
            )
            down = pypto.matmul(sw, w2_e, pypto.DT_FP32)
            pypto.set_vec_tile_shapes(min(tile_batch, vec_first), hidden_size)
            out = pypto.cast(down, pypto.DT_BF16)

            pypto.assemble(out, [e_start + tok_idx, 0], result)


def _check(sorted_tokens, w13_flat, w2_flat, expert_cumsum, result,
           num_experts, hidden_size, intermediate_size):
    assert sorted_tokens.dim() == 2 and sorted_tokens.dtype == torch.bfloat16
    assert w13_flat.dim() == 2 and w13_flat.dtype == torch.bfloat16
    assert w2_flat.dim() == 2 and w2_flat.dtype == torch.bfloat16
    assert expert_cumsum.dim() == 1 and expert_cumsum.dtype == torch.int32
    N, H = sorted_tokens.shape
    assert H == hidden_size
    assert w13_flat.shape == (num_experts * hidden_size, 2 * intermediate_size)
    assert w2_flat.shape == (num_experts * intermediate_size, hidden_size)
    assert expert_cumsum.shape[0] == num_experts + 1
    assert result.shape == sorted_tokens.shape


@allow_in_graph
def llada2_moe_grouped_gemm(
    sorted_tokens: torch.Tensor,
    w13_flat: torch.Tensor,
    w2_flat: torch.Tensor,
    expert_cumsum: torch.Tensor,
    result: torch.Tensor,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    """Grouped GEMM for all MoE experts in a single kernel call.

    Parameters
    ----------
    sorted_tokens : [N_total, H] BF16 — tokens pre-sorted by expert
    w13_flat : [E*H, 2*I] BF16 — flattened gate||up weights
    w2_flat : [E*I, H] BF16 — flattened down weights
    expert_cumsum : [E+1] INT32 — cumulative token counts per expert
    result : [N_total, H] BF16 — output buffer
    num_experts : number of experts
    hidden_size : hidden dimension H
    intermediate_size : intermediate dimension I
    """
    if isinstance(sorted_tokens, FakeTensor):
        return
    _check(sorted_tokens, w13_flat, w2_flat, expert_cumsum, result,
           num_experts, hidden_size, intermediate_size)
    llada2_moe_grouped_gemm_kernel(
        sorted_tokens, w13_flat, w2_flat, expert_cumsum, result,
        num_experts, hidden_size, intermediate_size,
    )
