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
"""
Lightning Indexer Prolog Quantization Module

This module implements the Lightning Indexer Prolog quantization computation
for DeepSeek serie models. It handles:
- Query computation with dynamic quantization
- Weight computation for indexer attention

Main Functions:
    - lightning_indexer_prolog_compute: Main computation function

Example:
    See test_lightning_indexer_prolog_quant.py for usage examples.
"""

import math
from dataclasses import dataclass
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import pypto


from deepseek_v4.common import inverse_rope_3d, quant_tensor


@dataclass
class IndexerPrologQuantConfig:
    unroll_list: list


def check_input_shape_dtype(
    qr, idx_wq_b, x, weights_proj, cos, sin, hadamard, qr_scale, idx_wq_b_scale
):
    q_lora_rank = 1024
    idx_nq = 64
    head_dim = 128
    rope_dim = 64
    h = 4096
    assert (
        len(qr.shape) == 2 and qr.size(1) == q_lora_rank
    ), f"qr shape need to be: (t, f{q_lora_rank}), but got: f{qr.shape}"
    assert (
        len(idx_wq_b.shape) == 2
        and idx_wq_b.size(0) == q_lora_rank
        and idx_wq_b.size(1) == idx_nq * head_dim
    ), f"idx_wq_b shape need to be: (f{q_lora_rank, idx_nq * head_dim}), but got: f{idx_wq_b.shape}"
    assert (
        len(x.shape) == 2 and x.size(1) == h
    ), f"x shape need to be: (t, f{h}), but got: f{x.shape}"
    assert (
        len(weights_proj.shape) == 2
        and weights_proj.size(0) == h
        and weights_proj.size(1) == idx_nq
    ), f"weights_proj shape need to be: (f{h, idx_nq}), but got: f{weights_proj.shape}"
    assert (
        len(cos.shape) == 2 and cos.size(1) == rope_dim
    ), f"cos shape need to be: (t, f{rope_dim}), but got: f{cos.shape}"
    assert (
        len(sin.shape) == 2 and sin.size(1) == rope_dim
    ), f"sin shape need to be: (t, f{rope_dim}), but got: f{sin.shape}"
    assert (
        len(hadamard.shape) == 2
        and hadamard.size(0) == head_dim
        and hadamard.size(1) == head_dim
    ), f"hadamard shape need to be: (f{head_dim, head_dim}), but got: f{hadamard.shape}"
    assert (
        len(qr_scale.shape) == 2 and qr_scale.size(1) == 1
    ), f"qr_scale shape need to be: (t, f{1}), but got: f{qr_scale.shape}"
    assert (
        len(idx_wq_b_scale.shape) == 2
        and idx_wq_b_scale.size(0) == idx_nq * head_dim
        and idx_wq_b_scale.size(1) == 1
    ), f"idx_wq_b_scale shape need to be: (f{idx_nq * head_dim, 1}), but got: f{idx_wq_b_scale.shape}"

    assert (
        qr.dtype == idx_wq_b.dtype == torch.int8
    ), f"expected qr and idx_wq_b dtype to be torch.int8, but got: f{qr.dtype} and f{idx_wq_b.dtype}"
    assert (
        x.dtype
        == weights_proj.dtype
        == cos.dtype
        == sin.dtype
        == hadamard.dtype
        == torch.bfloat16
    ), (
        f"expected x, weights_proj, cos, sin and hadamard dtype to be torch.bfloat16 but got: "
        f"f{x.dtype}, f{weights_proj.dtype}, f{cos.dtype}, f{sin.dtype} and f{hadamard.dtype}"
    )
    assert (
        qr_scale.dtype == idx_wq_b_scale.dtype == torch.float32
    ), (
        f"expected qr_scale and idx_wq_b_scale dtype to be torch.float32, but got: "
        f"f{qr_scale.dtype} and f{idx_wq_b_scale.dtype}"
    )


@allow_in_graph
def npu_quant_lightning_indexer_prolog(
    qr: torch.Tensor,
    idx_wq_b: torch.Tensor,
    x: torch.Tensor,
    weights_proj: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    hadamard: torch.Tensor,
    qr_scale: torch.Tensor,
    idx_wq_b_scale: torch.Tensor,
):
    """
    torch npu graph interface

    """

    q = torch.empty(
        [qr.size(0), weights_proj.size(1), hadamard.size(0)],
        dtype=qr.dtype,
        device=qr.device,
    )
    weights = torch.empty(
        [qr.size(0), weights_proj.size(1)],
        dtype=torch.float16,
        device=weights_proj.device,
    )
    q_scale = torch.empty(
        [qr.size(0), weights_proj.size(1)],
        dtype=torch.float16,
        device=weights_proj.device,
    )

    check_input_shape_dtype(
        qr, idx_wq_b, x, weights_proj, cos, sin, hadamard, qr_scale, idx_wq_b_scale
    )

    # tiling
    tile_config = IndexerPrologQuantConfig(unroll_list=[128, 64, 32, 16, 8, 1])

    # kernel
    if not isinstance(qr, FakeTensor):
        inputs = [qr, idx_wq_b, x, weights_proj, cos, sin, hadamard, qr_scale, idx_wq_b_scale, q, weights, q_scale]
        quant_lightning_indexer_prolog_kernel(*inputs, tile_config)

    return q, weights, q_scale


@pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {-1: 2, 1: 1},
        "vec_nbuffer_setting": {0: 2},
    },
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1
    },
)
def quant_lightning_indexer_prolog_kernel(
    qr: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT8),
    idx_wq_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8, format=pypto.TileOpFormat.TILEOP_NZ),
    x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    weights_proj: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_NZ),
    cos: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    hadamard: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    qr_scale: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    idx_wq_b_scale: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    q: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    weights: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
    q_scale: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
    tile_config
):
    """JIT-compiled wrapper for Lightning Indexer Prolog Quantization computation.

    This is the main entry point for the Lightning Indexer Prolog Quantization operator.
    It sets up optimization passes and runtime options before calling the core
    computation function in JIT decorator.

    Args:
        group       name           dtype     shape                               format
        INPUT 0	    qr	           DT_INT8	 (t, q_lora_rank)                    ND
        INPUT 1	    idx_wq_b	   DT_INT8	 (q_lora_rank, idx_nq * head_dim)	 ND
        INPUT 2	    x	           DT_BF16	 (t, h)	                             ND
        INPUT 3	    weights_proj   DT_BF16	 (h, idx_nq)	                     ND
        INPUT 4	    cos	           DT_BF16	 (t, rope_dim)                       ND
        INPUT 5	    sin	           DT_BF16	 (t, rope_dim)                       ND
        INPUT 6     hadamard       DT_BF16   (head_dim, head_dim)                ND
        INPUT 7     qr_scale       DT_FP32   (t, 1)                              ND
        INPUT 8     idx_wq_b_scale DT_FP32   (idx_nq * head_dim, 1)              ND
        OUTPUT 0	q              DT_INT8	 (t, idx_nq * head_dim)	             ND
        OUTPUT 1    weights        DT_FP16   (t, idx_nq)                         ND
        OUTPUT 2    q_scale        DT_FP16   (t, idx_nq)                         ND
        CONFIGS     tile_config    /          /                                  /
    Note:
        This function is decorated with @pypto.frontend.jit for JIT compilation.
        It configures pass options for memory optimization and calls the core
        computation function.
    """

    idx_wq_b.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    weights_proj.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    hadamard.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    x_dtype = x.dtype
    # dynamic axis
    t = qr.shape[0]
    # static axes
    q_lora_rank = qr.shape[1]
    h = x.shape[1]
    idx_nq = weights_proj.shape[1]
    head_dim = hadamard.shape[0]
    rope_dim = cos.shape[1]

    # Reshape inplace will not generate data move
    w_qb_scale = pypto.reshape(idx_wq_b_scale, [1, idx_nq * head_dim], inplace=True)
    hadamard_q = pypto.reshape(hadamard, [1, head_dim, head_dim], inplace=True)

    unroll_list = tile_config.unroll_list
    for t_idx, unrollLength in pypto.loop_unroll(
        0,
        t,
        1,
        name="IndexerPrologLoop",
        idx_name="t_idx",
        unroll_list=unroll_list,
    ):
        # use for perf optimization
        pypto.experimental.set_operation_options(combine_axis=True)
        t_tile = unrollLength
        qr_in = pypto.view(qr, [t_tile, q_lora_rank], [t_idx, 0])
        qs_in = pypto.view(qr_scale, [t_tile, 1], [t_idx, 0])
        pypto.set_semantic_label("Query-Linear")
        # (t_tile, q_lora_rank) @ (q_lora_rank, idx_nq * head_dim) --> (t_tile, idx_nq * head_dim)
        pypto.set_cube_tile_shapes(
            [128, 128], [256, 1024], [256, 256]
        )
        q_s32 = pypto.matmul(qr_in, idx_wq_b, pypto.DT_INT32)

        pypto.set_semantic_label("Query-Dequant")
        pypto.set_vec_tile_shapes(1, idx_nq * head_dim)
        q_f32 = pypto.cast(q_s32, pypto.DT_FP32)
        # (t_tile, idx_nq * head_dim), fp32, last dim brc
        q_f32 = q_f32 * qs_in
        # (t_tile, idx_nq * head_dim), fp32, first dim brc
        q_f32 = q_f32 * w_qb_scale
        q_cast = pypto.cast(q_f32, x_dtype)
        q_re = pypto.reshape(q_cast, [t_tile, idx_nq, head_dim])

        # UB view
        q_nope = pypto.view(q_re, [t_tile, idx_nq, head_dim - rope_dim], [0, 0, 0])
        q_rope = pypto.view(
            q_re, [t_tile, idx_nq, rope_dim], [0, 0, head_dim - rope_dim]
        )

        rope_cos = pypto.view(cos, [t_tile, rope_dim], [t_idx, 0])
        rope_sin = pypto.view(sin, [t_tile, rope_dim], [t_idx, 0])

        q_roped = inverse_rope_3d(q_rope, rope_cos, rope_sin)

        pypto.set_vec_tile_shapes(1, idx_nq, head_dim)
        q_assemble = pypto.tensor([t_tile, idx_nq, head_dim], x_dtype, "q_assemble")
        pypto.assemble(pypto.clone(q_nope), [0, 0, 0], q_assemble)
        pypto.assemble(q_roped, [0, 0, head_dim - rope_dim], q_assemble)

        pypto.set_semantic_label("Hadamard-Compute")
        # (t_tile, idx_nq, head_dim) @ (1, head_dim, head_dim) -> (t_tile, idx_nq, head_dim)
        pypto.set_cube_tile_shapes(
            [idx_nq, idx_nq], [head_dim, head_dim], [head_dim, head_dim]
        )
        q_hadamard = pypto.matmul(
            q_assemble, hadamard_q, x_dtype
        )  # (t_tile, idx_nq, head_dim)
        pypto.set_vec_tile_shapes(1, idx_nq, head_dim)
        q_res, q_scale_res = quant_tensor(q_hadamard)
        q_scale_out = pypto.reshape(q_scale_res, [t_tile, idx_nq])
        pypto.set_vec_tile_shapes(t_tile, idx_nq)
        q_scale_cast = pypto.cast(q_scale_out, pypto.DT_FP16)

        pypto.assemble(q_res, [t_idx, 0, 0], q)
        pypto.assemble(q_scale_cast, [t_idx, 0], q_scale)

        pypto.set_semantic_label("Weight-Compute")
        x_in = pypto.view(x, [t_tile, h], [t_idx, 0])
        # (t_tile, h) @ (h, idx_nq) --> (t_tile, idx_nq)
        pypto.set_cube_tile_shapes(
            [32, 64],
            [h // 4, h],
            [idx_nq // 4, idx_nq // 4],
        )
        pypto.set_vec_tile_shapes(t_tile, idx_nq)
        weights_fp32 = pypto.cast(
            pypto.matmul(x_in, weights_proj, x_dtype), pypto.DT_FP32
        )
        weights_mul = pypto.mul(
            weights_fp32, 1.0 / (math.sqrt(idx_nq) * math.sqrt(head_dim))
        )
        weights_fp16 = pypto.cast(weights_mul, pypto.DT_FP16)
        pypto.assemble(weights_fp16, [t_idx, 0], weights)


pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define(
    "quant_lightning_indexer_prolog("
    "Tensor qr, Tensor idx_wq_b, Tensor x, Tensor weights_proj, "
    "Tensor cos, Tensor sin, Tensor hadamard, "
    "Tensor qr_scale, Tensor idx_wq_b_scale"
    ") -> (Tensor, Tensor, Tensor)"
)


@torch.library.impl(pyptolib, "quant_lightning_indexer_prolog", "Meta")
def quant_lightning_indexer_prolog(
    qr, idx_wq_b, x, weights_proj, cos, sin, hadamard, qr_scale, idx_wq_b_scale
):
    q = torch.empty(
        [qr.size(0), weights_proj.size(1), hadamard.size(0)],
        dtype=qr.dtype,
        device=qr.device,
    )
    weights = torch.empty(
        [qr.size(0), hadamard.size(0)],
        dtype=torch.float16,
        device=weights_proj.device,
    )
    q_scale = torch.empty(
        [qr.size(0), weights_proj.size(1)],
        dtype=torch.float16,
        device=weights_proj.device,
    )
    return q, weights, q_scale


try:
    @torch.library.impl(pyptolib, "quant_lightning_indexer_prolog", "NPU")
    def quant_lightning_indexer_prolog(
        qr, idx_wq_b, x, weights_proj, cos, sin, hadamard, qr_scale, idx_wq_b_scale
    ):
        return npu_quant_lightning_indexer_prolog(
            qr, idx_wq_b, x, weights_proj, cos, sin, hadamard, qr_scale, idx_wq_b_scale
        )
except Exception as e:
    import logging
    if "could not parse dispatch key: NPU" in str(e):
        logging.warning("Skip: torchair not installed, skip NPU registration for operator 'IP'")
    else:
        logging.warning(f"Skip: Unexpected error: {e}")