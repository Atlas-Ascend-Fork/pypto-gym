# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""PyPTO implementation of QkvRmsNormRopeCache.

This module intentionally lives under ``pypto/custom`` and does not depend on
the AscendC implementation.  The exported ``qkv_rms_norm_rope_cache_wrapper``
matches the public AscendC argument order and returns the three required output
tensors.  The current target implementation supports the INT8 quantized PA_NZ
cache branch used by the two network cases; the Python wrapper only validates
arguments and dispatches.
"""

import collections
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import pypto
import torch


@dataclass(frozen=True)
class QkvTileConfig:
    q_group_heads: int
    q_vec_token_tile: int
    kv_vec_token_tile: int
    stitch_function_max_num: int
    device_sched_mode: int


TP4_TILE_CONFIG = QkvTileConfig(
    q_group_heads=16,
    q_vec_token_tile=4,
    kv_vec_token_tile=2,
    stitch_function_max_num=128,
    device_sched_mode=3,
)
TP1_TILE_CONFIG = QkvTileConfig(
    q_group_heads=16,
    q_vec_token_tile=4,
    kv_vec_token_tile=4,
    stitch_function_max_num=128,
    device_sched_mode=3,
)
GENERIC_TILE_CONFIG = QkvTileConfig(
    q_group_heads=16,
    q_vec_token_tile=2,
    kv_vec_token_tile=2,
    stitch_function_max_num=128,
    device_sched_mode=3,
)

TP4_Q_GROUP_HEADS = TP4_TILE_CONFIG.q_group_heads
TP4_Q_VEC_TOKEN_TILE = TP4_TILE_CONFIG.q_vec_token_tile
TP4_KV_VEC_TOKEN_TILE = TP4_TILE_CONFIG.kv_vec_token_tile
TP1_Q_GROUP_HEADS = TP1_TILE_CONFIG.q_group_heads
TP1_Q_VEC_TOKEN_TILE = TP1_TILE_CONFIG.q_vec_token_tile
TP1_KV_VEC_TOKEN_TILE = TP1_TILE_CONFIG.kv_vec_token_tile
GENERIC_Q_GROUP_HEADS = GENERIC_TILE_CONFIG.q_group_heads
GENERIC_Q_VEC_TOKEN_TILE = GENERIC_TILE_CONFIG.q_vec_token_tile
GENERIC_KV_VEC_TOKEN_TILE = GENERIC_TILE_CONFIG.kv_vec_token_tile
GENERIC_KV_GROUP_HEADS = 2
TP4_KERNEL_TILE_CONFIG = [TP4_Q_GROUP_HEADS, TP4_Q_VEC_TOKEN_TILE, TP4_KV_VEC_TOKEN_TILE, GENERIC_KV_GROUP_HEADS, 0]
TP1_KERNEL_TILE_CONFIG = [TP1_Q_GROUP_HEADS, TP1_Q_VEC_TOKEN_TILE, TP1_KV_VEC_TOKEN_TILE, GENERIC_KV_GROUP_HEADS, 0]
GENERIC_KERNEL_TILE_CONFIG = [
    GENERIC_Q_GROUP_HEADS,
    GENERIC_Q_VEC_TOKEN_TILE,
    GENERIC_KV_VEC_TOKEN_TILE,
    GENERIC_KV_GROUP_HEADS,
    1,
]

ScatterGroupParams = collections.namedtuple(
    "ScatterGroupParams",
    ["cache_out", "src", "tokens", "num_heads", "dim", "c0", "head_start", "vec_token_tile"],
)
QGroupParams = collections.namedtuple(
    "QGroupParams",
    ["qkv", "q_gamma", "cos_fp32", "sin_fp32", "q_out_out", "tokens", "num_q", "dim", "epsilon",
     "group_heads", "vec_token_tile"],
)
QTokenParams = collections.namedtuple("QTokenParams", QGroupParams._fields + ("token_tile",))
KvTokenParams = collections.namedtuple(
    "KvTokenParams",
    ["qkv", "k_gamma", "cos_fp32", "sin_fp32", "index", "k_scale", "v_scale", "k_cache_out",
     "v_cache_out", "tokens", "q_size", "k_size", "num_k", "num_v", "dim", "c0", "epsilon"],
)
QuantParams = collections.namedtuple(
    "QuantParams",
    ["qkv", "q_gamma", "k_gamma", "cos", "sin", "index", "k_cache", "v_cache", "k_scale", "v_scale",
     "q_out_out", "k_cache_out", "v_cache_out", "qkv_size", "head_nums", "epsilon",
     "q_group_heads", "q_vec_token_tile", "kv_vec_token_tile"],
)
QuantGenericParams = collections.namedtuple("QuantGenericParams", QuantParams._fields + ("kv_group_heads",))
QkvMeta = collections.namedtuple(
    "QkvMeta", ["tokens", "c0", "num_q", "num_k", "num_v", "dim", "q_size", "k_size"]
)


def _rotate_half(x: pypto.Tensor) -> pypto.Tensor:
    shape = x.shape
    dim = shape[len(shape) - 1]
    half = dim // 2
    left_shape = list(shape)
    left_shape[len(shape) - 1] = half
    offsets_0 = [0] * len(shape)
    offsets_1 = [0] * len(shape)
    offsets_1[len(shape) - 1] = half
    x1 = pypto.view(x, left_shape, offsets_0)
    x2 = pypto.view(x, left_shape, offsets_1)
    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def _rope_fp32(x: pypto.Tensor, cos_fp32: pypto.Tensor, sin_fp32: pypto.Tensor, vec_token_tile: int) -> pypto.Tensor:
    t = x.shape[0]
    n = x.shape[1]
    d = x.shape[2]
    x_fp32 = pypto.cast(x, pypto.DT_FP32)
    cos_3d = pypto.reshape(cos_fp32, [t, 1, d], valid_shape=[t, 1, d])
    sin_3d = pypto.reshape(sin_fp32, [t, 1, d], valid_shape=[t, 1, d])
    pypto.set_vec_tile_shapes(vec_token_tile, n, d)
    x_cos = pypto.mul(x_fp32, cos_3d)
    rotated_sin = pypto.mul(_rotate_half(x_fp32), sin_3d)
    return pypto.add(x_cos, rotated_sin)


def _rope(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor, vec_token_tile: int) -> pypto.Tensor:
    rope_fp32 = _rope_fp32(x, pypto.cast(cos, pypto.DT_FP32), pypto.cast(sin, pypto.DT_FP32), vec_token_tile)
    return pypto.cast(rope_fp32, x.dtype, pypto.CastMode.CAST_RINT)


def _rms_norm_fp32(x: pypto.Tensor, gamma: pypto.Tensor, epsilon: float, vec_token_tile: int) -> pypto.Tensor:
    dim = x.shape[len(x.shape) - 1]
    gamma_shape = [1] * len(x.shape)
    gamma_shape[len(x.shape) - 1] = dim
    x_fp32 = pypto.cast(x, pypto.DT_FP32)
    gamma_fp32 = pypto.cast(pypto.reshape(gamma, gamma_shape), pypto.DT_FP32)
    square = x_fp32 * x_fp32
    if dim == 128:
        half_shape = list(x.shape)
        half_shape[len(x.shape) - 1] = 64
        right_offsets = [0] * len(x.shape)
        right_offsets[len(x.shape) - 1] = 64
        left_square = pypto.view(square, half_shape, [0] * len(x.shape))
        right_square = pypto.view(square, half_shape, right_offsets)
        mean = pypto.sum(left_square + right_square, -1, keepdim=True) * (1.0 / dim)
    else:
        mean = pypto.sum(square, -1, keepdim=True) * (1.0 / dim)
    if len(x.shape) == 3:
        pypto.set_vec_tile_shapes(vec_token_tile, x.shape[1], 1)
    rms = pypto.sqrt(mean + epsilon)
    if len(x.shape) == 3:
        pypto.set_vec_tile_shapes(vec_token_tile, x.shape[1], dim)
    return pypto.div(x_fp32, rms, pypto.PrecisionType.INTRINSIC) * gamma_fp32


def _rms_norm(x: pypto.Tensor, gamma: pypto.Tensor, epsilon: float) -> pypto.Tensor:
    return pypto.cast(_rms_norm_fp32(x, gamma, epsilon, 1), x.dtype, pypto.CastMode.CAST_RINT)


def _quant_int8(x: pypto.Tensor, scale: pypto.Tensor) -> pypto.Tensor:
    shape = x.shape
    dim = shape[len(shape) - 1]
    scale_shape = [1] * len(shape)
    scale_shape[len(shape) - 2] = shape[len(shape) - 2]
    scale_shape[len(shape) - 1] = dim
    x_fp32 = pypto.cast(x, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    scale_fp32 = pypto.cast(pypto.reshape(scale, scale_shape), pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    quant_fp32 = x_fp32 / scale_fp32
    quant_int32 = pypto.cast(quant_fp32, pypto.DT_INT32, pypto.CastMode.CAST_RINT)
    quant_fp16 = pypto.cast(quant_int32, pypto.DT_FP16, pypto.CastMode.CAST_ROUND)
    return pypto.cast(quant_fp16, pypto.DT_INT8, pypto.CastMode.CAST_TRUNC, satmode=pypto.SaturationMode.ON)


def _scatter_pa_nz_int8_contiguous_page0(params: ScatterGroupParams):
    d1_per_head = params.dim // params.c0
    c1 = params.num_heads * d1_per_head
    pypto.set_vec_tile_shapes(params.vec_token_tile, c1, params.c0)
    src_3d = pypto.reshape(params.src, [params.tokens, c1, params.c0])
    pypto.set_vec_tile_shapes(params.vec_token_tile, c1, params.c0)
    src_c1_t_c0 = pypto.transpose(src_3d, 0, 1)
    pypto.set_vec_tile_shapes(1, c1, params.vec_token_tile, params.c0)
    src_pa_nz = pypto.reshape(src_c1_t_c0, [1, c1, params.tokens, params.c0])
    pypto.assemble(src_pa_nz, [0, params.head_start * d1_per_head, 0, 0], params.cache_out)


def _scatter_pa_nz_int8_indexed_group(params: ScatterGroupParams, cache_in: pypto.Tensor, index: pypto.Tensor):
    d1_per_head = params.dim // params.c0
    c1 = params.num_heads * d1_per_head
    c1_offset = params.head_start * d1_per_head
    block_size = cache_in.shape[2]

    pypto.set_vec_tile_shapes(params.vec_token_tile, c1, params.c0)
    src_3d = pypto.reshape(params.src, [params.tokens, c1, params.c0])
    pypto.set_vec_tile_shapes(params.vec_token_tile, c1, params.c0)
    src_c1_t_c0 = pypto.transpose(src_3d, 0, 1)
    for token_idx in pypto.loop(params.tokens, name="LOOP_INDEXED_CACHE_GROUP", idx_name="token_idx"):
        page_offset = index[token_idx]
        page_id = page_offset // block_size
        token_offset = page_offset % block_size
        token_tile = pypto.view(src_c1_t_c0, [c1, 1, params.c0], [0, token_idx, 0])
        pypto.assemble(
            pypto.reshape(token_tile, [1, c1, 1, params.c0]),
            [page_id, c1_offset, token_offset, 0],
            params.cache_out,
        )


def _compute_q_grouped(params: QGroupParams):
    for head_start in range(0, params.num_q, params.group_heads):
        pypto.set_vec_tile_shapes(params.vec_token_tile, params.group_heads * params.dim)
        q_2d = pypto.view(params.qkv, [params.tokens, params.group_heads * params.dim],
                          [0, head_start * params.dim])
        pypto.set_vec_tile_shapes(params.vec_token_tile, params.group_heads, params.dim)
        q_3d = pypto.reshape(q_2d, [params.tokens, params.group_heads, params.dim])
        q_norm = _rms_norm_fp32(q_3d, params.q_gamma, params.epsilon, params.vec_token_tile)
        cos_q = pypto.view(params.cos_fp32, [params.tokens, params.dim], [0, 0])
        sin_q = pypto.view(params.sin_fp32, [params.tokens, params.dim], [0, 0])
        q_rope = _rope_fp32(q_norm, cos_q, sin_q, params.vec_token_tile)
        pypto.set_vec_tile_shapes(params.vec_token_tile, params.group_heads * params.dim)
        q_res = pypto.reshape(q_rope, [params.tokens, params.group_heads * params.dim])
        q_res_bf16 = pypto.cast(q_res, pypto.DT_BF16, pypto.CastMode.CAST_RINT)
        pypto.assemble(q_res_bf16, [0, head_start * params.dim], params.q_out_out)


def _compute_q_token_fallback(params: QTokenParams):
    for token_base in pypto.loop(0, params.tokens, params.token_tile, name="LOOP_Q_TOKEN", idx_name="token_base"):
        for head_start in range(0, params.num_q, params.group_heads):
            pypto.set_vec_tile_shapes(params.token_tile, params.group_heads * params.dim)
            q_2d = pypto.view(params.qkv, [params.token_tile, params.group_heads * params.dim],
                              [token_base, head_start * params.dim])
            pypto.set_vec_tile_shapes(params.token_tile, params.group_heads, params.dim)
            q_3d = pypto.reshape(q_2d, [params.token_tile, params.group_heads, params.dim])
            q_norm = _rms_norm_fp32(q_3d, params.q_gamma, params.epsilon, params.token_tile)
            cos_q = pypto.view(params.cos_fp32, [params.token_tile, params.dim], [token_base, 0])
            sin_q = pypto.view(params.sin_fp32, [params.token_tile, params.dim], [token_base, 0])
            q_rope = _rope_fp32(q_norm, cos_q, sin_q, params.token_tile)
            pypto.set_vec_tile_shapes(params.token_tile, params.group_heads * params.dim)
            q_res = pypto.reshape(q_rope, [params.token_tile, params.group_heads * params.dim])
            q_res_bf16 = pypto.cast(q_res, pypto.DT_BF16, pypto.CastMode.CAST_RINT)
            pypto.assemble(q_res_bf16, [token_base, head_start * params.dim], params.q_out_out)


def _compute_kv_token_fallback(params: KvTokenParams):
    k_c1 = params.num_k * params.dim // params.c0
    v_c1 = params.num_v * params.dim // params.c0
    block_size = params.k_cache_out.shape[2]

    for token_idx in pypto.loop(params.tokens, name="LOOP_INDEXED_CACHE_FULL", idx_name="token_idx"):
        page_offset = params.index[token_idx]
        page_id = page_offset // block_size
        token_offset = page_offset % block_size

        pypto.set_vec_tile_shapes(1, params.k_size)
        k_2d = pypto.view(params.qkv, [1, params.k_size], [token_idx, params.q_size])
        pypto.set_vec_tile_shapes(1, params.num_k, params.dim)
        k_3d = pypto.reshape(k_2d, [1, params.num_k, params.dim])
        k_norm = _rms_norm_fp32(k_3d, params.k_gamma, params.epsilon, 1)
        cos_token = pypto.view(params.cos_fp32, [1, params.dim], [token_idx, 0])
        sin_token = pypto.view(params.sin_fp32, [1, params.dim], [token_idx, 0])
        k_rope = _rope_fp32(k_norm, cos_token, sin_token, 1)
        k_quant = _quant_int8(k_rope, params.k_scale)
        pypto.set_vec_tile_shapes(1, k_c1, 1, params.c0)
        k_tile = pypto.reshape(k_quant, [1, k_c1, 1, params.c0])
        pypto.assemble(k_tile, [page_id, 0, token_offset, 0], params.k_cache_out)

        pypto.set_vec_tile_shapes(1, params.num_v * params.dim)
        v_2d = pypto.view(params.qkv, [1, params.num_v * params.dim],
                          [token_idx, params.q_size + params.k_size])
        pypto.set_vec_tile_shapes(1, params.num_v, params.dim)
        v_3d = pypto.reshape(v_2d, [1, params.num_v, params.dim])
        v_quant = _quant_int8(v_3d, params.v_scale)
        pypto.set_vec_tile_shapes(1, v_c1, 1, params.c0)
        v_tile = pypto.reshape(v_quant, [1, v_c1, 1, params.c0])
        pypto.assemble(v_tile, [page_id, 0, token_offset, 0], params.v_cache_out)


def _qkv_meta(params: QuantParams):
    static_tokens = params.qkv_size[0] * params.qkv_size[1]
    c0 = params.k_cache.shape[3]
    num_q = params.head_nums[0]
    num_k = params.head_nums[1]
    num_v = params.head_nums[2]
    dim = params.qkv_size[3]
    q_size = num_q * dim
    k_size = num_k * dim
    return QkvMeta(static_tokens, c0, num_q, num_k, num_v, dim, q_size, k_size)


def _cos_sin_fp32(params: QuantParams, static_tokens: int, dim: int):
    pypto.set_vec_tile_shapes(params.kv_vec_token_tile, dim)
    cos_static = pypto.view(params.cos, [static_tokens, dim], [0, 0])
    sin_static = pypto.view(params.sin, [static_tokens, dim], [0, 0])
    return pypto.cast(cos_static, pypto.DT_FP32), pypto.cast(sin_static, pypto.DT_FP32)


def _q_group_params(params: QuantParams, cos_sin: tuple[pypto.Tensor, pypto.Tensor], meta: QkvMeta) -> QGroupParams:
    cos_fp32, sin_fp32 = cos_sin
    return QGroupParams(
        params.qkv, params.q_gamma, cos_fp32, sin_fp32, params.q_out_out,
        meta.tokens, meta.num_q, meta.dim, params.epsilon, params.q_group_heads, params.q_vec_token_tile,
    )


def _compute_quant(params: QuantParams):
    meta = _qkv_meta(params)
    v_size = meta.num_v * meta.dim
    cos_sin = _cos_sin_fp32(params, meta.tokens, meta.dim)
    cos_fp32, sin_fp32 = cos_sin
    _compute_q_grouped(_q_group_params(params, cos_sin, meta))

    pypto.set_vec_tile_shapes(params.kv_vec_token_tile, meta.k_size)
    k_2d_all = pypto.view(params.qkv, [meta.tokens, meta.k_size], [0, meta.q_size])
    v_2d_all = pypto.view(params.qkv, [meta.tokens, v_size], [0, meta.q_size + meta.k_size])
    pypto.set_vec_tile_shapes(params.kv_vec_token_tile, meta.num_k, meta.dim)
    k_3d_all = pypto.reshape(k_2d_all, [meta.tokens, meta.num_k, meta.dim])
    v_3d_all = pypto.reshape(v_2d_all, [meta.tokens, meta.num_v, meta.dim])
    k_norm_all = _rms_norm_fp32(k_3d_all, params.k_gamma, params.epsilon, params.kv_vec_token_tile)
    cos_all = pypto.view(cos_fp32, [meta.tokens, meta.dim], [0, 0])
    sin_all = pypto.view(sin_fp32, [meta.tokens, meta.dim], [0, 0])
    k_rope_all = _rope_fp32(k_norm_all, cos_all, sin_all, params.kv_vec_token_tile)

    pypto.set_vec_tile_shapes(params.kv_vec_token_tile, meta.num_k, meta.dim)
    k_quant_all = _quant_int8(k_rope_all, params.k_scale)
    v_quant_all = _quant_int8(v_3d_all, params.v_scale)
    _scatter_pa_nz_int8_contiguous_page0(
        ScatterGroupParams(params.k_cache_out, k_quant_all, meta.tokens, meta.num_k, meta.dim, meta.c0, 0,
                           params.kv_vec_token_tile)
    )
    _scatter_pa_nz_int8_contiguous_page0(
        ScatterGroupParams(params.v_cache_out, v_quant_all, meta.tokens, meta.num_v, meta.dim, meta.c0, 0,
                           params.kv_vec_token_tile)
    )


def _select_q_token_tile(static_tokens: int, dim: int) -> int:
    if dim > 256 or static_tokens % 2 != 0:
        return 1
    if dim > 256 or static_tokens % 4 != 0:
        return 2
    return 4


def _compute_quant_generic_fallback(params: QuantGenericParams):
    meta = _qkv_meta(params)
    cos_sin = _cos_sin_fp32(params, meta.tokens, meta.dim)
    cos_fp32, sin_fp32 = cos_sin
    _compute_q_token_fallback(
        QTokenParams(*_q_group_params(params, cos_sin, meta), _select_q_token_tile(meta.tokens, meta.dim))
    )
    _compute_kv_token_fallback(
        KvTokenParams(
            params.qkv, params.k_gamma, cos_fp32, sin_fp32, params.index, params.k_scale, params.v_scale,
            params.k_cache_out, params.v_cache_out, meta.tokens, meta.q_size, meta.k_size, meta.num_k,
            meta.num_v, meta.dim, meta.c0, params.epsilon,
        )
    )


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": TP4_TILE_CONFIG.stitch_function_max_num,
        "device_sched_mode": TP4_TILE_CONFIG.device_sched_mode,
    },
)
def qkv_rms_norm_rope_cache_quant_kernel_regular(
    qkv: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    q_gamma: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    k_gamma: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    index: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT64),
    k_cache: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    v_cache: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    k_scale: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    v_scale: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    q_out_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    k_cache_out: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    v_cache_out: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    qkv_size: list,
    head_nums: list,
    epsilon: float,
    tile_config: list,
):
    pypto.set_vec_tile_shapes(1, 128)
    _compute_quant(
        QuantParams(
            qkv, q_gamma, k_gamma, cos, sin, index, k_cache, v_cache, k_scale, v_scale,
            q_out_out, k_cache_out, v_cache_out, qkv_size, head_nums, epsilon,
            tile_config[0], tile_config[1], tile_config[2],
        )
    )


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": GENERIC_TILE_CONFIG.stitch_function_max_num,
        "device_sched_mode": GENERIC_TILE_CONFIG.device_sched_mode,
    },
)
def qkv_rms_norm_rope_cache_quant_kernel_generic(
    qkv: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    q_gamma: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    k_gamma: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    index: pypto.Tensor([pypto.STATIC], pypto.DT_INT64),
    generic_k_cache: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    generic_v_cache: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    generic_k_scale: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    generic_v_scale: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    generic_q_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    generic_k_cache_out: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    generic_v_cache_out: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    generic_qkv_size: list,
    generic_head_nums: list,
    generic_epsilon: float,
):
    pypto.set_vec_tile_shapes(1, 128)
    _compute_quant_generic_fallback(
        QuantGenericParams(
            qkv, q_gamma, k_gamma, cos, sin, index, generic_k_cache, generic_v_cache,
            generic_k_scale, generic_v_scale, generic_q_out, generic_k_cache_out,
            generic_v_cache_out, generic_qkv_size, generic_head_nums, generic_epsilon,
            GENERIC_Q_GROUP_HEADS, GENERIC_Q_VEC_TOKEN_TILE, GENERIC_KV_VEC_TOKEN_TILE,
            GENERIC_KV_GROUP_HEADS,
        )
    )


def _select_qkv_tile_config(head_nums: Tuple[int, int, int], dim: int) -> list[int]:
    num_q, num_k, num_v = head_nums
    if num_k > 4 or num_v > 4:
        return GENERIC_KERNEL_TILE_CONFIG
    if num_q > 64 or dim > 128:
        return GENERIC_KERNEL_TILE_CONFIG
    if num_q <= 16:
        return TP4_KERNEL_TILE_CONFIG
    return TP1_KERNEL_TILE_CONFIG


def qkv_rms_norm_rope_cache_wrapper(
    qkv: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    index: torch.Tensor,
    q_out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    k_offset: Optional[torch.Tensor] = None,
    v_offset: Optional[torch.Tensor] = None,
    qkv_size: Sequence[int] = (),
    head_nums: Sequence[int] = (),
    epsilon: float = 1e-6,
    cache_mode: str = "PA_NZ",
    is_output_qkv: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the PyPTO kernel and return ``(q_out, k_cache, v_cache)``.

    ``q_out``, ``k_cache`` and ``v_cache`` are treated like AscendC in/out
    tensors.  The public wrapper keeps these arguments in the AscendC order,
    while the private JIT kernel only receives the writable output buffers.
    """
    if qkv.dtype != torch.bfloat16:
        raise TypeError("this PyPTO implementation currently supports BF16 qkv only")
    if k_cache.dtype != v_cache.dtype:
        raise TypeError("k_cache and v_cache must have the same dtype")
    if cache_mode != "PA_NZ":
        raise NotImplementedError("this PyPTO implementation currently supports PA_NZ cache_mode only")
    if is_output_qkv:
        raise NotImplementedError("this PyPTO implementation currently supports is_output_qkv=False only")
    q_out_out = q_out
    k_cache_out = k_cache
    v_cache_out = v_cache
    if k_cache.dtype != torch.int8:
        raise TypeError("current network implementation requires int8 k_cache/v_cache")
    if k_scale is None or v_scale is None:
        raise ValueError("k_scale and v_scale are required for int8 cache")
    if k_offset is not None or v_offset is not None:
        raise NotImplementedError("asymmetric quantization is not supported yet")
    tile_config = _select_qkv_tile_config(tuple(head_nums), int(qkv_size[3]))
    if tile_config[4] == 1:
        qkv_rms_norm_rope_cache_quant_kernel_generic(
            qkv, q_gamma, k_gamma, cos, sin, index, k_cache, v_cache, k_scale, v_scale,
            q_out_out, k_cache_out, v_cache_out, list(qkv_size), list(head_nums), float(epsilon),
        )
    else:
        qkv_rms_norm_rope_cache_quant_kernel_regular(
            qkv, q_gamma, k_gamma, cos, sin, index, k_cache, v_cache, k_scale, v_scale,
            q_out_out, k_cache_out, v_cache_out, list(qkv_size), list(head_nums), float(epsilon),
            list(tile_config),
        )
    return q_out_out, k_cache_out, v_cache_out
