# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IndexerPrologueK 的 bit-exact CPU 参考实现（golden）。

三处 BF16 发布边界（GEMM、RMSNorm、RoPE）严格使用参考实现的舍入。MXFP4 使用
exponent 位操作，配合倒数乘 ``safe_amax * (1/6)``（非除法），
而非 ``log2``/``exp2`` 近似，因此 scale 字节与 E2M1 码可逐位比较。
"""

import struct

import numpy as np
import torch

FP4_GROUP_SIZE = 32
FP4_MAX = 6.0
FP32_EXP_SHIFT = 23
FP32_MANTISSA_MASK = 0x7FFFFF
FP32_EXP_MASK = 0xFF
E2M1_VALUES_PER_BYTE = 2
CACHE_ABI_DIM2 = 1

# 顺序有意义：多个阈值同时命中时后写入的覆盖前面的，与参考 ``_pack_mxfp4`` 链一致。
THRESHOLDS = (
    (5.0, 6),
    (3.5, 5),
    (2.5, 4),
    (1.75, 3),
    (1.25, 2),
    (0.75, 1),
    (0.25, 0),
)


def _bits_f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def value_and_scale_bytes(index_head_dim):
    return index_head_dim // E2M1_VALUES_PER_BYTE, index_head_dim // FP4_GROUP_SIZE


def scale_for_amax(amax):
    """返回某个 group 绝对值最大对应的 ``(scale_byte, scale_value)``。

    ``safe_amax = max(amax, 6*2^-126)``；``raw = safe_amax * (1/6)`` 在 FP32 下计算
    （倒数乘，与 device 一致）；随后只要尾数非零就把 FP32 阶码字段 +1，得到 >= raw 的
    下一个 2 的幂。
    """
    safe = max(float(amax), FP4_MAX * (2.0 ** -126))
    safe_f32 = np.float32(safe)
    reciprocal = np.float32(1.0 / FP4_MAX)
    raw_f32 = np.float32(safe_f32 * reciprocal)
    bits = int(raw_f32.view(np.uint32))
    exp = (bits >> FP32_EXP_SHIFT) & FP32_EXP_MASK
    if (bits & FP32_MANTISSA_MASK) != 0:
        exp += 1
    return exp & 0xFF, _bits_f32(exp << FP32_EXP_SHIFT)


def encode_group(values):
    """把 32 个 FP32 值编码为 16 个打包字节 + 1 个 UE8M0 scale 字节。"""
    amax = 0.0
    for value in values:
        magnitude = abs(float(value))
        if magnitude > amax:
            amax = magnitude
    exp, scale = scale_for_amax(amax)
    data = bytearray(FP4_GROUP_SIZE // 2)
    for pair in range(FP4_GROUP_SIZE // 2):
        packed = 0
        for half in range(2):
            value = float(values[2 * pair + half])
            code = 7
            for threshold, lower in THRESHOLDS:
                if abs(value) / scale <= threshold:
                    code = lower
            if value < 0.0:
                code += 8
            packed |= code << (4 * half)
        data[pair] = packed
    return bytes(data), exp


def pack_mxfp4_row(row_bf16_values):
    """编码一行 BF16（原始 python float 序列）-> (data, scale)。"""
    values = list(row_bf16_values)
    num_groups = len(values) // FP4_GROUP_SIZE
    data = bytearray(len(values) // E2M1_VALUES_PER_BYTE)
    scale = bytearray(num_groups)
    for group in range(num_groups):
        chunk = values[group * FP4_GROUP_SIZE:(group + 1) * FP4_GROUP_SIZE]
        gdata, gscale = encode_group(chunk)
        data[group * (FP4_GROUP_SIZE // 2):(group + 1) * (FP4_GROUP_SIZE // 2)] = gdata
        scale[group] = gscale
    return bytes(data), bytes(scale)


def owner_key(storage_mode, cache_row, cache_slots, group_size):
    """返回跨核划分工作用的目标归属 key。"""
    if storage_mode == 0:
        return cache_row
    raw_block_size = cache_slots * group_size
    block_id = cache_row // raw_block_size
    slot = cache_row % raw_block_size
    return block_id * cache_slots + slot // group_size


def scatter_offsets(storage_mode, cache_row, cache_slots, group_size, index_head_dim):
    """返回扁平 cache 内的 ``(block, dest1, data_base, scale_base)``。

    mode 0 下 ``dest1`` 是 slot，两个 base 均为 0（scale 位于独立 cache）。mode 1 下
    ``dest1`` 是 combined group 行，data/scale 字节区间按行内布局给出。
    """
    value_bytes, scale_bytes = value_and_scale_bytes(index_head_dim)
    if storage_mode == 0:
        block = cache_row // cache_slots
        slot = cache_row % cache_slots
        return block, slot, 0, 0
    raw_block_size = cache_slots * group_size
    block = cache_row // raw_block_size
    slot = cache_row % raw_block_size
    group_row = slot // group_size
    sub_slot = slot % group_size
    data_base = sub_slot * value_bytes
    scale_base = group_size * value_bytes + sub_slot * scale_bytes
    return block, group_row, data_base, scale_base


def mxfp4_encode_rows(key_bf16):
    """把 ``[T, D]`` BF16 编码为 E2M1 数据 ``[T, D/2]`` 与 scale ``[T, D/32]``。"""
    rows, dim = key_bf16.shape
    value_bytes, scale_bytes = value_and_scale_bytes(dim)
    data = torch.empty(rows, value_bytes, dtype=torch.uint8)
    scales = torch.empty(rows, scale_bytes, dtype=torch.uint8)
    float_rows = key_bf16.float().tolist()
    for row in range(rows):
        row_data, row_scale = pack_mxfp4_row(float_rows[row])
        data[row] = torch.frombuffer(bytearray(row_data), dtype=torch.uint8)
        scales[row] = torch.frombuffer(bytearray(row_scale), dtype=torch.uint8)
    return data, scales


def reference_projection(latent, wk):
    """``BF16(GEMM_FP32_ACC(latent, wk.T))``。"""
    return (latent.float() @ wk.float().T).to(torch.bfloat16)


def reference_norm_rope(projected, norm_weight, rope_sin, rope_cos, norm_eps):
    """RMSNorm（BF16 发布）+ 尾部相邻偶奇 RoPE（BF16 发布）。"""
    key_fp32 = projected.float()
    inverse_rms = torch.rsqrt(
        key_fp32.square().mean(dim=-1, keepdim=True) + norm_eps
    )
    key = (key_fp32 * inverse_rms * norm_weight).to(torch.bfloat16)

    rope_head_dim = rope_sin.shape[-1]
    tail = key[:, -rope_head_dim:].float()
    even = tail[:, 0::2]
    odd = tail[:, 1::2]
    rotated = torch.empty_like(tail)
    rotated[:, 0::2] = even * rope_cos[:, 0::2] - odd * rope_sin[:, 0::2]
    rotated[:, 1::2] = odd * rope_cos[:, 1::2] + even * rope_sin[:, 1::2]
    key[:, -rope_head_dim:] = rotated.to(torch.bfloat16)
    return key


def reference_scatter(
    k_cache, k_scale_cache, cache_index, packed_data, packed_scale,
    storage_mode, combined_block_size,
):
    """把打包行散写到 cache 副本（重复索引：最后一个 token 生效）。"""
    expected_cache = k_cache.clone()
    expected_scale = None if k_scale_cache is None else k_scale_cache.clone()
    cache_slots = int(k_cache.shape[1])
    value_bytes = int(packed_data.shape[-1])
    index_head_dim = 2 * value_bytes
    for row, cache_row_flat in enumerate(cache_index.tolist()):
        if cache_row_flat == -1:
            continue
        block, dest1, data_base, scale_base = scatter_offsets(
            storage_mode, cache_row_flat, cache_slots, combined_block_size,
            index_head_dim,
        )
        if storage_mode == 0:
            expected_cache[block, dest1, 0, :] = packed_data[row]
            if expected_scale is not None:
                expected_scale[block, dest1, 0, :] = packed_scale[row]
        else:
            scale_bytes = int(packed_scale.shape[-1])
            expected_cache[
                block, dest1, 0, data_base:data_base + value_bytes
            ] = packed_data[row]
            expected_cache[
                block, dest1, 0, scale_base:scale_base + scale_bytes
            ] = packed_scale[row]
    return expected_cache, expected_scale


def reference_forward(
    latent,
    wk,
    norm_weight,
    rope_sin,
    rope_cos,
    k_cache,
    cache_index,
    k_scale_cache=None,
    *,
    storage_mode=0,
    norm_eps=1e-6,
    combined_block_size=-1,
):
    """完整 CPU golden，返回 ``(expected_cache, expected_scale)``。"""
    projected = reference_projection(latent, wk)
    key = reference_norm_rope(projected, norm_weight, rope_sin, rope_cos, norm_eps)
    packed_data, packed_scale = mxfp4_encode_rows(key)
    return reference_scatter(
        k_cache, k_scale_cache, cache_index, packed_data, packed_scale,
        storage_mode, combined_block_size,
    )
