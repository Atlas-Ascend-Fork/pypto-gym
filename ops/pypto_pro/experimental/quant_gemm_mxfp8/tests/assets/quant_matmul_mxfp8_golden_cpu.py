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
"""Independent CPU FP64 golden and MXFP8/E8M0 byte decoding. No Torch required."""

import numpy as np


def fp8_table(kind):
    """Decode all 256 bit patterns independently of torch FP8 casts."""
    bits = np.arange(256, dtype=np.uint8)
    mantissa_bits, bias = (2, 15) if kind == "e5m2" else (3, 7)
    exp = (bits.astype(np.int32) & 127) >> mantissa_bits
    mant = bits.astype(np.int32) & ((1 << mantissa_bits) - 1)
    value = np.where(
        exp == 0,
        np.ldexp(mant.astype(np.float64) / (1 << mantissa_bits), 1 - bias),
        np.ldexp(1.0 + mant / float(1 << mantissa_bits), exp - bias),
    )
    if kind == "e5m2":
        value = np.where(exp == 31, np.where(mant == 0, np.inf, np.nan), value)
    else:
        value = np.where((exp == 15) & (mant == 7), np.nan, value)
    return np.where(bits & 128, -value, value)


def decode_e8m0(raw):
    if raw.dtype != np.uint8 or np.any(raw == 255):
        raise ValueError("Expected finite uint8 E8M0 encodings (0..254)")
    return np.ldexp(np.ones(raw.shape, dtype=np.float64), raw.astype(np.int32) - 127)


def round_bf16(value):
    """FP32 -> BF16 round-to-nearest-even, returned as FP32 for numpy."""
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite golden before BF16 rounding")
    bits = value.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def make_fp8_bytes(rng, shape, kind):
    # Independent signs, mantissas and moderate exponents, plus zeros and
    # subnormals. Avoid NaN/Inf and overflow so every output is comparable.
    mbits, bias = (2, 15) if kind == "e5m2" else (3, 7)
    exp = rng.integers(bias - 3, bias + 3, shape, dtype=np.uint8)
    mant = rng.integers(0, 1 << mbits, shape, dtype=np.uint8)
    sign = rng.integers(0, 2, shape, dtype=np.uint8) << 7
    raw = sign | (exp << mbits) | mant
    raw.reshape(-1)[::251] = 0
    raw.reshape(-1)[1::251] = 1  # smallest positive subnormal
    raw.reshape(-1)[2::251] = 129  # smallest negative subnormal
    return raw


def golden_matmul(x1, x2, s1, s2, kind, column_block=512):
    """Decode actual input bits, dequantize in FP64, return unbiased [M,N].

    Chunk only independent output columns: each FP64 dot still covers ALL K.
    This bounds the FP64 B/scale temporaries for N=32768 and K=8192.
    No intermediate FP32 product or partial-sum rounding is introduced.
    """
    m, k = x1.shape
    if x2.ndim != 2 or x2.shape[0] != k or k % 64:
        raise ValueError("expected rank-2 A[M,K], B[K,N], K divisible by 64")
    n = x2.shape[1]
    if s1.shape != (m, k // 64, 2) or s2.shape != (k // 64, n, 2):
        raise ValueError("invalid packed MX scale shapes")
    if column_block <= 0:
        raise ValueError("column_block must be positive")
    table = fp8_table(kind)
    a_scale = np.repeat(decode_e8m0(s1).reshape(m, k // 32), 32, axis=-1)
    a = table[x1] * a_scale
    del a_scale
    result = np.empty((m, n), dtype=np.float64)
    for start in range(0, n, column_block):
        stop = min(start + column_block, n)
        b_scale = np.repeat(
            decode_e8m0(s2[:, start:stop, :]).transpose(1, 0, 2).reshape(stop - start, k // 32),
            32, axis=-1,
        ).T
        b = table[x2[:, start:stop]] * b_scale
        result[:, start:stop] = a @ b
    return result
