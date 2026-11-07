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

"""Independent CPU mathematical/bit-pattern reference; no NPU imports."""
import torch


def _fp8_group(x, cache_dtype, round_scale):
    maximum = 448.0 if cache_dtype == torch.float8_e4m3fn else 57344.0
    amax = x.float().abs().max()
    s = torch.maximum(amax, torch.tensor(1.e-4, dtype=torch.float32)) * torch.tensor(1.0 / maximum, dtype=torch.float32)
    if round_scale:
        raw = int(s.view(torch.int32)) & 0xffffffff
        exponent = (raw >> 23) & 0xff
        mantissa = raw & 0x7fffff
        rounded_bits = ((exponent + int(mantissa != 0)) << 23) & 0xffffffff
        signed_bits = rounded_bits if rounded_bits < 0x80000000 else rounded_bits - 0x100000000
        s = torch.tensor(signed_bits, dtype=torch.int32).view(torch.float32)
    bounded = torch.clamp(x.float() / s, -maximum, maximum)
    q = bounded.to(cache_dtype).view(torch.uint8)
    # FP8 NaN data uses the canonical positive encoding 0x7f.
    q = torch.where(torch.isnan(bounded), torch.tensor(0x7f, dtype=torch.uint8), q)
    if bool(torch.isnan(s)):
        # A NaN scale is stored with the canonical BF16 encoding 0x7fff.
        sb = torch.tensor([0x7fff], dtype=torch.uint16).view(torch.uint8)
    else:
        sb = s.to(torch.bfloat16).reshape(1).view(torch.uint8)
    return q, sb


def _fp4_group(x):
    # Quantization levels and ties-to-even are expressed independently from VF.
    words = x.view(torch.int16).to(torch.int32) & 0xffff
    max_exp = int((words & 0x7f80).max())
    e = max(max_exp, 0x100) - 0x100
    scale_bits = e if max_exp != 0x7f80 else 0x7f81
    inverse_bits = 0x7f00 - e if max_exp != 0x7f80 else 0x7f81
    if max_exp == 0:
        scale_bits = inverse_bits = 0
    if e == 0x7f00:
        inverse_bits = 0x0040
    if max_exp == 0x7f80:
        # A non-finite group uses a NaN inverse and positive-zero E2M1 data.
        return (torch.zeros(x.numel() // 2, dtype=torch.uint8),
                torch.tensor([scale_bits], dtype=torch.uint16).view(torch.uint8))
    inverse = torch.tensor(inverse_bits, dtype=torch.uint16).view(torch.bfloat16)
    y = (x.float() * inverse.float()).to(torch.bfloat16).float()
    ay = y.abs()
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)
    distance = (ay[:, None] - levels[None, :]).abs()
    min_distance = distance.min(dim=1).values
    ties = distance == min_distance[:, None]
    # Stable argmin gives a nearest code. At an exact midpoint, an even
    # code wins; code zero is a legitimate result, never an unset sentinel.
    codes = distance.argmin(dim=1)
    even_codes = torch.tensor([0, 2, 4, 6], dtype=torch.int64)
    even_ties = ties[:, even_codes]
    nearest_even = even_codes[even_ties.to(torch.int64).argmax(dim=1)]
    codes = torch.where(even_ties.any(dim=1), nearest_even, codes)
    sign = ((words >> 12) & 8).to(torch.int64)
    codes = codes | sign
    packed = (codes[0::2] | (codes[1::2] << 4)).to(torch.uint8)
    scales = torch.tensor(scale_bits, dtype=torch.uint16).reshape(1).view(torch.uint8)
    return packed, scales


def golden_cache_bytes(initial_bytes, x, slots, cache_shape, cache_dtype,
                       quant_group_size=32, quant_mode=2, round_scale=True,
                       block_stride=0):
    """Return full physical byte backing after one call, preserving untouched bytes."""
    out = initial_bytes.detach().cpu().to(torch.uint8).flatten().clone()
    x = x.detach().cpu().to(torch.bfloat16)
    slots = slots.detach().cpu().flatten()
    t, d = x.shape
    data = d if quant_mode == 2 else d // 2
    bundle = data + 2 * (d // quant_group_size)
    if len(cache_shape) == 2:
        n, c = cache_shape
    else:
        b, bs, _, c = cache_shape
        n = b * bs
        stride = block_stride or bs * c
    for row in range(t):
        slot = int(slots[row])
        if not 0 <= slot < n:
            continue
        base = slot * c if len(cache_shape) == 2 else (slot // bs) * stride + (slot % bs) * bundle
        for g in range(d // quant_group_size):
            values = x[row, g * quant_group_size:(g + 1) * quant_group_size].contiguous()
            q, sb = _fp8_group(values, cache_dtype, round_scale) if quant_mode == 2 else _fp4_group(values)
            start = base + (g * quant_group_size if quant_mode == 2 else g * quant_group_size // 2)
            out[start:start + q.numel()] = q
            out[base + data + 2 * g:base + data + 2 * g + 2] = sb
        if len(cache_shape) == 2:
            out[base + bundle:base + ((bundle + 31) // 32) * 32] = 0
    return out


GUARD_BYTES = 64


def make_inputs(case):
    """Build the all-one input and guarded cache for a default configuration."""
    d, mode, group = case['d'], case['mode'], case['group']
    data = d if mode == 2 else d // 2
    bundle = data + 2 * (d // group)
    c = ((bundle + 31) // 32) * 32
    if case.get('layout4', False):
        block_stride = 2 * 4 * c
        shape, strides = (2, 4, 1, c), (block_stride, c, c, 1)
        body_bytes = 2 * block_stride
        slots = [0, 3, 4, -1]
    else:
        block_stride = 0
        shape, strides = (8, c), (c, 1)
        body_bytes = 8 * c
        slots = [1, -1, 3, 8]
    return dict(
        x=torch.ones((4, d), dtype=torch.bfloat16),
        slots=torch.tensor(slots, dtype=torch.int32),
        initial=torch.full((GUARD_BYTES + body_bytes + GUARD_BYTES,), 0x5a, dtype=torch.uint8),
        shape=shape, strides=strides, block_stride=block_stride)


def reference(inputs, case):
    """Return expected bytes, including both guards, padding and storage gaps."""
    initial = inputs['initial']
    body_and_right_guard = golden_cache_bytes(
        initial[GUARD_BYTES:], inputs['x'], inputs['slots'], inputs['shape'], case['dtype'],
        quant_group_size=case['group'], quant_mode=case['mode'],
        round_scale=True, block_stride=inputs['block_stride'])
    return torch.cat((initial[:GUARD_BYTES], body_and_right_guard))
