# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Uniform CPU inputs and independent FP64 reference for CompressorV2."""

from __future__ import annotations

from typing import NamedTuple

import torch


class CompressorInputs(NamedTuple):
    """Operator inputs in the public positional order."""

    x: torch.Tensor
    wkv: torch.Tensor
    wgate: torch.Tensor
    cache: torch.Tensor
    table: torch.Tensor
    cu_seqlens: torch.Tensor | None
    seqused: torch.Tensor | None
    start_pos: torch.Tensor | None


class _SequenceSpan(NamedTuple):
    """Input and cache positions for a single sequence."""

    begin: int
    count: int
    start: int
    prefix: int
    groups: int
    block: int


def make_inputs(case):
    """Generate uniform [-10, 10] inputs using the case's fixed CPU seed."""
    b, s, h, d = (case[key] for key in ("batch", "sequence", "hidden", "dim"))
    generator = torch.Generator().manual_seed(case["seed"])
    dtype = getattr(torch, case["dtype"])
    shape = (b, s, h) if case["layout"] == "BSH" else (b * s, h)
    cache = (
        torch.rand(
            (b, case["block_size"], 2 * d), generator=generator, dtype=torch.float32
        )
        * 20
        - 10
    )
    x = (torch.rand(shape, generator=generator, dtype=torch.float32) * 20 - 10).to(
        dtype
    )
    wkv = (torch.rand((d, h), generator=generator, dtype=torch.float32) * 20 - 10).to(
        dtype
    )
    wgate = (torch.rand((d, h), generator=generator, dtype=torch.float32) * 20 - 10).to(
        dtype
    )
    table = torch.tensor(case["blocks"], dtype=torch.int32)
    cu = (
        (torch.arange(b + 1, dtype=torch.int32) * s) if case["layout"] == "TH" else None
    )
    starts = torch.tensor(case["starts"], dtype=torch.int32)
    return CompressorInputs(x, wkv, wgate, cache, table, cu, None, starts)


def _sequence_span(values, batch, sequence_length, ratio):
    """Resolve the active input range and absolute start for one sequence."""
    _, table, cu, used, starts = values[3:]
    begin = batch * sequence_length if cu is None else int(cu[batch])
    count = sequence_length if cu is None else int(cu[batch + 1]) - begin
    if used is not None:
        count = int(used[batch])
    start = 0 if starts is None else int(starts[batch])
    prefix = start % ratio
    groups = (prefix + count) // ratio if count else 0
    return _SequenceSpan(begin, count, start, prefix, groups, int(table[batch]))


def _compress_group(old, projected, span, absolute, ratio, dim):
    """Combine historical and new FP64 projections for one complete group."""
    kvs, scores = [], []
    for pos in range(absolute, absolute + ratio):
        if pos < span.start:
            row = old[span.block, pos % old.shape[1]].double()
            kvs.append(row[:dim])
            scores.append(row[dim:])
        else:
            row = span.begin + pos - span.start
            kvs.append(projected[0][row])
            scores.append(projected[1][row])
    weight = torch.softmax(torch.stack(scores), dim=0)
    return (weight * torch.stack(kvs)).sum(0)


def _write_sequence_output(
    output, mask, old, projected, span, ratio, output_base, batch, is_bsh
):
    """Write valid groups using the public BSH or packed TH output layout."""
    dim = projected[0].shape[-1]
    for group in range(span.groups):
        absolute = span.start - span.prefix + group * ratio
        value = _compress_group(old, projected, span, absolute, ratio, dim)
        index = (batch, group) if is_bsh else output_base + group
        output[index] = value
        mask[index] = True


def _update_sequence_cache(cache, updated, projected, span, block_size, dim):
    """Replace exactly the final input rows retained by the ring cache."""
    for offset in range(max(0, span.count - block_size), span.count):
        slot = (span.start + offset) % block_size
        cache[span.block, slot, :dim] = projected[0][span.begin + offset].float()
        cache[span.block, slot, dim:] = projected[1][span.begin + offset].float()
        updated[span.block, slot] = True


def reference(values, ratio):
    """Independent FP64 contraction and grouping over the pre-call cache state."""
    x, wkv, wgate, old, _, cu, _, _ = values
    h, d = x.shape[-1], wkv.shape[0]
    b = x.shape[0] if x.ndim == 3 else cu.numel() - 1
    s = x.shape[1] if x.ndim == 3 else 0
    t = x.numel() // h
    projected = (
        x.double().reshape(t, h) @ wkv.double().T,
        x.double().reshape(t, h) @ wgate.double().T,
    )
    shape = (
        (b, (s + ratio - 1) // ratio, d) if x.ndim == 3 else (min(t, t // ratio + b), d)
    )
    output = torch.zeros(shape, dtype=torch.float64)
    mask = torch.zeros(shape[:-1], dtype=torch.bool)
    cache = old.clone()
    updated = torch.zeros_like(old, dtype=torch.bool)
    output_base = 0
    for batch in range(b):
        span = _sequence_span(values, batch, s, ratio)
        _write_sequence_output(
            output, mask, old, projected, span, ratio, output_base, batch, x.ndim == 3
        )
        _update_sequence_cache(cache, updated, projected, span, old.shape[1], d)
        output_base += span.groups
    return output.to(x.dtype), mask, cache, updated
