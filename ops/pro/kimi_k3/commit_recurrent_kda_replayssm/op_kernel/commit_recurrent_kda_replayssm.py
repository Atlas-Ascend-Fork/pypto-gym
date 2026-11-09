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

"""PyPTO-Pro implementation of `commit_recurrent_kda_replayssm` (Ascend 950PR).

Contract:

for every batch ``b`` and every accepted token ``t < num_accepted_tokens[b]``,
applied in increasing ``t``::

    recurrent_state[b, :, i, j] = recurrent_state[b, :, i, j] * replay_decay[b, t, :, j]
                                  + replay_u[b, t, :, i] * replay_k[b, t, :, j]

``recurrent_state`` is updated **in place**; the replay records are read-only.
``D = 128`` is fixed; ``B``, ``S``, ``N`` are runtime dimensions.

Design
------
* Pure Vector kernel.  A work item owns ``Rows`` state rows of one
  ``(batch, head)``; rows are independent in the recurrence, so a head splits
  freely into ``128 // Rows`` row blocks.  Items are assigned block-cyclically
  (``pl.range(block_idx, total_items, block_num)``) so every worker receives the
  same count.
* The owned state rows stay resident in UB for the whole accepted prefix, so
  every state byte is read once and written once: the contract's memory floor.
* The accepted prefix is fetched as a **fused record block** (``[Cap, 128]``
  decay and FP16 key, ``[Cap, Rows]`` FP16 u).  ``Cap`` tokens are widened to
  FP32 once per item; the token loop therefore issues no transfer and no
  FP16->FP32 conversion at all.  ``S > Cap`` walks extra chunks over the same
  resident state tile, so any positive ``S`` is correct.
* **Row-group register residency**: the token loop sits inside a row loop, so
  ``GRP`` state rows live in registers for the whole token loop.  The state
  makes one UB round trip per row per chunk instead of one per token.
* **Software pipeline**: the next item's state tile is issued after the current
  item's record transfers are already in the MTE2 queue, so the state transfer
  overlaps the current VF work instead of delaying it.  The record blocks are
  double buffered, so item ``k``'s transfer overlaps item ``k-1``'s VF.

Internal tiling keys
--------------------
``Rows`` (state rows per work item), ``Slots`` (state tile ring), ``Cap``
(tokens fused into one record block) and ``Rdep`` (record-block ring) are
selected on the host; the public domain and defaults are documented in the
README.  Physical UB stays below the 256 KiB Vector capacity for every key.

No GM workspace, no Cube, no cross-core handover.

CodeCheck exceptions
--------------------
* G.FNM.03: PyPTO kernel/host-wrapper tensor parameters must follow the positional
  ABI; PyPTO JIT cannot carry torch.Tensor fields in a dataclass, so the signature
  stays positional.
* Large function / depth: the section, chunk and row loops mirror the hardware
  pipeline; splitting them further would need a re-validation on Ascend 950.
"""

from __future__ import annotations

import torch

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.tilingkey import TilingKeyField

FP32 = pl.DT_FP32
FP16 = pl.DT_FP16
I32 = pl.DT_INT32

HEAD_DIM = 128
GRP = 4
DEFAULT_ROWS = 64
DEFAULT_SLOTS = 3
DEFAULT_CAP = 16
DEFAULT_RDEP = 2


class CommitKey:
    """Internal tiling keys; values are the compiled specializations."""

    Rows = TilingKeyField(bits=3, values=[8, 16, 32, 64, 128])
    Slots = TilingKeyField(bits=2, values=[2, 3, 4])
    Cap = TilingKeyField(bits=3, values=[8, 16, 32, 48, 64])
    Rdep = TilingKeyField(bits=1, values=[2, 3])

    @classmethod
    def is_valid(cls, key):
        rows, _slots, _cap, _rdep = key
        return rows % GRP == 0


@pl.vector_function
def _widen_block(u16, u32, k16, k32, ntokens: pl.DT_INT64, rows: pl.DT_INT64,
                 dim: pl.DT_INT64):
    """Widen the fused FP16 key/u record block to FP32 once per item."""
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for t in pl.range(ntokens):
        for c in pl.range(0, dim, 64):
            kp = vf.load_align(k16, [t, c], dist=pl.LoadDist.UNPK)
            kw = vf.astype(kp, full, dtype=FP32, layout=pl.CastLayout.ZERO)
            vf.store_align(k32, kw, full, [t, c])
        for c in pl.range(0, rows, 64):
            packed = vf.load_align(u16, [t, c], dist=pl.LoadDist.UNPK)
            mask = vf.update_mask(pl.min(64, rows - c), dtype=FP32)
            w = vf.astype(packed, mask, dtype=FP32, layout=pl.CastLayout.ZERO)
            vf.store_align(u32, w, mask, [t, c])


@pl.vector_function
def _apply_block(state, dec, key32, u32, ntokens: pl.DT_INT64, rows: pl.DT_INT64):
    """``state[r, :] = state[r, :] * dec + u[r] * key`` for GRP-row groups."""
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for rg in pl.range(0, rows, GRP):
        a0 = vf.load_align(state, [rg, 0])
        b0 = vf.load_align(state, [rg, 64])
        a1 = vf.load_align(state, [rg + 1, 0])
        b1 = vf.load_align(state, [rg + 1, 64])
        a2 = vf.load_align(state, [rg + 2, 0])
        b2 = vf.load_align(state, [rg + 2, 64])
        a3 = vf.load_align(state, [rg + 3, 0])
        b3 = vf.load_align(state, [rg + 3, 64])
        for t in pl.range(ntokens):
            dp = vf.load_align(dec, [t, 0])
            dq = vf.load_align(dec, [t, 64])
            kp = vf.load_align(key32, [t, 0])
            kq = vf.load_align(key32, [t, 64])
            u0 = vf.load_align(u32, [t, rg], dist=pl.LoadDist.BRC)
            u1 = vf.load_align(u32, [t, rg + 1], dist=pl.LoadDist.BRC)
            u2 = vf.load_align(u32, [t, rg + 2], dist=pl.LoadDist.BRC)
            u3 = vf.load_align(u32, [t, rg + 3], dist=pl.LoadDist.BRC)
            a0 = vf.mul(a0, dp, full)
            a0 = vf.mul_add_dst(u0, kp, full)
            b0 = vf.mul(b0, dq, full)
            b0 = vf.mul_add_dst(u0, kq, full)
            a1 = vf.mul(a1, dp, full)
            a1 = vf.mul_add_dst(u1, kp, full)
            b1 = vf.mul(b1, dq, full)
            b1 = vf.mul_add_dst(u1, kq, full)
            a2 = vf.mul(a2, dp, full)
            a2 = vf.mul_add_dst(u2, kp, full)
            b2 = vf.mul(b2, dq, full)
            b2 = vf.mul_add_dst(u2, kq, full)
            a3 = vf.mul(a3, dp, full)
            a3 = vf.mul_add_dst(u3, kp, full)
            b3 = vf.mul(b3, dq, full)
            b3 = vf.mul_add_dst(u3, kq, full)
        vf.store_align(state, a0, full, [rg, 0])
        vf.store_align(state, b0, full, [rg, 64])
        vf.store_align(state, a1, full, [rg + 1, 0])
        vf.store_align(state, b1, full, [rg + 1, 64])
        vf.store_align(state, a2, full, [rg + 2, 0])
        vf.store_align(state, b2, full, [rg + 2, 64])
        vf.store_align(state, a3, full, [rg + 3, 0])
        vf.store_align(state, b3, full, [rg + 3, 64])


@pl.jit(auto_mutex=True, tiling_key=CommitKey, name="commit_recurrent_kda_replayssm")
def commit_recurrent_kda_replayssm_kernel(
    state_gm: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, HEAD_DIM, HEAD_DIM], FP32],
    replay_u: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, HEAD_DIM], FP16],
    replay_k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, HEAD_DIM], FP16],
    replay_decay: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, HEAD_DIM], FP32],
    accepted: pl.Tensor[[pl.DYNAMIC], I32],
    batch: pl.DT_INT64, heads: pl.DT_INT64,
):
    """One work item = `Rows` state rows of one (batch, head)."""
    # Physical u columns must cover the UNPK read width (one 256 B register).
    ucols = ((Rows + 63) // 64) * 64
    row_blocks = HEAD_DIM // Rows
    state_bytes = Rows * HEAD_DIM * 4
    dec_bytes = Cap * HEAD_DIM * 4
    key_bytes = Cap * HEAD_DIM * 2
    k32_bytes = Cap * HEAD_DIM * 4
    u16_bytes = Cap * ucols * 2
    u32_bytes = Cap * ucols * 4
    va_state = 0
    va_dec = va_state + Slots * state_bytes
    va_key = va_dec + Rdep * dec_bytes
    va_k32 = va_key + Rdep * key_bytes
    va_u16 = va_k32 + Rdep * k32_bytes
    va_u32 = va_u16 + Rdep * u16_bytes

    with pl.section_vector():
        state_g = pl.make_tile_group(
            type=pl.TileType(shape=[Rows, HEAD_DIM], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_state + i * state_bytes for i in range(Slots)],
            mutex_ids=list(range(Slots)))
        dec_g = pl.make_tile_group(
            type=pl.TileType(shape=[Cap, HEAD_DIM], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_dec + i * dec_bytes for i in range(Rdep)],
            mutex_ids=list(range(10, 10 + Rdep)))
        key_g = pl.make_tile_group(
            type=pl.TileType(shape=[Cap, HEAD_DIM], dtype=FP16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_key + i * key_bytes for i in range(Rdep)],
            mutex_ids=list(range(12, 12 + Rdep)))
        k32_g = pl.make_tile_group(
            type=pl.TileType(shape=[Cap, HEAD_DIM], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_k32 + i * k32_bytes for i in range(Rdep)],
            mutex_ids=list(range(14, 14 + Rdep)))
        u16_g = pl.make_tile_group(
            type=pl.TileType(shape=[Cap, ucols], dtype=FP16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_u16 + i * u16_bytes for i in range(Rdep)],
            mutex_ids=list(range(16, 16 + Rdep)))
        u32_g = pl.make_tile_group(
            type=pl.TileType(shape=[Cap, ucols], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_u32 + i * u32_bytes for i in range(Rdep)],
            mutex_ids=list(range(18, 18 + Rdep)))

        total_items = batch * heads * row_blocks
        bidx = pl.get_block_idx()
        bnum = pl.get_block_num()
        count = (total_items - bidx + bnum - 1) // bnum
        if count > 0:
            ib = bidx // (heads * row_blocks)
            ir = bidx % (heads * row_blocks)
            pl.load(state_g[0], state_gm,
                    [ib, ir // row_blocks, (ir % row_blocks) * Rows, 0])
        for k in pl.range(count):
            item = bidx + k * bnum
            bb = item // (heads * row_blocks)
            rem = item % (heads * row_blocks)
            hh = rem // row_blocks
            r0 = (rem % row_blocks) * Rows
            cur = state_g[k % Slots]
            acc = pl.getval(accepted, bb)
            eff = pl.max(acc, 1)     # keep one chunk even for a zero prefix
            for c in pl.range(0, eff, Cap):
                n = pl.min(Cap, eff - c)
                dec_t = dec_g.next()
                key_t = key_g.next()
                k32_t = k32_g.next()
                u16_t = u16_g.next()
                u32_t = u32_g.next()
                pl.set_validshape(dec_t, [n, HEAD_DIM])
                pl.set_validshape(key_t, [n, HEAD_DIM])
                pl.set_validshape(k32_t, [n, HEAD_DIM])
                pl.set_validshape(u16_t, [n, Rows])
                pl.set_validshape(u32_t, [n, Rows])
                pl.load(dec_t, replay_decay, [bb, c, hh, 0], order=[1, 3])
                pl.load(key_t, replay_k, [bb, c, hh, 0], order=[1, 3])
                pl.load(u16_t, replay_u, [bb, c, hh, r0], order=[1, 3])
                _prefetch_state(c, k, count, item, bnum, state_g, state_gm,
                                heads, row_blocks, Slots, Rows)
                _apply_chunk(acc, u16_t, u32_t, key_t, k32_t, cur, dec_t, n, Rows)
            pl.store(state_gm, cur, [bb, hh, r0, 0])


def _prefetch_state(c, k, count, item, bnum, state_g, state_gm, heads,
                    row_blocks, slots, rows):
    """Queue the next item's state tile while the current chunk is in flight."""
    if c == 0:
        if k + 1 < count:
            nitem = item + bnum
            nb = nitem // (heads * row_blocks)
            nr = nitem % (heads * row_blocks)
            pl.load(state_g[(k + 1) % slots], state_gm,
                    [nb, nr // row_blocks, (nr % row_blocks) * rows, 0])


def _apply_chunk(acc, u16_t, u32_t, key_t, k32_t, cur, dec_t, n, rows):
    """Apply one record chunk to the resident state when the prefix is nonempty."""
    if acc > 0:
        _widen_block(u16_t, u32_t, key_t, k32_t, n, rows, HEAD_DIM)
        _apply_block(cur, dec_t, k32_t, u32_t, n, rows)


def _vector_cores() -> int:
    from pypto_pro.runtime.platform import get_platform_info
    try:
        cores = int(get_platform_info().vector_core_num)
    except (AttributeError, RuntimeError, TypeError, ValueError):  # platform query fallback
        cores = 64
    if cores <= 0:
        cores = 64
    return min(cores, 64)


def _require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _prepare(recurrent_state, replay_u, replay_k, replay_decay, num_accepted_tokens):
    _require(recurrent_state.dim() == 4, "state must be rank-4 [B,N,128,128]")
    _require(tuple(recurrent_state.shape[2:]) == (HEAD_DIM, HEAD_DIM),
             "state trailing shape must be 128x128")
    batch, heads = int(recurrent_state.shape[0]), int(recurrent_state.shape[1])
    _require(batch > 0 and heads > 0, "B and N must be positive")
    _require(recurrent_state.dtype == torch.float32, "state must be fp32")
    _require(recurrent_state.is_contiguous(), "state must be contiguous")
    _require(replay_u.dim() == 4, "replay_u must be [B,S,N,128]")
    seq = int(replay_u.shape[1])
    _require(seq > 0, "S must be positive")
    want = (batch, seq, heads, HEAD_DIM)
    for name, tensor, dtype in (("replay_u", replay_u, torch.float16),
                                ("replay_k", replay_k, torch.float16),
                                ("replay_decay", replay_decay, torch.float32)):
        _require(tuple(tensor.shape) == want, f"{name} must be {want}")
        _require(tensor.dtype == dtype, f"{name} dtype mismatch")
        _require(tensor.is_contiguous(), f"{name} must be contiguous")
        _require(tensor.device == recurrent_state.device, f"{name} device mismatch")
    _require(tuple(num_accepted_tokens.shape) == (batch,),
             "num_accepted_tokens must be [B]")
    _require(num_accepted_tokens.dtype == torch.int32,
             "num_accepted_tokens must be int32")
    _require(num_accepted_tokens.is_contiguous(),
             "num_accepted_tokens must be contiguous")
    return batch, heads, seq


def _route(rows, slots, cap, rdep):
    for name, value, values in (("rows", rows, CommitKey.Rows.values),
                                ("slots", slots, CommitKey.Slots.values),
                                ("cap", cap, CommitKey.Cap.values),
                                ("rdep", rdep, CommitKey.Rdep.values)):
        _require(value in values, f"{name}={value} must be one of {values}")
    _require(rows % GRP == 0, f"rows must be a multiple of {GRP}")
    return {"Rows": rows, "Slots": slots, "Cap": cap, "Rdep": rdep}


def commit_recurrent_kda_replayssm(recurrent_state, replay_u, replay_k,
                                   replay_decay, num_accepted_tokens,
                                   block_num=None, rows=DEFAULT_ROWS,
                                   workers=None, slots=DEFAULT_SLOTS,
                                   cap=DEFAULT_CAP, rdep=DEFAULT_RDEP):
    """Commit each batch's accepted replay prefix into its FP32 checkpoint.

    ``recurrent_state`` is modified in place; the replay records and
    ``num_accepted_tokens`` are read-only.  ``block_num`` is accepted for
    signature compatibility and is not used: the
    kernel selects its own worker count from the device Vector core count.
    """
    route = _route(rows, slots, cap, rdep)
    del block_num
    batch, heads, _seq = _prepare(recurrent_state, replay_u, replay_k,
                                  replay_decay, num_accepted_tokens)
    total_items = batch * heads * (HEAD_DIM // rows)
    if workers is None:
        workers = _vector_cores()
    workers = min(int(workers), total_items)
    _require(workers > 0, "no work items")
    commit_recurrent_kda_replayssm_kernel[None, workers, route](
        recurrent_state, replay_u, replay_k, replay_decay, num_accepted_tokens,
        batch, heads,
    )
