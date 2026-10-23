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
"""mixed_quant_sparse_flash_mla CPU golden（独立数学 oracle，纯 torch）。

三部分：
* host 侧 metadata 规划器（``plan_metadata`` / ``build_metadata``，与参考实现的独立
  host 规划器同构；测试用它生成 kernel 消费的 ``int32[1024]``）；
* 独立 CPU 参考 ``mixed_quant_sparse_flash_mla_golden_cpu``：直接从合同字节池
  解码 FP8 E4M3FN group32 / FP4 E2M1 group16 反量化，做 128-token 分块在线
  softmax（含 sinks、BF16 概率舍入、FP32 累加、FP64 LSE via logsumexp）；
* 输入构造适配层 ``build_inputs`` / ``reference``：按 ``SPEC.md`` 机器合同造输入。

case 参数来自本算子目录下的 ``SPEC.md`` 机器合同块（``p0_cases``）。
"""

import logging
import sys

import torch

logger = logging.getLogger(__name__)

#: Tensor argument order of the kernel signature.
_INPUT_ORDER = (
    "q",
    "ori_kv",
    "ori_sparse_indices",
    "ori_block_table",
    "ori_topk_length",
    "sinks",
    "metadata",
    "cmp_kv",
    "cmp_sparse_indices",
    "cmp_block_table",
    "cmp_topk_length",
    "cu_seqlens_q",
)

AIC_CORE_MAX_NUM = 36
MQSMLA_METADATA_TOTAL_SIZE = 1024
FA_METADATA_SIZE = 9
FA_CORE_ENABLE_INDEX = 0
FA_BN2_START_INDEX = 1
FA_M_START_INDEX = 2
FA_S2_START_INDEX = 3
FA_BN2_END_INDEX = 4
FA_M_END_INDEX = 5
FA_S2_END_INDEX = 6
FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7
FD_USED_VEC_NUM_WORD = 900

# FD (FlashDecode) reduction contract. One split = one row cut by a core
# boundary = two partials. Slot pair (2k, 2k+1) holds the head and tail partial
# of split k; the reducer is core k for k < num_splits.
FD_METADATA_SIZE = 8
FD_METADATA_BASE = AIC_CORE_MAX_NUM * FA_METADATA_SIZE  # 324
FD_CORE_ENABLE_INDEX = 0
FD_BN2_IDX_INDEX = 1
FD_M_IDX_INDEX = 2
FD_WORKSPACE_IDX_INDEX = 3
FD_WORKSPACE_NUM_INDEX = 4
FD_M_START_INDEX = 5
FD_M_NUM_INDEX = 6
FD_HEADS_PER_AIV = 32

TILE_N = 128  # KV tokens per S2 task (matches the kernel contract).
# Flat partition only when the modelled critical path shrinks by at least this
# factor. Measured on the board: a 1.15x-critical-path case nets ~0.98x (the
# barrier + partial traffic eat the balance gain), 1.67x nets 1.145x, so only
# clearly imbalanced cases are worth splitting.
FD_MIN_GAIN = 1.3
# Below this many total tiles the barrier + partial traffic dominates, so keep
# the whole-row schedule (tiny cases are already tile-starved, not imbalanced).
FD_MIN_TOTAL_TILES = 64
# Splitting a short row leaves one-tile parts, which exercise a different (and so
# far not fully characterised) tail path; short-row cases are not the ones FD is
# for anyway (their imbalance came from row granularity at a scale FD cannot fix
# better than the whole-row schedule), so only split rows long enough to matter.
FD_MIN_ROW_TILES = 4


def _row_tiles(w_ori, w_cmp, has_cmp):
    n = (int(w_ori) + TILE_N - 1) // TILE_N if int(w_ori) > 0 else 0
    if has_cmp and int(w_cmp) > 0:
        n += (int(w_cmp) + TILE_N - 1) // TILE_N
    return n


def _batch_of(cu_q, row):
    b = 0
    for i, v in enumerate(cu_q):
        if v <= row:
            b = i
    return b


def _get_cube_core_num(device=None):
    import torch_npu  # noqa: F401

    return int(torch.npu.get_device_properties(device).cube_core_num)


def plan_metadata_uniform(cu_q, len_ori, len_cmp=None, blocks=None):
    """Plan uniform whole-row partition.

    cu_q: list[int] prefix sums (length B+1, cu_q[0]==0, cu_q[-1]==rows).
    len_ori/len_cmp: list[int] per-row valid lengths (values only affect
    future FD planning; uniform planning does not read them).
    Returns torch.int32[1024] CPU tensor.
    """
    rows = int(cu_q[-1])
    if blocks is None:
        blocks = rows
    if not 0 < blocks <= AIC_CORE_MAX_NUM:
        raise ValueError(f"blocks must be in 1..{AIC_CORE_MAX_NUM}, got {blocks}")
    if len(cu_q) < 2 or cu_q[0] != 0:
        raise ValueError("cu_q must start with 0 and have at least 2 entries")
    md = [0] * MQSMLA_METADATA_TOTAL_SIZE
    rows_per_core = rows // blocks
    extra = rows % blocks
    start = 0
    for core in range(blocks):
        count = rows_per_core + (1 if core < extra else 0)
        end = start + count
        if count > 0:
            bn2_s, bn2_e = _batch_of(cu_q, start), _batch_of(cu_q, end)
            m_s = start - cu_q[bn2_s]
            m_e = end - cu_q[bn2_e]
            fa = FA_METADATA_SIZE * core
            md[fa + FA_CORE_ENABLE_INDEX] = 1
            md[fa + FA_BN2_START_INDEX] = bn2_s
            md[fa + FA_M_START_INDEX] = m_s
            md[fa + FA_S2_START_INDEX] = 0
            md[fa + FA_BN2_END_INDEX] = bn2_e
            md[fa + FA_M_END_INDEX] = m_e
            md[fa + FA_S2_END_INDEX] = 0
            md[fa + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = 0
        start = end
    md[FD_USED_VEC_NUM_WORD] = 0
    return torch.tensor(md, dtype=torch.int32)


def _uniform_critical(rows, row_tiles, blocks):
    """Max tiles on any core under the whole-row uniform schedule."""
    ublocks = min(blocks, rows)
    if ublocks <= 0:
        return 0
    worst = 0
    start = 0
    for c in range(ublocks):
        cnt = rows // ublocks + (1 if c < rows % ublocks else 0)
        worst = max(worst, sum(row_tiles[start : start + cnt]))
        start += cnt
    return worst


def plan_metadata_fd(cu_q, len_ori, len_cmp=None, has_cmp=True, blocks=28, min_gain=FD_MIN_GAIN):
    """Plan the flat ``(row, tile)`` interval schedule with 2-way FD partials.

    Returns ``(metadata_int32_1024, block_dim)`` when the critical path shrinks
    by at least ``min_gain`` and every split row stays 2-way; returns ``None``
    otherwise (caller falls back to ``plan_metadata_uniform``).

    The interval end is exclusive when the core stops on a row boundary
    (``s2_end == 0``) and inclusive when it stops inside a row, matching the
    kernel's ``row_count = m_end - m_start + (s2_end > 0)``.
    """
    rows = int(cu_q[-1])
    if rows <= 0 or blocks <= 0:
        return None
    row_tiles = [_row_tiles(len_ori[r], (len_cmp[r] if has_cmp and len_cmp else 0), has_cmp) for r in range(rows)]
    total = sum(row_tiles)
    if total < FD_MIN_TOTAL_TILES or max(row_tiles) < FD_MIN_ROW_TILES:
        return None

    ub_max = _uniform_critical(rows, row_tiles, blocks)
    n_blocks = min(blocks, total)
    flat = [total * c // n_blocks for c in range(n_blocks + 1)]
    fl_max = max(flat[c + 1] - flat[c] for c in range(n_blocks))
    if fl_max <= 0 or ub_max < fl_max * min_gain:
        return None

    rstart = []
    acc = 0
    for r in range(rows):
        rstart.append(acc)
        acc += row_tiles[r]

    def row_at(flat_pos):
        r = 0
        for i in range(rows):
            if rstart[i] <= flat_pos:
                r = i
        return r

    splits = []
    for c in range(1, n_blocks):
        f = flat[c]
        r = row_at(f)
        if rstart[r] < f < rstart[r] + row_tiles[r]:
            splits.append((c, r))
    if len(splits) > n_blocks or len(splits) * 2 > 2 * AIC_CORE_MAX_NUM:
        return None
    # A row cut by two consecutive boundaries would need a 3-way merge.
    for i in range(1, len(splits)):
        if splits[i][1] == splits[i - 1][1]:
            return None

    md = [0] * MQSMLA_METADATA_TOTAL_SIZE
    for c in range(n_blocks):
        f0, f1 = flat[c], flat[c + 1]
        if f1 == f0:
            continue
        r0, r1 = row_at(f0), row_at(f1 - 1)
        e = f1 - rstart[r1]
        r1e = r1 if e < row_tiles[r1] else r1 + 1
        b0, b1 = _batch_of(cu_q, r0), _batch_of(cu_q, r1e)
        fa = FA_METADATA_SIZE * c
        md[fa + FA_CORE_ENABLE_INDEX] = 1
        md[fa + FA_BN2_START_INDEX] = b0
        md[fa + FA_M_START_INDEX] = r0 - cu_q[b0]
        md[fa + FA_S2_START_INDEX] = f0 - rstart[r0]
        md[fa + FA_BN2_END_INDEX] = b1
        md[fa + FA_M_END_INDEX] = r1e - cu_q[b1]
        md[fa + FA_S2_END_INDEX] = 0 if e >= row_tiles[r1] else e

    # Split k occupies workspace slots (2k head, 2k+1 tail); reducer is core k.
    # FA word 7/8 carry "slot + 1" for the partial this core must write (0 = none).
    for k, (c, r) in enumerate(splits):
        head_ws = 2 * k
        tail_ws = 2 * k + 1
        md[FA_METADATA_SIZE * c + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = head_ws + 1
        md[FA_METADATA_SIZE * (c - 1) + 8] = tail_ws + 1
        b = _batch_of(cu_q, r)
        for half in range(2):
            base = FD_METADATA_BASE + (2 * k + half) * FD_METADATA_SIZE
            md[base + FD_CORE_ENABLE_INDEX] = 1
            md[base + FD_BN2_IDX_INDEX] = b
            md[base + FD_M_IDX_INDEX] = r - cu_q[b]
            md[base + FD_WORKSPACE_IDX_INDEX] = head_ws
            md[base + FD_WORKSPACE_NUM_INDEX] = 2
            md[base + FD_M_START_INDEX] = FD_HEADS_PER_AIV * half
            md[base + FD_M_NUM_INDEX] = FD_HEADS_PER_AIV
    md[FD_USED_VEC_NUM_WORD] = len(splits)
    return torch.tensor(md, dtype=torch.int32), n_blocks


def plan_metadata(cu_q, len_ori, len_cmp=None, has_cmp=True, blocks=28, min_gain=FD_MIN_GAIN):
    """Dispatch: flat FD schedule when it pays, else uniform whole rows.

    Returns ``(metadata_int32_1024, block_dim)``. ``len_ori``/``len_cmp`` are
    per-row effective widths (already clamped to K1/K2) and are required for the
    FD cost model.
    """
    if len_ori is not None:
        fd = plan_metadata_fd(cu_q, len_ori, len_cmp, has_cmp, blocks, min_gain)
        if fd is not None:
            return fd
    ublocks = min(blocks, max(int(cu_q[-1]), 1))
    return plan_metadata_uniform(cu_q, len_ori, len_cmp, blocks=ublocks), ublocks


def build_metadata(cu_q_tensor, len_ori_tensor, len_cmp_tensor=None, has_cmp=True, blocks=None, k_ori=None, k_cmp=None):
    """Tensor-input convenience wrapper: reads values on host (D2H)."""
    cu_q = [int(v) for v in cu_q_tensor.cpu().reshape(-1).tolist()]
    w_o = [int(v) for v in len_ori_tensor.cpu().reshape(-1).tolist()]
    w_c = None
    if len_cmp_tensor is not None:
        w_c = [int(v) for v in len_cmp_tensor.cpu().reshape(-1).tolist()]
    if k_ori is not None:
        w_o = [min(k_ori, max(v, 0)) for v in w_o]
    if w_c is not None and k_cmp is not None:
        w_c = [min(k_cmp, max(v, 0)) for v in w_c]
    if blocks is None:
        blocks = _get_cube_core_num()
    return plan_metadata(cu_q, w_o, w_c or ([0] * len(w_o)), has_cmp, blocks)


D = 512
N1 = 64
TILE_N = 128
FP4_LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])


def load_case_params():
    import json
    import re
    from pathlib import Path as _Path

    text = (_Path(__file__).resolve().parents[2] / "SPEC.md").read_text()
    block = re.findall(r"(?ms)^```json machine-contract\s*\n(.*?)^```\s*$", text)[0]
    contract = json.loads(block)
    return {case["name"]: case for case in contract["p0_cases"]}


_CASE_LAYOUTS = {
    "mixed_decode": [1, 1],
    "ori_only": [1, 1],
    "empty_and_tail": [3, 0, 5],
    "padded_pages": [2, 3],
    "single_token_k": [4],
    "narrow_k_tail": [4],
    "cmp_only_rows": [3],
    "multirow_42": [42],
    "selective_fd_72": [6] * 12,
    "omit_lengths": [1, 1],
    "decode_1": [1] * 32,
    "decode_6": [6] * 32,
    "decode_6_long_s2": [6] * 12,
}
_CASE_ORDER = list(_CASE_LAYOUTS)


# B3 capability: the kernel now supports the full block table (bt = pages,
# ≤1024).  Cases whose page count was already ≤128 are byte-identical to the
# previous min(pages, 128) contract because min(pages, 128) == pages for them.
def _batch_of(cu, row):
    lo = 0
    for b, v in enumerate(cu):
        if v <= row:
            lo = b
    return lo


def _decode_ori_rows(rows):
    data = rows[..., :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = rows[..., 512:].contiguous().view(torch.bfloat16).float()
    return (data * scales.repeat_interleave(32, dim=-1)).bfloat16()


def _decode_cmp_rows(rows):
    data = rows[..., :256].contiguous()
    scales = rows[..., 256:].contiguous().view(torch.bfloat16).float()
    codes = torch.stack(((data & 0xF).long(), (data >> 4).long()), dim=-1)
    values = FP4_LUT[codes.reshape(*data.shape[:-1], 512)]
    return (values * scales.repeat_interleave(16, dim=-1)).bfloat16()


def _gather_logical(pool, block_table, indices, lengths, cu, side):
    rows_n = indices.shape[0]
    bs = pool.shape[1]
    width = pool.shape[-1]
    pool_flat = pool.reshape(-1, width)
    logical = torch.zeros(rows_n, indices.shape[-1], D).bfloat16()
    for r in range(rows_n):
        b = _batch_of(cu, r)
        n = int(lengths[r])
        if n == 0:
            continue
        idx = indices[r, :n].long()
        page = torch.where(idx >= 0, idx // bs, torch.zeros_like(idx))
        off = torch.where(idx >= 0, idx % bs, torch.zeros_like(idx))
        phys = block_table[b][page]
        flat = (phys.long() * bs + off).clamp(max=pool_flat.shape[0] - 1)
        packed = pool_flat[flat]
        if side == "ori":
            logical[r, :n] = _decode_ori_rows(packed)
        else:
            logical[r, :n] = _decode_cmp_rows(packed)
    return logical


def mixed_quant_sparse_flash_mla_golden_cpu(
    q,
    ori_kv,
    ori_sparse_indices,
    ori_block_table,
    ori_topk_length,
    sinks,
    metadata,
    cmp_kv,
    cmp_sparse_indices,
    cmp_block_table,
    cmp_topk_length,
    cu_seqlens_q,
    **params,
):
    """Implement the SPEC formula on CPU; return SPEC output dtypes."""
    scale = float(params["softmax_scale"])
    has_cmp = bool(params.get("has_cmp", True))
    rows = q.shape[0]
    cu = cu_seqlens_q.tolist()
    len_o = ori_topk_length.reshape(-1)
    logical_o = _gather_logical(ori_kv, ori_block_table, ori_sparse_indices.reshape(rows, -1), len_o, cu, "ori")
    if has_cmp:
        len_c = cmp_topk_length.reshape(-1)
        logical_c = _gather_logical(cmp_kv, cmp_block_table, cmp_sparse_indices.reshape(rows, -1), len_c, cu, "cmp")
    else:
        len_c = torch.zeros(rows, dtype=torch.int32)
        logical_c = None
    qf = q.float()
    sinks_f = sinks.float()
    out = torch.zeros(rows, N1, D)
    lse_full = torch.zeros(1, rows, N1)
    for row in range(rows):
        sides = [logical_o[row, : int(len_o[row])].float()]
        if has_cmp:
            sides.append(logical_c[row, : int(len_c[row])].float())
        sides = [s for s in sides if s.shape[0] > 0]
        if not sides:
            out[row] = 0
            lse_full[0, row] = sinks_f
            continue
        kv64 = torch.cat(sides).double()
        logits = qf[row].double() @ kv64.T * scale
        all_logits = torch.cat((logits, sinks_f.double()[:, None]), dim=1)
        lse_full[0, row] = torch.logsumexp(all_logits, dim=1).float()
        row_max = sinks_f.clone()
        row_sum = torch.ones(N1)
        accumulator = torch.zeros(N1, D)
        for values in sides:
            for start in range(0, len(values), TILE_N):
                tile = values[start : start + TILE_N]
                scores = qf[row] @ tile.T * scale
                new_max = torch.maximum(row_max, scores.max(dim=1).values)
                correction = torch.exp(row_max - new_max)
                weights = torch.exp(scores - new_max[:, None])
                row_sum = row_sum * correction + weights.sum(dim=1)
                accumulator = accumulator * correction[:, None] + weights.bfloat16().float() @ tile
                row_max = new_max
        out[row] = (accumulator / row_sum[:, None]).bfloat16()
    return {"out": out.bfloat16(), "lse": lse_full.float()}


def _build_case_tensors(case, device=torch.device("cpu")):
    import numpy as np

    p = dict(case)
    generator = torch.Generator().manual_seed(42)
    spans = _CASE_LAYOUTS[p["name"]]
    batch, rows = len(spans), sum(spans)
    has_cmp = p["has_cmp"]
    q = (torch.randn((rows, N1, D), generator=generator) * 0.25).bfloat16()
    sinks = torch.randn(N1, generator=generator)
    cu = torch.tensor([0, *np.cumsum(spans).tolist()], dtype=torch.int32)
    tensors = {"q": q, "sinks": sinks, "cu_seqlens_q": cu}
    for side, fp4 in (("ori", False), ("cmp", True)):
        tokens = p[f"s_{side}"]
        bs = p[f"bs_{side}"]
        width = p[f"k_{side}"]
        group = 16 if fp4 else 32
        row_bytes = 320 if fp4 else 544
        pages = (tokens + bs - 1) // bs
        window = pages
        shape = (batch, pages * bs, D)
        scales = (0.125 + torch.rand((*shape[:-1], D // group), generator=generator) * 0.375).bfloat16()
        if fp4:
            codes = torch.randint(0, 16, shape, generator=generator, dtype=torch.uint8)
            encoded = codes[..., 0::2] | (codes[..., 1::2] << 4)
        else:
            values = (torch.randn(shape, generator=generator) * 2).to(torch.float8_e4m3fn)
            encoded = values.view(torch.uint8)
        packed = torch.cat((encoded, scales.view(torch.uint8)), dim=-1)
        table = torch.randperm(batch * pages, generator=generator).reshape(batch, pages)[:, :window].int().contiguous()
        packed_pages = packed.reshape(batch, pages, bs, -1)
        physical = torch.empty_like(packed_pages.reshape(batch * pages, bs, -1))
        physical[table.long().flatten()] = packed_pages[:, :window].reshape(batch * window, bs, physical.shape[-1])
        pool = physical.reshape(batch * pages, bs, 1, row_bytes)
        stride0 = bs * row_bytes
        if p.get("kv_axis0_noncontiguous", False) and side == "ori":
            stride0 = bs * row_bytes + 64
            backing = torch.full((batch * pages * stride0,), 0xA5, dtype=torch.uint8)
            pool = torch.as_strided(backing, (batch * pages, bs, 1, row_bytes), (stride0, row_bytes, row_bytes, 1))
            pool.copy_(physical.reshape(batch * pages, bs, 1, row_bytes))
        indices = torch.randint(min(tokens, window * bs), (rows, width), generator=generator, dtype=torch.int32)
        lengths = torch.full((rows,), width, dtype=torch.int32)
        if p.get("vary_lengths", False):
            lengths = torch.tensor(([0, 1, min(127, width), width] * ((rows + 3) // 4))[:rows], dtype=torch.int32)
        if p.get("full_lengths", False):
            lengths.fill_(width)
        empty_cmp = side == "cmp" and not has_cmp
        empty_ori = side == "ori" and p.get("zero_ori", False)
        if empty_cmp or empty_ori:
            lengths.zero_()
        indices[torch.arange(width)[None, :] >= lengths[:, None]] = -1
        if side == "ori":
            tensors["ori_kv"] = pool
            tensors["ori_sparse_indices"] = indices.reshape(rows, 1, -1)
            tensors["ori_block_table"] = table
            tensors["ori_topk_length"] = lengths.reshape(rows, 1)
            tensors["ori_kv_stride0"] = stride0
        else:
            tensors["cmp_kv"] = pool
            tensors["cmp_sparse_indices"] = indices.reshape(rows, 1, -1)
            tensors["cmp_block_table"] = table
            tensors["cmp_topk_length"] = lengths.reshape(rows, 1)
            tensors["cmp_kv_stride0"] = stride0
    if not has_cmp:
        tensors["cmp_kv"] = torch.zeros(1, 1, 1, 320, dtype=torch.uint8)
        tensors["cmp_sparse_indices"] = torch.zeros(rows, 1, 1, dtype=torch.int32)
        tensors["cmp_block_table"] = torch.zeros(batch, 1, dtype=torch.int32)
        tensors["cmp_topk_length"] = torch.zeros(rows, 1, dtype=torch.int32)
        tensors["cmp_kv_stride0"] = 320
    blocks = min(28, max(1, rows))
    tensors["metadata"] = plan_metadata_uniform(cu.tolist(), None, None, blocks=blocks)
    ordered = [tensors.get(name) for name in _INPUT_ORDER]
    scalars = {
        "ori_kv_stride0": tensors["ori_kv_stride0"],
        "cmp_kv_stride0": tensors["cmp_kv_stride0"],
        "block_dim": blocks,
    }
    if device is not None and str(device) != "cpu":
        ordered = [t.to(device) for t in ordered]
    return ordered, scalars


def _make_inputs(device=torch.device("cpu")):
    """Return every SPEC P0 as (case_name, positional_tensor_args, scalar_kwargs)."""
    spec_cases = load_case_params()
    cases = []
    for name in _CASE_ORDER:
        case = dict(spec_cases[name]["params"])
        case["name"] = name
        args, scalars = _build_case_tensors(case, device)
        cases.append((name, args, scalars))
    return cases


def _validate():
    """Check every P0's output shape/dtype and math invariants."""
    spec_cases = load_case_params()
    ok = True
    for name in _CASE_ORDER:
        case = dict(spec_cases[name]["params"])
        case["name"] = name
        args, _ = _build_case_tensors(case)
        result = mixed_quant_sparse_flash_mla_golden_cpu(*args, **case)
        checks = {
            "out_dtype": result["out"].dtype == torch.bfloat16,
            "out_shape": tuple(result["out"].shape) == tuple(args[0].shape),
            "out_finite": bool(torch.isfinite(result["out"].float().view(-1)).all()),
            "lse_dtype": result["lse"].dtype == torch.float32,
            "lse_shape": tuple(result["lse"].shape) == (1, args[0].shape[0], N1),
            "lse_finite": bool(torch.isfinite(result["lse"].view(-1)).all()),
        }
        len_o = args[4].reshape(-1)
        len_c = args[10].reshape(-1) if case["has_cmp"] else torch.zeros(args[0].shape[0], dtype=torch.int32)
        empty = (len_o == 0) & (len_c == 0)
        if bool(empty.any()):
            eo = result["out"].reshape(args[0].shape[0], N1, D)[empty]
            checks["empty_out_zero"] = bool(torch.count_nonzero(eo) == 0)
            el = result["lse"][0, empty]
            checks["empty_lse_sink"] = bool(torch.equal(el, args[5].expand(int(empty.sum()), N1)))
        case_ok = all(bool(v) for v in checks.values())
        ok = ok and case_ok
        logger.info("%s %s", name, "PASS" if case_ok else f"FAIL {checks}")
    return ok


def _case(name):
    case = dict(load_case_params()[name]["params"])
    case["name"] = name
    return case


def build_inputs(name, device=torch.device("cpu")):
    case = _case(name)
    ordered, _scalars = _build_case_tensors(case, device)
    inputs = dict(zip(_INPUT_ORDER, ordered))
    # FD 规划：host 侧按 (row,tile) 均衡调度（长行且失衡足够大时才切行）
    w_o = [min(int(case["k_ori"]), max(int(v), 0)) for v in inputs["ori_topk_length"].reshape(-1).tolist()]
    w_c = (
        [min(int(case["k_cmp"]), max(int(v), 0)) for v in inputs["cmp_topk_length"].reshape(-1).tolist()]
        if case["has_cmp"]
        else [0] * len(w_o)
    )
    cu = [int(v) for v in inputs["cu_seqlens_q"].reshape(-1).tolist()]
    md, block_dim = plan_metadata(cu, w_o, w_c, bool(case["has_cmp"]), 28)
    inputs["metadata"] = md
    return inputs, block_dim, case


def reference(name, inputs):
    case = _case(name)
    ordered = [inputs[k] for k in _INPUT_ORDER]
    return mixed_quant_sparse_flash_mla_golden_cpu(*ordered, **case)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = sys.argv[1] if len(sys.argv) > 1 else "mixed_decode"
    ins, bd, _ = build_inputs(n)
    logger.info("%s %s block_dim %s", n, {k: tuple(v.shape) for k, v in ins.items()}, bd)
