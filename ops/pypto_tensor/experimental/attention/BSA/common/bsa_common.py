#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
BSA (Block Sparse Attention) Shared Configuration and Utilities

Contains all shared code used by both forward and backward:
  - BSAConfig / DEFAULT_CONFIG
  - Golden helpers (_resolve_defaults, _block_ranges, _is_valid_mask)
  - Impl helpers (_pad_to_block_aligned, _is_dense_mask, _make_jit_opts)
  - Compacted KV builders (shared by forward and backward dQ)
  - Mask generation (generate_block_sparse_mask)
"""

import math
import os
from collections import namedtuple
from dataclasses import dataclass
import torch


@dataclass
class BSAConfig:
    """Centralized configuration for BSA (Block Sparse Attention)."""

    # --- Shape Constraints ---
    head_dim: int = 128
    block_shape_x: int = 256
    block_shape_y: int = 512

    # --- Data Types ---
    torch_dtype: torch.dtype = torch.float16
    accum_torch_dtype: torch.dtype = torch.float32

    # --- Numerical Constants ---
    large_neg: float = -65504.0
    lse_init: float = float('-inf')
    lse_pad_value: float = 1e30

    # --- Precision Tolerance (per BSA.md) ---
    fwd_atol: float = 0.0001
    fwd_rtol: float = 0.0078125
    bwd_atol: float = 0.0001
    bwd_rtol: float = 0.0078125

    # --- Derived helpers ---
    @property
    def softmax_scale(self) -> float:
        return self.head_dim ** -0.5

    # --- Validation ---
    def __post_init__(self):
        assert self.head_dim == 128, f"head_dim must be 128, got {self.head_dim}"
        assert self.block_shape_x % 64 == 0
        assert self.block_shape_y % 64 == 0
        assert self.block_shape_y >= 128
        assert self.torch_dtype == torch.float16


DEFAULT_CONFIG = BSAConfig()
BSAForwardResult = namedtuple('BSAForwardResult', ['o', 'lse'])
BSABackwardResult = namedtuple('BSABackwardResult', ['d_q', 'd_k', 'd_v'])
ResolveDefaultsResult = namedtuple('ResolveDefaultsResult',
    ['b', 'hq', 'hkv', 'sq', 'skv', 'd', 'bx', 'by', 'scale', 'asq', 'askv'])
SparseQDkdvResult = namedtuple('SparseQDkdvResult',
    ['q_compact', 'do_compact', 'o_compact', 'lse_compact', 'inner_mask', 'max_inner'])

_SparseKVConfig = namedtuple('_SparseKVConfig',
    ['b', 'hq', 'hkv', 'sq', 'skv', 'sq_pad', 'skv_pad',
     'num_qb', 'num_kb', 'bx', 'by', 'd', 'device',
     'actual_seq_lengths', 'actual_seq_lengths_kv',
     'q_2d', 'do_2d', 'o_2d', 'lse_2d'])
_KVFillConfig = namedtuple('_KVFillConfig',
    ['hkv', 'skv_pad', 'bx', 'by', 'd', 'device', 'total_qblocks', 'k_2d', 'v_2d'])
_QDkdvFillConfig = namedtuple('_QDkdvFillConfig',
    ['hq', 'sq_pad', 'bx', 'by', 'd', 'device', 'total_kv', 'q_2d', 'do_2d', 'o_2d', 'lse_2d'])
_KVBoundaryCfg = namedtuple('_KVBoundaryCfg', ['max_sel', 'bx', 'num_kb'])
_QBoundaryCfg = namedtuple('_QBoundaryCfg', ['max_sel', 'bx', 'num_qb'])
_QBoundaryInnerCfg = namedtuple('_QBoundaryInnerCfg', ['max_inner', 'bx', 'num_qb'])
_CollectKVCfg = namedtuple('_CollectKVCfg', ['b', 'hq', 'hkv', 'num_qb', 'num_kb'])
_CollectQCfg = namedtuple('_CollectQCfg', ['b', 'hq', 'hkv', 'num_qb', 'num_kb'])
_ValidQForKVBlockCfg = namedtuple('_ValidQForKVBlockCfg',
    ['b_idx', 'h_kv_idx', 'v_blk', 'group', 'num_qb', 'nkv_cols'])

# Shared Q/K/V preparation result (used by both FWD and BWD impls)
_QKVPrepared = namedtuple('_QKVPrepared',
    ['bx', 'by', 'b', 'hq', 'hkv', 'sq', 'skv', 'd',
     'num_qb', 'num_kb', 'sq_pad', 'skv_pad',
     'q_2d', 'k_2d', 'v_2d'])

# Sparse KV building result (used by dispatch functions in FWD and BWD)
_SparseKVResult = namedtuple('_SparseKVResult',
    ['k_compact', 'v_compact', 'valid_mask', 'max_sel'])


def _prepare_and_build_sparse_kv(call_inputs):
    """Shared Q/K/V preparation + sparse KV building (used by both FWD and BWD).

    Encapsulates _prepare_qkv_2d and _build_sparse_kv_cached with default
    FWD-style config (q_2d=None, do_2d=None, o_2d=None, lse_2d=None),
    eliminating duplicate preparation code in fwd/bwd impl files.
    Returns (prepared, sparse_kv_result) as namedtuples.
    """
    prepared = _prepare_qkv_2d(
        call_inputs.query, call_inputs.key, call_inputs.value,
        call_inputs.block_shape, call_inputs.cfg)
    k_compact, v_compact, valid_mask, max_sel = _build_sparse_kv_cached(
        call_inputs.block_sparse_mask, prepared.k_2d, prepared.v_2d,
        _SparseKVConfig(
            b=prepared.b, hq=prepared.hq, hkv=prepared.hkv,
            sq=prepared.sq, skv=prepared.skv,
            sq_pad=prepared.sq_pad, skv_pad=prepared.skv_pad,
            num_qb=prepared.num_qb, num_kb=prepared.num_kb,
            bx=prepared.bx, by=prepared.by, d=prepared.d,
            device=call_inputs.query.device,
            actual_seq_lengths=call_inputs.actual_seq_lengths,
            actual_seq_lengths_kv=call_inputs.actual_seq_lengths_kv,
            q_2d=None, do_2d=None, o_2d=None, lse_2d=None))
    return prepared, _SparseKVResult(
        k_compact=k_compact, v_compact=v_compact,
        valid_mask=valid_mask, max_sel=max_sel)


_VEC_TILE_LOAD = (128, 128)
_CUBE_TILE = (128, 128)
_VEC_TILE_OUTPUT = (16, 128, 128)
_CUBE_TILE_LIST = list(_CUBE_TILE)
_DEVICE_SCHED_MODE = 3
_VEC_NBUFFER_SETTING = {}
_CUBE_NBUFFER_SETTING = {-1: 4}
_CUBE_L1_REUSE_SETTING = {-1: 16}
_RUNTIME_DEBUG_MODE = int(os.environ.get('BSA_RUNTIME_DEBUG_MODE', '0'))
_MASK_CACHE = {}
_Q_DKDV_CACHE = {}


def _snapshot_output_dirs(search_bases):
    """Collect timestamped output subdirectories across all search bases."""
    all_dirs = {}  # name -> (base, full_path)
    for base in search_bases:
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            full = os.path.join(base, name)
            if os.path.isdir(full):
                all_dirs[name] = (base, full)
    return all_dirs


def _find_newest_created_dir(before, search_bases):
    """Find the newest output dir created since *before*.

    Looks for dirs with program.json (compilation marker), optionally also
    merged_swimlane.json (runtime profiling marker). The swimlane file may
    not exist yet at compile time — it's written after kernel execution.

    Args:
        before: dict from _snapshot_output_dirs() taken before kernel
            compilation — keys are dir names, values are (base, full_path).
        search_bases: list of directory paths to search for output subdirs.
    """
    after = _snapshot_output_dirs(search_bases)
    created = {name: after[name] for name in after if name not in before}
    best_dir, best_mt = None, 0.0
    for _, (base, full) in created.items():
        # Prefer dirs with merged_swimlane.json (runtime profiling complete),
        # but also accept dirs with program.json (compilation happened).
        swim = os.path.join(full, "merged_swimlane.json")
        prog = os.path.join(full, "program.json")
        if os.path.isfile(swim):
            mt = os.path.getmtime(swim)
            if mt > best_mt:
                best_dir, best_mt = full, mt
        elif os.path.isfile(prog) and best_dir is None:
            mt = os.path.getmtime(prog)
            best_dir = full
            best_mt = mt
    return best_dir


def _make_jit_opts(cfg, total_outer=None, *, extra_pass_options=None,
                    extra_runtime_options=None):
    """Build JIT options for pypto.frontend.jit.

    Args:
        extra_pass_options: optional dict merged into pass_options.
        extra_runtime_options: optional dict merged into runtime_options.
    """
    pass_opts = {
        "cube_l1_reuse_setting": _CUBE_L1_REUSE_SETTING,
        "vec_nbuffer_setting": _VEC_NBUFFER_SETTING,
        "cube_nbuffer_setting": _CUBE_NBUFFER_SETTING,
    }
    if extra_pass_options:
        pass_opts.update(extra_pass_options)
    rt_opts = {
        "device_sched_mode": _DEVICE_SCHED_MODE,
    }
    if extra_runtime_options:
        rt_opts.update(extra_runtime_options)
    return dict(
        runtime_options=rt_opts,
        pass_options=pass_opts,
        debug_options={"runtime_debug_mode": _RUNTIME_DEBUG_MODE},
    )


def _resolve_defaults(query, key, block_shape_x, block_shape_y,
                      actual_seq_lengths, actual_seq_lengths_kv,
                      scale_value, cfg):
    """Resolve default parameters and validate shapes."""
    b, hq, sq, d = query.shape
    _, hkv, skv, _ = key.shape

    bx = block_shape_x or cfg.block_shape_x
    by = block_shape_y or cfg.block_shape_y
    scale = scale_value if scale_value is not None else cfg.softmax_scale
    asq = actual_seq_lengths if actual_seq_lengths is not None else torch.full([b], sq, dtype=torch.int64)
    askv = actual_seq_lengths_kv if actual_seq_lengths_kv is not None else torch.full([b], skv, dtype=torch.int64)

    assert d == cfg.head_dim, f"head_dim must be {cfg.head_dim}, got {d}"
    assert hq >= hkv and hq % hkv == 0, \
        f"GQA constraint: hq({hq}) >= hkv({hkv}) and hq % hkv == 0"

    return ResolveDefaultsResult(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, d=d,
                                  bx=bx, by=by, scale=scale, asq=asq, askv=askv)


def _block_ranges(seq_len, block_size, actual_len):
    """Yield (block_index, start, end) for each block up to actual_len."""
    num_blocks = math.ceil(actual_len / block_size)
    for blk in range(num_blocks):
        start = blk * block_size
        end = min(start + block_size, actual_len)
        yield blk, start, end


def _is_valid_mask(block_sparse_mask, b, h_q, u, v):
    """Check whether mask[b, h_q, u, v] is True, with bounds guard."""
    if u >= block_sparse_mask.shape[2] or v >= block_sparse_mask.shape[3]:
        return False
    return block_sparse_mask[b, h_q, u, v].item()


def _pad_to_block_aligned(tensor, block_size):
    b, h, s, d = tensor.shape
    s_padded = math.ceil(s / block_size) * block_size
    if s == s_padded:
        return tensor, s_padded
    return torch.nn.functional.pad(tensor, (0, 0, 0, s_padded - s), mode='constant', value=0.0), s_padded


def _prepare_qkv_2d(query, key, value, block_shape, cfg):
    """Common setup: resolve block shapes, compute derived dims, pad & reshape Q/K/V to 2D.

    Shared by both forward and backward implementations.
    Returns _QKVPrepared namedtuple grouping all derived shape values and 2D tensors.
    """
    bx = block_shape[0] if block_shape else cfg.block_shape_x
    by = block_shape[1] if block_shape else cfg.block_shape_y

    b, hq, sq, d = query.shape
    _, hkv, skv, _ = key.shape

    num_qb = math.ceil(sq / bx)
    num_kb = math.ceil(skv / by)
    sq_pad = num_qb * bx
    skv_pad = num_kb * by

    q_pad, _ = _pad_to_block_aligned(query, bx)
    k_pad, _ = _pad_to_block_aligned(key, by)
    v_pad, _ = _pad_to_block_aligned(value, by)

    q_2d = q_pad.reshape(b * hq * sq_pad, d)
    k_2d = k_pad.reshape(b * hkv * skv_pad, d)
    v_2d = v_pad.reshape(b * hkv * skv_pad, d)

    return _QKVPrepared(
        bx=bx, by=by, b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, d=d,
        num_qb=num_qb, num_kb=num_kb, sq_pad=sq_pad, skv_pad=skv_pad,
        q_2d=q_2d, k_2d=k_2d, v_2d=v_2d)


def _is_dense_mask(block_sparse_mask):
    return block_sparse_mask.all().item()


def _apply_kv_boundary_mask(valid_mask, qblock_info, kv_cfg, remaining_kv_per_batch):
    """Zero out columns in valid_mask for the last KV block's padded rows.

    Supports per-batch actual_seq_lengths_kv for non-aligned sequences.
    When all batch items have the same actual length (aligned case),
    remaining_kv_per_batch is a single value (backward compatible).

    Args:
        kv_cfg: _KVBoundaryCfg(max_sel, bx, num_kb) namedtuple.
        remaining_kv_per_batch: either a single int (all batches same, aligned case)
            or a list/tensor of per-batch remaining KV lengths (non-aligned case).
    """
    max_sel, bx, num_kb = kv_cfg.max_sel, kv_cfg.bx, kv_cfg.num_kb

    def _zero_kv_boundary(qblk_idx, valid_kv_blocks, rem_kv):
        for j, v_blk in enumerate(valid_kv_blocks):
            if v_blk == num_kb - 1:
                m_start = qblk_idx * max_sel * bx + j * bx
                valid_mask[m_start:m_start + bx, int(rem_kv):] = 0.0

    if isinstance(remaining_kv_per_batch, (int, float)):
        for i, (b_idx, h_kv_idx, valid_v) in enumerate(qblock_info):
            _zero_kv_boundary(i, valid_v, remaining_kv_per_batch)
    else:
        for i, (b_idx, h_kv_idx, valid_v) in enumerate(qblock_info):
            _zero_kv_boundary(i, valid_v, remaining_kv_per_batch[b_idx])


def _apply_q_boundary_mask(valid_mask, total_qblocks, q_cfg, remaining_q_per_batch):
    """Zero out rows in valid_mask for the last Q block's padded rows.

    Supports per-batch actual_seq_lengths for non-aligned sequences.
    When all batch items have the same actual length (aligned case),
    remaining_q_per_batch is a single value (backward compatible).

    Args:
        q_cfg: _QBoundaryCfg(max_sel, bx, num_qb) namedtuple.
        remaining_q_per_batch: either a single int (all batches same) or per-batch list/tensor.
    """
    max_sel, bx, num_qb = q_cfg.max_sel, q_cfg.bx, q_cfg.num_qb

    def _zero_q_boundary(qblk_idx, rem_q_len):
        if qblk_idx % num_qb == num_qb - 1 and rem_q_len < bx:
            for j in range(max_sel):
                m_dst = qblk_idx * max_sel * bx + j * bx
                valid_mask[m_dst + rem_q_len:m_dst + bx, :] = 0.0

    if isinstance(remaining_q_per_batch, (int, float)):
        rem_q = int(remaining_q_per_batch)
        for i in range(total_qblocks):
            _zero_q_boundary(i, rem_q)
    else:
        b_count = len(remaining_q_per_batch)
        hq_total = total_qblocks // (b_count * num_qb) if b_count > 0 else total_qblocks // num_qb
        for i in range(total_qblocks):
            b_idx = i // (hq_total * num_qb)
            _zero_q_boundary(i, remaining_q_per_batch[b_idx])


def _collect_valid_kv_per_qblock(block_sparse_mask, kv_cfg, actual_seq_lengths_kv=None, by=None):
    """Phase 1 of _build_sparse_kv: collect valid KV indices per Q block.

    Args:
        kv_cfg: _CollectKVCfg(b, hq, hkv, num_qb, num_kb) namedtuple.
        actual_seq_lengths_kv: per-batch actual KV lengths (for non-aligned filtering).
        by: KV block size (block_shape_y).
    """
    b, hq, hkv, num_qb, num_kb = kv_cfg.b, kv_cfg.hq, kv_cfg.hkv, kv_cfg.num_qb, kv_cfg.num_kb
    group = hq // hkv
    nkv_cols = block_sparse_mask.shape[3]
    qblock_info = []
    max_sel = 0
    # 追踪没有真实有效 KV 的 dummy Q blocks ----
    # 当某个 Q block 在 actual KV 范围内没有任何 mask=True 的 KV block 时，
    # valid_v 为空，会回退到 [0]（dummy block）。这些 dummy block 会导致
    # kernel 的 online softmax 产生非零输出，需要记录并在后续清零。
    dummy_qblocks = set()

    for flat_idx in range(b * hq):
        b_idx = flat_idx // hq
        h_q = flat_idx % hq
        h_kv_idx = h_q // group
        # 限制每个batch的KV block搜索范围 ----
        # 搜索范围是 range(num_kb)，num_kb 来自 skv_max（所有batch最大值），
        # shorter batch 会收集到超出其真实 KV 长度的 block（全padding数据）。
        # 对每个 batch 计算 actual_num_kb，只在真实范围内搜索。
        if actual_seq_lengths_kv is not None and by is not None:
            # 例：batch 0 skv=2264, by=512 → actual_num_kb=5（blocks 0-4 有效）
            # batch 1 skv=3603, by=512 → actual_num_kb=8（blocks 0-7 有效）
            actual_num_kb = math.ceil(int(actual_seq_lengths_kv[b_idx]) / by)
            max_v = min(actual_num_kb, min(num_kb, nkv_cols))
        else:
            max_v = min(num_kb, nkv_cols)
        for u in range(num_qb):
            valid_v = [v for v in range(max_v)
                       if block_sparse_mask[b_idx, h_q, u, v].item()]
            if not valid_v:
                # 该 Q block 在实际范围内没有有效 KV block → 使用 dummy block 0
                valid_v = [0]
                # 记录 dummy Q block 的全局索引，后续在 _build_sparse_kv_cached 中清零
                qblock_idx = flat_idx * num_qb + u
                dummy_qblocks.add(qblock_idx)
            qblock_info.append((b_idx, h_kv_idx, valid_v))
            max_sel = max(max_sel, len(valid_v))

    # 新增 dummy_qblocks：没有真实有效 KV 的 Q block 索引集合
    return qblock_info, max(max_sel, 1), dummy_qblocks


def _fill_compacted_kv(qblock_info, max_sel, valid_mask, fill_cfg):
    """Phase 2 of _build_sparse_kv: build compacted K/V tensors.

    Args:
        fill_cfg: _KVFillConfig namedtuple including k_2d, v_2d as data fields.
    """
    hkv, skv_pad, bx, by, d, device, total_qblocks, k_2d, v_2d = (
        fill_cfg.hkv, fill_cfg.skv_pad, fill_cfg.bx, fill_cfg.by,
        fill_cfg.d, fill_cfg.device, fill_cfg.total_qblocks,
        fill_cfg.k_2d, fill_cfg.v_2d)
    k_compact = torch.zeros(total_qblocks * max_sel * by, d,
                             dtype=torch.float16, device=device)
    v_compact = torch.zeros(total_qblocks * max_sel * by, d,
                            dtype=torch.float16, device=device)
    valid_mask_new = torch.zeros(total_qblocks * max_sel * bx, by,
                                  dtype=torch.float32, device=device)

    for i, (b_idx, h_kv_idx, valid_v) in enumerate(qblock_info):
        for j, v_blk in enumerate(valid_v):
            src = (b_idx * hkv + h_kv_idx) * skv_pad + v_blk * by
            dst = i * max_sel * by + j * by
            k_compact[dst:dst + by] = k_2d[src:src + by]
            v_compact[dst:dst + by] = v_2d[src:src + by]
            valid_mask_new[i * max_sel * bx + j * bx:i * max_sel * bx + j * bx + bx, :] = 1.0

        first_src = (b_idx * hkv + h_kv_idx) * skv_pad + valid_v[0] * by
        for j in range(len(valid_v), max_sel):
            dst = i * max_sel * by + j * by
            k_compact[dst:dst + by] = k_2d[first_src:first_src + by]
            v_compact[dst:dst + by] = v_2d[first_src:first_src + by]

    return k_compact, v_compact, valid_mask_new


def _build_sparse_kv_cached(block_sparse_mask, k_2d, v_2d, shape_cfg):
    """Build compacted K/V tensors with caching: reuse mask structure if unchanged.

    Caches qblock_info/max_sel keyed on mask data_ptr + shape, so repeated calls
    with the same mask skip the expensive Python-level block traversal.
    Then fills compact K/V tensors and applies boundary masking for non-aligned
    sequences.

    Supports non-aligned sequences via per-batch actual_seq_lengths in shape_cfg.

    Args:
        shape_cfg: _SparseKVConfig namedtuple grouping all shape parameters,
            including actual_seq_lengths and actual_seq_lengths_kv.
    """
    global _MASK_CACHE
    b, hq, hkv, sq, skv = shape_cfg.b, shape_cfg.hq, shape_cfg.hkv, shape_cfg.sq, shape_cfg.skv
    sq_pad, skv_pad = shape_cfg.sq_pad, shape_cfg.skv_pad
    num_qb, num_kb = shape_cfg.num_qb, shape_cfg.num_kb
    bx, by, d, device = shape_cfg.bx, shape_cfg.by, shape_cfg.d, shape_cfg.device
    actual_seq_lengths = shape_cfg.actual_seq_lengths
    actual_seq_lengths_kv = shape_cfg.actual_seq_lengths_kv

    total_qblocks = b * hq * num_qb


    cache_key = (block_sparse_mask.data_ptr(), b, hq, hkv, num_qb, num_kb, bx, by)

    kv_collect_cfg = _CollectKVCfg(b=b, hq=hq, hkv=hkv, num_qb=num_qb, num_kb=num_kb)
    # 扩展key cache以包含 per-batch 实际 KV 长度 ----
    # key cache 只含 mask 指针+形状，不同 batch 会出现配置共享错误缓存。
    # 当 actual_seq_lengths_kv 不同时，per-batch 过滤结果不同，必须区分缓存。
    if actual_seq_lengths_kv is not None:
        aslk_tuple = tuple(int(actual_seq_lengths_kv[i]) for i in range(b))
        cache_key = cache_key + ('aslk',) + aslk_tuple
    cached = _MASK_CACHE.get(cache_key)
    if cached is not None:
        # 缓存命中：解包三元组（含 dummy_qblocks）
        qblock_info, max_sel, dummy_qblocks = cached
    else:
        # 传递 actual_seq_lengths_kv 进行 per-batch 过滤 ----
        qblock_info, max_sel, dummy_qblocks = _collect_valid_kv_per_qblock(
            block_sparse_mask, kv_collect_cfg,
            actual_seq_lengths_kv=actual_seq_lengths_kv, by=by)
        _MASK_CACHE[cache_key] = (qblock_info, max_sel, dummy_qblocks)

    fill_cfg = _KVFillConfig(hkv=hkv, skv_pad=skv_pad, bx=bx, by=by,
                              d=d, device=device, total_qblocks=total_qblocks,
                              k_2d=k_2d, v_2d=v_2d)
    k_compact, v_compact, valid_mask = _fill_compacted_kv(
        qblock_info, max_sel, None, fill_cfg)

    # ==== Non-aligned: 集中处理 per-batch boundary + dummy + out-of-range ====
    # num_kb/num_qb 来自 sq_max/skv_max（全局最大值），而 boundary masking
    # 公式用 per-batch 实际长度去减全局 block 数，shorter batch 算出负值导致 Python
    # 负索引切片破坏有效数据。
    # 在有 actual_seq_lengths 信息时，绕过原有 boundary 函数，直接在此处用一个
    # 循环集中完成所有 per-batch 修正（dummy 清零 + KV boundary + Q boundary）
    if actual_seq_lengths_kv is not None or actual_seq_lengths is not None:
        # Step 1: 清零 dummy Q blocks 的 valid_mask
        # 这些 Q blocks 在实际范围内没有有效 KV block，valid_mask 设为全零以避免
        # kernel 的 online softmax 产生非零输出。
        for dummy_idx in dummy_qblocks:
            mask_start = dummy_idx * max_sel * bx
            mask_end = mask_start + max_sel * bx
            valid_mask[mask_start:mask_end, :] = 0.0

        # Step 2: Per-batch KV boundary masking
        # 对每个 Q block 检查其 KV block 是否在该 batch 的实际范围内，
        # 超范围则全部清零，最后一个有效 block 则只清零 padding 列。
        if actual_seq_lengths_kv is not None:
            for i, (b_idx, h_kv_idx, valid_v) in enumerate(qblock_info):
                skv_val = int(actual_seq_lengths_kv[b_idx])
                actual_num_kb = math.ceil(skv_val / by)
                last_valid_rem = skv_val - (actual_num_kb - 1) * by
                for j, v_blk in enumerate(valid_v):
                    m_start = i * max_sel * bx + j * bx
                    if v_blk == actual_num_kb - 1:
                        if last_valid_rem < by:
                            valid_mask[m_start:m_start + bx, last_valid_rem:] = 0.0
                    elif v_blk >= actual_num_kb:
                        valid_mask[m_start:m_start + bx, :] = 0.0

        # Step 3: Per-batch Q boundary masking
        # 超出 batch 实际 Q 范围的 block 全部清零，最后一个有效 Q block 只清零 padding 行。
        if actual_seq_lengths is not None:
            b_count = len(actual_seq_lengths)
            hq_total = total_qblocks // (b_count * num_qb) if b_count > 0 else total_qblocks // num_qb
            for i in range(total_qblocks):
                b_idx = i // (hq_total * num_qb)
                sq_val = int(actual_seq_lengths[b_idx])
                actual_num_qb = math.ceil(sq_val / bx)
                u = i % num_qb
                if u == actual_num_qb - 1:
                    last_valid_rem = sq_val - (actual_num_qb - 1) * bx
                    if last_valid_rem < bx:
                        for j in range(max_sel):
                            m_dst = i * max_sel * bx + j * bx
                            valid_mask[m_dst + last_valid_rem:m_dst + bx, :] = 0.0
                elif u >= actual_num_qb:
                    for j in range(max_sel):
                        m_dst = i * max_sel * bx + j * bx
                        valid_mask[m_dst:m_dst + bx, :] = 0.0
    else:
        # ==== Aligned path: 使用原有 boundary masking 函数（不修改） ====
        if skv_pad > skv:
            remaining_kv_per_batch = skv - (num_kb - 1) * by
        else:
            remaining_kv_per_batch = by
        kv_boundary_cfg = _KVBoundaryCfg(max_sel=max_sel, bx=bx, num_kb=num_kb)
        _apply_kv_boundary_mask(
            valid_mask, qblock_info, kv_boundary_cfg, remaining_kv_per_batch)

        if sq_pad > sq:
            remaining_q_per_batch = sq - (num_qb - 1) * bx
        else:
            remaining_q_per_batch = bx
        q_boundary_cfg = _QBoundaryCfg(max_sel=max_sel, bx=bx, num_qb=num_qb)
        _apply_q_boundary_mask(valid_mask, total_qblocks, q_boundary_cfg,
                               remaining_q_per_batch)

    return k_compact, v_compact, valid_mask, max_sel


def _collect_valid_q_for_kvblock(block_sparse_mask, cfg):
    """Collect valid Q block indices for one KV block (extracted to reduce nesting).

    Args:
        cfg: _ValidQForKVBlockCfg(b_idx, h_kv_idx, v_blk, group, num_qb, nkv_cols).
    """
    b_idx, h_kv_idx, v_blk, group, num_qb, nkv_cols = (
        cfg.b_idx, cfg.h_kv_idx, cfg.v_blk, cfg.group, cfg.num_qb, cfg.nkv_cols)
    valid_q = []
    for g_idx in range(group):
        h_q = h_kv_idx * group + g_idx
        for u in range(num_qb):
            if v_blk < nkv_cols and block_sparse_mask[b_idx, h_q, u, v_blk].item():
                valid_q.append((h_q, u))
    return valid_q


def _collect_valid_q_per_kvblock(block_sparse_mask, q_cfg):
    """Phase 1 of _build_sparse_q_dkdv: collect valid Q indices per KV block.

    Args:
        q_cfg: _CollectQCfg(b, hq, hkv, num_qb, num_kb) namedtuple.
    """
    b, hq, hkv, num_qb, num_kb = q_cfg.b, q_cfg.hq, q_cfg.hkv, q_cfg.num_qb, q_cfg.num_kb
    group = hq // hkv
    nkv_cols = block_sparse_mask.shape[3]
    kvblock_info = []
    max_inner = 0

    for flat_idx in range(b * hkv):
        b_idx = flat_idx // hkv
        h_kv_idx = flat_idx % hkv
        for v_blk in range(num_kb):
            valid_q = _collect_valid_q_for_kvblock(
                block_sparse_mask, _ValidQForKVBlockCfg(
                    b_idx=b_idx, h_kv_idx=h_kv_idx, v_blk=v_blk,
                    group=group, num_qb=num_qb, nkv_cols=nkv_cols))
            kvblock_info.append((b_idx, valid_q))
            max_inner = max(max_inner, len(valid_q))

    return kvblock_info, max(max_inner, 1)


def _fill_compacted_q(kvblock_info, max_inner, fill_cfg):
    """Phase 2 of _build_sparse_q_dkdv: build compacted Q/dO/O/LSE tensors.

    Args:
        fill_cfg: _QDkdvFillConfig namedtuple including q_2d, do_2d, o_2d, lse_2d.
    """
    hq, sq_pad, bx, by, d, device, total_kv, q_2d, do_2d, o_2d, lse_2d = (
        fill_cfg.hq, fill_cfg.sq_pad, fill_cfg.bx, fill_cfg.by,
        fill_cfg.d, fill_cfg.device, fill_cfg.total_kv,
        fill_cfg.q_2d, fill_cfg.do_2d, fill_cfg.o_2d, fill_cfg.lse_2d)
    q_compact = torch.zeros(total_kv * max_inner * bx, d, dtype=torch.float16, device=device)
    do_compact = torch.zeros(total_kv * max_inner * bx, d, dtype=torch.float16, device=device)
    o_compact = torch.zeros(total_kv * max_inner * bx, d, dtype=torch.float16, device=device)
    lse_compact = torch.full([total_kv * max_inner * bx, 1], 1e30, dtype=torch.float32, device=device)
    inner_mask = torch.zeros(total_kv * max_inner * bx, by, dtype=torch.float32, device=device)

    for i, (b_idx, valid_q) in enumerate(kvblock_info):
        for j, (h_q, u) in enumerate(valid_q):
            src = (b_idx * hq + h_q) * sq_pad + u * bx
            dst = i * max_inner * bx + j * bx
            q_compact[dst:dst + bx] = q_2d[src:src + bx]
            do_compact[dst:dst + bx] = do_2d[src:src + bx]
            o_compact[dst:dst + bx] = o_2d[src:src + bx]
            lse_compact[dst:dst + bx] = lse_2d[src:src + bx]
            inner_mask[dst:dst + bx, :] = 1.0

        if valid_q:
            h_q0, u0 = valid_q[0]
            src0 = (b_idx * hq + h_q0) * sq_pad + u0 * bx
            for j in range(len(valid_q), max_inner):
                dst = i * max_inner * bx + j * bx
                q_compact[dst:dst + bx] = q_2d[src0:src0 + bx]
                do_compact[dst:dst + bx] = do_2d[src0:src0 + bx]
                o_compact[dst:dst + bx] = o_2d[src0:src0 + bx]
                lse_compact[dst:dst + bx] = lse_2d[src0:src0 + bx]

    return q_compact, do_compact, o_compact, lse_compact, inner_mask


def _apply_q_boundary_inner_mask(inner_mask, kvblock_info, inner_cfg, remaining_q_per_batch):
    """Zero out rows in inner_mask for the last Q block's padded rows (dK/dV path).

    Supports per-batch actual_seq_lengths for non-aligned sequences.

    Args:
        inner_cfg: _QBoundaryInnerCfg(max_inner, bx, num_qb) namedtuple.
        remaining_q_per_batch: either a single int (aligned case) or per-batch list/tensor.
    """
    max_inner, bx, num_qb = inner_cfg.max_inner, inner_cfg.bx, inner_cfg.num_qb

    def _zero_inner_boundary(kvblk_idx, valid_q_blocks, rem_q_len):
        for j, (h_q, u) in enumerate(valid_q_blocks):
            if u == num_qb - 1 and rem_q_len < bx:
                m_dst = kvblk_idx * max_inner * bx + j * bx
                inner_mask[m_dst + rem_q_len:m_dst + bx, :] = 0.0

    if isinstance(remaining_q_per_batch, (int, float)):
        rem_q = int(remaining_q_per_batch)
        for i, (b, valid_q) in enumerate(kvblock_info):
            _zero_inner_boundary(i, valid_q, rem_q)
    else:
        for i, (b_idx, valid_q) in enumerate(kvblock_info):
            _zero_inner_boundary(i, valid_q, remaining_q_per_batch[b_idx])


def _build_sparse_q_dkdv_cached(block_sparse_mask, shape_cfg):
    """Build compacted Q/dO/O/LSE tensors for dK/dV computation, with caching.

    Caches kvblock_info/max_inner keyed on mask data_ptr + shape, so repeated
    calls with the same mask skip the expensive Python-level block traversal.
    Then fills compact tensors and applies boundary masking for non-aligned Q lengths.

    Args:
        shape_cfg: _SparseKVConfig namedtuple grouping all parameters,
            including actual_seq_lengths for boundary masking,
            actual_seq_lengths_kv for per-batch KV boundary masking,
            and q_2d/do_2d/o_2d/lse_2d data tensors for fill.
    """
    global _Q_DKDV_CACHE
    b, hq, hkv, sq, sq_pad = shape_cfg.b, shape_cfg.hq, shape_cfg.hkv, shape_cfg.sq, shape_cfg.sq_pad
    num_qb, num_kb = shape_cfg.num_qb, shape_cfg.num_kb
    bx, by, d, device = shape_cfg.bx, shape_cfg.by, shape_cfg.d, shape_cfg.device
    actual_seq_lengths = shape_cfg.actual_seq_lengths
    actual_seq_lengths_kv = shape_cfg.actual_seq_lengths_kv

    total_kv = b * hkv * num_kb

    cache_key = (block_sparse_mask.data_ptr(), b, hq, hkv, num_qb, num_kb, bx, by)

    q_collect_cfg = _CollectQCfg(b=b, hq=hq, hkv=hkv, num_qb=num_qb, num_kb=num_kb)
    cached = _Q_DKDV_CACHE.get(cache_key)
    if cached is not None:
        kvblock_info, max_inner = cached
    else:
        kvblock_info, max_inner = _collect_valid_q_per_kvblock(
            block_sparse_mask, q_collect_cfg)
        _Q_DKDV_CACHE[cache_key] = (kvblock_info, max_inner)

    q_fill_cfg = _QDkdvFillConfig(hq=hq, sq_pad=sq_pad, bx=bx, by=by,
                                  d=d, device=device, total_kv=total_kv,
                                  q_2d=shape_cfg.q_2d, do_2d=shape_cfg.do_2d,
                                  o_2d=shape_cfg.o_2d, lse_2d=shape_cfg.lse_2d)
    q_compact, do_compact, o_compact, lse_compact, inner_mask = _fill_compacted_q(
        kvblock_info, max_inner, q_fill_cfg)

    # ==== Non-aligned: 集中处理 per-batch Q/KV boundary masking ====
    # 与 _build_sparse_kv_cached 中的处理方式类似，在有 actual_seq_lengths 信息时，
    # 绕过原有 _apply_q_boundary_inner_mask 函数，直接在此处集中完成所有 per-batch
    if actual_seq_lengths is not None or actual_seq_lengths_kv is not None:
        # Step 1: Per-batch KV boundary masking
        # 对 ghost KV blocks（超出该 batch 实际 KV 范围）清零 inner_mask 整个区域，
        # 对最后一个有效 KV block 清零 padding 列（超出 actual_kv 的列位置）。
        if actual_seq_lengths_kv is not None:
            for i, (b_idx, valid_q) in enumerate(kvblock_info):
                v_blk = i % num_kb
                skv_val = int(actual_seq_lengths_kv[b_idx])
                actual_num_kb = math.ceil(skv_val / by)
                if v_blk >= actual_num_kb:
                    mask_start = i * max_inner * bx
                    mask_end = mask_start + max_inner * bx
                    inner_mask[mask_start:mask_end, :] = 0.0
                elif v_blk == actual_num_kb - 1:
                    last_valid_rem = skv_val - (actual_num_kb - 1) * by
                    if last_valid_rem < by:
                        for j, (h_q, u) in enumerate(valid_q):
                            m_start = i * max_inner * bx + j * bx
                            inner_mask[m_start:m_start + bx, last_valid_rem:] = 0.0

        # Step 2: Per-batch Q boundary masking
        # 对 ghost Q blocks（超出该 batch 实际 Q 范围）清零 inner_mask 所有行，
        # 对最后一个有效 Q block 清零 padding 行（超出 actual_sq 的行位置）。
        if actual_seq_lengths is not None:
            for i, (b_idx, valid_q) in enumerate(kvblock_info):
                for j, (h_q, u) in enumerate(valid_q):
                    sq_val = int(actual_seq_lengths[b_idx])
                    actual_num_qb = math.ceil(sq_val / bx)
                    m_dst = i * max_inner * bx + j * bx
                    if u >= actual_num_qb:
                        inner_mask[m_dst:m_dst + bx, :] = 0.0
                    elif u == actual_num_qb - 1:
                        last_valid_rem = sq_val - (actual_num_qb - 1) * bx
                        if last_valid_rem < bx:
                            inner_mask[m_dst + last_valid_rem:m_dst + bx, :] = 0.0
    else:
        # ==== Aligned path: 使用原有 boundary masking 函数（不修改） ====
        if sq_pad > sq:
            remaining_q_per_batch = sq - (num_qb - 1) * bx
        else:
            remaining_q_per_batch = bx
        inner_cfg = _QBoundaryInnerCfg(max_inner=max_inner, bx=bx, num_qb=num_qb)
        _apply_q_boundary_inner_mask(
            inner_mask, kvblock_info, inner_cfg,
            remaining_q_per_batch)

    padded_rows = (inner_mask.sum(dim=-1) == 0)
    lse_compact[padded_rows] = 1e30
    q_compact[padded_rows] = 0.0
    do_compact[padded_rows] = 0.0
    o_compact[padded_rows] = 0.0

    return SparseQDkdvResult(q_compact=q_compact, do_compact=do_compact,
                                o_compact=o_compact, lse_compact=lse_compact,
                                inner_mask=inner_mask, max_inner=max_inner)


def generate_block_sparse_mask(
    batch, head_num_q, num_q_blocks, num_kv_blocks,
    sparsity=0.5, device="cpu", seed=None,
):
    """Generate a random block-sparse mask with given sparsity ratio."""
    if seed is not None:
        torch.manual_seed(seed)

    mask = torch.rand(batch, head_num_q, num_q_blocks, num_kv_blocks, device=device) < sparsity

    any_valid = mask.any(dim=-1)
    empty_qblocks = ~any_valid

    if empty_qblocks.any():
        forced_kv = torch.randint(0, num_kv_blocks,
                                  empty_qblocks.shape, device=device)
        b_idx, h_idx, u_idx = torch.where(empty_qblocks)
        mask[b_idx, h_idx, u_idx, forced_kv[b_idx, h_idx, u_idx]] = True

    return mask
