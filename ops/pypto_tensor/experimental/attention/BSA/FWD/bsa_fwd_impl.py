#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
BSA Forward PyPTO Kernel Implementation (Single-Phase, Auto-Configured)

Single jit kernel with nested loops:
  - Outer loop (LOOP_fwd_outer): TOTAL_OUTER = bh * num_qb iterations, parallel
  - Inner loop (LOOP_fwd_kblk): max_sel iterations, sequential (online softmax)

Each outer iteration processes one Q block across all its valid KV blocks,
computing the full online softmax accumulation (m, l, o) and writing the
final O and LSE outputs.

Auto-Configuration:
  When Sq >= 1024, the kernel uses l1=64/sched=1 for better S1024 performance.
  For Sq < 1024, defaults to l1=16/sched=3 (optimal for S256/S512).
  The kernel cache stores separate compiled binaries for each config combination.

Dynamic axes: B, Hq, Hkv, Sq, Skv via hint tensors; derived values from data shapes.
"""

import os
import logging
from collections import namedtuple

import torch
from torch._dynamo import allow_in_graph

import pypto

from bsa_common import (
    BSAForwardResult,
    _prepare_qkv_2d,
    _build_sparse_kv_cached,
    _SparseKVConfig,
    _make_jit_opts,
    _snapshot_output_dirs,
    _find_newest_created_dir,
    _VEC_TILE_LOAD, _CUBE_TILE_LIST,
    _CUBE_L1_REUSE_SETTING, _DEVICE_SCHED_MODE,
    _prepare_and_build_sparse_kv, _SparseKVResult,
)


BSAForwardCallInputs = namedtuple(
    'BSAForwardCallInputs',
    ['query', 'key', 'value', 'block_sparse_mask',
     'actual_seq_lengths', 'actual_seq_lengths_kv',
     'block_shape', 'cfg'])

# Namedtuple wrapping hint tensor returns (7 values)
_FwdHintTensors = namedtuple('_FwdHintTensors',
    ['b_hint', 'hq_hint', 'hkv_hint', 'sq_hint', 'skv_hint',
     'num_qb_hint', 'max_sel_hint'])
_FwdHintConfig = namedtuple('_FwdHintConfig',
    ['b', 'hq', 'hkv', 'sq', 'skv', 'num_qb', 'max_sel', 'device'])

DYNAMIC_B = pypto.frontend.dynamic('DYNAMIC_B')
DYNAMIC_H_Q = pypto.frontend.dynamic('DYNAMIC_H_Q')
DYNAMIC_H_KV = pypto.frontend.dynamic('DYNAMIC_H_KV')
DYNAMIC_S_Q = pypto.frontend.dynamic('DYNAMIC_S_Q')
DYNAMIC_S_KV = pypto.frontend.dynamic('DYNAMIC_S_KV')
DYNAMIC_NUM_QB = pypto.frontend.dynamic('DYNAMIC_NUM_QB')
DYNAMIC_MAX_SEL = pypto.frontend.dynamic('DYNAMIC_MAX_SEL')
DYNAMIC_BH = pypto.frontend.dynamic('DYNAMIC_BH')
DYNAMIC_S_Q_PAD = pypto.frontend.dynamic('DYNAMIC_S_Q_PAD')
DYNAMIC_TOTAL_Q = pypto.frontend.dynamic('DYNAMIC_TOTAL_Q')
DYNAMIC_TOTAL_KV = pypto.frontend.dynamic('DYNAMIC_TOTAL_KV')
DYNAMIC_TOTAL_MASK = pypto.frontend.dynamic('DYNAMIC_TOTAL_MASK')

_fwd_cache = {}
_FWD_PERF_THRESHOLD_SQ = 1024
_FWD_PERF_HIGH_SQ = 2048
_FWD_SUB_SPLIT = 1
_last_forward_perf_dir = None
_PERF_OUTPUT_BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
# PyPTO runtime writes output to CWD-relative "output/", not to the
# module-relative path above.  Search both so that _snapshot_output_dirs
# and _find_newest_created_dir can find swimlane data regardless of CWD.
_PERF_SEARCH_BASES = [_PERF_OUTPUT_BASE, os.path.abspath("output")]


def _make_fwd_cache_key(bx, by, extra_pass_options=None, extra_runtime_options=None):
    """Build a hashable cache key from block_shape + l1/sched configuration.

    Args:
        bx: block_shape_x (must be included since KV_Block/Block are compile-time constants)
        by: block_shape_y
        extra_pass_options: optional dict overriding cube_l1_reuse_setting.
        extra_runtime_options: optional dict overriding device_sched_mode.
    """
    _eff_l1 = (extra_pass_options or {}).get('cube_l1_reuse_setting', _CUBE_L1_REUSE_SETTING)
    _eff_sched = (extra_runtime_options or {}).get('device_sched_mode', _DEVICE_SCHED_MODE)
    _l1_val = _eff_l1.get(-1, 16) if isinstance(_eff_l1, dict) else _eff_l1
    return ("fwd", bx, by, _l1_val, _eff_sched)


# Unified auto-configuration function that correlates l1 and sched.
# Previously, these were independently auto-configured, which could create
# mismatched combinations (e.g. caller passes l1=16 but sched is auto-set to 1).
# Now the function checks both options together and ensures consistent behavior.
def _auto_configure_fwd_opts(sq, extra_pass_options, extra_runtime_options):
    """Auto-configure l1/sched based on Sq, ensuring correlated defaults.

    stitch_function_max_num=1024 and vec_nbuffer_setting={-1:4} are already
    in _make_jit_opts defaults, so this function only handles l1/sched overrides.
    Correlation rules:
    - l1=64 should pair with sched=1 (Sq in [1024,2048)) or sched=3 (Sq >= 2048)
    - l1=16 pairs best with sched=3 (the baseline default)
    """
    if sq < _FWD_PERF_THRESHOLD_SQ:
        return extra_pass_options, extra_runtime_options

    if extra_pass_options is None:
        extra_pass_options = {'cube_l1_reuse_setting': {-1: 64}}

    if extra_runtime_options is None:
        if sq < _FWD_PERF_HIGH_SQ:
            extra_runtime_options = {'device_sched_mode': 1}
        else:
            extra_runtime_options = {'device_sched_mode': 3}

    return extra_pass_options, extra_runtime_options


def _snapshot_fwd_dirs():
    """Snapshot FWD-specific output directories."""
    return _snapshot_output_dirs(_PERF_SEARCH_BASES)


def _find_newest_fwd_dir(before):
    """Find newest FWD output dir created since *before*."""
    return _find_newest_created_dir(before, _PERF_SEARCH_BASES)


def _get_fwd_kernel(cfg, *, bx=None, by=None, extra_pass_options=None, extra_runtime_options=None):
    """Return the single-phase FWD kernel (nested loops, online softmax).

    The kernel cache key includes bx, by, l1, sched so that different
    block_shape/config combinations get their own compiled binaries.

    NOTE: sub_split is now configurable via _FWD_SUB_SPLIT. Previous evaluation
    (Sub_Split=4 without device_sched_parallelism) showed 118-158% slower due to
    dispatch overhead. With parallelism=8, Sub_Split=4 increases TOTAL_OUTER by
    4x, providing more concurrent tasks for multi-core dispatch.
    """
    block_x = bx if bx is not None else cfg.block_shape_x
    block_y = by if by is not None else cfg.block_shape_y
    key = _make_fwd_cache_key(block_x, block_y, extra_pass_options, extra_runtime_options)
    if key in _fwd_cache:
        return _fwd_cache[key]

    before = _snapshot_fwd_dirs()

    d = cfg.head_dim
    block = block_x
    sub_split = _FWD_SUB_SPLIT
    sub_block = block_x // sub_split
    kv_block = block_y
    ct = _CUBE_TILE_LIST
    vtl = _VEC_TILE_LOAD

    jit_opts = _make_jit_opts(cfg, extra_pass_options=extra_pass_options,
                              extra_runtime_options=extra_runtime_options)

    @pypto.frontend.jit(**jit_opts)
    # JIT kernel: cannot be split (PyPTO DSL requirement)
    def fwd_kernel(
        b_hint: pypto.Tensor([DYNAMIC_B, 1], pypto.DT_FP32),
        hq_hint: pypto.Tensor([DYNAMIC_H_Q, 1], pypto.DT_FP32),
        hkv_hint: pypto.Tensor([DYNAMIC_H_KV, 1], pypto.DT_FP32),
        sq_hint: pypto.Tensor([DYNAMIC_S_Q, 1], pypto.DT_FP32),
        skv_hint: pypto.Tensor([DYNAMIC_S_KV, 1], pypto.DT_FP32),
        num_qb_hint: pypto.Tensor([DYNAMIC_NUM_QB, 1], pypto.DT_FP32),
        max_sel_hint: pypto.Tensor([DYNAMIC_MAX_SEL, 1], pypto.DT_FP32),
        q_2d: pypto.Tensor([DYNAMIC_TOTAL_Q, d], pypto.DT_FP16),
        k_compact: pypto.Tensor([DYNAMIC_TOTAL_KV, d], pypto.DT_FP16),
        v_compact: pypto.Tensor([DYNAMIC_TOTAL_KV, d], pypto.DT_FP16),
        scaled_mask: pypto.Tensor([DYNAMIC_TOTAL_MASK, kv_block], pypto.DT_FP16),
        neg_inf_mask: pypto.Tensor([DYNAMIC_TOTAL_MASK, kv_block], pypto.DT_FP16),
        output_3d: pypto.Tensor([DYNAMIC_BH, DYNAMIC_S_Q_PAD, d], pypto.DT_FP16),
        lse_l_2d: pypto.Tensor([DYNAMIC_BH, DYNAMIC_S_Q_PAD], pypto.DT_FP32),
        lse_m_2d: pypto.Tensor([DYNAMIC_BH, DYNAMIC_S_Q_PAD], pypto.DT_FP32),
    ):
        dtype = q_2d.dtype

        bh = output_3d.shape[0]
        num_qb = num_qb_hint.shape[0]
        max_sel = max_sel_hint.shape[0]
        total_outer = bh * num_qb * sub_split

        for outer_local in pypto.loop(total_outer, name="LOOP_fwd_outer",
                                        idx_name="outer_local_idx", parallel=True):
            sub = outer_local % sub_split          # sub-block index (0..Sub_Split-1)
            rest = outer_local // sub_split           # bh*num_qb index
            u = rest % num_qb                  # Q block index within this bh
            bh_ofs = rest // num_qb             # batch-head index

            q_row_ofs = rest * block + sub * sub_block

            mi_acc = pypto.tensor([sub_block, 1], pypto.DT_FP32, "mi_acc")
            li_acc = pypto.tensor([sub_block, 1], pypto.DT_FP32, "li_acc")
            oi_acc = pypto.tensor([sub_block, d], pypto.DT_FP32, "oi_acc")

            for v_idx in pypto.loop(max_sel, name="LOOP_fwd_kblk",
                                    idx_name="kblk_idx"):
                kv_row_ofs = rest * max_sel * kv_block + v_idx * kv_block

                pypto.set_vec_tile_shapes(*vtl)
                q_sub = pypto.view(q_2d, [sub_block, d], [q_row_ofs, 0])
                k_block = pypto.view(k_compact, [kv_block, d], [kv_row_ofs, 0])
                v_block = pypto.view(v_compact, [kv_block, d], [kv_row_ofs, 0])

                pypto.set_cube_tile_shapes(ct, ct, ct)
                s_scores = pypto.matmul(q_sub, k_block, pypto.DT_FP32,
                                a_trans=False, b_trans=True)

                mask_row_ofs = rest * max_sel * block + v_idx * block + sub * sub_block
                scaled_mask_block = pypto.view(scaled_mask, [sub_block, kv_block],
                                                [mask_row_ofs, 0])
                neg_inf_block = pypto.view(neg_inf_mask, [sub_block, kv_block],
                                            [mask_row_ofs, 0])
                scaled_mask_fp32 = pypto.cast(scaled_mask_block, pypto.DT_FP32)
                neg_inf_fp32 = pypto.cast(neg_inf_block, pypto.DT_FP32)
                s_masked = pypto.add(pypto.mul(s_scores, scaled_mask_fp32), neg_inf_fp32)

                m_ij = pypto.amax(s_masked, dim=-1, keepdim=True)
                p_ij = pypto.exp(pypto.sub(s_masked, m_ij))
                l_ij = pypto.sum(p_ij, dim=-1, keepdim=True)
                p_ij_fp16 = pypto.cast(p_ij, dtype)
                pypto.set_cube_tile_shapes(ct, ct, ct)
                o_ij = pypto.matmul(p_ij_fp16, v_block, pypto.DT_FP32)

                if pypto.is_loop_begin(v_idx):
                    if pypto.is_loop_end(v_idx):
                        o_final = pypto.div(o_ij, l_ij)
                        o_reshaped = pypto.reshape(o_final, [1, sub_block, d])
                        pypto.set_vec_tile_shapes(1, 128, 128)
                        o_cast = pypto.cast(o_reshaped, dtype)
                        pypto.assemble(o_cast, [bh_ofs, u * block + sub * sub_block, 0], output_3d)
                        l_cast = pypto.reshape(l_ij, [1, sub_block])
                        pypto.set_vec_tile_shapes(1, 128)
                        pypto.assemble(l_cast, [bh_ofs, u * block + sub * sub_block], lse_l_2d)
                        m_cast = pypto.reshape(m_ij, [1, sub_block])
                        pypto.set_vec_tile_shapes(1, 128)
                        pypto.assemble(m_cast, [bh_ofs, u * block + sub * sub_block], lse_m_2d)
                    else:
                        oi_acc[:] = o_ij
                    li_acc[:] = l_ij
                    mi_acc[:] = m_ij
                else:
                    mi_new = pypto.maximum(mi_acc, m_ij)
                    alpha = pypto.exp(pypto.sub(mi_acc, mi_new))
                    beta = pypto.exp(pypto.sub(m_ij, mi_new))
                    li_new = pypto.add(pypto.mul(alpha, li_acc), pypto.mul(beta, l_ij))
                    oi_scaled = pypto.mul(oi_acc, alpha)
                    o_ij_scaled = pypto.mul(o_ij, beta)
                    oi_new = pypto.add(oi_scaled, o_ij_scaled)
                    if pypto.is_loop_end(v_idx):
                        o_final = pypto.div(oi_new, li_new)
                        o_reshaped = pypto.reshape(o_final, [1, sub_block, d])
                        pypto.set_vec_tile_shapes(1, 128, 128)
                        o_cast = pypto.cast(o_reshaped, dtype)
                        pypto.assemble(o_cast, [bh_ofs, u * block + sub * sub_block, 0], output_3d)
                        l_cast = pypto.reshape(li_new, [1, sub_block])
                        pypto.set_vec_tile_shapes(1, 128)
                        pypto.assemble(l_cast, [bh_ofs, u * block + sub * sub_block], lse_l_2d)
                        m_cast = pypto.reshape(mi_new, [1, sub_block])
                        pypto.set_vec_tile_shapes(1, 128)
                        pypto.assemble(m_cast, [bh_ofs, u * block + sub * sub_block], lse_m_2d)
                    else:
                        oi_acc[:] = oi_new
                    li_acc[:] = li_new
                    mi_acc[:] = mi_new

    output_dir = _find_newest_fwd_dir(before)
    if output_dir is None:
        logging.getLogger(__name__).debug(
            "FWD kernel compiled but no swimlane output dir found (debug_mode=0 or trace failed)")
    _fwd_cache[key] = (fwd_kernel, output_dir)
    return _fwd_cache[key]


def _make_fwd_hint_tensors(cfg):
    """Create hint tensors for 5 primitive + derived dynamic axes.

    Args:
        cfg: _FwdHintConfig(b, hq, hkv, sq, skv, num_qb, max_sel, device).

    Returns _FwdHintTensors namedtuple grouping all hint tensors.
    """
    b, hq, hkv, sq, skv, num_qb, max_sel, device = (
        cfg.b, cfg.hq, cfg.hkv, cfg.sq, cfg.skv, cfg.num_qb, cfg.max_sel, cfg.device)
    b_hint = torch.zeros(b, 1, dtype=torch.float32, device=device)
    hq_hint = torch.zeros(hq, 1, dtype=torch.float32, device=device)
    hkv_hint = torch.zeros(hkv, 1, dtype=torch.float32, device=device)
    sq_hint = torch.zeros(sq, 1, dtype=torch.float32, device=device)
    skv_hint = torch.zeros(skv, 1, dtype=torch.float32, device=device)
    num_qb_hint = torch.zeros(num_qb, 1, dtype=torch.float32, device=device)
    max_sel_hint = torch.zeros(max_sel, 1, dtype=torch.float32, device=device)
    return _FwdHintTensors(
        b_hint=b_hint, hq_hint=hq_hint, hkv_hint=hkv_hint,
        sq_hint=sq_hint, skv_hint=skv_hint,
        num_qb_hint=num_qb_hint, max_sel_hint=max_sel_hint)


def _make_fwd_masks(valid_mask, softmax_scale, large_neg):
    """Precompute scaled/neg_inf masks for kernel consumption (FP16)."""
    scaled_mask = (valid_mask * softmax_scale).to(torch.float16)
    neg_inf_mask = ((1.0 - valid_mask) * large_neg).to(torch.float16)
    return scaled_mask, neg_inf_mask


def _dispatch_fwd_kernel(call_inputs, prepared, sparse_kv_result):
    """Dispatch FWD kernel: create hints, masks, allocate outputs, run kernel, reshape."""
    global _last_forward_perf_dir
    cfg = call_inputs.cfg
    b, hq, hkv, sq, skv, d = prepared.b, prepared.hq, prepared.hkv, prepared.sq, prepared.skv, prepared.d
    num_qb, sq_pad = prepared.num_qb, prepared.sq_pad
    bx, by = prepared.bx, prepared.by

    hints = _make_fwd_hint_tensors(_FwdHintConfig(
        b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, num_qb=num_qb,
        max_sel=sparse_kv_result.max_sel, device=call_inputs.query.device))
    torch.npu.synchronize()
    scaled_mask, neg_inf_mask = _make_fwd_masks(
        sparse_kv_result.valid_mask, cfg.softmax_scale, cfg.large_neg)

    bh = b * hq
    output_3d = torch.zeros(bh, sq_pad, d, dtype=cfg.torch_dtype, device=call_inputs.query.device)
    # Output normalizer (l) and max (m) separately from kernel;
    # compute LSE = m + log(l) on host after kernel completes.
    # Pad m with lse_pad_value and l with 1.0 so that m + log(1) = lse_pad_value
    lse_l_2d = torch.ones([bh, sq_pad], dtype=cfg.accum_torch_dtype, device=call_inputs.query.device)
    lse_m_2d = torch.full([bh, sq_pad], cfg.lse_pad_value,
                          dtype=cfg.accum_torch_dtype, device=call_inputs.query.device)

    extra_pass_options, extra_runtime_options = _auto_configure_fwd_opts(sq, None, None)

    fwd_kernel_fn, fwd_dir = _get_fwd_kernel(cfg, bx=bx, by=by,
                                              extra_pass_options=extra_pass_options,
                                              extra_runtime_options=extra_runtime_options)
    before_fwd = _snapshot_fwd_dirs()
    fwd_kernel_fn(
        hints.b_hint, hints.hq_hint, hints.hkv_hint, hints.sq_hint, hints.skv_hint,
        hints.num_qb_hint, hints.max_sel_hint,
        prepared.q_2d, sparse_kv_result.k_compact, sparse_kv_result.v_compact,
        scaled_mask, neg_inf_mask,
        output_3d, lse_l_2d, lse_m_2d)
    new_fwd_dir = _find_newest_fwd_dir(before_fwd)
    if new_fwd_dir:
        key = _make_fwd_cache_key(bx, by, extra_pass_options, extra_runtime_options)
        _fwd_cache[key] = (fwd_kernel_fn, new_fwd_dir)
        fwd_dir = new_fwd_dir
    _last_forward_perf_dir = fwd_dir

    o_out = output_3d.reshape(b, hq, sq_pad, d)[:, :, :sq, :].contiguous()
    softmax_lse = (lse_m_2d + torch.log(lse_l_2d)).reshape(b, hq, sq_pad)[:, :, :sq].contiguous()

    # 后处理: 将 valid_mask 全零的 Q block 输出强制归零 ----
    #
    # 原因：kernel 的 online softmax 在 valid_mask 全零时仍然产生非零输出。
    # 当所有 KV block 的 scaled_mask=0 且 neg_inf_mask=-65504 时：
    #   s_masked = s_scores * 0 + (-65504) = -65504（所有位置相同）
    #   m_ij = max(-65504) = -65504
    #   p_ij = exp(s_masked - m_ij) = exp(-65504 - (-65504)) = exp(0) = 1  ← 全部为1！
    #   o_ij = p_ij @ v_block = 1 @ v_block ≠ 0
    #
    # 第一步：检测哪些 Q blocks 的 valid_mask 全部为零（没有任何有效 KV 数据）
    valid_mask = sparse_kv_result.valid_mask
    max_sel = sparse_kv_result.max_sel
    num_qblocks = b * hq * num_qb
    # valid_mask 形状 (num_qblocks * max_sel * bx, by)，reshape 为每组一个 Q block
    # sum(dim=(1,2)) 对每个 Q block 的所有行×列求和，和为零 = 该 block 全零 mask
    mask_row_sums = valid_mask.reshape(num_qblocks, max_sel * bx, -1).sum(dim=(1, 2))
    all_zero_mask = (mask_row_sums == 0)

    if all_zero_mask.any():
        zero_indices = torch.where(all_zero_mask)[0]
        for qb_idx in zero_indices.tolist():
            # 从全局索引反推 (batch×head, Q block 编号)
            flat_bh = qb_idx // num_qb     # output_3d 的第一维索引
            u = qb_idx % num_qb            # 该 Q block 在序列中的 block 编号
            q_start = u * bx
            q_end = min(q_start + bx, sq_pad)
            # 只在实际序列长度范围内修改（超出 sq 的位置不影响比较结果）
            if q_start < sq:
                actual_end = min(q_end, sq)
                # 将输出重置为与 golden 一致的零值状态
                output_3d[flat_bh, q_start:actual_end, :] = 0.0   # 输出归零
                lse_l_2d[flat_bh, q_start:actual_end] = 1.0        # l=1 → log(1)=0
                lse_m_2d[flat_bh, q_start:actual_end] = cfg.lse_init  # m=-inf → LSE = -inf
        # 因为直接修改了底层张量，需重新 reshape 得到最终输出
        o_out = output_3d.reshape(b, hq, sq_pad, d)[:, :, :sq, :].contiguous()
        softmax_lse = (lse_m_2d + torch.log(lse_l_2d)).reshape(b, hq, sq_pad)[:, :, :sq].contiguous()


    return BSAForwardResult(o=o_out, lse=softmax_lse)


@allow_in_graph
def block_sparse_attention_forward(call_inputs):
    """BSA Forward — single-phase kernel with auto-configured l1/sched.

    The kernel uses online softmax accumulation (m/l/o) across KV blocks,
    producing O and LSE in a single pass per Q block.

    Args:
        call_inputs: BSAForwardCallInputs namedtuple containing:
            query: [B, Hq, Sq, D] FP16 tensor
            key: [B, Hkv, Skv, D] FP16 tensor
            value: [B, Hkv, Skv, D] FP16 tensor
            block_sparse_mask: [B, Hq, num_qb, num_kb] bool mask
            actual_seq_lengths: per-batch Q lengths (None = all Sq)
            actual_seq_lengths_kv: per-batch KV lengths (None = all Skv)
            block_shape: (bx, by) or None for defaults
            cfg: BSAConfig instance

    Returns:
        BSAForwardResult(o=O, lse=softmax_lse) where O is [B, Hq, Sq, D]
        and lse is [B, Hq, Sq]
    """
    prepared, sparse_kv_result = _prepare_and_build_sparse_kv(call_inputs)
    return _dispatch_fwd_kernel(call_inputs, prepared, sparse_kv_result)

