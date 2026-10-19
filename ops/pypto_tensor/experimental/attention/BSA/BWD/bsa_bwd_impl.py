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
BSA Backward PyPTO Kernel Implementation (Fully Dynamic Axis Pattern)

Single compiled kernel handles ALL (B, Hq, Hkv, Sq, Skv) combinations — no
per-shape re-compilation. Dynamic axes B, Hq, Hkv, Sq, Skv are passed via
hint tensors; derived values are computed inside the kernel from data shapes.

The kernel contains two phases in one merged dispatch:
  - Phase 1 (LOOP_dqd_sub): compute dQ per Q block across valid KV blocks
  - Phase 2 (LOOP_dkdvd_kblk): accumulate dK/dV per KV block across Q blocks

Key optimizations: mask preprocessing, d_row precompute outside kernel,
FP32 accumulators as kernel inputs, parallel outer loops.
"""

import os
from collections import namedtuple
import torch
from torch._dynamo import allow_in_graph
import pypto

from bsa_common import (
    BSABackwardResult,
    _pad_to_block_aligned, _prepare_qkv_2d,
    _build_sparse_kv_cached,
    _build_sparse_q_dkdv_cached,
    _SparseKVConfig,
    _make_jit_opts,
    _snapshot_output_dirs,
    _find_newest_created_dir,
    _VEC_TILE_LOAD, _CUBE_TILE_LIST,
    _prepare_and_build_sparse_kv, _SparseKVResult,
)


BSABackwardCallInputs = namedtuple(
    'BSABackwardCallInputs',
    ['dout', 'query', 'key', 'value', 'attention_out', 'softmax_lse',
     'block_sparse_mask',
     'actual_seq_lengths', 'actual_seq_lengths_kv',
     'block_shape', 'cfg'])

# Namedtuple wrappers for multi-value returns
_BwdHintTensors = namedtuple('_BwdHintTensors',
    ['b_hint', 'hq_hint', 'hkv_hint', 'sq_hint', 'skv_hint',
     'num_qb_hint', 'num_kb_hint', 'max_sel_hint', 'max_inner_hint'])
_BwdHintConfig = namedtuple('_BwdHintConfig',
    ['b', 'hq', 'hkv', 'sq', 'skv', 'num_qb', 'num_kb', 'max_sel', 'max_inner', 'device'])
_BwdMasksConfig = namedtuple('_BwdMasksConfig',
    ['valid_mask', 'inner_mask', 'softmax_scale', 'large_neg', 'do_2d', 'o_2d', 'do_c', 'o_c'])
_BwdMasks = namedtuple('_BwdMasks',
    ['scaled_mask_dq', 'neg_inf_mask_dq', 'd_row_dq',
     'scaled_mask_dkdv', 'neg_inf_mask_dkdv', 'd_row_dkdv'])
_BwdSpecificInputs = namedtuple('_BwdSpecificInputs',
    ['do_2d', 'o_2d', 'lse_2d', 'dq_2d', 'dk_2d', 'dv_2d',
     'q_c', 'do_c', 'o_c', 'lse_c', 'inner_mask', 'max_inner'])


DYNAMIC_B = pypto.frontend.dynamic('DYNAMIC_B')
DYNAMIC_H_Q = pypto.frontend.dynamic('DYNAMIC_H_Q')
DYNAMIC_H_KV = pypto.frontend.dynamic('DYNAMIC_H_KV')
DYNAMIC_S_Q = pypto.frontend.dynamic('DYNAMIC_S_Q')
DYNAMIC_S_KV = pypto.frontend.dynamic('DYNAMIC_S_KV')

DYNAMIC_NUM_QB = pypto.frontend.dynamic('DYNAMIC_NUM_QB')
DYNAMIC_NUM_KB = pypto.frontend.dynamic('DYNAMIC_NUM_KB')
DYNAMIC_MAX_SEL = pypto.frontend.dynamic('DYNAMIC_MAX_SEL')
DYNAMIC_MAX_INNER = pypto.frontend.dynamic('DYNAMIC_MAX_INNER')

DYNAMIC_TOTAL_Q = pypto.frontend.dynamic('DYNAMIC_TOTAL_Q')
DYNAMIC_TOTAL_KV_COMPACT = pypto.frontend.dynamic('DYNAMIC_TOTAL_KV_COMPACT')
DYNAMIC_TOTAL_MASK = pypto.frontend.dynamic('DYNAMIC_TOTAL_MASK')
DYNAMIC_TOTAL_Q_DKDV = pypto.frontend.dynamic('DYNAMIC_TOTAL_Q_DKDV')
DYNAMIC_TOTAL_MASK_DKDV = pypto.frontend.dynamic('DYNAMIC_TOTAL_MASK_DKDV')
DYNAMIC_TOTAL_KV = pypto.frontend.dynamic('DYNAMIC_TOTAL_KV')

_BWD_PASS_OPTS = {
    "cube_l1_reuse_setting": {-1: 64},
    "vec_nbuffer_setting": {},
}
_BWD_RT_OPTS = {}
_BWD_OPTIMAL_UNROLL = [4, 2, 1]

_bwd_cache = {}
_PERF_OUTPUT_BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
# PyPTO runtime writes output to CWD-relative "output/", not to the
# module-relative path above.  Search both so that _snapshot_output_dirs
# and _find_newest_created_dir can find swimlane data regardless of CWD.
_PERF_SEARCH_BASES = [_PERF_OUTPUT_BASE, os.path.abspath("output")]
_last_backward_perf_dirs = {}


def _snapshot_bwd_dirs():
    """Snapshot BWD-specific output directories."""
    return _snapshot_output_dirs(_PERF_SEARCH_BASES)


def _find_newest_bwd_dir(before):
    """Find newest BWD output dir created since *before*."""
    return _find_newest_created_dir(before, _PERF_SEARCH_BASES)


def _get_bwd_kernel(cfg, *, bx=None, by=None, extra_runtime_options=None, inner_unroll=None):
    """Return the BWD kernel for given configuration.

    Args:
        inner_unroll: unroll_list for inner loops (LOOP_dqd_kblk and LOOP_dkdvd_inner).
            None means no unroll. [4,2,1] is the optimized value for sched=3.
            Must be None when sched=1 (compilation failure with sched=1+unroll).
    """
    block_x = bx if bx is not None else cfg.block_shape_x
    block_y = by if by is not None else cfg.block_shape_y
    _sched = (extra_runtime_options or {}).get('device_sched_mode', 3)
    _unroll_key = tuple(inner_unroll) if inner_unroll else ()
    key = ("bwd", block_x, block_y, _sched, _unroll_key)
    if key in _bwd_cache:
        return _bwd_cache[key]

    _bwd_rt = _BWD_RT_OPTS.copy()
    if extra_runtime_options:
        _bwd_rt.update(extra_runtime_options)

    before = _snapshot_bwd_dirs()

    softmax_scale = cfg.softmax_scale
    d = cfg.head_dim
    large_neg = cfg.large_neg
    block = block_x
    kv_block = block_y
    sub_split = block_x // 64
    sub_block = block_x // sub_split
    ct = _CUBE_TILE_LIST
    vtl = _VEC_TILE_LOAD

    @pypto.frontend.jit(**_make_jit_opts(cfg, extra_pass_options=_BWD_PASS_OPTS,
                                          extra_runtime_options=_bwd_rt))
    # JIT kernel: cannot be split (PyPTO DSL requirement)
    def kernel(
        b_hint: pypto.Tensor([DYNAMIC_B, 1], pypto.DT_FP32),
        hq_hint: pypto.Tensor([DYNAMIC_H_Q, 1], pypto.DT_FP32),
        hkv_hint: pypto.Tensor([DYNAMIC_H_KV, 1], pypto.DT_FP32),
        sq_hint: pypto.Tensor([DYNAMIC_S_Q, 1], pypto.DT_FP32),
        skv_hint: pypto.Tensor([DYNAMIC_S_KV, 1], pypto.DT_FP32),
        num_qb_hint: pypto.Tensor([DYNAMIC_NUM_QB, 1], pypto.DT_FP32),
        num_kb_hint: pypto.Tensor([DYNAMIC_NUM_KB, 1], pypto.DT_FP32),
        max_sel_hint: pypto.Tensor([DYNAMIC_MAX_SEL, 1], pypto.DT_FP32),
        max_inner_hint: pypto.Tensor([DYNAMIC_MAX_INNER, 1], pypto.DT_FP32),
        q_2d: pypto.Tensor([DYNAMIC_TOTAL_Q, d], pypto.DT_FP16),
        k_compact: pypto.Tensor([DYNAMIC_TOTAL_KV_COMPACT, d], pypto.DT_FP16),
        v_compact: pypto.Tensor([DYNAMIC_TOTAL_KV_COMPACT, d], pypto.DT_FP16),
        do_2d: pypto.Tensor([DYNAMIC_TOTAL_Q, d], pypto.DT_FP16),
        lse_2d: pypto.Tensor([DYNAMIC_TOTAL_Q, 1], pypto.DT_FP32),
        scaled_mask_dq: pypto.Tensor([DYNAMIC_TOTAL_MASK, kv_block], pypto.DT_FP16),
        neg_inf_mask_dq: pypto.Tensor([DYNAMIC_TOTAL_MASK, kv_block], pypto.DT_FP16),
        d_row_dq: pypto.Tensor([DYNAMIC_TOTAL_Q, 1], pypto.DT_FP32),
        dq_2d: pypto.Tensor([DYNAMIC_TOTAL_Q, d], pypto.DT_FP32),
        q_compact: pypto.Tensor([DYNAMIC_TOTAL_Q_DKDV, d], pypto.DT_FP16),
        do_compact: pypto.Tensor([DYNAMIC_TOTAL_Q_DKDV, d], pypto.DT_FP16),
        lse_compact: pypto.Tensor([DYNAMIC_TOTAL_Q_DKDV, 1], pypto.DT_FP32),
        scaled_mask_dkdv: pypto.Tensor([DYNAMIC_TOTAL_MASK_DKDV, kv_block], pypto.DT_FP16),
        neg_inf_mask_dkdv: pypto.Tensor([DYNAMIC_TOTAL_MASK_DKDV, kv_block], pypto.DT_FP16),
        d_row_dkdv: pypto.Tensor([DYNAMIC_TOTAL_Q_DKDV, 1], pypto.DT_FP32),
        k_2d: pypto.Tensor([DYNAMIC_TOTAL_KV, d], pypto.DT_FP16),
        v_2d: pypto.Tensor([DYNAMIC_TOTAL_KV, d], pypto.DT_FP16),
        dk_2d: pypto.Tensor([DYNAMIC_TOTAL_KV, d], pypto.DT_FP32),
        dv_2d: pypto.Tensor([DYNAMIC_TOTAL_KV, d], pypto.DT_FP32),
    ):
        dtype = q_2d.dtype

        # Derive runtime dimensions from hint tensors
        b_dim = b_hint.shape[0]
        hq_dim = hq_hint.shape[0]
        hkv_dim = hkv_hint.shape[0]
        bh = b_dim * hq_dim
        bhkv = b_dim * hkv_dim
        num_qb_rt = num_qb_hint.shape[0]
        num_kb_rt = num_kb_hint.shape[0]
        max_sel_rt = max_sel_hint.shape[0]
        max_inner_rt = max_inner_hint.shape[0]

        sq_pad = num_qb_rt * block
        skv_pad = num_kb_rt * kv_block

        total_dq_outer = bh * num_qb_rt * sub_split
        total_dkdv_outer = bhkv * num_kb_rt

        # ===== Phase 1: dQ computation =====
        for outer in pypto.loop(total_dq_outer, name="LOOP_dqd_sub",
                                idx_name="dq_outer_idx", parallel=True):
            sub = outer % sub_split
            rest1 = outer // sub_split

            q_row_ofs = rest1 * block + sub * sub_block

            pypto.set_vec_tile_shapes(*vtl)
            q_sub = pypto.view(q_2d, [sub_block, d], [q_row_ofs, 0])
            do_sub = pypto.view(do_2d, [sub_block, d], [q_row_ofs, 0])
            lse_sub = pypto.view(lse_2d, [sub_block, 1], [q_row_ofs, 0])

            d_row = pypto.view(d_row_dq, [sub_block, 1], [q_row_ofs, 0])

            dq_acc = pypto.tensor([sub_block, d], pypto.DT_FP32, "dq_acc")

            for v_idx in pypto.loop(max_sel_rt, name="LOOP_dqd_kblk", idx_name="v_idx",
                                    unroll_list=inner_unroll):
                kv_row_ofs = rest1 * max_sel_rt * kv_block + v_idx * kv_block

                pypto.set_vec_tile_shapes(*vtl)
                k_block = pypto.view(k_compact, [kv_block, d], [kv_row_ofs, 0])
                v_block = pypto.view(v_compact, [kv_block, d], [kv_row_ofs, 0])

                mask_row_ofs = rest1 * max_sel_rt * block + v_idx * block + sub * sub_block
                scaled_mask_block = pypto.view(scaled_mask_dq, [sub_block, kv_block],
                                                [mask_row_ofs, 0])
                neg_inf_block = pypto.view(neg_inf_mask_dq, [sub_block, kv_block],
                                            [mask_row_ofs, 0])

                pypto.set_cube_tile_shapes(ct, ct, ct)
                s_scores = pypto.matmul(q_sub, k_block, pypto.DT_FP32,
                                a_trans=False, b_trans=True)

                scaled_mask_fp32 = pypto.cast(scaled_mask_block, pypto.DT_FP32)
                neg_inf_fp32 = pypto.cast(neg_inf_block, pypto.DT_FP32)
                s_masked = pypto.add(pypto.mul(s_scores, scaled_mask_fp32), neg_inf_fp32)

                p_probs = pypto.exp(pypto.sub(s_masked, lse_sub))

                pypto.set_cube_tile_shapes(ct, ct, ct)
                do_v = pypto.matmul(do_sub, v_block, pypto.DT_FP32,
                                    a_trans=False, b_trans=True)

                dp = pypto.mul(p_probs, pypto.sub(do_v, d_row))

                dp_fp16 = pypto.cast(dp, dtype)
                pypto.set_cube_tile_shapes(ct, ct, ct)
                dq_contrib = pypto.matmul(dp_fp16, k_block, pypto.DT_FP32)
                dq_contrib = pypto.mul(dq_contrib, softmax_scale)

                if pypto.is_loop_begin(v_idx):
                    dq_acc[:] = dq_contrib
                else:
                    dq_new = pypto.add(dq_acc, dq_contrib)
                    dq_acc[:] = dq_new

                if pypto.is_loop_end(v_idx):
                    pypto.assemble(dq_acc, [q_row_ofs, 0], dq_2d)

        # ===== Phase 2: dK/dV computation =====
        for outer in pypto.loop(total_dkdv_outer, name="LOOP_dkdvd_kblk",
                                idx_name="dkdv_outer_idx", parallel=True):
            kv_row_ofs = outer * kv_block

            pypto.set_vec_tile_shapes(*vtl)
            k_block = pypto.view(k_2d, [kv_block, d], [kv_row_ofs, 0])
            v_block = pypto.view(v_2d, [kv_block, d], [kv_row_ofs, 0])

            dk_acc = pypto.tensor([kv_block, d], pypto.DT_FP32, "dk_acc")
            dv_acc = pypto.tensor([kv_block, d], pypto.DT_FP32, "dv_acc")

            for inner in pypto.loop(max_inner_rt, name="LOOP_dkdvd_inner", idx_name="inner_idx",
                                unroll_list=inner_unroll):
                q_row_ofs = outer * max_inner_rt * block + inner * block

                d_row = pypto.view(d_row_dkdv, [block, 1], [q_row_ofs, 0])

                pypto.set_vec_tile_shapes(*vtl)
                q_block = pypto.view(q_compact, [block, d], [q_row_ofs, 0])
                do_block = pypto.view(do_compact, [block, d], [q_row_ofs, 0])
                lse_block = pypto.view(lse_compact, [block, 1], [q_row_ofs, 0])

                mask_row_ofs = outer * max_inner_rt * block + inner * block
                scaled_mask_block = pypto.view(scaled_mask_dkdv, [block, kv_block],
                                                [mask_row_ofs, 0])
                neg_inf_block = pypto.view(neg_inf_mask_dkdv, [block, kv_block],
                                            [mask_row_ofs, 0])

                pypto.set_cube_tile_shapes(ct, ct, ct)
                s_scores = pypto.matmul(q_block, k_block, pypto.DT_FP32,
                                a_trans=False, b_trans=True)

                scaled_mask_fp32 = pypto.cast(scaled_mask_block, pypto.DT_FP32)
                neg_inf_fp32 = pypto.cast(neg_inf_block, pypto.DT_FP32)
                s_masked = pypto.add(pypto.mul(s_scores, scaled_mask_fp32), neg_inf_fp32)

                p_probs = pypto.exp(pypto.sub(s_masked, lse_block))

                pypto.set_cube_tile_shapes(ct, ct, ct)
                do_v = pypto.matmul(do_block, v_block, pypto.DT_FP32,
                                    a_trans=False, b_trans=True)

                dp = pypto.mul(p_probs, pypto.sub(do_v, d_row))

                dp_fp16 = pypto.cast(dp, dtype)
                pypto.set_cube_tile_shapes(ct, ct, ct)
                dk_contrib = pypto.matmul(dp_fp16, q_block, pypto.DT_FP32,
                                          a_trans=True, b_trans=False)
                dk_contrib = pypto.mul(dk_contrib, softmax_scale)

                p_fp16 = pypto.cast(p_probs, dtype)
                pypto.set_cube_tile_shapes(ct, ct, ct)
                dv_contrib = pypto.matmul(p_fp16, do_block, pypto.DT_FP32,
                                          a_trans=True, b_trans=False)

                if pypto.is_loop_begin(inner):
                    dk_acc[:] = dk_contrib
                    dv_acc[:] = dv_contrib
                else:
                    dk_new = pypto.add(dk_acc, dk_contrib)
                    dv_new = pypto.add(dv_acc, dv_contrib)
                    dk_acc[:] = dk_new
                    dv_acc[:] = dv_new

                if pypto.is_loop_end(inner):
                    pypto.assemble(dk_acc, [kv_row_ofs, 0], dk_2d)
                    pypto.assemble(dv_acc, [kv_row_ofs, 0], dv_2d)

    output_dir = _find_newest_bwd_dir(before)
    _bwd_cache[key] = (kernel, output_dir)
    return _bwd_cache[key]


def _make_bwd_hint_tensors(cfg):
    """Create hint tensors for 5 primitive + derived dynamic axes.

    Args:
        cfg: _BwdHintConfig(b, hq, hkv, sq, skv, num_qb, num_kb, max_sel, max_inner, device).

    Returns _BwdHintTensors namedtuple grouping all hint tensors.
    """
    b, hq, hkv, sq, skv, num_qb, num_kb, max_sel, max_inner, device = (
        cfg.b, cfg.hq, cfg.hkv, cfg.sq, cfg.skv, cfg.num_qb, cfg.num_kb,
        cfg.max_sel, cfg.max_inner, cfg.device)
    b_hint = torch.zeros(b, 1, dtype=torch.float32, device=device)
    hq_hint = torch.zeros(hq, 1, dtype=torch.float32, device=device)
    hkv_hint = torch.zeros(hkv, 1, dtype=torch.float32, device=device)
    sq_hint = torch.zeros(sq, 1, dtype=torch.float32, device=device)
    skv_hint = torch.zeros(skv, 1, dtype=torch.float32, device=device)
    num_qb_hint = torch.zeros(num_qb, 1, dtype=torch.float32, device=device)
    num_kb_hint = torch.zeros(num_kb, 1, dtype=torch.float32, device=device)
    max_sel_hint = torch.zeros(max_sel, 1, dtype=torch.float32, device=device)
    max_inner_hint = torch.zeros(max_inner, 1, dtype=torch.float32, device=device)
    return _BwdHintTensors(
        b_hint=b_hint, hq_hint=hq_hint, hkv_hint=hkv_hint,
        sq_hint=sq_hint, skv_hint=skv_hint,
        num_qb_hint=num_qb_hint, num_kb_hint=num_kb_hint,
        max_sel_hint=max_sel_hint, max_inner_hint=max_inner_hint)


def _make_bwd_masks(cfg):
    """Precompute scaled/neg_inf masks and d_row vectors for dQ and dK/dV phases.

    Args:
        cfg: _BwdMasksConfig(valid_mask, inner_mask, softmax_scale, large_neg, do_2d, o_2d, do_c, o_c).

    Returns _BwdMasks namedtuple grouping all precomputed mask tensors.
    """
    valid_mask, inner_mask, softmax_scale, large_neg = (
        cfg.valid_mask, cfg.inner_mask, cfg.softmax_scale, cfg.large_neg)
    do_2d, o_2d, do_c, o_c = cfg.do_2d, cfg.o_2d, cfg.do_c, cfg.o_c
    scaled_mask_dq = (valid_mask * softmax_scale).to(torch.float16)
    neg_inf_mask_dq = ((1.0 - valid_mask) * large_neg).to(torch.float16)
    d_row_dq = (do_2d.to(torch.float32) * o_2d.to(torch.float32)).sum(dim=-1, keepdim=True)

    scaled_mask_dkdv = (inner_mask * softmax_scale).to(torch.float16)
    neg_inf_mask_dkdv = ((1.0 - inner_mask) * large_neg).to(torch.float16)
    d_row_dkdv = (do_c.to(torch.float32) * o_c.to(torch.float32)).sum(dim=-1, keepdim=True)

    padded_rows_dkdv = (inner_mask.sum(dim=-1) == 0).unsqueeze(-1)
    d_row_dkdv[padded_rows_dkdv] = 0.0

    return _BwdMasks(
        scaled_mask_dq=scaled_mask_dq, neg_inf_mask_dq=neg_inf_mask_dq,
        d_row_dq=d_row_dq, scaled_mask_dkdv=scaled_mask_dkdv,
        neg_inf_mask_dkdv=neg_inf_mask_dkdv, d_row_dkdv=d_row_dkdv)


def _prepare_bwd_specific_inputs(call_inputs, prepared):
    """BWD-specific: pad dO/O/LSE, allocate dQ/dK/dV, build sparse Q/dkdv."""
    bx = prepared.bx
    b, hq, hkv, sq, d = prepared.b, prepared.hq, prepared.hkv, prepared.sq, prepared.d
    sq_pad, skv_pad = prepared.sq_pad, prepared.skv_pad
    cfg = call_inputs.cfg

    do_pad, _ = _pad_to_block_aligned(call_inputs.dout, bx)
    o_pad, _ = _pad_to_block_aligned(call_inputs.attention_out, bx)

    do_2d = do_pad.reshape(b * hq * sq_pad, d)
    o_2d = o_pad.reshape(b * hq * sq_pad, d)

    lse_pad = torch.full([b, hq, sq_pad], cfg.lse_pad_value,
                         dtype=cfg.accum_torch_dtype, device=call_inputs.query.device)
    lse_pad[:, :, :sq] = call_inputs.softmax_lse
    lse_2d = lse_pad.reshape(b * hq * sq_pad, 1)

    dq_2d = torch.zeros(b * hq * sq_pad, d, dtype=torch.float32, device=call_inputs.query.device)
    dk_2d = torch.zeros(b * hkv * skv_pad, d, dtype=torch.float32, device=call_inputs.query.device)
    dv_2d = torch.zeros(b * hkv * skv_pad, d, dtype=torch.float32, device=call_inputs.query.device)

    dkdv_result = _build_sparse_q_dkdv_cached(
        call_inputs.block_sparse_mask,
        _SparseKVConfig(b=b, hq=hq, hkv=prepared.hkv, sq=sq, skv=prepared.skv,
                        sq_pad=sq_pad, skv_pad=skv_pad,
                        num_qb=prepared.num_qb, num_kb=prepared.num_kb,
                        bx=bx, by=prepared.by, d=d, device=call_inputs.query.device,
                        actual_seq_lengths=call_inputs.actual_seq_lengths,
                        actual_seq_lengths_kv=call_inputs.actual_seq_lengths_kv,
                        q_2d=prepared.q_2d, do_2d=do_2d, o_2d=o_2d, lse_2d=lse_2d))
    return _BwdSpecificInputs(
        do_2d=do_2d, o_2d=o_2d, lse_2d=lse_2d,
        dq_2d=dq_2d, dk_2d=dk_2d, dv_2d=dv_2d,
        q_c=dkdv_result.q_compact, do_c=dkdv_result.do_compact,
        o_c=dkdv_result.o_compact, lse_c=dkdv_result.lse_compact,
        inner_mask=dkdv_result.inner_mask, max_inner=dkdv_result.max_inner)


def _dispatch_bwd_kernel(call_inputs, prepared, bwd_inputs, sparse_kv_result):
    """Dispatch BWD kernel: create hints, masks, run kernel, reshape outputs."""
    global _last_backward_perf_dirs
    _last_backward_perf_dirs = {}
    cfg = call_inputs.cfg
    b, hq, hkv, sq, skv, d = prepared.b, prepared.hq, prepared.hkv, prepared.sq, prepared.skv, prepared.d
    num_qb, num_kb = prepared.num_qb, prepared.num_kb
    sq_pad, skv_pad = prepared.sq_pad, prepared.skv_pad
    bx, by = prepared.bx, prepared.by

    hints = _make_bwd_hint_tensors(_BwdHintConfig(
        b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, num_qb=num_qb, num_kb=num_kb,
        max_sel=sparse_kv_result.max_sel, max_inner=bwd_inputs.max_inner,
        device=call_inputs.query.device))

    masks = _make_bwd_masks(_BwdMasksConfig(
        valid_mask=sparse_kv_result.valid_mask, inner_mask=bwd_inputs.inner_mask,
        softmax_scale=cfg.softmax_scale, large_neg=cfg.large_neg,
        do_2d=bwd_inputs.do_2d, o_2d=bwd_inputs.o_2d,
        do_c=bwd_inputs.do_c, o_c=bwd_inputs.o_c))

    torch.npu.synchronize()

    bwd_rt_opts = {'device_sched_mode': 3}
    inner_unroll = _BWD_OPTIMAL_UNROLL

    bwd_kernel_fn, bwd_dir = _get_bwd_kernel(cfg, bx=bx, by=by,
                                               extra_runtime_options=bwd_rt_opts,
                                               inner_unroll=inner_unroll)
    before_bwd = _snapshot_bwd_dirs()
    bwd_kernel_fn(
        hints.b_hint, hints.hq_hint, hints.hkv_hint, hints.sq_hint, hints.skv_hint,
        hints.num_qb_hint, hints.num_kb_hint, hints.max_sel_hint, hints.max_inner_hint,
        prepared.q_2d, sparse_kv_result.k_compact, sparse_kv_result.v_compact,
        bwd_inputs.do_2d, bwd_inputs.lse_2d,
        masks.scaled_mask_dq, masks.neg_inf_mask_dq, masks.d_row_dq, bwd_inputs.dq_2d,
        bwd_inputs.q_c, bwd_inputs.do_c, bwd_inputs.lse_c,
        masks.scaled_mask_dkdv, masks.neg_inf_mask_dkdv, masks.d_row_dkdv,
        prepared.k_2d, prepared.v_2d, bwd_inputs.dk_2d, bwd_inputs.dv_2d)
    new_bwd_dir = _find_newest_bwd_dir(before_bwd)
    if new_bwd_dir:
        _unroll_key = tuple(_BWD_OPTIMAL_UNROLL)
        key = ("bwd", bx, by, 3, _unroll_key)
        _bwd_cache[key] = (bwd_kernel_fn, new_bwd_dir)
        bwd_dir = new_bwd_dir
    _last_backward_perf_dirs["dQ"] = bwd_dir
    _last_backward_perf_dirs["dK/dV"] = bwd_dir

    # 后处理: 将 valid_mask 全零的 Q block 的 dQ 强制归零 ----
    # 原因：kernel 的 online softmax 在 valid_mask 全零时，
    # 由于 LSE 可能为 -inf，产生 NaN/inf 的 dQ。
    valid_mask = sparse_kv_result.valid_mask
    max_sel = sparse_kv_result.max_sel
    num_qblocks = b * hq * num_qb
    mask_row_sums = valid_mask.reshape(num_qblocks, max_sel * bx, -1).sum(dim=(1, 2))
    all_zero_mask = (mask_row_sums == 0)

    if all_zero_mask.any():
        dq_3d = bwd_inputs.dq_2d.reshape(b * hq, sq_pad, d)
        zero_indices = torch.where(all_zero_mask)[0]
        for qb_idx in zero_indices.tolist():
            flat_bh = qb_idx // num_qb
            u = qb_idx % num_qb
            q_start = u * bx
            q_end = min(q_start + bx, sq_pad)
            dq_3d[flat_bh, q_start:q_end, :] = 0.0

    dq = bwd_inputs.dq_2d.reshape(b, hq, sq_pad, d)[:, :, :sq, :].to(torch.float16).contiguous()
    dk = bwd_inputs.dk_2d.reshape(b, hkv, skv_pad, d)[:, :, :skv, :].to(torch.float16).contiguous()
    dv = bwd_inputs.dv_2d.reshape(b, hkv, skv_pad, d)[:, :, :skv, :].to(torch.float16).contiguous()
    return BSABackwardResult(d_q=dq, d_k=dk, d_v=dv)


@allow_in_graph
def block_sparse_attention_backward(call_inputs):
    """BSA Backward — merged dQ+dK/dV single kernel with auto-configured sched.

    Args:
        call_inputs: BSABackwardCallInputs namedtuple containing:
            dout, query, key, value, attention_out, softmax_lse,
            block_sparse_mask, actual_seq_lengths, actual_seq_lengths_kv,
            block_shape, cfg

    Returns:
        BSABackwardResult(d_q, d_k, d_v) where gradients are [B, H, S, D] FP16
    """
    prepared, sparse_kv_result = _prepare_and_build_sparse_kv(call_inputs)
    bwd_inputs = _prepare_bwd_specific_inputs(call_inputs, prepared)
    return _dispatch_bwd_kernel(call_inputs, prepared, bwd_inputs, sparse_kv_result)