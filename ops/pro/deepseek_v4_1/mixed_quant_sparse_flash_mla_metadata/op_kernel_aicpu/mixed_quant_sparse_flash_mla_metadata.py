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
"""MQSMLA metadata planner as an AICPU kernel (device-side, no host readback).

Same plan as the host entry ``mixed_quant_sparse_flash_mla_metadata``: uniform whole-row split, or the flat
``(row, tile)`` FlashDecode split when it saves at least ``FD_MIN_SAVED_TILES`` of the modelled critical path.
The kernel reads every length tensor on device and writes the ``int32[1024]`` metadata in place.
"""

import torch

import pypto_pro.language as pl

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
FA_FD_TAIL_WORKSPACE_IDX_INDEX = 8

FD_USED_VEC_NUM_WORD = 900
FD_METADATA_SIZE = 8
FD_METADATA_BASE = AIC_CORE_MAX_NUM * FA_METADATA_SIZE
FD_CORE_ENABLE_INDEX = 0
FD_BN2_IDX_INDEX = 1
FD_M_IDX_INDEX = 2
FD_WORKSPACE_IDX_INDEX = 3
FD_WORKSPACE_NUM_INDEX = 4
FD_M_START_INDEX = 5
FD_M_NUM_INDEX = 6
FD_HEADS_PER_AIV = 32

TILE_N = 128
FD_MIN_SAVED_TILES = 4
FD_MIN_SAVED_TILES_LONG = 2
FD_MIN_ROWS_LONG = 96
FD_TARGET_TILES_PER_CORE = 4
FD_MIN_ROW_TILES = 5


def _ceil_div(a, b):
    if b == 0:
        return 0
    return a // b + (1 if a % b else 0)


def _batch_of_from(cu_q, n_cu, row, start):
    """从 start 开始单调前进求 batch（调用方保证 row 不回退），避免每次 O(n_cu) 全扫。"""
    b = start
    for i in pl.range(start, n_cu):
        if pl.getval(cu_q, i) <= row:
            b = i
    return b


def _row_tiles(len_ori, len_cmp, row, has_cmp, k_ori, k_cmp):
    w = pl.max(pl.getval(len_ori, row), 0)
    if k_ori > 0:
        w = pl.min(w, k_ori)
    n = _ceil_div(w, TILE_N)
    if has_cmp == 1:
        wc = pl.max(pl.getval(len_cmp, row), 0)
        if k_cmp > 0:
            wc = pl.min(wc, k_cmp)
        n = n + _ceil_div(wc, TILE_N)
    return n


def _fd_splits(len_ori, len_cmp, rows, n_blocks, total, has_cmp, k_ori, k_cmp):
    """单次前扫同时统计 split 数与去重：返回 count；命中同一行两次时返回 -count。"""
    count = 0
    ok = 1
    prev = -1
    r = 0
    st = 0
    for c in pl.range(1, n_blocks):
        f = total * c // n_blocks
        while r < rows:
            rt = _row_tiles(len_ori, len_cmp, r, has_cmp, k_ori, k_cmp)
            if st + rt > f:
                break
            st = st + rt
            r = r + 1
        if r < rows:
            rt = _row_tiles(len_ori, len_cmp, r, has_cmp, k_ori, k_cmp)
            if st < f:
                if r == prev:
                    ok = 0
                prev = r
                count = count + 1
    if ok == 0:
        return 0 - count
    return count


def _write_uniform(md, cu_q, n_cu, rows, ublocks):
    rows_per_core = rows // ublocks
    extra = rows % ublocks
    start = 0
    bc0 = 0
    bc1 = 0
    for core in pl.range(ublocks):
        count = rows_per_core + (1 if core < extra else 0)
        end = start + count
        if count > 0:
            b0 = _batch_of_from(cu_q, n_cu, start, bc0)
            b1 = _batch_of_from(cu_q, n_cu, end, bc1)
            bc0 = b0
            bc1 = b1
            fa = FA_METADATA_SIZE * core
            pl.setval(md, fa + FA_CORE_ENABLE_INDEX, 1)
            pl.setval(md, fa + FA_BN2_START_INDEX, b0)
            pl.setval(md, fa + FA_M_START_INDEX, start - pl.getval(cu_q, b0))
            pl.setval(md, fa + FA_S2_START_INDEX, 0)
            pl.setval(md, fa + FA_BN2_END_INDEX, b1)
            pl.setval(md, fa + FA_M_END_INDEX, end - pl.getval(cu_q, b1))
            pl.setval(md, fa + FA_S2_END_INDEX, 0)
            pl.setval(md, fa + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, 0)
        start = end


def _write_fd(md, cu_q, n_cu, len_ori, len_cmp, rows, n_blocks, total, has_cmp, k_ori, k_cmp):
    """FD 描述写入：行游标单调前进，总代价 O(rows + n_blocks)（原实现是 O(rows*n_blocks)）。"""
    r0 = 0
    st0 = 0
    r1 = 0
    st1 = 0
    bc0 = 0
    bc1 = 0
    for c in pl.range(n_blocks):
        f0 = total * c // n_blocks
        f1 = total * (c + 1) // n_blocks
        while r0 < rows:
            rt = _row_tiles(len_ori, len_cmp, r0, has_cmp, k_ori, k_cmp)
            if st0 + rt > f0:
                break
            st0 = st0 + rt
            r0 = r0 + 1
        while r1 < rows:
            rt = _row_tiles(len_ori, len_cmp, r1, has_cmp, k_ori, k_cmp)
            if st1 + rt > f1 - 1:
                break
            st1 = st1 + rt
            r1 = r1 + 1
        if f1 != f0:
            rt1 = _row_tiles(len_ori, len_cmp, r1, has_cmp, k_ori, k_cmp)
            e = f1 - st1
            r1e = r1 if e < rt1 else r1 + 1
            b0 = _batch_of_from(cu_q, n_cu, r0, bc0)
            b1 = _batch_of_from(cu_q, n_cu, r1e, bc1)
            bc0 = b0
            bc1 = b1
            fa = FA_METADATA_SIZE * c
            pl.setval(md, fa + FA_CORE_ENABLE_INDEX, 1)
            pl.setval(md, fa + FA_BN2_START_INDEX, b0)
            pl.setval(md, fa + FA_M_START_INDEX, r0 - pl.getval(cu_q, b0))
            pl.setval(md, fa + FA_S2_START_INDEX, f0 - st0)
            pl.setval(md, fa + FA_BN2_END_INDEX, b1)
            pl.setval(md, fa + FA_M_END_INDEX, r1e - pl.getval(cu_q, b1))
            pl.setval(md, fa + FA_S2_END_INDEX, 0 if e >= rt1 else e)

    n_splits = 0
    r = 0
    st = 0
    bc2 = 0
    for c in pl.range(1, n_blocks):
        f = total * c // n_blocks
        while r < rows:
            rt = _row_tiles(len_ori, len_cmp, r, has_cmp, k_ori, k_cmp)
            if st + rt > f:
                break
            st = st + rt
            r = r + 1
        if r < rows:
            rt = _row_tiles(len_ori, len_cmp, r, has_cmp, k_ori, k_cmp)
            if st < f:
                head_ws = 2 * (c - 1)
                tail_ws = 2 * (c - 1) + 1
                pl.setval(md, FA_METADATA_SIZE * c + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, head_ws + 1)
                pl.setval(md, FA_METADATA_SIZE * (c - 1) + FA_FD_TAIL_WORKSPACE_IDX_INDEX, tail_ws + 1)
                b = _batch_of_from(cu_q, n_cu, r, bc2)
                bc2 = b
                for half in pl.range(2):
                    base = FD_METADATA_BASE + (2 * (c - 1) + half) * FD_METADATA_SIZE
                    pl.setval(md, base + FD_CORE_ENABLE_INDEX, 1)
                    pl.setval(md, base + FD_BN2_IDX_INDEX, b)
                    pl.setval(md, base + FD_M_IDX_INDEX, r - pl.getval(cu_q, b))
                    pl.setval(md, base + FD_WORKSPACE_IDX_INDEX, head_ws)
                    pl.setval(md, base + FD_WORKSPACE_NUM_INDEX, 2)
                    pl.setval(md, base + FD_M_START_INDEX, FD_HEADS_PER_AIV * half)
                    pl.setval(md, base + FD_M_NUM_INDEX, FD_HEADS_PER_AIV)
                n_splits = n_splits + 1
    pl.setval(md, FD_USED_VEC_NUM_WORD, n_splits)


def _write_zeros(md):
    for i in pl.range(MQSMLA_METADATA_TOTAL_SIZE):
        pl.setval(md, i, 0)


@pl.jit(mode="aicpu_kernel", arch="3510")
def mixed_quant_sparse_flash_mla_metadata(
    cu_seqlens_q: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    ori_topk_length: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    cmp_topk_length: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    metadata: pl.Tensor[[MQSMLA_METADATA_TOTAL_SIZE], pl.DT_INT32, pl.Output],
    blocks: pl.DT_INT32,
    has_cmp: pl.DT_INT32,
    k_ori: pl.DT_INT32,
    k_cmp: pl.DT_INT32,
):
    n_cu = pl.numel(cu_seqlens_q)
    rows = pl.getval(cu_seqlens_q, n_cu - 1)
    _write_zeros(metadata)
    # 单趟同时求 total / min_tiles / ub_max：每行只读一次长度，避免三趟重复的标量 GM 读。
    total = 0
    min_tiles = 0
    ub_max = 0
    _u = pl.min(blocks, rows)
    _base = 0
    _extra = 0
    _next_end = 0
    _s = 0
    _cur = 0
    if _u > 0:
        _base = rows // _u
        _extra = rows % _u
        _next_end = _base + (1 if _extra > 0 else 0)
    for r in pl.range(rows):
        rt = _row_tiles(ori_topk_length, cmp_topk_length, r, has_cmp, k_ori, k_cmp)
        total = total + rt
        if r == 0:
            min_tiles = rt
        else:
            min_tiles = pl.min(min_tiles, rt)
        _s = _s + rt
        if r + 1 == _next_end:
            ub_max = pl.max(ub_max, _s)
            _s = 0
            _cur = _cur + 1
            _next_end = _next_end + _base + (1 if _cur < _extra else 0)

    fd_ok = 1
    n_blocks = 0
    fd_floor = 0
    # 短行（< FD_MIN_ROW_TILES 个 tile）落在 FD 切分边界会破坏 partial/LSE 归约；
    # 混合长度行回退 uniform，与 host/C++ 规划器保持一致。
    if min_tiles < FD_MIN_ROW_TILES:
        fd_ok = 0
    if rows <= 0 or blocks <= 0:
        fd_ok = 0
    if fd_ok == 1:
        n_blocks = pl.min(pl.max(total // FD_TARGET_TILES_PER_CORE, rows), blocks)
        if n_blocks <= 0:
            fd_ok = 0
    if fd_ok == 1:
        fd_floor = _ceil_div(total, n_blocks)
        saved_tiles = ub_max - fd_floor
        if saved_tiles < FD_MIN_SAVED_TILES:
            if saved_tiles < FD_MIN_SAVED_TILES_LONG or rows < FD_MIN_ROWS_LONG:
                fd_ok = 0
    if fd_ok == 1:
        fl_max = 0
        for c in pl.range(n_blocks):
            d = total * (c + 1) // n_blocks - total * c // n_blocks
            fl_max = pl.max(fl_max, d)
        if fl_max <= 0:
            fd_ok = 0
    if fd_ok == 1:
        n_splits = _fd_splits(
            ori_topk_length, cmp_topk_length, rows, n_blocks, total, has_cmp, k_ori, k_cmp
        )
        if n_splits < 0:
            fd_ok = 0
        else:
            if n_splits > n_blocks or n_splits * 2 > 2 * AIC_CORE_MAX_NUM:
                fd_ok = 0

    if fd_ok == 1:
        _write_fd(
            metadata, cu_seqlens_q, n_cu, ori_topk_length, cmp_topk_length,
            rows, n_blocks, total, has_cmp, k_ori, k_cmp,
        )
    else:
        ublocks = pl.min(blocks, pl.max(rows, 1))
        _write_uniform(metadata, cu_seqlens_q, n_cu, rows, ublocks)


def _cube_core_num(device):
    return int(torch.npu.get_device_properties(device).cube_core_num)


def mixed_quant_sparse_flash_mla_metadata_aicpu(
    ori_topk_length,
    cmp_topk_length=None,
    *,
    cu_seqlens_q=None,
    has_cmp_kv=True,
    block_num=None,
    k_ori=0,
    k_cmp=0,
):
    """Plan ``int32[1024]`` metadata on the AICPU core; no tensor value is read on the host."""
    if not isinstance(ori_topk_length, torch.Tensor) or ori_topk_length.dtype != torch.int32:
        raise ValueError("ori_topk_length must be an int32 torch.Tensor")
    if cmp_topk_length is not None:
        if not isinstance(cmp_topk_length, torch.Tensor) or cmp_topk_length.dtype != torch.int32:
            raise ValueError("cmp_topk_length must be an int32 torch.Tensor or None")
    rows = int(ori_topk_length.shape[0])
    device = ori_topk_length.device
    if cu_seqlens_q is None:
        cu_q = torch.tensor([0, rows], dtype=torch.int32, device=device)
    else:
        cu_q = cu_seqlens_q
    blocks = _cube_core_num(device.index) if block_num is None else int(block_num)
    if not 0 < blocks <= AIC_CORE_MAX_NUM:
        raise ValueError(f"block_num must be in 1..{AIC_CORE_MAX_NUM}, got {blocks}")
    has_cmp = 1 if (has_cmp_kv and cmp_topk_length is not None) else 0
    metadata = torch.empty(MQSMLA_METADATA_TOTAL_SIZE, dtype=torch.int32, device=device)
    mixed_quant_sparse_flash_mla_metadata(
        cu_q, ori_topk_length, cmp_topk_length if has_cmp else None,
        metadata, blocks, has_cmp, int(k_ori), int(k_cmp),
    )
    return metadata
