# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F821
"""AICPU task planner; unchanged sequence and assignment semantics."""
import pypto_pro.language as pl

I32 = pl.DT_INT32


def _write_vector_ranges(plan, rows, vector_workers):
    """Assign consecutive output rows to the vector workers."""
    rows_per_worker = rows // vector_workers
    extra_rows = rows % vector_workers
    for core in pl.range(vector_workers):
        first = core * rows_per_worker + pl.min(core, extra_rows)
        count = rows_per_worker
        if core < extra_rows:
            count = count + 1
        pl.setval(plan, 1 + 3 * rows + 2 * core, first)
        pl.setval(plan, 1 + 3 * rows + 2 * core + 1, first + count)


def _has_active_batch(batches, used_q, used_k, has_used_q, has_used_k):
    """Check whether the empty-output shortcut can skip the Cube plan."""
    any_active = 0
    for b in pl.range(batches):
        active = 1
        if has_used_q != 0:
            if pl.getval(used_q, b) <= 0:
                active = 0
        if has_used_k != 0:
            if pl.getval(used_k, b) <= 0:
                active = 0
        if active != 0:
            any_active = 1
            break
    return any_active


def _write_batch_tasks(plan, b, query, key, distribution, shards):
    """Write the eight-word Cube task records in worker order."""
    begin, end, qlength, qgroups = query
    kstart, klength, fast = key
    workers, tasks_per_worker, extra_tasks, task_base = distribution
    split, shard_tiles, tile_n, total_tiles = shards
    for qgroup in pl.range(qgroups):
        qbase = begin + qgroup * 6
        qvalid = pl.min(6, pl.max(0, end - qbase))
        task_k = klength
        task_start = kstart
        if fast != 0:
            task_start = 0
        elif qlength == 0:
            task_k = 0
        for shard in pl.range(split):
            item = (b * qgroups + qgroup) * split + shard
            owner = item % workers
            first = owner * tasks_per_worker + pl.min(owner, extra_tasks)
            packed = first + item // workers
            record = task_base + 8 * packed
            pl.setval(plan, record, qbase)
            pl.setval(plan, record + 1, qvalid)
            pl.setval(plan, record + 2, task_k)
            pl.setval(plan, record + 3, task_start)
            pl.setval(plan, record + 4, shard)
            pl.setval(plan, record + 5, b)
            key_begin = shard * shard_tiles * tile_n
            key_tiles = pl.min(qvalid, 1) * pl.max(
                0, pl.min(shard_tiles, total_tiles - shard * shard_tiles)
            )
            pl.setval(plan, record + 6, key_begin)
            pl.setval(plan, record + 7, key_tiles)


def _write_batches(plan, sequences, present, shape, distribution, shards):
    """Expand device sequence lengths into row metadata and Cube task records."""
    cu_q, cu_k, used_q, used_k, residual = sequences
    has_cu_q, has_cu_k, has_used_q, has_used_k, has_residual = present
    rows, max_k, batches, qgroups, fast = shape
    any_active = 0
    for b in pl.range(batches):
        begin = 0
        end = rows
        if has_cu_q != 0:
            begin = pl.getval(cu_q, b)
            end = pl.getval(cu_q, b + 1)
        qlength = end - begin
        if has_used_q != 0:
            qlength = pl.getval(used_q, b)
        kstart = 0
        klength = max_k
        if has_cu_k != 0:
            kstart = pl.getval(cu_k, b)
            klength = pl.getval(cu_k, b + 1) - kstart
        if has_used_k != 0:
            klength = pl.getval(used_k, b)
        rem = 0
        if has_residual != 0:
            rem = pl.getval(residual, b)
        active = 1
        if has_used_q != 0 and qlength <= 0:
            active = 0
        if has_used_k != 0 and klength <= 0:
            active = 0
        if active != 0:
            any_active = 1
        for row in pl.range(begin, end):
            record = 1 + 3 * row
            pl.setval(plan, record, qlength - (row - begin))
            pl.setval(plan, record + 1, klength)
            pl.setval(plan, record + 2, rem)
        _write_batch_tasks(plan, b, (begin, end, qlength, qgroups),
                           (kstart, klength, fast), distribution, shards)
    return any_active


@pl.jit(mode="aicpu_kernel", arch="3510")
def quant_lightning_indexer_metadata(
    cu_q: pl.Tensor[[pl.DYNAMIC], I32, pl.Input],
    cu_k: pl.Tensor[[pl.DYNAMIC], I32, pl.Input],
    used_q: pl.Tensor[[pl.DYNAMIC], I32, pl.Input],
    used_k: pl.Tensor[[pl.DYNAMIC], I32, pl.Input],
    residual: pl.Tensor[[pl.DYNAMIC], I32, pl.Input],
    plan: pl.Tensor[[pl.DYNAMIC], I32, pl.Output],
    rows: pl.DT_INT64, max_q: pl.DT_INT64,
    max_k: pl.DT_INT64, schedule: pl.DT_INT64,
):
    """Build query/key task boundaries on AICPU from device sequence arrays.

    The compute kernel's existing six-query groups and key shards are preserved.
    Schedule bits 0Ã¢â‚¬â€œ7 encode the split, 8Ã¢â‚¬â€œ23 the tile width, 24Ã¢â‚¬â€œ31 the Cube
    count, 32Ã¢â‚¬â€œ39 the Vector count and 40Ã¢â‚¬â€œ46 the flags. Sequence lengths come
    from the input tensors.
    """
    split = schedule & 255
    tile_n = (schedule >> 8) & 65535
    workers = (schedule >> 24) & 255
    vector_workers = (schedule >> 32) & 255
    flags = schedule >> 40
    fast = flags & 1
    has_cu_q = (flags >> 1) & 1
    has_cu_k = (flags >> 2) & 1
    has_used_q = (flags >> 3) & 1
    has_used_k = (flags >> 4) & 1
    has_residual = (flags >> 5) & 1
    allow_empty = (flags >> 6) & 1
    batches = 1
    if has_cu_q != 0:
        batches = cu_q.shape[0] - 1
    _write_vector_ranges(plan, rows, vector_workers)
    # A fully empty call only consumes the AIV row ranges and the empty flag.
    # Decide this on AICPU before producing unused Cube tasks or row operands.
    any_active = 0
    if allow_empty != 0:
        any_active = _has_active_batch(batches, used_q, used_k, has_used_q, has_used_k)
    if allow_empty == 0 or any_active != 0:
        qgroups = (max_q + 5) // 6
        tasks = batches * qgroups * split
        task_base = 1 + 3 * rows + 2 * vector_workers + 2 * workers
        total_tiles = (max_k + tile_n - 1) // tile_n
        shard_tiles = (total_tiles + split - 1) // split
        tasks_per_worker = tasks // workers
        extra_tasks = tasks % workers
        for core in pl.range(workers):
            first = core * tasks_per_worker + pl.min(core, extra_tasks)
            count = tasks_per_worker
            if core < extra_tasks:
                count = count + 1
            pl.setval(plan, 1 + 3 * rows + 2 * vector_workers + 2 * core, first)
            pl.setval(plan, 1 + 3 * rows + 2 * vector_workers + 2 * core + 1, first + count)
        any_active = _write_batches(
            plan, (cu_q, cu_k, used_q, used_k, residual),
            (has_cu_q, has_cu_k, has_used_q, has_used_k, has_residual),
            (rows, max_k, batches, qgroups, fast),
            (workers, tasks_per_worker, extra_tasks, task_base),
            (split, shard_tiles, tile_n, total_tiles),
        )
    pl.setval(plan, 0, 1 - any_active)
