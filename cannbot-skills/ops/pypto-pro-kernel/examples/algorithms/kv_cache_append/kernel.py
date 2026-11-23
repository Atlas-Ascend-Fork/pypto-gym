# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。分页 K/V cache 原地写入，重复槽由输入顺序最后的 token 获得。

cache 是连续 FP32 [块,页长,头,维]，new 是连续 FP32 [T,头,维]。调用者提供互不重叠的
张量，同设备 INT32 slots[T] 的值在 [0,块×页长) 内。展平后一个 token 恰好占一行。
单 worker 按序写；多 worker 先查后缀，放弃已被覆盖的写入，从而为每个目的槽确定唯一
写者。后缀比较为 O(T²)，仅面向 T≤256 的短追加。slot 表驻留 UB，auto_mutex 保护双槽。
行宽 Hkv×D 须为 8 的倍数且不超过 2048。
参见[展平前的坐标约定](../../../references/patterns/indexed-gather.md#preserve-coordinates-before-flattening)。

example-query: {\
  "term_groups":[\
    ["KV cache","paged cache","分页缓存","KV 缓存"],\
    ["duplicate slots","repeated slots","重复槽"],\
    ["last writer","last-writer","最后"]],\
  "exclude":[{"term_groups":[\
    ["use scatter-add","accumulate duplicate slots","重复槽累加","需要累加"]],\
    "reason":"Duplicate destinations retain the final input token in order."}]}
"""

import torch
import pypto_pro.language as pl

_DATA_DTYPE = pl.DT_FP32
_SLOT_DTYPE = pl.DT_INT32
_MAX_W = 2048
_MAX_TOKENS = 256
_MAX_WORKERS = 32
_ROW_BYTES = _MAX_W * 4


@pl.jit(auto_mutex=True)
def kv_cache_append_kernel(
    k_cache: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], _DATA_DTYPE],
    v_cache: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], _DATA_DTYPE],
    new_k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], _DATA_DTYPE],
    new_v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], _DATA_DTYPE],
    slots: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], _SLOT_DTYPE],
    scan_end: pl.DT_INT64,
):
    """Scatter `T` rows into the two caches, last-writer-wins, in place."""
    with pl.section_vector():
        row_type = pl.TileType(shape=[1, _MAX_W], dtype=_DATA_DTYPE,
                               target_memory=pl.MemorySpace.Vec,
                               valid_shape=[-1, -1])
        slot_type = pl.TileType(shape=[1, _MAX_TOKENS], dtype=_SLOT_DTYPE,
                                target_memory=pl.MemorySpace.Vec,
                                valid_shape=[-1, -1])
        k_group = pl.make_tile_group(type=row_type, addrs=[0, _ROW_BYTES],
                                     mutex_ids=[0, 1], depth=2)
        v_group = pl.make_tile_group(type=row_type, addrs=[2 * _ROW_BYTES, 3 * _ROW_BYTES],
                                     mutex_ids=[2, 3], depth=2)
        slot_group = pl.make_tile_group(type=slot_type, addrs=[4 * _ROW_BYTES],
                                        mutex_ids=[4], depth=1)

        tokens = new_k.shape[0]
        width = new_k.shape[1]

        # 各 worker 独立读取紧密的 slot 表，随后可在 UB 中反复查找。
        slot_tile = slot_group.next()
        pl.set_validshape(slot_tile, [1, tokens])
        pl.load(slot_tile, slots, [0, 0])

        for token in pl.range(pl.get_block_idx(), tokens, pl.get_block_num()):
            slot = pl.getval(slot_tile, token)

            # 先发出与 slot 无关的行 DMA，再进行后缀比较。
            k_tile = k_group.next()
            pl.set_validshape(k_tile, [1, width])
            pl.load(k_tile, new_k, [token, 0])
            v_tile = v_group.next()
            pl.set_validshape(v_tile, [1, width])
            pl.load(v_tile, new_v, [token, 0])

            # 单 worker 可按序覆盖。多 worker 只允许该槽的最后一个 token 写回。
            is_last = 1
            for later in pl.range(token + 1, scan_end):
                diff = pl.getval(slot_tile, later) - slot
                is_last = is_last * pl.min(pl.max(diff, -diff), 1)

            if is_last != 0:
                pl.store(k_cache, k_tile, [slot, 0])
                pl.store(v_cache, v_tile, [slot, 0])


def _check_shapes(k_cache, v_cache, new_k, new_v, slot_mapping):
    if not isinstance(k_cache, torch.Tensor) or k_cache.device.type != "npu":
        raise ValueError("k_cache must be an NPU tensor")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("k_cache and v_cache must be 4D tensors")
    if new_k.dim() != 3 or new_v.dim() != 3:
        raise ValueError("new_k and new_v must be 3D tensors")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")
    if new_k.shape != new_v.shape:
        raise ValueError("new_k and new_v must have the same shape")
    if tuple(new_k.shape[1:]) != tuple(k_cache.shape[2:]):
        raise ValueError("new token H/D dimensions must match cache")
    if slot_mapping.dim() != 1 or slot_mapping.shape[0] != new_k.shape[0]:
        raise ValueError("slot_mapping must be [T]")


def _check_storage(k_cache, v_cache, new_k, new_v, slot_mapping):
    if k_cache.dtype != torch.float32:
        raise TypeError("k_cache dtype must be float32")
    if slot_mapping.dtype != torch.int32:
        raise TypeError("slot_mapping dtype must be int32")
    for name, t in (("v_cache", v_cache), ("new_k", new_k), ("new_v", new_v)):
        if t.dtype != k_cache.dtype:
            raise TypeError(name + " dtype must match k_cache")
    for name, t in (("k_cache", k_cache), ("v_cache", v_cache), ("new_k", new_k),
                    ("new_v", new_v), ("slot_mapping", slot_mapping)):
        if t.device != k_cache.device:
            raise ValueError(name + " must be on the same device as k_cache")
        if not t.is_contiguous():
            raise ValueError(name + " must be contiguous")


def run(k_cache: torch.Tensor, v_cache: torch.Tensor, new_k: torch.Tensor,
        new_v: torch.Tensor, slot_mapping: torch.Tensor, *, workers=8):
    """Append T new K/V tokens into their slot_mapping cache slots (in place)."""
    _check_shapes(k_cache, v_cache, new_k, new_v, slot_mapping)
    _check_storage(k_cache, v_cache, new_k, new_v, slot_mapping)
    num_blocks, page_block_size, hkv, head_dim = k_cache.shape
    tokens = int(new_k.shape[0])
    num_slots = num_blocks * page_block_size
    width = hkv * head_dim
    if not 1 <= workers <= _MAX_WORKERS:
        raise ValueError("workers must be in [1, 32]")
    if not 0 < num_slots < 2**31 or not 0 < width <= _MAX_W or width % 8:
        raise ValueError("require positive INT32 slot count and width divisible by 8, at most 2048")
    if tokens == 0:
        return k_cache, v_cache
    if tokens > _MAX_TOKENS:
        raise ValueError("token count %d exceeds supported %d" % (tokens, _MAX_TOKENS))

    # 连续维合并成零拷贝视图；对原 cache 的写入直接对调用方可见。
    k_flat = k_cache.reshape(num_slots, width)
    v_flat = v_cache.reshape(num_slots, width)
    nk_flat = new_k.reshape(tokens, width)
    nv_flat = new_v.reshape(tokens, width)
    slots = slot_mapping.reshape(1, tokens)

    workers = min(workers, tokens)
    kv_cache_append_kernel[None, workers](
        k_flat, v_flat, nk_flat, nv_flat, slots, tokens if workers > 1 else 0)
    return k_cache, v_cache
