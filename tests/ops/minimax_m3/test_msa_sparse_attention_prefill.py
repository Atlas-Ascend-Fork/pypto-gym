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
"""
MiniMax M3 MSA Main Branch Test Module

Data flow:
  - Q: [n, hq, dh]
  - K/V blocks: [topk, bk, hkv, dh] (gathered from raw KV by topk_indices)
  - topk_indices: [topk] int32 (selected KV block original indices, must include last block)
  - causal_mask: [bk, bk] (fixed lower-triangular)
  - block_mask: [num_blocks, topk, bk, bk] (precomputed from topk_indices + causal_mask)
  - Kernel: loop hkv(parallel) -> g -> qb -> i(topk), online softmax
  - For each selected KV block topk_indices[i], it contributes to all query blocks
    whose index >= topk_indices[i]; when query block index == topk_indices[i],
    causal mask is applied (within-block lower-triangular).
  - All causal logic is precomputed into block_mask on host; kernel has zero
    dynamic conditions (required by PyPTO AssignMemoryType pass).

Parameters: Hq=64, Hkv=4 (GQA group=16), topk=16, N=2048, head_dim=128, Bk=128.
"""

import os
import sys
import random
import logging

import torch
import torch_npu
import pytest
import pypto

_DIR = os.path.dirname(os.path.abspath(__file__))
_p = _DIR
while _p != "/" and not os.path.isdir(os.path.join(_p, "src")):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)

from ops.pypto_tensor.minimax.msa_main_branch_impl import msa_main_branch


HQ = 64
HKV = 4
TOPK = 16
HEAD_DIM = 128
BK = 128
GROUP = HQ // HKV
MASK_NEG = -1e9


def gen_inputs(n, dtype=torch.float32):
    query = torch.randn(n, HQ, HEAD_DIM, dtype=dtype)
    key = torch.randn(n, HKV, HEAD_DIM, dtype=dtype)
    value = torch.randn(n, HKV, HEAD_DIM, dtype=dtype)
    return query, key, value


def gen_topk_indices(n, seed=42):
    num_blocks = (n + BK - 1) // BK
    rng = random.Random(seed)
    all_blocks = list(range(num_blocks))
    rng.shuffle(all_blocks)
    ksel = min(TOPK, num_blocks)
    selected = all_blocks[:ksel]
    if (num_blocks - 1) not in selected:
        replace_idx = rng.randrange(len(selected))
        selected[replace_idx] = num_blocks - 1
    return torch.tensor(selected, dtype=torch.int32)


def gen_causal_mask():
    row_idx = torch.arange(BK).view(-1, 1)
    col_idx = torch.arange(BK).view(1, -1)
    return torch.where(col_idx <= row_idx, 0.0, float(MASK_NEG)).to(torch.float32)


def gen_block_mask(n, topk_indices, causal_mask):
    """Precompute per (qb, i) mask. Returns [num_blocks, topk, bk, bk].

    For each (qb, i) with kv_seq = topk_indices[i]:
      - kv_seq > qb: MASK_NEG (block not yet reachable, softmax skips it)
      - kv_seq == qb: causal_mask (within-block lower-triangular), but the
        last query block may be partial (n % bk != 0): rows beyond n are
        masked MASK_NEG so padding query rows produce zero contribution.
      - kv_seq < qb: 0.0 (full attention, no mask)
    """
    num_blocks = (n + BK - 1) // BK
    block_mask = torch.full(
        (num_blocks, TOPK, BK, BK), float(MASK_NEG), dtype=torch.float32
    )
    last_q_valid = n - (num_blocks - 1) * BK if n % BK else BK
    for qb in range(num_blocks):
        qb_valid = last_q_valid if qb == num_blocks - 1 else BK
        for i, kv_seq in enumerate(topk_indices):
            if kv_seq > qb:
                pass
            elif kv_seq == qb:
                block_mask[qb, i, :qb_valid, :] = causal_mask[:qb_valid, :]
            else:
                block_mask[qb, i, :qb_valid, :] = 0.0
    return block_mask


def golden_msa_main_branch(n, query, key_blocks, value_blocks, topk_indices, causal_mask):
    """
    Golden: sparse softmax attention on selected KV blocks.

    query : [n, hq, dh]
    key_blocks : [topk, bk, hkv, dh]
    value_blocks : [topk, bk, hkv, dh]
    topk_indices : [topk] int32 (original block indices)
    causal_mask : [bk, bk]
    """
    num_blocks = (n + BK - 1) // BK
    padded_n = num_blocks * BK
    scale = 1.0 / (HEAD_DIM ** 0.5)
    query_padded = torch.zeros(padded_n, HQ, HEAD_DIM, dtype=torch.float32)
    query_padded[:n] = query
    q_4d = query_padded.reshape(num_blocks, BK, HQ, HEAD_DIM)
    output = torch.zeros(n, HQ, HEAD_DIM, dtype=torch.float32)

    for h in range(HQ):
        h_kv = h // GROUP
        for qb in range(num_blocks):
            q = q_4d[qb, :, h, :]
            scores_list = []
            v_list = []
            for i, kv_seq in enumerate(topk_indices):
                if kv_seq > qb:
                    continue
                k = key_blocks[i, :, h_kv, :]
                v = value_blocks[i, :, h_kv, :]
                s = torch.matmul(q, k.t()) * scale
                if kv_seq == qb:
                    s = s + causal_mask
                scores_list.append(s)
                v_list.append(v)
            if not scores_list:
                continue
            scores_all = torch.cat(scores_list, dim=-1)
            v_all = torch.cat(v_list, dim=0)
            p = torch.softmax(scores_all, dim=-1)
            out = torch.matmul(p, v_all)
            qi = qb * BK
            qe = min(qi + BK, n)
            output[qi:qe, h, :] = out[:qe - qi]
    return output


def compare(actual, expected, name="tensor", atol_abs=1e-3, atol_rel=1e-3):
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape mismatch actual={tuple(actual.shape)} expected={tuple(expected.shape)}"
        )
    if not torch.isfinite(actual).all():
        raise AssertionError(f"{name}: actual contains non-finite values")
    if not torch.isfinite(expected).all():
        raise AssertionError(f"{name}: expected contains non-finite values")

    diff = torch.abs(actual - expected)
    tol = atol_abs + atol_rel * torch.abs(expected)
    out_count = (diff > tol).sum().item()
    total = actual.numel()
    max_diff = diff.max().item()

    if out_count > 0:
        bad_idx = torch.nonzero(diff > tol, as_tuple=False)[0].tolist()
        idx = tuple(bad_idx)
        raise AssertionError(
            f"{name}: {out_count}/{total} out of tolerance, "
            f"max_diff={max_diff:.6e}, atol_abs={atol_abs}, atol_rel={atol_rel}, "
            f"first_bad_idx={bad_idx}, "
            f"actual={actual[idx].item():.9e}, expected={expected[idx].item():.9e}"
        )
    logging.info(
        f"[PASS] {name}: shape={tuple(actual.shape)}, max_diff={max_diff:.6e}"
    )


def do_test(n, seed=42, max_n=None):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    if max_n is None:
        max_n = max(2048, ((n + 127) // 128) * 128)

    torch.manual_seed(0)
    query, key, value = gen_inputs(n, torch.float32)
    topk_indices = gen_topk_indices(n, seed=seed)
    causal_mask = gen_causal_mask()

    num_blocks = (n + BK - 1) // BK
    padded_n = num_blocks * BK
    key_padded = torch.zeros(padded_n, HKV, HEAD_DIM, dtype=torch.float32)
    key_padded[:n] = key
    value_padded = torch.zeros(padded_n, HKV, HEAD_DIM, dtype=torch.float32)
    value_padded[:n] = value
    key_4d = key_padded.reshape(num_blocks, BK, HKV, HEAD_DIM)
    value_4d = value_padded.reshape(num_blocks, BK, HKV, HEAD_DIM)
    ksel = len(topk_indices)
    key_blocks_4d = key_4d[topk_indices.long()]
    value_blocks_4d = value_4d[topk_indices.long()]
    if ksel < TOPK:
        pad_k = torch.zeros(TOPK - ksel, BK, HKV, HEAD_DIM, dtype=torch.float32)
        pad_v = torch.zeros(TOPK - ksel, BK, HKV, HEAD_DIM, dtype=torch.float32)
        key_blocks_4d = torch.cat([key_blocks_4d, pad_k], dim=0)
        value_blocks_4d = torch.cat([value_blocks_4d, pad_v], dim=0)

    block_mask = gen_block_mask(n, topk_indices, causal_mask)

    golden = golden_msa_main_branch(n, query, key_blocks_4d, value_blocks_4d, topk_indices, causal_mask)

    query_npu = query.npu()
    key_blocks_npu = key_blocks_4d.reshape(TOPK * BK, HKV, HEAD_DIM).npu()
    value_blocks_npu = value_blocks_4d.reshape(TOPK * BK, HKV, HEAD_DIM).npu()
    block_mask_npu = block_mask.npu()
    output_npu = torch.zeros_like(golden).npu()

    kernel = msa_main_branch(HQ, HKV, HEAD_DIM, BK, TOPK, max_n=max_n)
    kernel(query_npu, key_blocks_npu, value_blocks_npu, block_mask_npu, output_npu)
    torch.npu.synchronize()

    compare(output_npu.cpu(), golden, name=f"msa_main_branch_n{n}")


@pytest.mark.soc("950")
def test_aligned_4096():
    do_test(n=4096, seed=55, max_n=4096)


@pytest.mark.soc("950")
def test_aligned_2048():
    do_test(n=2048, seed=42)


@pytest.mark.soc("950")
def test_aligned_1024():
    do_test(n=1024, seed=43)


@pytest.mark.soc("950")
def test_unaligned_1500():
    do_test(n=1500, seed=45)


@pytest.mark.soc("950")
def test_unaligned_500():
    do_test(n=500, seed=47)


@pytest.mark.soc("950")
def test_unaligned_1():
    do_test(n=1, seed=52)


@pytest.mark.soc("950")
def test_unaligned_64():
    do_test(n=64, seed=53)


@pytest.mark.soc("950")
def test_unaligned_100():
    do_test(n=100, seed=54)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_aligned_2048()
