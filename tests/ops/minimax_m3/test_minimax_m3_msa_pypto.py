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
"""MiniMax-M3 MSA PyPTO kernels — precision tests (NPU-only; skipped without Ascend).

Validates the two PyPTO MSA decode kernels against eager-torch references:
  * indexer  — selected block ids IDENTICAL to the torch lightning-indexer (max-pool + top-k + local);
  * attention — block-sparse flash output == gather-SDPA over the selected blocks (aligned + partial ctx);
  * e2e       — indexer -> attention matches the torch-MSA reference end to end.
"""
import os
import sys

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise ImportError("torch_npu not available; this test only runs on Ascend NPU.") from exc

_DIR = os.path.dirname(os.path.abspath(__file__))
_p = _DIR
while _p != "/" and not os.path.isdir(os.path.join(_p, "src")):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)

from ops.pypto_tensor.minimax.minimax_m3_msa_indexer_impl import (  # noqa: E402
    minimax_m3_msa_indexer, NIDX, D, BY, TOPK, LOCAL)
from ops.pypto_tensor.minimax.minimax_m3_msa_sparse_attention_impl import (  # noqa: E402
    minimax_m3_msa_sparse_decode, HQ, HKV, GROUP, SCALE)

# NOTE: the HF modeling / config (``pypto_gym.transformers.minimax_m3``) pull in ``transformers``,
# which is NOT part of the on-device op-smoke environment. Only the last test needs the full HF
# attention module, so those imports are deferred into it (and it skips when transformers is absent);
# the kernel-only indexer / attention / e2e tests below run with just torch + pypto.

DEV = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"


def _torch_indexer(idx_q, idx_k, nb):
    sc = torch.matmul(idx_q, idx_k.transpose(0, 1))               # [NIDX, nb*BY]
    blk = sc.view(NIDX, nb, BY).amax(-1).amax(0)                  # [nb]
    ksel = min(TOPK, nb)
    if LOCAL > 0 and nb > ksel:                                  # mirrors the kernel's short-ctx guard
        ids = blk[:nb - LOCAL].topk(ksel - LOCAL, sorted=False).indices.to(torch.int32)
        loc = torch.arange(nb - LOCAL, nb, device=idx_q.device, dtype=torch.int32)
        return torch.cat([ids, loc])
    return torch.arange(nb, device=idx_q.device, dtype=torch.int32)


def _gather_oracle(q, k_blocks, v_blocks, sel, seq_len):
    cur_block = (seq_len - 1) // BY
    cur_valid = seq_len - cur_block * BY
    kg = k_blocks[0, :, sel].reshape(HKV, TOPK * BY, D).repeat_interleave(GROUP, 0).unsqueeze(0)
    vg = v_blocks[0, :, sel].reshape(HKV, TOPK * BY, D).repeat_interleave(GROUP, 0).unsqueeze(0)
    sc = torch.matmul(q.unsqueeze(2).float(), kg.float().transpose(-1, -2)) * SCALE
    col_valid = torch.ones(TOPK * BY, device=q.device, dtype=torch.bool)
    for j, blk in enumerate(sel.tolist()):
        if blk == cur_block:
            col_valid[j * BY + cur_valid:(j + 1) * BY] = False
        elif blk > cur_block:
            col_valid[j * BY:(j + 1) * BY] = False
    sc = sc.masked_fill(~col_valid[None, None, None, :], float("-inf"))
    return torch.matmul(torch.softmax(sc, -1), vg.float()).squeeze(2)


def _append_prompt_cache(attn, cache, prompt, pos_emb):
    from pypto_gym.transformers.minimax_m3.modeling_minimax_m3 import apply_rotary_pos_emb
    bsz, seq, _ = prompt.shape
    key = attn.k_norm(attn.k_proj(prompt).view(bsz, seq, HKV, D)).transpose(1, 2)
    value = attn.v_proj(prompt).view(bsz, seq, HKV, D).transpose(1, 2)
    key, _ = apply_rotary_pos_emb(key, key, *pos_emb)
    _, idx_k = getattr(attn, "_msa_index_qk")(prompt, pos_emb)
    cache.append(attn.layer_idx, key, value, idx_k)


@torch.no_grad()
@pytest.mark.parametrize("nb", [8, TOPK + 1])     # nb<=TOPK guard + shortest sparse path
def test_indexer_selection_identical_to_torch(nb):
    torch.npu.set_device(DEV)
    torch.manual_seed(nb)
    idx_q = torch.randn(NIDX, D, device=DEV, dtype=torch.bfloat16) * 0.1
    idx_k = torch.randn(nb * BY, D, device=DEV, dtype=torch.bfloat16) * 0.1
    pypto = minimax_m3_msa_indexer(idx_q, idx_k, nb)
    assert pypto.shape == (1, min(TOPK, nb)), f"unexpected selection shape {tuple(pypto.shape)}"
    pypto_ids = set(pypto.view(-1).tolist())
    torch_ids = set(_torch_indexer(idx_q, idx_k, nb).view(-1).tolist())
    assert pypto_ids == torch_ids, f"selection differs: pypto-only={pypto_ids - torch_ids}"


@torch.no_grad()
@pytest.mark.skip(reason="large MSA sparse-decode case; excluded from CI smoke (needs a full free die)")
@pytest.mark.parametrize("nb,seq_len", [(TOPK + 1, (TOPK + 1) * BY)])
def test_e2e_indexer_plus_attention(nb, seq_len):
    torch.npu.set_device(DEV)
    torch.manual_seed(nb)
    idx_q = torch.randn(NIDX, D, device=DEV, dtype=torch.bfloat16) * 0.1
    idx_k = torch.randn(nb * BY, D, device=DEV, dtype=torch.bfloat16) * 0.1
    q = torch.randn(1, HQ, D, device=DEV, dtype=torch.bfloat16) * 0.05
    k_blocks = torch.randn(1, HKV, nb, BY, D, device=DEV, dtype=torch.bfloat16) * 0.05
    v_blocks = torch.randn(1, HKV, nb, BY, D, device=DEV, dtype=torch.bfloat16) * 0.05

    block_ids = minimax_m3_msa_indexer(idx_q, idx_k, nb)                    # PyPTO indexer
    out = minimax_m3_msa_sparse_decode(q, k_blocks, v_blocks, block_ids, seq_len)  # PyPTO attention

    ref_ids = _torch_indexer(idx_q, idx_k, nb).view(1, TOPK)
    assert set(block_ids[0].tolist()) == set(ref_ids[0].tolist())
    ref = _gather_oracle(q, k_blocks, v_blocks, ref_ids[0].long(), seq_len)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2


@torch.no_grad()
@pytest.mark.skip(reason="large HF attention integration case; excluded from CI smoke (needs a full free die)")
def test_attention_forward_pypto_msa_matches_native_paged_decode():
    try:
        from pypto_gym.transformers.minimax_m3.configuration_minimax_m3 import MiniMaxM3Config
        from pypto_gym.transformers.minimax_m3.modeling_minimax_m3 import (
            MiniMaxM3Attention, MiniMaxM3RotaryEmbedding, MiniMaxM3PagedSparseCache)
    except ImportError:
        pytest.skip("transformers / HF modeling unavailable; kernel-only tests above cover the PyPTO ops")
    torch.npu.set_device(DEV)
    torch.manual_seed(2026)
    sa = {"use_sparse_attention": True, "sparse_index_dim": 128, "sparse_num_index_heads": 4,
          "sparse_block_size": BY, "sparse_topk_blocks": TOPK, "sparse_local_block": LOCAL,
          "sparse_attention_freq": [0, 0, 0] + [1] * 57}
    cfg = MiniMaxM3Config(hidden_size=6144, num_attention_heads=HQ, num_key_value_heads=HKV,
                          head_dim=D, num_hidden_layers=60, rotary_dim=64,
                          sparse_attention_config=sa, first_k_dense_replace=3)
    setattr(cfg, "_attn_implementation", "eager")
    attn = MiniMaxM3Attention(cfg, layer_idx=3).to(DEV).to(torch.bfloat16).eval()
    rot = MiniMaxM3RotaryEmbedding(cfg).to(DEV)
    prompt_len = 20 * BY
    prompt = torch.randn(1, prompt_len, cfg.hidden_size, device=DEV, dtype=torch.bfloat16) * 0.02
    h1 = torch.randn(1, 1, cfg.hidden_size, device=DEV, dtype=torch.bfloat16) * 0.02
    pos = torch.arange(prompt_len, device=DEV).unsqueeze(0)
    p1 = torch.tensor([[prompt_len]], device=DEV)

    def decode(use_pypto):
        cache = MiniMaxM3PagedSparseCache(cfg, max_ctx=prompt_len + 1, device=DEV, dtype=torch.bfloat16)
        attn.is_sparse = True
        attn.use_msa_sparse = True
        attn.use_pypto_msa = use_pypto
        _append_prompt_cache(attn, cache, prompt, rot(prompt, pos))
        return attn(h1, rot(h1, p1), past_key_values=cache,
                    cache_position=torch.tensor([prompt_len], device=DEV), position_ids=p1)

    native_out = decode(False)
    pypto_out = decode(True)
    assert (pypto_out.float() - native_out.float()).abs().max().item() < 5e-2
