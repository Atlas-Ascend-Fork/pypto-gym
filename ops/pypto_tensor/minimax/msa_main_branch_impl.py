#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
MiniMax M3 MSA Main Branch Implementation Module (flash attention, GQA-batched)

GQA-batched flash attention: all 16 Q heads sharing one KV head are batched
into the cube M dimension (per query-block tile).  K/V/mask are loaded once
per chunk and reused 16x, reducing HBM traffic dramatically.

Loop order: h_kv -> n_block -> g(inner chunk) -> c(kv chunk)
  This keeps oi on-chip (one head at a time) while K/V/mask are loaded
  in the outer n_block/c loops and reused across all 16 Q heads.

Reference: MiniMax M3 Technical Report (arXiv:2606.13392v2), Equation 8.
"""

import os

import torch
import pypto

_MAX_N = 2048
_NTILE = int(os.environ.get("MSA_NTILE", "512"))
_NQ_TILE = int(os.environ.get("MSA_NQ_TILE", "128"))  # query-block tile size
_DTYPE = os.environ.get("MSA_DTYPE", "fp32")  # fp32 | fp16 | bf16

_DT_MAP = {
    "fp32": (pypto.DT_FP32, torch.float32),
    "fp16": (pypto.DT_FP16, torch.float16),
    "bf16": (pypto.DT_BF16, torch.bfloat16),
}


def msa_main_branch(hq, hkv, dh, bk, topk, max_n=None):
    """max_n: query 最大行长（默认 _MAX_N=2048，可用 MSA_MAX_N 环境变量或参数覆盖，如 4096）。
    max_n 必须是 bk 的整数倍。
    """
    if max_n is None:
        max_n = int(os.environ.get("MSA_MAX_N", str(_MAX_N)))
    group = hq // hkv
    total_heads = hkv * group
    kv_len = topk * bk
    ntile = _NTILE
    nchunks = kv_len // ntile
    nq_tile = _NQ_TILE
    nq_blocks = max_n // nq_tile
    pt_dt, torch_dt = _DT_MAP[_DTYPE]
    scale = 1.0 / (dh ** 0.5)

    @pypto.frontend.jit(
        runtime_options={
            "stitch_function_max_num": 64,
            "device_sched_mode": 1,
        },
        pass_options={
            "ooo_sched_mode": "HLF",
            "sg_set_tunevf_mode": 1,
            "vec_nbuffer_setting": {"DEFAULT": 1},
            "cube_l1_reuse_setting": {"DEFAULT": 2},
            "cube_nbuffer_setting": {"DEFAULT": 2},
        },
        codegen_options={
            "vf_options": "-mllvm -cce-vf-enable-vloopv2-recognizer=true"
        },
        host_options={"compile_timeout": 3600}
    )
    def kernel(
        query: pypto.Tensor([pypto.DYNAMIC, hq, dh], pt_dt),
        key_blocks: pypto.Tensor([topk * bk, hkv, dh], pt_dt),
        value_blocks: pypto.Tensor([topk * bk, hkv, dh], pt_dt),
        block_mask: pypto.Tensor([max_n, kv_len], pypto.DT_FP32),
        output: pypto.Tensor([pypto.DYNAMIC, hq, dh], pypto.DT_FP32),
    ):
        pypto.experimental.auto_mix_partition(1)

        n = query.shape[0]

        query_2d = pypto.reshape(query, [n, hq * dh], inplace=True)
        output_2d = pypto.reshape(output, [n, hq * dh], inplace=True)
        key_2d = pypto.reshape(key_blocks, [kv_len, hkv * dh], inplace=True)
        value_2d = pypto.reshape(value_blocks, [kv_len, hkv * dh], inplace=True)
        mask_view = pypto.view(block_mask, [max_n, kv_len], [0, 0],
                               valid_shape=[n, kv_len])

        pypto.set_vec_tile_shapes(128, 128)

        for hg in pypto.loop(total_heads, name="LOOP_HG", idx_name="hg"):
            h_kv = hg // group
            g = hg - h_kv * group
            h_idx = h_kv * group + g
            h_kv_col = h_kv * dh
            q_col = h_idx * dh

            k_all = pypto.view(key_2d, [kv_len, dh], [0, h_kv_col])
            v_all = pypto.view(value_2d, [kv_len, dh], [0, h_kv_col])
            q_all = pypto.view(query_2d, [max_n, dh], [0, q_col], valid_shape=[n, dh])

            mi = pypto.full([max_n, 1], float("-inf"), pypto.DT_FP32, valid_shape=[n, 1])
            li = pypto.full([max_n, 1], 0.0, pypto.DT_FP32, valid_shape=[n, 1])
            oi = pypto.full([max_n, dh], 0.0, pypto.DT_FP32, valid_shape=[n, dh])

            for c in range(nchunks):
                kv_off = c * ntile
                k_ch = pypto.view(k_all, [ntile, dh], [kv_off, 0])
                v_ch = pypto.view(v_all, [ntile, dh], [kv_off, 0])
                mask_ch = pypto.view(block_mask, [max_n, ntile], [0, kv_off],
                                     valid_shape=[n, ntile])

                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                raw = pypto.matmul(q_all, k_ch, pypto.DT_FP32, b_trans=True)

                scaled = pypto.mul(raw, scale)
                masked = pypto.add(scaled, mask_ch)
                m_c = pypto.amax(masked, dim=-1, keepdim=True)
                shifted = pypto.sub(masked, m_c)
                p = pypto.exp(shifted)
                l_c = pypto.sum(p, dim=-1, keepdim=True)

                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                p_cast = pypto.cast(p, pt_dt) if pt_dt != pypto.DT_FP32 else p
                pv = pypto.matmul(p_cast, v_ch, pypto.DT_FP32)

                if c == 0:
                    mi[:] = m_c
                    li[:] = l_c
                    oi[:] = pv
                else:
                    mi_new = pypto.maximum(mi, m_c)
                    alpha = pypto.exp(pypto.sub(mi, mi_new))
                    beta = pypto.exp(pypto.sub(m_c, mi_new))
                    li[:] = pypto.add(pypto.mul(alpha, li), pypto.mul(beta, l_c))
                    oi[:] = pypto.add(pypto.mul(oi, alpha), pypto.mul(pv, beta))
                    mi[:] = mi_new

            out = pypto.div(oi, li)
            pypto.assemble(out, [0, q_col], output_2d)

    def wrapper(query, key_blocks, value_blocks, block_mask, output):
        mask_r = block_mask.transpose(1, 2).reshape(
            block_mask.shape[0] * bk, kv_len
        ).contiguous()
        if mask_r.shape[0] < max_n:
            pad = torch.full((max_n - mask_r.shape[0], kv_len),
                             float("-1e9"), dtype=torch.float32,
                             device=mask_r.device)
            mask_r = torch.cat([mask_r, pad], dim=0)
        q_c = query
        k_c = key_blocks
        v_c = value_blocks
        if _DTYPE != "fp32":
            q_c = q_c.to(torch_dt)
            k_c = k_c.to(torch_dt)
            v_c = v_c.to(torch_dt)
        kernel(q_c, k_c, v_c, mask_r, output)

    return wrapper
