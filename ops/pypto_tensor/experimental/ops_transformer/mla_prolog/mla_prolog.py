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
MLA Prolog PyPTO Kernel Implementation
Multi-Head Latent Attention 前处理算子 (支持动态序列长度)
"""

import os
import sys
import argparse
import logging
from typing import Tuple
from dataclasses import dataclass
import pypto
import torch
import torch_npu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'deepseek_v32_exp'))
from utils.compare import compare

logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_device_id():
    return int(os.environ.get('TILE_FWK_DEVICE_ID', 0))


T = pypto.DYNAMIC
TILE_T = 8
HE = 256
HCQ = 64
HCKV = 32
N = 4
D = 16
DR = 8
HALF_DR = DR // 2


@dataclass
class MLAPrologParams:
    weight_dq: torch.Tensor
    weight_uq_qr: torch.Tensor
    weight_uk: torch.Tensor
    weight_dkv_kr: torch.Tensor
    rmsnorm_gamma_cq: torch.Tensor
    rmsnorm_gamma_ckv: torch.Tensor
    rope_sin: torch.Tensor
    rope_cos: torch.Tensor
    epsilon_cq: float = 1e-5
    epsilon_ckv: float = 1e-5


@pypto.frontend.jit()
def mla_prolog_kernel(
    token_x: pypto.Tensor([T, HE], pypto.DT_BF16),
    weight_dq: pypto.Tensor([HE, HCQ], pypto.DT_BF16),
    weight_uq_qr: pypto.Tensor([HCQ, N * (D + DR)], pypto.DT_BF16),
    weight_uk: pypto.Tensor([N, D, HCKV], pypto.DT_BF16),
    weight_dkv_kr: pypto.Tensor([HE, HCKV + DR], pypto.DT_BF16),
    rmsnorm_gamma_cq: pypto.Tensor([HCQ], pypto.DT_BF16),
    rmsnorm_gamma_ckv: pypto.Tensor([HCKV], pypto.DT_BF16),
    rope_sin: pypto.Tensor([T, DR], pypto.DT_BF16),
    rope_cos: pypto.Tensor([T, DR], pypto.DT_BF16),
    query: pypto.Tensor([T, N, HCKV], pypto.DT_BF16),
    query_rope: pypto.Tensor([T, N, DR], pypto.DT_BF16),
    c_kv_out: pypto.Tensor([T, HCKV], pypto.DT_BF16),
    k_r_out: pypto.Tensor([T, DR], pypto.DT_BF16),
):
    pypto.set_codegen_options(support_dynamic_aligned=True)
    pypto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])
    pypto.set_vec_tile_shapes(32, 64)

    seq_len = token_x.shape[0]
    t_loop = (seq_len + TILE_T - 1) // TILE_T

    for t_idx in pypto.loop(t_loop, name="LOOP_T", idx_name="t_idx"):
        t_offset = t_idx * TILE_T
        t_offset_end = min(t_offset + TILE_T, seq_len)
        valid_t = t_offset_end - t_offset

        x_tile = pypto.view(token_x, [TILE_T, HE], [t_offset, 0], valid_shape=[valid_t, HE])
        sin_tile = pypto.view(rope_sin, [TILE_T, DR], [t_offset, 0], valid_shape=[valid_t, DR])
        cos_tile = pypto.view(rope_cos, [TILE_T, DR], [t_offset, 0], valid_shape=[valid_t, DR])

        mm_cq = pypto.matmul(x_tile, weight_dq, pypto.DT_BF16)
        squared = mm_cq * mm_cq
        mean_sq = pypto.sum(squared, dim=-1, keepdim=True)
        mean_sq = mean_sq / HCQ
        rms = pypto.sqrt(mean_sq + 1e-5)
        c_q = mm_cq / rms
        c_q = c_q * rmsnorm_gamma_cq

        mm_qc_qr = pypto.matmul(c_q, weight_uq_qr, pypto.DT_BF16)
        qc_qr_split = N * D
        mm_qc = mm_qc_qr[:, :qc_qr_split]
        mm_qr = mm_qc_qr[:, qc_qr_split:]

        q0 = pypto.matmul(mm_qc[:, 0 * D:1 * D], weight_uk[0, :, :], pypto.DT_BF16)
        q1 = pypto.matmul(mm_qc[:, 1 * D:2 * D], weight_uk[1, :, :], pypto.DT_BF16)
        q2 = pypto.matmul(mm_qc[:, 2 * D:3 * D], weight_uk[2, :, :], pypto.DT_BF16)
        q3 = pypto.matmul(mm_qc[:, 3 * D:4 * D], weight_uk[3, :, :], pypto.DT_BF16)

        q01 = pypto.concat([q0, q1], dim=-1)
        q23 = pypto.concat([q2, q3], dim=-1)
        q_all = pypto.concat([q01, q23], dim=-1)
        query_tile = pypto.reshape(q_all, [TILE_T, N, HCKV], valid_shape=[valid_t, N, HCKV])
        pypto.assemble(query_tile, [t_offset, 0, 0], query)

        sin_h = sin_tile[:, :HALF_DR]
        cos_h = cos_tile[:, :HALF_DR]

        qr0 = mm_qr[:, 0 * DR:1 * DR]
        qr0_even = qr0[:, :HALF_DR]
        qr0_odd = qr0[:, HALF_DR:]
        qr0_out_even = qr0_even * cos_h - qr0_odd * sin_h
        qr0_out_odd = qr0_odd * cos_h + qr0_even * sin_h
        qr0_rope = pypto.concat([qr0_out_even, qr0_out_odd], dim=-1)

        qr1 = mm_qr[:, 1 * DR:2 * DR]
        qr1_even = qr1[:, :HALF_DR]
        qr1_odd = qr1[:, HALF_DR:]
        qr1_out_even = qr1_even * cos_h - qr1_odd * sin_h
        qr1_out_odd = qr1_odd * cos_h + qr1_even * sin_h
        qr1_rope = pypto.concat([qr1_out_even, qr1_out_odd], dim=-1)

        qr2 = mm_qr[:, 2 * DR:3 * DR]
        qr2_even = qr2[:, :HALF_DR]
        qr2_odd = qr2[:, HALF_DR:]
        qr2_out_even = qr2_even * cos_h - qr2_odd * sin_h
        qr2_out_odd = qr2_odd * cos_h + qr2_even * sin_h
        qr2_rope = pypto.concat([qr2_out_even, qr2_out_odd], dim=-1)

        qr3 = mm_qr[:, 3 * DR:4 * DR]
        qr3_even = qr3[:, :HALF_DR]
        qr3_odd = qr3[:, HALF_DR:]
        qr3_out_even = qr3_even * cos_h - qr3_odd * sin_h
        qr3_out_odd = qr3_odd * cos_h + qr3_even * sin_h
        qr3_rope = pypto.concat([qr3_out_even, qr3_out_odd], dim=-1)

        qr01 = pypto.concat([qr0_rope, qr1_rope], dim=-1)
        qr23 = pypto.concat([qr2_rope, qr3_rope], dim=-1)
        qr_all = pypto.concat([qr01, qr23], dim=-1)
        query_rope_tile = pypto.reshape(qr_all, [TILE_T, N, DR], valid_shape=[valid_t, N, DR])
        pypto.assemble(query_rope_tile, [t_offset, 0, 0], query_rope)

        mm_ckv_kr = pypto.matmul(x_tile, weight_dkv_kr, pypto.DT_BF16)
        mm_ckv = mm_ckv_kr[:, :HCKV]
        mm_kr = mm_ckv_kr[:, HCKV:]

        sq_ckv = mm_ckv * mm_ckv
        mean_sq_ckv = pypto.sum(sq_ckv, dim=-1, keepdim=True)
        mean_sq_ckv = mean_sq_ckv / HCKV
        rms_ckv = pypto.sqrt(mean_sq_ckv + 1e-5)
        c_kv_normed = mm_ckv / rms_ckv
        c_kv_normed = c_kv_normed * rmsnorm_gamma_ckv
        pypto.assemble(c_kv_normed, [t_offset, 0], c_kv_out)

        k_even = mm_kr[:, :HALF_DR]
        k_odd = mm_kr[:, HALF_DR:]
        k_out_even = k_even * cos_h - k_odd * sin_h
        k_out_odd = k_odd * cos_h + k_even * sin_h
        k_r_result = pypto.concat([k_out_even, k_out_odd], dim=-1)
        pypto.assemble(k_r_result, [t_offset, 0], k_r_out)


def mla_prolog_golden(
    token_x: torch.Tensor,
    params: MLAPrologParams,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    seq_len = token_x.shape[0]

    def rms_norm(x, gamma, epsilon):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + epsilon)
        return gamma * (x / rms)

    def apply_rope(x, sin, cos):
        shape = x.shape
        last_dim = shape[-1]
        half_dim = last_dim // 2
        x_even = x[..., :half_dim]
        x_odd = x[..., half_dim:]
        if sin.shape[-1] == last_dim:
            sin_half = sin[..., :half_dim]
            cos_half = cos[..., :half_dim]
        else:
            sin_half = sin
            cos_half = cos
        out_even = x_even * cos_half - x_odd * sin_half
        out_odd = x_odd * cos_half + x_even * sin_half
        return torch.cat([out_even, out_odd], dim=-1)

    mm_cq = torch.matmul(token_x, params.weight_dq)
    mm_cq_bf16 = mm_cq.bfloat16()
    c_q = rms_norm(mm_cq_bf16.float(), params.rmsnorm_gamma_cq.float(), params.epsilon_cq)
    c_q_bf16 = c_q.bfloat16()

    mm_qc_qr = torch.matmul(c_q_bf16, params.weight_uq_qr)
    mm_qc_qr_bf16 = mm_qc_qr.bfloat16()
    qc_qr_split = N * D
    mm_qc = mm_qc_qr_bf16[:, : qc_qr_split]
    mm_qr = mm_qc_qr_bf16[:, qc_qr_split:]

    q0 = torch.matmul(mm_qc[:, 0 * D: 1 * D], params.weight_uk[0, :, :]).bfloat16()
    q1 = torch.matmul(mm_qc[:, 1 * D: 2 * D], params.weight_uk[1, :, :]).bfloat16()
    q2 = torch.matmul(mm_qc[:, 2 * D: 3 * D], params.weight_uk[2, :, :]).bfloat16()
    q3 = torch.matmul(mm_qc[:, 3 * D: 4 * D], params.weight_uk[3, :, :]).bfloat16()

    q01 = torch.cat([q0, q1], dim=-1)
    q23 = torch.cat([q2, q3], dim=-1)
    q_all = torch.cat([q01, q23], dim=-1)
    query = q_all.reshape(seq_len, N, HCKV)

    sin_h = params.rope_sin[:, : HALF_DR].bfloat16()
    cos_h = params.rope_cos[:, : HALF_DR].bfloat16()

    def apply_rope_1d(x_1d):
        x_even = x_1d[:, : HALF_DR]
        x_odd = x_1d[:, HALF_DR:]
        out_even = x_even * cos_h - x_odd * sin_h
        out_odd = x_odd * cos_h + x_even * sin_h
        return torch.cat([out_even, out_odd], dim=-1)

    qr0 = apply_rope_1d(mm_qr[:, 0 * DR: 1 * DR])
    qr1 = apply_rope_1d(mm_qr[:, 1 * DR: 2 * DR])
    qr2 = apply_rope_1d(mm_qr[:, 2 * DR: 3 * DR])
    qr3 = apply_rope_1d(mm_qr[:, 3 * DR: 4 * DR])

    qr01 = torch.cat([qr0, qr1], dim=-1)
    qr23 = torch.cat([qr2, qr3], dim=-1)
    qr_all = torch.cat([qr01, qr23], dim=-1)
    query_rope = qr_all.reshape(seq_len, N, DR).bfloat16()

    mm_ckv_kr = torch.matmul(token_x, params.weight_dkv_kr)
    mm_ckv_kr_bf16 = mm_ckv_kr.bfloat16()
    hckv_actual = params.weight_dkv_kr.shape[1] - DR
    mm_ckv = mm_ckv_kr_bf16[:, : hckv_actual]
    mm_kr = mm_ckv_kr_bf16[:, hckv_actual:]

    c_kv_normed = rms_norm(mm_ckv.float(), params.rmsnorm_gamma_ckv.float(), params.epsilon_ckv)
    k_r_rope = apply_rope(mm_kr, params.rope_sin, params.rope_cos)

    return query.bfloat16(), query_rope.bfloat16(), c_kv_normed.bfloat16(), k_r_rope.bfloat16()


def test_mla_prolog(device_id=None, run_mode: str = "npu"):
    logging.info("=" * 60)
    logging.info("Test: MLA Prolog (Dynamic Sequence Length)")
    logging.info("=" * 60)

    torch.manual_seed(42)

    device = f'npu:{device_id}' if device_id is not None else 'cpu'

    test_seq_lens = [8, 16, 32]

    for seq_len in test_seq_lens:
        logging.info(f"\nTesting seq_len={seq_len} (dynamic)")

        token_x = torch.randn(seq_len, HE, dtype=torch.bfloat16, device=device)
        weight_dq = torch.randn(HE, HCQ, dtype=torch.bfloat16, device=device)
        weight_uq_qr = torch.randn(HCQ, N * (D + DR), dtype=torch.bfloat16, device=device)
        weight_uk = torch.randn(N, D, HCKV, dtype=torch.bfloat16, device=device)
        weight_dkv_kr = torch.randn(HE, HCKV + DR, dtype=torch.bfloat16, device=device)
        rmsnorm_gamma_cq = torch.randn(HCQ, dtype=torch.bfloat16, device=device)
        rmsnorm_gamma_ckv = torch.randn(HCKV, dtype=torch.bfloat16, device=device)
        rope_sin = torch.randn(seq_len, DR, dtype=torch.bfloat16, device=device)
        rope_cos = torch.randn(seq_len, DR, dtype=torch.bfloat16, device=device)

        query = torch.empty(seq_len, N, HCKV, dtype=torch.bfloat16, device=device)
        query_rope = torch.empty(seq_len, N, DR, dtype=torch.bfloat16, device=device)
        c_kv_out = torch.empty(seq_len, HCKV, dtype=torch.bfloat16, device=device)
        k_r_out = torch.empty(seq_len, DR, dtype=torch.bfloat16, device=device)

        mla_prolog_kernel(
            token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr,
            rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos,
            query, query_rope, c_kv_out, k_r_out
        )

        params = MLAPrologParams(
            weight_dq=weight_dq,
            weight_uq_qr=weight_uq_qr,
            weight_uk=weight_uk,
            weight_dkv_kr=weight_dkv_kr,
            rmsnorm_gamma_cq=rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv=rmsnorm_gamma_ckv,
            rope_sin=rope_sin,
            rope_cos=rope_cos,
        )

        golden_query, golden_query_rope, golden_c_kv, golden_k_r = mla_prolog_golden(token_x, params)

        if run_mode == "npu":
            compare(query.cpu(), golden_query.cpu(), "query", 0.005, 0.0078125, 0.005)
            compare(query_rope.cpu(), golden_query_rope.cpu(), "query_rope", 0.005, 0.0078125, 0.05)
            compare(c_kv_out.cpu(), golden_c_kv.cpu(), "c_kv_out", 0.005, 0.0078125, 0.005)
            compare(k_r_out.cpu(), golden_k_r.cpu(), "k_r_out", 0.005, 0.0078125, 0.05)
            logging.info(f"✓ seq_len={seq_len} all precision comparisons passed")


def main():
    parser = argparse.ArgumentParser(description="PyPTO MLA Prolog Kernel")
    parser.add_argument('--run_mode', type=str, default='npu', choices=["npu"])
    args = parser.parse_args()

    logging.info("\n" + "=" * 60)
    logging.info("PyPTO MLA Prolog Kernel (Dynamic Sequence Length)")
    logging.info("=" * 60 + "\n")

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        torch.npu.set_device(device_id)
        logging.info("Running on NPU...")

    test_mla_prolog(device_id, args.run_mode)


if __name__ == "__main__":
    main()
