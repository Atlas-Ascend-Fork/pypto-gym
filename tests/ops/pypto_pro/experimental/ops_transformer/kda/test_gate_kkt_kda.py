# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""gate_kkt_kda — fused gate_cumsum + kkt_kda test harness.

Also validates the newly-fused aqk output (pivot decomposition, fp16 ws_aqk)
against the RefKDA golden implementation.
"""

import logging
import os
import sys

import torch
import torch_npu
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_REPO_ROOT = _THIS_DIR
while not os.path.isdir(os.path.join(_REPO_ROOT, "src")):
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
    if _REPO_ROOT == os.path.dirname(_REPO_ROOT):
        break
_SRC_DIR = _REPO_ROOT
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ref_kda import RefKDA
from kda_test_config import (TEST_SHAPES, CHUNK_SIZE, K_DIM, HV, C, K, HC, DEVICE,
                              require_a5, make_tril_mask, make_kda_base_inputs, run_main)

from ops.pypto_pro.experimental.ops_transformer.kda.gate_kkt_kda_impl import (
    run_gate_kkt_kda,
)


def _seq_ranges(T, cu_seqlens):
    if cu_seqlens is None:
        return [(0, T)]
    return [(cu_seqlens[i], cu_seqlens[i + 1]) for i in range(len(cu_seqlens) - 1)]


def golden_aqk_workspace(qf, kf, g_cs, chunk_size, num_chunks, cu_seqlens):
    """Golden masked aqk in ws_aqk layout [total_work*C, C] fp32.

    work_id = chunk_id * HV + head_id, chunks enumerated in t-base order.
    """
    B, T, HVd, Kd = qf.shape
    ws = torch.zeros(num_chunks * HVd * chunk_size, chunk_size, dtype=torch.float32)
    wi = 0
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for tb in range(bos, eos, chunk_size):
            s, e = tb, min(tb + chunk_size, eos)
            c_len = e - s
            for h in range(HVd):
                qc = qf[0, s:e, h].float()          # [c_len, K]
                kc = kf[0, s:e, h].float()
                gc = g_cs[0, s:e, h].float()
                q_eff = qc * torch.exp(gc)
                k_eff = kc * torch.exp(-gc)
                aqk = torch.tril(q_eff @ k_eff.transpose(0, 1), diagonal=0)
                ws[wi * chunk_size: wi * chunk_size + c_len,
                   :c_len] = aqk
                wi += 1
    return ws


def _run_case(T, cu_seqlens, label, device):
    require_a5(DEVICE)

    q, k, v, g_log, beta_sig, scale = make_kda_base_inputs(T, device)
    ref = RefKDA(torch.float32 if device != "cpu" else torch.float64)
    st = ref.full_pipeline(q.cpu(), k.cpu(), v.cpu(), g_log.cpu(),
                           beta_sig.cpu(), cu_seqlens, scale, CHUNK_SIZE)

    L_tril = make_tril_mask(CHUNK_SIZE, device)
    mask_strict = make_tril_mask(CHUNK_SIZE, device, diagonal=-1)
    mask_incl = make_tril_mask(CHUNK_SIZE, device, diagonal=0)

    if cu_seqlens is None:
        num_chunks = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    else:
        num_chunks = sum((cu_seqlens[i + 1] - cu_seqlens[i] + CHUNK_SIZE - 1) // CHUNK_SIZE
                         for i in range(len(cu_seqlens) - 1))

    g_cs = torch.empty(1, HV, T, K_DIM, device=DEVICE, dtype=torch.float32)
    L_out = torch.empty(1, HV, T, CHUNK_SIZE,
                        device=DEVICE, dtype=torch.float16)

    beta_bntd = beta_sig.half().permute(0, 2, 1).contiguous()
    qf_h = st.qf.half()
    tw = num_chunks * HV
    nc = min(torch.npu.get_device_properties(0).cube_core_num, tw)
    ws_aqk = run_gate_kkt_kda(g_log.half(), qf_h, k.half(), beta_bntd,
                              L_tril, mask_strict, mask_incl,
                              g_cs, L_out, nc, cu_seqlens)

    g_cs_npu = g_cs.cpu().float().permute(0, 2, 1, 3)
    npu = L_out.cpu().float().permute(0, 2, 1, 3)

    g_cs_ref = st.g_cs.float().cpu()
    L_ref = st.L.float().cpu()

    aqk_npu = ws_aqk.cpu().float()
    aqk_ref = golden_aqk_workspace(st.qf, st.kf, st.g_cs, CHUNK_SIZE, num_chunks, cu_seqlens)

    g_cs_diff = (g_cs_npu - g_cs_ref).abs().max().item()
    diff = (npu - L_ref).abs().max().item()
    aqk_diff = (aqk_npu - aqk_ref).abs().max().item()
    logging.info("  [%s] cores=%d  g_cs diff: %.3e  L diff: %.3e  aqk diff: %.3e",
                 label, nc, g_cs_diff, diff, aqk_diff)
    torch.testing.assert_close(g_cs_npu, g_cs_ref, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(npu, L_ref, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(aqk_npu, aqk_ref, rtol=5e-3, atol=5e-3)
    logging.info("  [%s] PASS", label)


@pytest.mark.soc("950")
@pytest.mark.parametrize("T,cu", TEST_SHAPES)
def test_gate_kkt_kda(T, cu):
    cu_str = f"cu={cu}" if cu else "cu=None"
    _run_case(T, cu, f"T={T}, {cu_str}", DEVICE)


if __name__ == "__main__":
    run_main("gate_kkt_kda fused block programming test (A5)", TEST_SHAPES, _run_case)