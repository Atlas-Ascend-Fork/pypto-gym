# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""chunk_o_kda — fused output pass test harness."""

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
from kda_test_config import (TEST_SHAPES, CHUNK_SIZE, K_DIM, V_DIM, HV, C, K, V, HC, DEVICE,
                              require_a5, make_kda_base_inputs, make_tril_mask, run_main)

from ops.pypto_pro.experimental.ops_transformer.kda.chunk_o_kda_impl import (
    run_chunk_o_kda,
)
from ops.pypto_pro.experimental.ops_transformer.kda.gate_kkt_kda_impl import (
    run_gate_kkt_kda,
)


def _run_case(T, cu_seqlens, label, device, rtol=5e-3, atol=5e-3):
    """Run chunk_o_kda on NPU and compare with CPU golden."""
    require_a5(DEVICE)
    q, k, v, g_log, beta_sig, scale = make_kda_base_inputs(T, device)
    ref = RefKDA(torch.float32 if device != "cpu" else torch.float64)
    st = ref.full_pipeline(q.cpu(), k.cpu(), v.cpu(), g_log.cpu(),
                           beta_sig.cpu(), cu_seqlens, scale, CHUNK_SIZE)

    if cu_seqlens is None:
        num_chunks = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    else:
        num_chunks = sum((cu_seqlens[i + 1] - cu_seqlens[i] + CHUNK_SIZE - 1) // CHUNK_SIZE
                         for i in range(len(cu_seqlens) - 1))
    total_work = num_chunks * HV
    num_cores = min(torch.npu.get_device_properties(0).cube_core_num, total_work)

    torch.manual_seed(42)
    v_corr = torch.randn(1, T, HV, V_DIM, dtype=torch.float32) * 0.1
    s_snapshots = torch.randn(HV, num_chunks, K_DIM, V_DIM, dtype=torch.float32) * 0.1

    q_d = st.qf.half().to(device)
    k_d = st.kf.half().to(device)
    vcorr_bntd = v_corr.half().permute(0, 2, 1, 3).contiguous().to(device)
    s_d = s_snapshots.half().to(device)
    g_bntd = st.g_cs.permute(0, 2, 1, 3).contiguous().to(device)
    o_npu = torch.empty(1, T, HV, V_DIM, device=DEVICE, dtype=torch.float16)

    # gate_kkt 先算 aqk（共享 ki_a），chunk_o 只读 ws_aqk
    L_tril = make_tril_mask(C, device)
    mask_strict = make_tril_mask(C, device, diagonal=-1)
    mask_incl = make_tril_mask(C, device, diagonal=0)
    g_cs_buf = torch.empty(1, HV, T, K_DIM, device=DEVICE, dtype=torch.float32)
    L_out_buf = torch.empty(1, HV, T, C, device=DEVICE, dtype=torch.float16)
    beta_bntd = beta_sig.half().permute(0, 2, 1).contiguous()
    ws_aqk = run_gate_kkt_kda(g_log.half(), q_d, k_d, beta_bntd,
                              L_tril, mask_strict, mask_incl,
                              g_cs_buf, L_out_buf, num_cores, cu_seqlens)

    run_chunk_o_kda(q_d, vcorr_bntd, s_d, g_bntd, o_npu, ws_aqk,
                    num_chunks, num_cores, cu_seqlens)

    o_ref = ref.chunk_o_kda(
        st.qf.half(), st.kf.half(), v_corr.half(),
        s_snapshots.permute(1, 0, 2, 3), st.g_cs, CHUNK_SIZE, cu_seqlens
    ).float().cpu()

    npu_out = o_npu.cpu().float()
    diff = (npu_out - o_ref).abs().max().item()
    logging.info("  [%s] cores=%d  max abs diff: %.3e", label, num_cores, diff)
    torch.testing.assert_close(npu_out, o_ref, rtol=rtol, atol=atol)
    logging.info("  [%s] PASS", label)


@pytest.mark.soc("950")
@pytest.mark.parametrize("T,cu", TEST_SHAPES)
def test_chunk_o_kda(T, cu):
    cu_str = f"cu={cu}" if cu else "cu=None"
    _run_case(T, cu, f"T={T}, {cu_str}", DEVICE)


if __name__ == "__main__":
    run_main("chunk_o_kda fused head-major block programming test (A5)", TEST_SHAPES, _run_case)
