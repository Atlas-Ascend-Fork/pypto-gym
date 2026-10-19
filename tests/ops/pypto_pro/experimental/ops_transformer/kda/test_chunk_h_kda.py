# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""chunk_h_kda — sequential recurrent state pass test harness."""

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
                              require_a5, make_cu_seqlens_tensor, make_kda_base_inputs,
                              alloc_chunk_h_workspaces, run_main)

from ops.pypto_pro.experimental.ops_transformer.kda.chunk_h_kda_impl import (
    run_chunk_h_kda,
)


def _run_case(T, cu_seqlens, label, device):
    require_a5(DEVICE)

    q, k, v, g_log, beta_sig, scale = make_kda_base_inputs(T, device)
    ref = RefKDA(torch.float32 if device != "cpu" else torch.float64)
    st = ref.full_pipeline(q.cpu(), k.cpu(), v.cpu(), g_log.cpu(),
                           beta_sig.cpu(), cu_seqlens, scale, CHUNK_SIZE)

    num_chunks = st.s_snapshots.shape[0]

    k_bs = k.cpu().half().to(device)
    w_hm = st.w.permute(0, 2, 1, 3).contiguous().to(device)
    u_hm = st.u.permute(0, 2, 1, 3).contiguous().to(device)
    g_hm = st.g_cs.permute(0, 2, 1, 3).contiguous().to(device)

    s_npu = torch.empty(HV, num_chunks, K_DIM, V_DIM, device=DEVICE, dtype=torch.float16)
    vcorr_npu = torch.empty(1, HV, T, V_DIM, device=DEVICE, dtype=torch.float16)

    run_chunk_h_kda(k_bs, w_hm, u_hm, g_hm, s_npu, vcorr_npu,
                    cu_seqlens)

    s_ref = st.s_snapshots.float().cpu()
    vcorr_ref = st.v_corr.float().cpu()

    s_diff = (s_npu.cpu().float() - s_ref.permute(1, 0, 2, 3)).abs().max().item()
    vcorr_diff = (vcorr_npu.cpu().float() - vcorr_ref.permute(0, 2, 1, 3)).abs().max().item()
    logging.info("  [%s] s_diff: %.3e  vcorr_diff: %.3e",
                 label, s_diff, vcorr_diff)

    torch.testing.assert_close(s_npu.cpu().float(), s_ref.permute(1, 0, 2, 3), rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(vcorr_npu.cpu().float(), vcorr_ref.permute(0, 2, 1, 3),
                                rtol=5e-3, atol=5e-3)
    logging.info("  [%s] PASS", label)


@pytest.mark.soc("950")
@pytest.mark.parametrize("T,cu", TEST_SHAPES)
def test_chunk_h_kda(T, cu):
    cu_str = f"cu={cu}" if cu else "cu=None"
    _run_case(T, cu, f"T={T}, {cu_str}", DEVICE)


if __name__ == "__main__":
    run_main("chunk_h_kda block programming test (A5)", TEST_SHAPES, _run_case)
