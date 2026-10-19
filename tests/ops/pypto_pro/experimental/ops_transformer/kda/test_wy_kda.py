# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""wy_kda — WY representation test harness."""

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
from kda_test_config import (TEST_SHAPES, CHUNK_SIZE, C, K, V, DEVICE,
                              require_a5, make_kda_base_inputs, run_main)

from ops.pypto_pro.experimental.ops_transformer.kda.wy_kda_impl import (
    wy_kda_block,
)


def _run_case(T, cu_seqlens, label, device):
    require_a5(DEVICE)

    q, k, v, g_log, beta_sig, scale = make_kda_base_inputs(T, device)
    ref = RefKDA(torch.float32 if device != "cpu" else torch.float64)
    st = ref.full_pipeline(q.cpu(), k.cpu(), v.cpu(), g_log.cpu(),
                           beta_sig.cpu(), cu_seqlens, scale, CHUNK_SIZE)

    k_d = k.half()
    v_d = v.half()
    g_cs_bntd = st.g_cs.permute(0, 2, 1, 3).contiguous().to(device)
    beta_d = beta_sig.half()
    A_inv_bntd = st.A_inv.permute(0, 2, 1, 3).contiguous().to(device)

    u_npu, w_npu = wy_kda_block(k_d, v_d, g_cs_bntd, beta_d, A_inv_bntd,
                                 CHUNK_SIZE, cu_seqlens)

    u_ref = st.u.float().cpu()
    w_ref = st.w.float().cpu()

    u_npu_bsnd = u_npu.cpu().float().permute(0, 2, 1, 3)
    w_npu_bsnd = w_npu.cpu().float().permute(0, 2, 1, 3)
    u_diff = (u_npu_bsnd - u_ref).abs().max().item()
    w_diff = (w_npu_bsnd - w_ref).abs().max().item()
    logging.info("  [%s] u_diff: %.3e  w_diff: %.3e", label, u_diff, w_diff)

    torch.testing.assert_close(u_npu_bsnd, u_ref, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(w_npu_bsnd, w_ref, rtol=1e-2, atol=0.5)
    logging.info("  [%s] PASS", label)


@pytest.mark.soc("950")
@pytest.mark.parametrize("T,cu", TEST_SHAPES)
def test_wy_kda(T, cu):
    cu_str = f"cu={cu}" if cu else "cu=None"
    _run_case(T, cu, f"T={T}, {cu_str}", DEVICE)


if __name__ == "__main__":
    run_main("wy_kda block programming test (A5) — fused kernel", TEST_SHAPES, _run_case)
