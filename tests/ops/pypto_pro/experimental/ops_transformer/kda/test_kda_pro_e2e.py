# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end KDA block-programming pipeline test.

Links all 5 NPU block kernels and compares against ``RefKDA.full_pipeline``
(double-precision CPU reference).

Stages (all on NPU, all head-major):
  [1] gate_kkt_kda   — fused gate_cumsum + kkt_kda (g→g_cs→L in one kernel)
  [2] inversion_kda  — (I + L)^{-1}  (cube Neumann series, fp16 input)
  [3] wy_kda         — u and w transforms (WY rep, A2 approach)
  [4] chunk_h_kda    — sequential state pass
  [5] chunk_o_kda    — output pass
"""

import logging
import os
import sys
import time

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

from kda_test_config import (TEST_SHAPES, CHUNK_SIZE, K_DIM, V_DIM, HV, DEVICE,
                              require_a5, make_kda_base_inputs)
from ref_kda import RefKDA

from ops.pypto_pro.experimental.ops_transformer.kda.gate_kkt_kda_impl import (
    run_gate_kkt_kda,
)
from ops.pypto_pro.experimental.ops_transformer.kda.inversion_kda_impl import (
    inversion_kda_cube,
)
from ops.pypto_pro.experimental.ops_transformer.kda.wy_kda_impl import (
    wy_kda_block,
)
from ops.pypto_pro.experimental.ops_transformer.kda.chunk_h_kda_impl import (
    run_chunk_h_kda,
)
from ops.pypto_pro.experimental.ops_transformer.kda.chunk_o_kda_impl import (
    run_chunk_o_kda,
)


H = HV
SCALE = K_DIM ** -0.5

RTOL = 5e-3
ATOL = 5e-3


def block_kda_pipeline(
    q, k, v, g_log, beta_sig, cu_seqlens_list, device, num_cores=8,
):
    T = q.shape[1]
    HVd = v.shape[2]
    G = HVd // H
    max_cores = torch.npu.get_device_properties(0).cube_core_num

    if cu_seqlens_list is None:
        num_chunks = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    else:
        num_chunks = sum((cu_seqlens_list[i + 1] - cu_seqlens_list[i] + CHUNK_SIZE - 1) // CHUNK_SIZE
                         for i in range(len(cu_seqlens_list) - 1))

    qf = (q.half().repeat_interleave(G, dim=2) * SCALE)
    kf = k.half().repeat_interleave(G, dim=2)
    vf = v.half()
    bf = beta_sig.half()
    g_log_h = g_log.half()

    q_d = qf.to(device)
    k_d = kf.to(device)
    v_d = vf.to(device)
    b_d = bf.to(device)
    g_d = g_log_h.to(device)

    L_tril = torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE,
                                    dtype=torch.float32)).to(device)
    mask_strict = torch.tril(
        torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32),
        diagonal=-1).to(device)
    mask_incl = torch.tril(
        torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32),
        diagonal=0).to(device)
    g_cs = torch.empty(1, HVd, T, K_DIM, device=DEVICE, dtype=torch.float32)
    L_out = torch.empty(1, HVd, T, CHUNK_SIZE,
                        device=DEVICE, dtype=torch.float16)
    b_bntd = b_d.permute(0, 2, 1).contiguous()
    tw = num_chunks * HVd
    ws_aqk = run_gate_kkt_kda(g_d, q_d, k_d, b_bntd, L_tril, mask_strict,
                              mask_incl, g_cs, L_out,
                              min(max_cores, tw), cu_seqlens_list)

    A_inv = inversion_kda_cube(L_out, cu_seqlens_list)

    u_out, w_out = wy_kda_block(k_d, v_d, g_cs, b_d, A_inv,
                                 CHUNK_SIZE, cu_seqlens_list, num_cores=max_cores)

    s_snap = torch.empty(HVd, num_chunks, K_DIM, V_DIM,
                         device=DEVICE, dtype=torch.float16)
    vcorr = torch.empty(1, HVd, T, V_DIM,
                        device=DEVICE, dtype=torch.float16)

    run_chunk_h_kda(k_d, w_out, u_out, g_cs, s_snap, vcorr,
                    cu_seqlens_list, num_cores=max_cores)

    o_npu = torch.empty(1, T, HVd, V_DIM,
                        device=DEVICE, dtype=torch.float16)
    tw6 = num_chunks * HVd
    run_chunk_o_kda(q_d, vcorr, s_snap, g_cs, o_npu, ws_aqk,
                     num_chunks, min(max_cores, tw6),
                     cu_seqlens_list)

    return o_npu


def _run_one(T, cu_seqlens_list, device):
    require_a5(DEVICE)
    torch.npu.manual_seed(0)

    q, k, v, g_log, beta_sig, scale = make_kda_base_inputs(T, device)

    t0 = time.time()
    o_npu = block_kda_pipeline(
        q, k, v, g_log, beta_sig, cu_seqlens_list, device, num_cores=8,
    )
    dt = time.time() - t0

    ref = RefKDA(torch.float32 if device != "cpu" else torch.double)
    st = ref.full_pipeline(
        q.cpu(), k.cpu(), v.cpu(), g_log.cpu(), beta_sig.cpu(),
        cu_seqlens_list, scale, CHUNK_SIZE
    )

    npu = o_npu.cpu().float()
    golden = st.o.cpu().float()
    diff = (npu - golden).abs().max().item()
    cu_str = f"cu={cu_seqlens_list}" if cu_seqlens_list else "cu=None"
    label = f"T={T}, {cu_str}"
    logging.info("  [%s] diff=%.3e  %.2fs", label, diff, dt)
    torch.testing.assert_close(npu, golden, rtol=RTOL, atol=ATOL)
    logging.info("  [%s] PASS", label)


@pytest.mark.soc("950")
@pytest.mark.parametrize("T,cu", TEST_SHAPES)
def test_kda_e2e_cube(T, cu):
    _run_one(T, cu, DEVICE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("KDA e2e block-programming test (A5)")
    logging.info("=" * 60)
    idx = None
    device = DEVICE
    for arg in sys.argv[1:]:
        if arg.isdigit():
            idx = int(arg) - 1
        elif arg == "cpu":
            device = "cpu"
    shapes = [TEST_SHAPES[idx]] if idx is not None else TEST_SHAPES
    for i, (T, cu) in enumerate(shapes, 1):
        cu_str = f"cu={cu}" if cu else "cu=None"
        logging.info("--- Case %d: T=%d, %s ---", i, T, cu_str)
        _run_one(T, cu, device)
    logging.info("\nAll KDA e2e tests passed!")
