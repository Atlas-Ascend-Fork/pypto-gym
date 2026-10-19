# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""inversion_kda — cube-based Neumann series inversion test harness."""

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
from kda_test_config import (TEST_SHAPES as _BASE_SHAPES, C, DEVICE,
                              require_a5, make_kda_base_inputs, run_main)

from ops.pypto_pro.experimental.ops_transformer.kda.inversion_kda_impl import (
    inversion_kda_cube,
)

TEST_SHAPES = _BASE_SHAPES + [
    (385, None),
    (258, [0, 129, 258]),
    (520, [0, 130, 260, 520]),
    (770, [0, 1, 129, 257, 770]),
]

CS = C


def _run_case(T, cu_seqlens, label, device):
    require_a5(DEVICE)

    q, k, v, g_log, beta_sig, scale = make_kda_base_inputs(T, device)
    ref = RefKDA(torch.float32 if device != "cpu" else torch.float64)
    st = ref.full_pipeline(q.cpu(), k.cpu(), v.cpu(), g_log.cpu(),
                           beta_sig.cpu(), cu_seqlens, scale, CS)

    A_fp32_hm = st.L.float().permute(0, 2, 1, 3).contiguous().to(device)
    A_inv = inversion_kda_cube(A_fp32_hm, cu_seqlens)

    golden = ref.inversion_kda(st.L.float(), CS, cu_seqlens).float().cpu()
    npu = A_inv.cpu().float().permute(0, 2, 1, 3)

    diff = (npu - golden).abs().max().item()
    logging.info("  [%s] max abs diff: %.3e", label, diff)
    torch.testing.assert_close(npu, golden, rtol=1e-3, atol=1e-3)
    logging.info("  [%s] PASS", label)


@pytest.mark.soc("950")
@pytest.mark.parametrize("T,cu", TEST_SHAPES)
def test_inversion_kda_cube(T, cu):
    cu_str = f"cu={cu}" if cu else "cu=None"
    _run_case(T, cu, f"T={T}, {cu_str}", DEVICE)


if __name__ == "__main__":
    run_main("inversion_kda_cube block programming test (A5) — fused kernel", TEST_SHAPES, _run_case)
