#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""kda_cp_main PyPTO-Pro test: one test function per case.

The kernel and the host wrapper live under ``op_kernel/kda_cp_main.py``; the CPU
golden / input factory and the numeric comparator live under ``tests/assets/``.
Golden semantics: the state is bf16-rounded at every chunk boundary and delta is
rounded to bf16 once before the downstream matmuls; tolerance
``atol=3e-3, rtol=3e-2``.

Run from the repository root (``conftest.py`` injects ``tests/``,
``tests/assets/`` and ``op_kernel/`` into ``sys.path``)::

    TILE_FWK_DEVICE_ID=<device-id> pytest ops/pro/kimi_k3/kda_cp_main/tests \
        --mode jit -q
"""

import os

import pytest
import torch
import torch_npu  # noqa: F401  (registers the NPU backend)

from kda_cp_main import CHUNK, D, flash_kda_cp_main
from kda_cp_main_golden_cpu import (
    chunk_cap,
    kda_cp_main_golden,
    make_inputs,
)
from kda_cp_main_precision import check_output

SHAPES = (
    ("BNSD", 17, 1, 3),
    ("BSND", 49, 1, 5),
    ("TND", 577, 1, 96),
    ("BNSD", 64, 1, 1),
    ("BNSD", 200, 2, 3),
    ("BSND", 129, 1, 2),
    ("BNSD", 128, 1, 32),
    ("BNSD", 193, 1, 30),
)
DEVICE_ID = os.environ.get("TILE_FWK_DEVICE_ID", "0")
DEVICE = f"npu:{DEVICE_ID}"

pytestmark = pytest.mark.soc("950")


def require_a5():
    """Skip when the visible device is not the Ascend 950 family."""
    try:
        torch.npu.set_device(DEVICE)
    except RuntimeError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")


def _run_case(layout, batch, nv, seq_len, seed=19):
    require_a5()
    device = torch.device(DEVICE)
    inputs = make_inputs(layout, batch, nv, seq_len, device, seed)
    out, final_state = flash_kda_cp_main(
        None, None, inputs["decay"], inputs["qg"], inputs["kg"], inputs["u"], inputs["w"],
        inputs["mqk"], inputs["state"], layout, None, inputs["cu"],
    )
    torch.npu.synchronize()
    ref_out, ref_state = kda_cp_main_golden(inputs, layout, batch, nv, seq_len)
    check_output(out, final_state, ref_out, ref_state)


@pytest.mark.parametrize("layout,seq_len,batch,nv", [list(s) for s in SHAPES])
def test_layouts_and_tails(layout, seq_len, batch, nv):
    _run_case(layout, batch, nv, seq_len)


def test_dual_chain_pairs():
    """Cover the multi-item-core path (2*B*Nv > cores)."""
    _run_case("BNSD", 1, 32, 256)


def test_multi_seq_tnd():
    require_a5()
    device = torch.device(DEVICE)
    g = torch.Generator().manual_seed(7)
    nv, lens = 2, (100, 37, 64)
    total = sum(lens)
    batch = len(lens)

    def rand_bf16(shape):
        return (torch.randn(shape, generator=g) * 0.02).bfloat16().to(device)

    qg, kg, u, w = (rand_bf16((total, nv, D)) for _ in range(4))
    cap = chunk_cap(batch, total)
    mqk = torch.tril(torch.randn((batch, nv, cap, CHUNK, CHUNK), generator=g) * 0.02)
    mqk = mqk.bfloat16().to(device)
    decay = (torch.rand((batch, nv, cap, D), generator=g) * 0.1 + 0.9).to(device)
    state = (torch.randn((batch, nv, D, D), generator=g) * 0.02).to(device)
    cu = torch.tensor([0, 100, 137, 201], dtype=torch.int32, device=device)

    out, final_state = flash_kda_cp_main(
        None, None, decay, qg, kg, u, w, mqk, state, "TND", None, cu)
    torch.npu.synchronize()

    # golden consumes per-sequence token slices of the packed tensors
    cu_cpu = cu.cpu().tolist()
    for s in range(batch):
        seq_len = cu_cpu[s + 1] - cu_cpu[s]
        sub = {
            "qg": qg[cu_cpu[s]:cu_cpu[s + 1]],
            "kg": kg[cu_cpu[s]:cu_cpu[s + 1]],
            "u": u[cu_cpu[s]:cu_cpu[s + 1]],
            "w": w[cu_cpu[s]:cu_cpu[s + 1]],
            "mqk": mqk[s:s + 1],
            "decay": decay[s:s + 1],
            "state": state[s:s + 1],
        }
        ref_out, ref_state = kda_cp_main_golden(sub, "TND", 1, nv, seq_len)
        torch.testing.assert_close(out[cu_cpu[s]:cu_cpu[s + 1]].cpu(), ref_out.cpu(),
                                   atol=3e-3, rtol=3e-2)
        torch.testing.assert_close(final_state[s:s + 1].cpu(), ref_state.cpu(),
                                   atol=3e-3, rtol=3e-2)


def test_stability_repeat():
    for _ in range(3):
        _run_case("BNSD", 1, 2, 130, seed=23)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
