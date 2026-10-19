# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared scaffold for the KDA chunk op tests.

Holds the boilerplate ``test_kda_chunk.py`` uses: the common imports, the
``_HAS_NPU`` torch_npu detection, the logging config, the sys.path inserts for
the impl + golden dirs, and the golden op import.
"""

__all__ = ["_HAS_NPU", "torch_npu", "_naive_recurrent_kda", "make_kda_inputs",
           "KdaCase", "KdaInputs", "chunk_grid",
           "HEAD_DIM", "RTOL", "ATOL"]

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional

import torch

# Single source of truth for the op-test envelope. The chunk tests
# iterate these grids AND test_cases_sync.py asserts test_cases.json documents
# exactly the same set — so the JSON cannot silently drift from what runs.
HEAD_DIM = 128          # head_k_dim == head_v_dim (real Kimi config)
RTOL = ATOL = 6e-3      # tolerance vs the fp32 naive-recurrent golden

logging.basicConfig(level=logging.INFO, format="%(message)s")

try:
    import torch_npu
    _HAS_NPU = True
except ImportError:
    torch_npu = None
    logging.info("torch_npu not available; this test only runs on Ascend NPU.")
    _HAS_NPU = False

# impl lives under src/.../kimi_linear_48b_a3b/kda; golden is same-dir.
_IMPL = (Path(__file__).resolve().parents[3]
         / "ops" / "pypto_tensor" / "kimi_linear_48b_a3b" / "kda")
sys.path.insert(0, str(_IMPL))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kda_golden import _naive_recurrent_kda  # noqa: E402


@dataclass
class KdaCase:
    """Spec for one KDA test case (shared by the chunk tests).

    gate: 'rand' -> realistic log-gate g=-5*rand in [-5,0]; 'ones' -> worst-case
    g=-5 everywhere.
    """
    batch: int
    num_heads: int
    seq_len: int
    with_state: bool = True
    gate: str = "rand"
    dtype: torch.dtype = torch.bfloat16
    seed: int = 0


class KdaInputs(NamedTuple):
    """Tensors for one KDA case, as built by ``make_kda_inputs``."""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    h0: Optional[torch.Tensor]
    scale: float


def make_kda_inputs(case, dev):
    """Build a reproducible ``KdaInputs`` for ``case`` on device ``dev``."""
    torch.manual_seed(case.seed)
    head_k_dim = head_v_dim = 128
    b, h, t, dt = case.batch, case.num_heads, case.seq_len, case.dtype
    q = (torch.randn(b, t, h, head_k_dim) * 0.5).to(dt).to(dev)
    k = (torch.randn(b, t, h, head_k_dim) * 0.5).to(dt).to(dev)
    v = (torch.randn(b, t, h, head_v_dim) * 0.5).to(dt).to(dev)
    # NOTE: probed from KimiDeltaAttention.forward (see test_cases.json "probe"):
    # the real model emits g in fp32 (fused_kda_gate) and beta in fp32
    # (.float().sigmoid()), while q/k/v are bf16. Match those real dtypes.
    if case.gate == "ones":
        g = (-5.0 * torch.ones(b, t, h, head_k_dim)).to(torch.float32).to(dev)
    else:
        g = (-5.0 * torch.rand(b, t, h, head_k_dim)).to(torch.float32).to(dev)
    beta = torch.sigmoid(torch.randn(b, t, h)).to(torch.float32).to(dev)
    h0 = (torch.randn(b, h, head_v_dim, head_k_dim) * 0.3).float().to(dev) if case.with_state else None
    scale = head_k_dim ** -0.5
    return KdaInputs(q, k, v, g, beta, h0, scale)


def chunk_grid():
    """Return the 42-case chunk/prefill op-test grid (shared with test_cases_sync)."""
    cases = []
    # realistic gate then worst-case g=-5 gate; seq_len/batch/heads sweep.
    for gate in ("rand", "ones"):
        for seq_len in (64, 128, 300, 512, 1000):
            for batch in (1, 2):
                for num_heads in (4, 32):
                    cases.append(KdaCase(batch, num_heads, seq_len, gate=gate))
    # no-initial-state path.
    cases.append(KdaCase(1, 4, 128, with_state=False, gate="rand"))
    cases.append(KdaCase(1, 32, 300, with_state=False, gate="rand"))
    return cases
