# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end dispatch-equivalence test for the kimi_linear_48b_a3b KDA integration.

The op test (``test_kda_chunk.py``) checks the PyPTO kernel against the
naive-recurrent golden. This test instead exercises the
**integration glue** the model actually runs: ``_dispatch_kda`` (modeling_kimi.py)
picks the PyPTO wrapper as ``primary_fn`` when ``USE_PTO_KDA`` is set and falls
back to the torch ``chunk_kda`` on ``NotImplementedError``. On Ascend that torch
fallback resolves (via the ``fla`` ImportError branch) to ``vec_chunk_kda`` in
``kimi_fla_compat``.

Two things are asserted, mirroring exactly the kwargs dict the model builds
(modeling_kimi.py:625-633):

1. **Branch equivalence** — the PyPTO primary path and the torch fallback path
   produce the same output/state (within the op tolerance). This is the
   end-to-end claim ("swapping the fused kernel in does not change results"),
   which the kernel-vs-golden op tests do not directly cover.
2. **Fallback trigger** — on an out-of-envelope shape (K != 128) the wrapper
   raises ``NotImplementedError`` and ``_dispatch_kda`` returns the torch
   result, i.e. the model degrades gracefully instead of crashing.

The 3-line ``_dispatch_kda`` contract is mirrored here as ``_dispatch_kda`` and
pinned to the real source by ``test_dispatch_source_matches`` so the mirror
cannot silently drift from modeling_kimi.py.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

import pypto

from _kda_test_common import _HAS_NPU, torch_npu, make_kda_inputs, KdaCase

from kda_chunk_impl import kda_chunk_wrapper

# vec_chunk_kda is the torch op the model falls back to on Ascend (the `fla`
# ImportError branch in modeling_kimi maps `chunk_kda` -> this drop-in).
_COMPAT_DIR = (Path(__file__).resolve().parents[3]
               / "src" / "pypto_gym" / "transformers" / "kimi_linear_48b_a3b")
sys.path.insert(0, str(_COMPAT_DIR))
from kimi_fla_compat import vec_chunk_kda  # noqa: E402


def _dispatch_kda(primary_fn, fallback_fn, kda_kwargs):
    """Mirror of modeling_kimi.py ``_dispatch_kda`` (pinned by the source test).

    Run the KDA op, falling back to the torch implementation if the PyPTO
    wrapper raises NotImplementedError (off-bound-device or unsupported shape).
    """
    try:
        return primary_fn(**kda_kwargs)
    except NotImplementedError:
        return fallback_fn(**kda_kwargs)


def _model_kwargs(case, dev):
    """Build the exact kda_kwargs dict the modeling layer passes (no `scale`;
    both wrapper and vec_chunk_kda default scale = K**-0.5)."""
    q, k, v, g, beta, h0, _scale = make_kda_inputs(case, dev)
    return dict(q=q, k=k, v=v, g=g, beta=beta, initial_state=h0,
                output_final_state=True, use_qk_l2norm_in_kernel=True,
                cu_seqlens=None)


def _assert_close(a, b, name):
    np.testing.assert_allclose(a.cpu().float().numpy(), b.cpu().float().numpy(),
                               rtol=6e-3, atol=6e-3, err_msg=name)


def test_branch_equivalence(npu_device):
    """PyPTO primary path == torch fallback path on the model's kwargs dict."""
    max_err = 0.0
    for case in (KdaCase(1, 4, 128, gate="rand"),
                 KdaCase(2, 32, 300, gate="rand"),
                 KdaCase(1, 32, 512, gate="ones"),          # worst-case gate
                 KdaCase(1, 4, 128, with_state=False, gate="rand")):
        kw = _model_kwargs(case, npu_device)
        o_p, s_p = _dispatch_kda(kda_chunk_wrapper, vec_chunk_kda, kw)
        # torch reference branch (what _dispatch_kda returns when USE_PTO_KDA is off).
        o_t, s_t = vec_chunk_kda(**kw)
        name = f"B{case.batch}H{case.num_heads}T{case.seq_len}_state{case.with_state}_g{case.gate}"
        _assert_close(o_p, o_t, f"{name}: out branch mismatch")
        _assert_close(s_p, s_t, f"{name}: state branch mismatch")
        err = max((o_p.cpu().float() - o_t.cpu().float()).abs().max().item(),
                  (s_p.cpu().float() - s_t.cpu().float()).abs().max().item())
        logging.info(f"[equiv {name}] pypto-vs-torch max_diff={err:.3e}")
        max_err = max(max_err, err)
    logging.info(f"[equiv summary] max abs error pypto-vs-torch-fallback = {max_err:.3e}")


def test_fallback_trigger(npu_device):
    """Out-of-envelope shape (K=64) -> wrapper raises -> dispatch returns torch."""
    case = KdaCase(1, 4, 128, gate="rand")
    q, k, v, g, beta, h0, _ = make_kda_inputs(case, npu_device)
    # Truncate K (head_k_dim) to 64 so the wrapper's envelope check rejects it.
    q64, k64, g64 = q[..., :64], k[..., :64], g[..., :64]
    h064 = h0[..., :64] if h0 is not None else None
    kw = dict(q=q64, k=k64, v=v, g=g64, beta=beta, initial_state=h064,
              output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=None)

    # Wrapper must reject the unsupported shape ...
    raised = False
    try:
        kda_chunk_wrapper(**kw)
    except NotImplementedError:
        raised = True
    assert raised, "kda_chunk_wrapper must raise NotImplementedError for K != 128"

    # ... and _dispatch_kda must transparently return the torch fallback result.
    o_disp, s_disp = _dispatch_kda(kda_chunk_wrapper, vec_chunk_kda, kw)
    o_ref, s_ref = vec_chunk_kda(**kw)
    _assert_close(o_disp, o_ref, "fallback out mismatch")
    _assert_close(s_disp, s_ref, "fallback state mismatch")
    logging.info("[fallback] K=64 -> NotImplementedError -> torch fallback OK")


def test_dispatch_source_matches():
    """Pin the mirrored _dispatch_kda to the real modeling_kimi source so the
    test cannot drift from the integration it claims to cover.
    """
    src = (_COMPAT_DIR / "modeling_kimi.py").read_text()
    for needle in ("def _dispatch_kda(primary_fn, fallback_fn, kda_kwargs):",
                   "return primary_fn(**kda_kwargs)",
                   "except NotImplementedError:",
                   "return fallback_fn(**kda_kwargs)"):
        assert needle in src, f"modeling_kimi._dispatch_kda contract changed: missing {needle!r}"
    logging.info("[source] mirrored _dispatch_kda matches modeling_kimi.py")


def main():
    pypto.set_host_options(compile_monitor_enable=1, compile_timeout=10,
                           compile_timeout_stage=5,
                           compile_monitor_print_interval=2)
    test_dispatch_source_matches()
    if not _HAS_NPU:
        logging.info("skip NPU cases: no NPU")
        return
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    npu_device = f"npu:{device_id}"
    test_branch_equivalence(npu_device)
    test_fallback_trigger(npu_device)
    logging.info("[PRECISION_PASS]")
    logging.info("All tests passed!")


if __name__ == "__main__":
    main()
