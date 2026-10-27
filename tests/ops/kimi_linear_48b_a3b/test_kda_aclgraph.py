# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ACLGraph / torch.library registration test for the KDA kernels.

The chunk kernel is registered as ``pypto::kda_chunk_kimi``
(Meta + NPU keys) so they can be captured by torch.compile / aclgraph. This test
asserts two properties of that registration without needing torchair:

1. **Op == wrapper** — the registered NPU op (`kda_*_pypto`) returns bit-for-bit
   the same result as the eager `kda_*_wrapper` it delegates to, so routing the
   model through the registered op under aclgraph cannot change numerics.
2. **Meta shape inference** — the Meta ("fake") impl produces the correct output
   shapes/dtypes under ``FakeTensorMode`` (what the graph tracer relies on), so a
   capture pass can infer shapes without launching the kernel.
"""

import logging
import os

import torch

import pypto

from _kda_test_common import _HAS_NPU, torch_npu, make_kda_inputs, KdaCase

from kda_chunk_impl import kda_chunk_wrapper, kda_chunk_pypto


def _assert_identical(a, b, name):
    assert torch.equal(a.cpu(), b.cpu()), f"{name}: registered op != wrapper (not bit-identical)"


def test_registered_op_matches_wrapper(npu_device):
    """The registered NPU op must equal the eager wrapper, bit-for-bit."""
    for case, wrapper, pypto_fn, tag in (
            (KdaCase(1, 4, 128, gate="rand"), kda_chunk_wrapper, kda_chunk_pypto, "chunk"),):
        q, k, v, g, beta, h0, scale = make_kda_inputs(case, npu_device)
        ow, sw = wrapper(q, k, v, g, beta, h0, scale=scale)
        op, sp = pypto_fn(q, k, v, g, beta, initial_state=h0, scale=scale)
        _assert_identical(op.float(), ow.float(), f"{tag} out")
        _assert_identical(sp, sw, f"{tag} state")
        logging.info(f"[aclgraph {tag}] registered pypto::kda_{tag}_kimi == wrapper (bit-identical)")


def test_torch_compile_fullgraph(npu_device):
    """Registered op composes into a torch.compile fullgraph (no graph break), bit-exact."""
    # aot_eager exercises Meta fake-prop + the NPU impl, which a capture pass relies on.
    # The real torchair aclgraph (reduce-overhead) capture is covered by
    # test_aclgraph_capture; torchair GE-graph mode would still need a per-op GE converter.
    q, k, v, g, beta, h0, scale = make_kda_inputs(KdaCase(1, 4, 128, gate="rand"), npu_device)

    class _M(torch.nn.Module):
        def forward(self, q, k, v, g, beta, h0):
            return kda_chunk_pypto(q, k, v, g, beta, initial_state=h0, scale=scale)

    m = _M()
    eo, es = m(q, k, v, g, beta, h0)
    for backend in ("aot_eager", "eager"):
        cm = torch.compile(m, backend=backend, fullgraph=True, dynamic=False)
        co, cs = cm(q, k, v, g, beta, h0)
        _assert_identical(co.float(), eo.float(), f"compile[{backend}] out")
        _assert_identical(cs, es, f"compile[{backend}] state")
        logging.info(f"[aclgraph compile] fullgraph torch.compile(backend={backend}) "
                     f"traces pypto::kda_chunk_kimi with no graph break (bit-exact)")


def test_meta_shape_inference():
    """The Meta impl infers output shapes/dtypes under FakeTensorMode."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    dev = "npu" if _HAS_NPU else "cpu"
    b, t, h, d = 1, 100, 32, 128
    with FakeTensorMode():
        q = torch.empty(b, t, h, d, dtype=torch.bfloat16, device=dev)
        k = torch.empty(b, t, h, d, dtype=torch.bfloat16, device=dev)
        v = torch.empty(b, t, h, d, dtype=torch.bfloat16, device=dev)
        gg = torch.empty(b, t, h, d, dtype=torch.float32, device=dev)
        beta = torch.empty(b, t, h, dtype=torch.float32, device=dev)
        out, st = torch.ops.pypto.kda_chunk_kimi(q, k, v, gg, beta, None, d ** -0.5, True)
        assert tuple(out.shape) == (b, t, h, d) and out.dtype == torch.bfloat16, "chunk meta out"
        assert tuple(st.shape) == (b, h, d, d) and st.dtype == torch.float32, "chunk meta state"
    logging.info("[aclgraph meta] Meta shape inference OK for kda_chunk_kimi")


def test_aclgraph_capture(npu_device):
    """Real torchair aclgraph (reduce-overhead) capture + replay is bit-exact.

    This is the end-to-end proof that the registered op is graph-capturable: it
    works because the wrapper does NOT call torch_npu.npu.synchronize() (which
    would abort capture); the kernel is stream-ordered instead. Skips cleanly if
    torchair is unavailable.
    """
    try:
        import torchair
    except ImportError:
        logging.info("[aclgraph capture] torchair unavailable; skipped")
        return
    q, k, v, g, beta, h0, scale = make_kda_inputs(KdaCase(1, 4, 128, gate="rand"), npu_device)

    class _M(torch.nn.Module):
        def forward(self, q, k, v, g, beta, h0):
            return kda_chunk_pypto(q, k, v, g, beta, initial_state=h0, scale=scale)

    m = _M()
    eo, es = m(q, k, v, g, beta, h0)
    cfg = torchair.CompilerConfig()
    cfg.mode = "reduce-overhead"          # aclgraph / npugraph capture
    cm = torch.compile(m, backend=torchair.get_npu_backend(compiler_config=cfg), dynamic=False)
    for _ in range(3):                    # warmup capture + replays
        co, cs = cm(q, k, v, g, beta, h0)
    _assert_identical(co.float(), eo.float(), "aclgraph capture out")
    _assert_identical(cs, es, "aclgraph capture state")
    logging.info("[aclgraph capture] torchair reduce-overhead capture+replay bit-exact (diff=0)")


def main():
    pypto.set_host_options(compile_monitor_enable=1, compile_timeout=10,
                           compile_timeout_stage=5,
                           compile_monitor_print_interval=2)
    test_meta_shape_inference()
    if not _HAS_NPU:
        logging.info("skip NPU op-equality check: no NPU")
        return
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    npu_device = f"npu:{device_id}"
    test_registered_op_matches_wrapper(npu_device)
    test_torch_compile_fullgraph(npu_device)
    test_aclgraph_capture(npu_device)
    logging.info("[PRECISION_PASS]")
    logging.info("All tests passed!")


if __name__ == "__main__":
    main()
