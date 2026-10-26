# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical + graph-capture tests for the kimi_linear_48b_a3b KDA fused prefill PyPTO kernel.
"""

import logging
import os
from collections import namedtuple

import numpy as np
import torch

from _kda_test_common import _HAS_NPU, torch_npu, HEAD_DIM

from kimi_fla_compat import (
    ShortConvolution,
    _naive_recurrent_kda,
    fused_kda_gate,
)
from kda_fused_prefill_impl import (
    KdaPrefillInputs,
    KdaPrefillWeights,
    kda_fused_prefill,
)

RMS_EPS = 1e-5
CONV_K = 4
RTOL_O = ATOL_O = 1.5e-2
RTOL_S = ATOL_S = 1e-4

Case = namedtuple("Case", "shape seed cache gate state",
                  defaults=(7, False, "rand", True))

_CASES = (
    Case((1, 128, 4)),
    Case((2, 300, 4)),
    Case((1, 512, 32)),
    Case((1, 512, 32), gate="ones"),
    Case((1, 300, 32), state=False),
    Case((2, 300, 32), cache=True),
)


def _conv_module(weight, device):
    m = ShortConvolution(weight.shape[0], CONV_K, activation="silu").to(
        device=device, dtype=weight.dtype)
    with torch.no_grad():
        m.weight.copy_(weight.reshape(weight.shape[0], 1, CONV_K))
    return m


def _conv_silu_fp32(x, w, hist=None):
    """fp32 conv+silu on (bf16) x with optional [B, p, 3] pre-history --
    the same math the kernel's conv branch computes."""
    b, t, h, d = x.shape
    xf = x.reshape(b, t, h * d).float().permute(0, 2, 1)
    left = (hist if hist is not None else
            torch.zeros(b, h * d, CONV_K - 1, device=xf.device))
    xc = torch.cat([left, xf], dim=2)
    y = torch.nn.functional.conv1d(
        xc, w.reshape(h * d, 1, CONV_K).float(), groups=h * d)
    return torch.nn.functional.silu(y[..., :t]).permute(0, 2, 1).view(b, t, h, d)


def make_fused_inputs(shape, dev, seed=7, with_cache=False, gate="rand"):
    """Model-realistic fused-prefill call args (grouped API tuples)."""
    b, t, h = shape
    torch.manual_seed(seed)
    p = h * HEAD_DIM
    q = (torch.randn(b, t, h, HEAD_DIM) * 0.5).to(torch.bfloat16).to(dev)
    k = (torch.randn(b, t, h, HEAD_DIM) * 0.5).to(torch.bfloat16).to(dev)
    v = (torch.randn(b, t, h, HEAD_DIM) * 0.5).to(torch.bfloat16).to(dev)
    wq = (torch.randn(p, CONV_K) * 0.3).to(torch.bfloat16).to(dev)
    wk = (torch.randn(p, CONV_K) * 0.3).to(torch.bfloat16).to(dev)
    wv = (torch.randn(p, CONV_K) * 0.3).to(torch.bfloat16).to(dev)
    if gate == "ones":
        g_raw = torch.full((b, t, p), 10.0, device=dev).to(torch.bfloat16)
    else:
        g_raw = torch.randn(b, t, p, device=dev).to(torch.bfloat16)
    g_gate = torch.randn(b, t, p, device=dev).to(torch.bfloat16)
    b_raw = torch.randn(b, t, h, device=dev).to(torch.bfloat16)
    dt_bias = torch.randn(p, dtype=torch.float32, device=dev)
    a_log = (torch.log(torch.rand(h, dtype=torch.float32, device=dev)
                    .uniform_(1, 16)).view(1, 1, h, 1))   # model param shape
    o_w = torch.randn(HEAD_DIM, dtype=torch.float32, device=dev) * 0.2 + 1.0
    h0 = (torch.randn(b, h, HEAD_DIM, HEAD_DIM) * 0.3).to(dev)
    cache = tuple((torch.randn(b, p, CONV_K - 1) * 0.3).to(dev)
                  for _ in range(3)) if with_cache else None
    inputs = KdaPrefillInputs(q=q, k=k, v=v, g_raw=g_raw, g_gate=g_gate,
                              b_raw=b_raw)
    weights = KdaPrefillWeights(dt_bias=dt_bias, a_log=a_log, o_norm_w=o_w,
                                conv_wq=wq, conv_wk=wk, conv_wv=wv)
    return inputs, weights, h0, cache


@torch.no_grad()
def _golden_prefill(inputs, weights, dev, initial_state=None, conv_cache=None):
    """ALL-fp32 golden: fp32 conv+silu -> gate -> beta -> exact naive
    recurrence -> fp32 o_norm. The conv HISTORY comes from the bf16
    ShortConvolution modules (the decode-seeding contract, checked
    bit-exact separately)."""
    q, k, v = inputs.q, inputs.k, inputs.v
    b, t, h, d = q.shape
    cs = conv_cache if conv_cache is not None else (None, None, None)
    q_c = _conv_silu_fp32(q, weights.conv_wq, cs[0])
    k_c = _conv_silu_fp32(k, weights.conv_wk, cs[1])
    v_c = _conv_silu_fp32(v, weights.conv_wv, cs[2])
    g = fused_kda_gate(inputs.g_raw, weights.a_log, HEAD_DIM, g_bias=weights.dt_bias)
    beta = inputs.b_raw.float().sigmoid()
    o, st = _naive_recurrent_kda(
        q_c, k_c, v_c, g, beta, initial_state=initial_state,
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    var = o.pow(2).mean(-1, keepdim=True)
    o = (o * torch.rsqrt(var + RMS_EPS) * weights.o_norm_w
         * torch.sigmoid(inputs.g_gate.view(b, t, h, d).float()))

    convs = (_conv_module(weights.conv_wq, dev),
             _conv_module(weights.conv_wk, dev),
             _conv_module(weights.conv_wv, dev))
    _, cq = convs[0](q.reshape(b, t, h * d), cache=cs[0], output_final_state=True)
    _, ck = convs[1](k.reshape(b, t, h * d), cache=cs[1], output_final_state=True)
    _, cv = convs[2](v.reshape(b, t, h * d), cache=cs[2], output_final_state=True)
    return o, st, (cq, ck, cv)


def run_case(dev, case):
    """Run one fused prefill case and assert it matches the fp32 golden."""
    inputs, weights, h0, cache = make_fused_inputs(case.shape, dev, case.seed,
                                                    case.cache, case.gate)
    initial = h0 if case.state else None
    o_ref, s_ref, cs_ref = _golden_prefill(inputs, weights, dev,
                                            initial_state=initial,
                                            conv_cache=cache)
    o_f, s_f, cs_f = kda_fused_prefill(inputs, weights, initial_state=initial,
                                        output_final_state=True, conv_cache=cache)

    b, t, h = case.shape
    name = (f"B{b}H{h}T{t}{'_cache' if case.cache else ''}_g{case.gate}"
            f"{'' if case.state else '_nostate'}")
    do = (o_f.float() - o_ref).abs().max().item()
    ds = (s_f - s_ref).abs().max().item()
    logging.info("[%s] out=%.2e (<=1 bf16 ULP) state=%.2e conv_history=bit-exact",
                 name, do, ds)

    np.testing.assert_allclose(o_f.float().cpu().numpy(), o_ref.cpu().numpy(),
                               rtol=RTOL_O, atol=ATOL_O, err_msg=f"{name}: out mismatch")
    np.testing.assert_allclose(s_f.cpu().numpy(), s_ref.cpu().numpy(),
                               rtol=RTOL_S, atol=RTOL_S, err_msg=f"{name}: state mismatch")
    # the conv history seeds decode — it must be bit-exact
    for a, r in zip(cs_f, cs_ref):
        assert torch.equal(a.cpu(), r.cpu()), f"{name}: conv history not bit-exact"
    return do, ds


def test_kda_fused_prefill():
    """True-precision test vs the all-fp32 golden (format-floor error)."""
    if not _HAS_NPU:
        return
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    dev = f"npu:{device_id}"

    worst_o = worst_s = 0.0
    for case in _CASES:
        do, ds = run_case(dev, case)
        worst_o, worst_s = max(worst_o, do), max(worst_s, ds)

    logging.info("\n[summary] worst true error: out=%.2e (bf16 format floor), "
                 "state=%.2e (fp32 WY)", worst_o, worst_s)


def test_kda_fused_prefill_graph_capture():
    """NPUGraph capture+replay regression check, bit-exact vs the eager call."""
    if not _HAS_NPU:
        return
    if not hasattr(torch_npu.npu, "NPUGraph"):
        logging.info("[graph] torch.npu.NPUGraph unavailable; skipped")
        return
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    dev = f"npu:{device_id}"

    inputs, weights, h0, _cache = make_fused_inputs((1, 128, 4), dev)
    kwargs = {"initial_state": h0, "output_final_state": True}

    # eager reference (also triggers the JIT compile, which must stay OUTSIDE
    # the capture region)
    o_e, s_e, cs_e = kda_fused_prefill(inputs, weights, **kwargs)
    o_ref, s_ref = o_e.clone(), s_e.clone()
    cs_ref = tuple(c.clone() for c in cs_e)
    torch_npu.npu.synchronize()

    # capture the same invocation (static input tensors, same as the bench's
    # stage capture), then replay and compare
    g = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(g):
        o_c, s_c, cs_c = kda_fused_prefill(inputs, weights, **kwargs)
    g.replay()
    torch_npu.npu.synchronize()

    assert torch.equal(o_c, o_ref), "graph replay: out differs from eager"
    assert torch.equal(s_c, s_ref), "graph replay: state differs from eager"
    for a, r in zip(cs_c, cs_ref):
        assert torch.equal(a, r), "graph replay: conv history differs from eager"
    logging.info("[graph] NPUGraph capture+replay bit-exact (out/state/conv-history)")


def main():
    """Direct-script entry: run both prefill tests."""
    test_kda_fused_prefill()
    test_kda_fused_prefill_graph_capture()
    logging.info("[PRECISION_PASS]")
    logging.info("All tests passed!")


if __name__ == "__main__":
    if _HAS_NPU:
        main()
    else:
        logging.info("skip: no NPU")
