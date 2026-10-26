# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Model-integration dispatch test for the kimi_linear_48b_a3b KDA prefill path.

Exercises the ACTUAL modeling hook ``_try_kda_fused_prefill`` (modeling_kimi.py)
with a minimal stub layer and a stub ``kimi_linear_48b_a3b_pto_kernels`` module,
without loading the model:

1. ``USE_PTO_KDA`` off -> hook returns None, fused operator not called;
2. ineligible prefill (``q_len <= 64`` / attention mask / training) -> None,
   operator not called;
3. operator raising ``NotImplementedError`` -> None (fallback protocol);
4. success path -> operator called with the layer's projections, the result
   passes through ``layer.o_proj``, and the recurrent/conv states are written
   back to the cache.

Numerical correctness of the fused kernel itself is covered by
``test_kda_fused_prefill.py`` (which also carries the graph-capture regression
check); this file deliberately does not duplicate that coverage.
"""

import functools
import logging
import os
import sys
import types

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")

# bootstrap the repo layout onto sys.path: src/ carries the pypto_gym modeling
# package, the repo root carries the top-level ops package with the kernels
# (same walk-up pattern as tests/ops/phi_3_mini_4k_instruct)
_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "src")):
    _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, "src"))
sys.path.insert(0, _p)

from pypto_gym.transformers.kimi_linear_48b_a3b.modeling_kimi import (  # noqa: E402
    _try_kda_fused_prefill as try_kda_fused_prefill)

_PTO_MOD = "kimi_linear_48b_a3b_pto_kernels"

B, T, H, D = 1, 128, 4, 128
P = H * D


def _fill_proj(hs, value):
    """Stub projection: [B,T,hidden] -> [B,T,P] constant fill."""
    return torch.full((hs.shape[0], hs.shape[1], P), value, device=hs.device)


def _zeros_proj(hs, last_dim):
    """Stub projection: [B,T,hidden] -> [B,T,last_dim] zeros."""
    return torch.zeros(hs.shape[0], hs.shape[1], last_dim, device=hs.device)


class _StubLayer:
    """Minimal stand-in for KimiDeltaAttention: only the attributes the hook
    touches. Projections return fixed tensors; o_proj is a real Linear so the
    test proves the hook's result actually passes through it."""

    def __init__(self, device):
        self.layer_idx = 0
        self.training = False
        self.num_heads = H
        self.head_dim = D
        self.conv_size = 4
        self.config = types.SimpleNamespace(gate_lower_bound=-5.0)
        # projections: [B,T,hidden] -> [B,T,P]
        self.q_proj = functools.partial(_fill_proj, value=0.1)
        self.k_proj = functools.partial(_fill_proj, value=0.2)
        self.v_proj = functools.partial(_fill_proj, value=0.3)
        self.f_a_proj = functools.partial(_zeros_proj, last_dim=D)
        self.g_a_proj = functools.partial(_zeros_proj, last_dim=D)
        self.f_b_proj = functools.partial(_zeros_proj, last_dim=P)
        self.g_b_proj = functools.partial(_zeros_proj, last_dim=P)
        self.b_proj = functools.partial(_zeros_proj, last_dim=H)
        self.dt_bias = torch.zeros(P)
        # "A_log" mirrors the model's checkpoint-key attribute (see the
        # naming note on KimiDeltaAttention.A_log in modeling_kimi.py);
        # setattr keeps the non-snake_case mirror name out of the linter.
        setattr(self, "A_log", torch.zeros(1, 1, H, 1))
        self.o_norm = types.SimpleNamespace(weight=torch.ones(D), eps=1e-5)
        self.q_conv1d = types.SimpleNamespace(
            weight=torch.zeros(P, 1, self.conv_size))
        self.k_conv1d = types.SimpleNamespace(
            weight=torch.zeros(P, 1, self.conv_size))
        self.v_conv1d = types.SimpleNamespace(
            weight=torch.zeros(P, 1, self.conv_size))
        torch.manual_seed(0)
        self.o_proj = torch.nn.Linear(P, 8, bias=False,
                                      dtype=torch.bfloat16)  # model dtype


class _StubCache:
    """Minimal stand-in for KimiDynamicCache: the two lists the hook indexes."""

    def __init__(self, layer_idx, conv_state=None):
        self.conv_states = [None] * (layer_idx + 1)
        self.recurrent_states = [None] * (layer_idx + 1)
        self.conv_states[layer_idx] = conv_state


def _install_stub_pto(op, use_flag=True):
    """Install a stub kimi_linear_48b_a3b_pto_kernels module; returns a restore
    callable. ``op`` is the fake kda_fused_prefill (a plain function)."""
    saved = sys.modules.get(_PTO_MOD)
    mod = types.ModuleType(_PTO_MOD)
    mod.USE_PTO_KDA = use_flag
    mod.kda_fused_prefill = op
    sys.modules[_PTO_MOD] = mod

    def restore():
        if saved is None:
            sys.modules.pop(_PTO_MOD, None)
        else:
            sys.modules[_PTO_MOD] = saved
    return restore


def _run_hook(layer, hidden_states, cache_params=None,
              attention_mask=None, cu_seqlens=None):
    return try_kda_fused_prefill(
        layer, hidden_states, cache_params, attention_mask, cu_seqlens)


def _counting_op(calls):
    def op(*a, **kw):
        calls.append(1)
    return op


def test_hook_disabled():
    """USE_PTO_KDA off -> None, operator not called."""
    calls = []
    restore = _install_stub_pto(_counting_op(calls), use_flag=False)
    try:
        layer = _StubLayer("cpu")
        hs = torch.zeros(B, T, 8)
        assert _run_hook(layer, hs) is None
        assert not calls
    finally:
        restore()


def test_hook_ineligible():
    """Representative ineligible guards -> None, operator not called."""
    calls = []
    restore = _install_stub_pto(_counting_op(calls))
    try:
        layer = _StubLayer("cpu")
        hs = torch.zeros(B, T, 8)
        # short prefill (decode-side, q_len <= 64)
        assert _run_hook(layer, torch.zeros(B, 32, 8)) is None
        # varlen / masked batch
        mask = torch.ones(B, T, dtype=torch.long)
        assert _run_hook(layer, hs, attention_mask=mask) is None
        # training
        layer.training = True
        assert _run_hook(layer, hs) is None
        layer.training = False
        assert not calls
    finally:
        restore()


def test_hook_fallback():
    """Operator raising NotImplementedError -> None (fallback protocol)."""
    def raising_op(*a, **kw):
        raise NotImplementedError("off-bound NPU")

    restore = _install_stub_pto(raising_op)
    try:
        layer = _StubLayer("cpu")
        hs = torch.zeros(B, T, 8)
        assert _run_hook(layer, hs) is None
    finally:
        restore()


def test_hook_success():
    """Success path: operator called, o_proj applied, states written back."""
    calls = []
    fake_o = torch.full((B, T, H, D), 0.5, dtype=torch.bfloat16)
    fake_state = torch.full((B, H, D, D), 0.25)
    fake_cs = tuple(torch.full((B, P, 3), 0.75) for _ in range(3))
    conv_in = torch.full((B, P, 3), 0.125)

    def fake_op(inputs, weights, initial_state=None, output_final_state=True,
                conv_cache=None, **kw):
        calls.append({"inputs": inputs, "weights": weights,
                      "conv_cache": conv_cache, "initial_state": initial_state})
        return fake_o, fake_state, fake_cs

    restore = _install_stub_pto(fake_op)
    try:
        layer = _StubLayer("cpu")
        hs = torch.zeros(B, T, 8)
        st_in = torch.full((B, H, D, D), 0.0625)
        cache = _StubCache(layer.layer_idx, conv_state=conv_in)
        cache.recurrent_states[layer.layer_idx] = st_in

        out = _run_hook(layer, hs, cache_params=cache)

        # the operator ran exactly once, with the layer's projections and the
        # cache's conv history / recurrent state
        assert len(calls) == 1
        c = calls[0]
        assert c["inputs"].q.shape == (B, T, H, D)
        assert c["inputs"].v.shape == (B, T, H, D)
        assert c["inputs"].g_raw.shape == (B, T, P)
        assert c["inputs"].b_raw.shape == (B, T, H)
        assert c["weights"].a_log is getattr(layer, "A_log")
        assert c["conv_cache"] is conv_in
        assert c["initial_state"] is st_in

        # the hook's return value is o_proj applied to the operator's output
        expected = layer.o_proj(fake_o.reshape(B, T, P))
        assert out.shape == expected.shape
        assert torch.equal(out, expected)

        # recurrent + conv states written back into the cache
        assert cache.recurrent_states[layer.layer_idx] is fake_state
        assert cache.conv_states[layer.layer_idx] is fake_cs
    finally:
        restore()


def test_hook_success_no_cache():
    """Success path without a cache: operator still called, nothing to write."""
    calls = []

    def fake_op(*a, **kw):
        calls.append(1)
        return (torch.zeros(B, T, H, D, dtype=torch.bfloat16),
                torch.zeros(B, H, D, D),
                tuple(torch.zeros(B, P, 3) for _ in range(3)))

    restore = _install_stub_pto(fake_op)
    try:
        layer = _StubLayer("cpu")
        hs = torch.zeros(B, T, 8)
        out = _run_hook(layer, hs, cache_params=None)
        assert out is not None
        assert len(calls) == 1
    finally:
        restore()


def main():
    """Direct-script entry: run all hook tests."""
    test_hook_disabled()
    test_hook_ineligible()
    test_hook_fallback()
    test_hook_success()
    test_hook_success_no_cache()
    logging.info("All tests passed!")


if __name__ == "__main__":
    main()
