#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reference for commit_recurrent_kda_replayssm, with CPU and NPU modes.

The math has exactly one implementation (torch, batch/head vectorized and
token serial). NumPy inputs run it on CPU and get NumPy back; torch inputs run
it on their device. CPU mode is the independent FP32 reference used by the
precision tests, and NPU mode is the performance baseline.
"""

from __future__ import annotations

import logging

import numpy as np

_LOG = logging.getLogger(__name__)


def _is_torch(value):
    try:
        import torch
    except ImportError:
        return False
    return isinstance(value, torch.Tensor)


def reference(state, replay_u, replay_k, replay_decay, num_accepted_tokens):
    """Return the committed state; NumPy inputs run on CPU, torch inputs on device."""
    torch_in = _is_torch(state)
    if not torch_in:
        import torch
        state = torch.tensor(np.asarray(state))
        replay_u = torch.as_tensor(np.asarray(replay_u))
        replay_k = torch.as_tensor(np.asarray(replay_k))
        replay_decay = torch.as_tensor(np.asarray(replay_decay))
        num_accepted_tokens = torch.as_tensor(np.asarray(num_accepted_tokens))
    out = _reference_impl(state, replay_u, replay_k, replay_decay,
                          num_accepted_tokens)
    return out if torch_in else out.numpy()


def _reference_impl(recurrent_state, replay_u, replay_k, replay_decay,
                    num_accepted_tokens):
    """Single torch implementation: batch/head vectorized, token serial."""
    import torch

    state = recurrent_state
    u_all = replay_u.to(torch.float32)
    k_all = replay_k.to(torch.float32)
    dec_all = replay_decay.to(torch.float32)
    accepted = num_accepted_tokens.to(torch.int64)

    batch = u_all.shape[0]
    max_accepted = int(accepted.max().item()) if batch else 0
    for t in range(max_accepted):
        decay = dec_all[:, t]                    # [B, N, D] -> key columns
        u = u_all[:, t]                          # [B, N, D] -> value rows
        k = k_all[:, t]                          # [B, N, D] -> key columns
        updated = state * decay.unsqueeze(-2) + u.unsqueeze(-1) * k.unsqueeze(-2)
        active = (t < accepted).view(batch, 1, 1, 1)
        state = torch.where(active, updated, state)
    recurrent_state.copy_(state)
    return recurrent_state


def make_inputs(batch, seq=8, heads=6, dim=128, accepted=4, seed=20260921,
                device=None):
    """Build the benchmark-shape inputs; ``device`` switches to torch tensors."""
    rng = np.random.default_rng(seed)
    state = np.zeros((batch, heads, dim, dim), dtype=np.float32)
    replay_u = (rng.standard_normal((batch, seq, heads, dim)) * 0.02).astype(np.float16)
    replay_k = (rng.standard_normal((batch, seq, heads, dim)) * 0.02).astype(np.float16)
    replay_decay = (1.0 / (1.0 + np.exp(
        -rng.standard_normal((batch, seq, heads, dim))))).astype(np.float32)
    if isinstance(accepted, int):
        acc = np.full((batch,), accepted, dtype=np.int32)
    else:
        acc = np.asarray(accepted, dtype=np.int32)
    if device is None:
        return state, replay_u, replay_k, replay_decay, acc
    import torch
    dev = torch.device(device)
    return (
        torch.tensor(state, device=dev),
        torch.from_numpy(np.ascontiguousarray(replay_u)).to(torch.float16).to(dev),
        torch.from_numpy(np.ascontiguousarray(replay_k)).to(torch.float16).to(dev),
        torch.from_numpy(replay_decay).to(dev),
        torch.from_numpy(acc).to(dev),
    )


# ===== numerical comparison criteria =====
ATOL = 1e-6
RTOL = 1e-6
REL_FLOOR = 1e-3


def floored_relative_error(candidate, ref, floor=REL_FLOOR):
    """max |cand - ref| / max(|ref|, floor); ``floor`` keeps near-zero entries finite."""
    candidate = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if candidate.shape != ref.shape:
        raise ValueError(
            f"shape mismatch: candidate {candidate.shape} vs ref {ref.shape}"
        )
    scale = np.maximum(np.abs(ref), floor)
    return float(np.max(np.abs(candidate - ref) / scale))


def violations(candidate, ref, atol=ATOL, rtol=RTOL):
    """Return (count, worst_abs_margin, worst_rel_error) for the contract rule."""
    candidate = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if candidate.shape != ref.shape:
        raise ValueError(
            f"shape mismatch: candidate {candidate.shape} vs ref {ref.shape}"
        )
    error = np.abs(candidate - ref)
    limit = atol + rtol * np.abs(ref)
    count = int(np.count_nonzero(error > limit))
    margin = float(np.max(error - limit)) if error.size else 0.0
    return count, margin, floored_relative_error(candidate, ref)


def assert_close(candidate, ref, atol=ATOL, rtol=RTOL, label="state"):
    """Raise AssertionError when the contract rule fails."""
    count, margin, rel = violations(candidate, ref, atol, rtol)
    if count:
        raise AssertionError(
            f"{label}: {count} element(s) outside atol={atol} rtol={rtol}; "
            f"worst margin {margin:.3e}, floored rel err {rel:.3e}"
        )
    return rel


# ===== self-check: NPU mode against CPU mode =====

CASES = (
    ("bench_shape_accepted4", 16, 8, 6, 4),
    ("mixed_accepted", 7, 4, 6, [0, 4, 1, 0, 2, 3, 0]),
    ("multichunk_tail", 2, 40, 3, [40, 39]),
    ("accepted_zero", 4, 5, 2, 0),
    ("tiny", 3, 3, 2, 2),
)


def _validate(device=None):
    import torch
    if device is None:
        if not torch.npu.is_available() or torch.npu.device_count() == 0:
            _LOG.info("NPU is unavailable; CPU reference self-check only")
            return
        device = torch.device("npu:0")
    _LOG.info("=" * 60)
    _LOG.info("commit_recurrent_kda_replayssm gen_golden 自检（NPU vs CPU）")
    _LOG.info("Device: %s", device)
    _LOG.info("=" * 60)
    for name, batch, seq, heads, accepted in CASES:
        cpu = make_inputs(batch, seq, heads, accepted=accepted)
        expect = reference(*cpu)
        args = make_inputs(batch, seq, heads, accepted=accepted, device=device)
        got = reference(*args).cpu().numpy()
        rel = floored_relative_error(got, expect)
        _LOG.info("  [%s] floored rel err %.3e", name, rel)
        if rel >= 1e-6:
            raise AssertionError(f"{name}: floored rel err {rel:.3e}")
    _LOG.info("自检完成")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _validate()
