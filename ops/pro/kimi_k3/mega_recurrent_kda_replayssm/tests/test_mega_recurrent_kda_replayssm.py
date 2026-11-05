# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mega_recurrent_kda_replayssm PyPTO-Pro test: one test function per case, plus a ``main`` entry.

The kernel and the host wrapper live under ``op_kernel/mega_recurrent_kda_replayssm.py``;
the CPU golden and the Scheme A comparator live under ``tests/assets/``. Each ``test_xxx``
builds deterministic inputs, runs the public wrapper on the NPU and checks the
per-element contract (``atol=rtol=5e-3``) plus the state invariants against the
independent CPU golden, for TP16/8/4 (N=6/12/24), B=4..512 and the 8/24/28/32 AIC tiers.

Run: ``python test_mega_recurrent_kda_replayssm.py``
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
from typing import NamedTuple

import pytest
import torch

import torch_npu  # noqa: F401

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_THIS_DIR)
_ASSETS_DIR = os.path.join(_THIS_DIR, "assets")
_REPO = _OP_DIR
while not os.path.isdir(os.path.join(_REPO, "ops")):
    _REPO = os.path.dirname(_REPO)
    if _REPO == os.path.dirname(_REPO):
        break
for _p in (_ASSETS_DIR, _OP_DIR, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_NZ = getattr(torch_npu.Format, "FRACTAL_NZ", None)
LOGGER = logging.getLogger(__name__)


def _load_module(name, path):
    """Load one sibling module by path (keeps ``python test_xxx.py`` working)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_M = _load_module(
    "mega_recurrent_kda_replayssm_kernel",
    os.path.join(_OP_DIR, "op_kernel", "mega_recurrent_kda_replayssm.py"))
_GOLDEN = _load_module(
    "mega_recurrent_kda_replayssm_golden_cpu",
    os.path.join(_ASSETS_DIR, "mega_recurrent_kda_replayssm_golden_cpu.py"))
_PRECISION = _load_module(
    "mega_recurrent_kda_replayssm_precision",
    os.path.join(_ASSETS_DIR, "mega_recurrent_kda_replayssm_precision.py"))

check_precision = _PRECISION.check_precision
CONV_K = _M.CONV_K
D = _M.D
DEFAULT_HEADS = _M.DEFAULT_HEADS
H = _M.H
S = _M.S

#: Outputs visible through the public API: the returned ``out`` plus the four
#: in-place state tensors (``conv_state`` / ``replay_k`` / ``replay_u`` /
#: ``replay_decay``).  The remaining Module-1..4 workspaces are kernel-internal.
_OUT_NAMES = ("out", "conv_state", "replay_k", "replay_u", "replay_decay")


class GoldenRefs(NamedTuple):
    """Reference Module-1..4 values, one field per historical tuple position."""

    raw_qkv: torch.Tensor
    panel_gate: torch.Tensor
    ws_beta_logit: torch.Tensor
    raw_decay: torch.Tensor
    decay_a: torch.Tensor
    ws_qkv: torch.Tensor
    panel_gate_dup: torch.Tensor
    conv_state: torch.Tensor
    replay_k: torch.Tensor
    gated: torch.Tensor
    replay_u: torch.Tensor
    replay_decay: torch.Tensor
    output: torch.Tensor


def _get_device():
    """Device from env (TILE_FWK_DEVICE_ID), default npu:0 — not hardcoded to a card."""
    return f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"


def _pc_metric_lines(summary):
    """Re-emit the merged precision comparator's per-output metrics as `key=value` lines.

    The engine prints ``matched_ratio   : 1.0000    thresh=...`` style lines; this
    reformats each output block into the single canonical line
    ``output[i] matched_ratio=<...> max_abs_error=<...>`` so the metric is always
    present in full (no truncated engine tail).
    """
    lines = []
    for block in summary.split("; "):
        mr = re.search(r"matched_ratio\s*:\s*([0-9.eE+-]+)", block)
        ma = re.search(r"max_abs_error\s*:\s*([0-9.eE+-]+)", block)
        mq = re.match(r"(output\[\d+\])", block.strip())
        idx = mq.group(1) if mq else "output[0]"
        if mr and ma:
            lines.append(f"{idx} matched_ratio={mr.group(1)} max_abs_error={ma.group(1)}")
    return lines


def _assert_precision(actual, golden, label="", atol=5e-3, rtol=5e-3):
    """Precision gate.

    1. **Frozen contract** (acceptance): per-tensor and per-element
       ``|a - g| <= atol + rtol*|g|`` with ``atol=rtol=5e-3``, exactly as the shipped
       harness does (``module_interfaces.yaml`` ``composition_verification``).
    2. the merged comparator (supplementary evidence).  Its per-dtype defaults are
       dtype-generic, not this operator's contract: for FP16 ``replay_k`` it uses
       rtol=atol=2**-9=1.95e-3, ~2.5x stricter than the frozen 5e-3.  Where (2) is
       stricter than (1) and (1) passes, (2) is reported as a note.

    The engine's per-output ``matched_ratio=`` / ``max_abs_error=`` metrics are
    always logged in full.
    """
    max_abs = {}
    for name, a, g in zip(_OUT_NAMES, actual, golden):
        af = a.cpu().float()
        gf = g.cpu().float()
        if not torch.isfinite(af).all():
            raise AssertionError(f"{label}: {name} contains non-finite values")
        diff = (af - gf).abs()
        max_abs[name] = float(diff.max())
        if bool((diff > atol + rtol * gf.abs()).any()):
            raise AssertionError(
                f"{label}: {name} violates the frozen atol=rtol={atol} contract "
                f"(max-abs {max_abs[name]:.3e})")
    pc_passed, pc_summary = check_precision(actual, golden)
    LOGGER.info("  [%s] frozen-contract PASS (atol=rtol=%s)", label, atol)
    for line in _pc_metric_lines(pc_summary):
        LOGGER.info("  [%s] comparator %s", label, line)
    LOGGER.info("  [%s] max_abs per tensor: %s", label,
                " ".join(f"{k}={v:.3e}" for k, v in max_abs.items()))
    if not pc_passed:
        LOGGER.info(
            "  [%s] comparator note: its dtype-generic default is stricter "
            "than the operator's frozen contract here (FP16 replay_k default "
            "rtol=atol=1.95e-3 vs frozen 5e-3); the frozen contract is satisfied.", label)
    return max_abs


def _make_inputs(batch, num_heads=DEFAULT_HEADS, seed=20260921, conv_layout=1):
    """Deterministic CPU inputs — mirrors the shipped test harness / staged golden."""
    gen = torch.Generator().manual_seed(seed)
    n = int(num_heads)
    p = n * D

    def randn(shape, dtype=torch.bfloat16, scale=0.02):
        return (torch.randn(shape, generator=gen) * scale).to(dtype)

    if conv_layout == 0:
        conv_w = randn((3, n, CONV_K, D))
    elif conv_layout == 2:
        conv_w = randn((3, CONV_K, D))
    else:
        conv_w = randn((CONV_K, 3 * p))

    conv_idx = torch.randperm(batch, generator=gen).int() + 1
    conv_idx[::7] = 0
    conv_acc = torch.arange(batch, dtype=torch.int32) % (S + 1)
    return dict(
        hidden_states=randn((batch, S, H)),
        qkv_projection_weight=randn((3 * p, H)),
        decay_projection_a_weight=randn((D, H)),
        decay_projection_b_weight=randn((p, D)),
        beta_projection_weight=randn((n, H)),
        output_gate_projection_weight=randn((p, H)),
        output_norm_weight=randn((D,), torch.float32, scale=0.2),
        output_projection_weight=randn((H, p)),
        conv1d_weight=conv_w,
        conv_state=randn((batch + 1, S + 2, 3 * p)),
        recurrent_state=randn((batch, n, D, D), torch.float32, scale=0.002),
        replay_u=torch.empty((batch, S, n, D), dtype=torch.float16),
        replay_k=torch.empty((batch, S, n, D), dtype=torch.float16),
        replay_decay=torch.empty((batch, S, n, D), dtype=torch.float32),
        conv_state_indices=conv_idx,
        conv_num_accepted_tokens=conv_acc,
        a_log=randn((n,), torch.float32, scale=0.1),
        dt_bias=randn((n, D), torch.float32, scale=0.1),
    )


_NZ_WEIGHTS = ("qkv_projection_weight", "decay_projection_a_weight",
               "decay_projection_b_weight", "beta_projection_weight",
               "output_gate_projection_weight", "output_projection_weight")


def _eval_golden(golden, cpu):
    """Reference Module-1..4 values (dev-only).

    The wrapper comparison uses the API-visible entries: ``out`` plus the four
    in-place state tensors; the remaining workspaces document the stage contract.
    """
    b = int(cpu["hidden_states"].shape[0])
    t = b * S
    beta_heads = int(cpu["beta_projection_weight"].shape[0])
    n = beta_heads if beta_heads in (6, 12, 24) else DEFAULT_HEADS
    p = n * D
    conv_w = golden.canonical_conv_weight(cpu["conv1d_weight"], n, D)
    ref = golden.run_golden(
        cpu["hidden_states"], cpu["qkv_projection_weight"],
        cpu["decay_projection_a_weight"], cpu["decay_projection_b_weight"],
        cpu["beta_projection_weight"], cpu["output_gate_projection_weight"],
        cpu["output_norm_weight"], cpu["output_projection_weight"], conv_w,
        cpu["conv_state"].clone(), cpu["recurrent_state"], cpu["conv_state_indices"],
        cpu["conv_num_accepted_tokens"], cpu["a_log"], cpu["dt_bias"],
        128 ** -0.5, -5.0, 1e-5, num_heads=n)
    decay_a_ref = golden.linear_reference(
        cpu["hidden_states"], cpu["decay_projection_a_weight"], torch.bfloat16)
    raw_decay_ref = golden.linear_reference(
        decay_a_ref, cpu["decay_projection_b_weight"], torch.bfloat16)
    beta_logit = golden.linear_reference(
        cpu["hidden_states"], cpu["beta_projection_weight"], torch.bfloat16).reshape(t, n)
    kbw = 16 if n <= 16 else 32
    ws_beta_logit = beta_logit.new_zeros(t, kbw)
    ws_beta_logit[:, :n] = beta_logit
    ws_qkv = torch.cat((ref["query"].reshape(t, p), ref["key"].reshape(t, p),
                        ref["value"].reshape(t, p)), dim=-1)
    panel_gate = ref["gate"].reshape(t, p)
    return GoldenRefs(
        ref["raw_qkv"].reshape(t, 3 * p), panel_gate, ws_beta_logit,
        raw_decay_ref.reshape(t, p), decay_a_ref.reshape(t, D), ws_qkv, panel_gate,
        ref["conv_state"], ref["replay_k"], ref["gated"].reshape(t, p),
        ref["replay_u"], ref["replay_decay"], ref["output"])


def run_case(batch, cores=None, seed=20260921, conv_layout=1, num_heads=DEFAULT_HEADS):
    """Build inputs, run the public wrapper once, compare the API-visible outputs.

    The wrapper returns only ``out``; ``conv_state`` / ``replay_*`` are the caller's
    tensors updated in place, so they are compared against the golden directly.
    """
    torch.npu.set_device(_get_device())
    dev = _get_device()
    cpu = _make_inputs(batch, num_heads=num_heads, seed=seed, conv_layout=conv_layout)
    npu = {k: v.to(dev) for k, v in cpu.items()}
    for name in _NZ_WEIGHTS:
        npu[name] = torch_npu.npu_format_cast(npu[name], _NZ)
    rec_before = npu["recurrent_state"].clone()
    conv_before = npu["conv_state"].clone()
    label = label_of(batch, cores, conv_layout)

    golden = _eval_golden(_GOLDEN, cpu)

    old_grid = _M.select_grid
    if cores is not None:
        def _fixed_grid():
            return int(cores)

        _M.select_grid = _fixed_grid
    try:
        out = _M.mega_recurrent_kda_replayssm_wrapper(
            npu["hidden_states"], npu["qkv_projection_weight"],
            npu["decay_projection_a_weight"], npu["decay_projection_b_weight"],
            npu["beta_projection_weight"], npu["output_gate_projection_weight"],
            npu["output_norm_weight"], npu["output_projection_weight"],
            npu["conv1d_weight"], npu["conv_state"], npu["recurrent_state"],
            npu["replay_u"], npu["replay_k"], npu["replay_decay"],
            npu["conv_state_indices"], npu["conv_num_accepted_tokens"],
            npu["a_log"], npu["dt_bias"], 128 ** -0.5, -5.0, 1e-5)
    finally:
        _M.select_grid = old_grid
    torch.npu.synchronize()

    # recurrent_state is read-only (zero tolerance, as in the shipped harness)
    if not torch.equal(npu["recurrent_state"].cpu(), rec_before.cpu()):
        raise AssertionError(f"{label}: recurrent_state mutated")
    # conv_state slots not referenced by conv_state_indices must stay untouched
    used = {int(i) for i in npu["conv_state_indices"].cpu().tolist() if int(i) != 0}
    unused = [i for i in range(int(npu["conv_state"].shape[0])) if i not in used]
    if unused and not torch.equal(npu["conv_state"][unused].cpu(), conv_before[unused].cpu()):
        raise AssertionError(f"{label}: conv_state wrote {len(unused)} unused slot(s)")

    actual = (out, npu["conv_state"], npu["replay_k"], npu["replay_u"],
              npu["replay_decay"])
    golden_visible = (golden.output, golden.conv_state, golden.replay_k,
                      golden.replay_u, golden.replay_decay)
    return _assert_precision(actual, golden_visible, label=label)


def label_of(batch, cores, conv_layout):
    return f"delivery B={batch} cores={cores} conv_layout={conv_layout}"


@pytest.mark.soc("950")
def test_mega_recurrent_kda_replayssm_b16_auto():
    run_case(16, cores=None)


@pytest.mark.soc("950")
def test_mega_recurrent_kda_replayssm_b32_g28():
    run_case(32, cores=28)


@pytest.mark.soc("950")
def test_mega_recurrent_kda_replayssm_b48_g24():
    run_case(48, cores=24)


@pytest.mark.soc("950")
def test_mega_recurrent_kda_replayssm_b128_g28():
    run_case(128, cores=28)


@pytest.mark.soc("950")
def test_mega_recurrent_kda_replayssm_b512_g28():
    run_case(512, cores=28)


@pytest.mark.soc("950")
def test_mega_recurrent_kda_replayssm_b16_conv_layout0():
    run_case(16, cores=8, conv_layout=0)


@pytest.mark.soc("950")
def test_mega_recurrent_kda_replayssm_b16_conv_layout2():
    run_case(16, cores=8, conv_layout=2)


@pytest.mark.soc("950")
def test_mega_recurrent_kda_replayssm_n12_b16():
    run_case(16, cores=28, num_heads=12)


@pytest.mark.soc("950")
def test_mega_recurrent_kda_replayssm_n12_b256():
    run_case(256, cores=28, num_heads=12)


@pytest.mark.soc("950")
def test_mega_recurrent_kda_replayssm_n24_b16():
    run_case(16, cores=28, num_heads=24)


@pytest.mark.soc("950")
def test_mega_recurrent_kda_replayssm_n24_b128():
    run_case(128, cores=28, num_heads=24)


@pytest.mark.soc("950")
def test_mega_recurrent_kda_replayssm_n24_b4():
    run_case(4, cores=28, num_heads=24)


@pytest.mark.soc("950")
def test_mega_recurrent_kda_replayssm_n12_b8():
    run_case(8, cores=28, num_heads=12)


@pytest.mark.soc("950")
def test_mega_recurrent_kda_replayssm_n12_b64_g8():
    run_case(64, cores=8, num_heads=12)


@pytest.mark.soc("950")
def test_mega_recurrent_kda_replayssm_n24_b32_g8():
    run_case(32, cores=8, num_heads=24)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for batch, cores in ((16, None), (32, 28), (48, 24), (128, 28), (512, 28)):
        run_case(batch, cores=cores)
    run_case(16, cores=8, conv_layout=0)
    run_case(16, cores=8, conv_layout=2)
    # TP8 / TP4 (N = 12 / 24), including the small-batch m-tile rungs
    for batch, heads, cores in ((16, 12, 28), (8, 12, 28), (256, 12, 28),
                                (4, 24, 28), (16, 24, 28), (128, 24, 28)):
        run_case(batch, cores=cores, num_heads=heads)
    # 8 AIC grid: the m3_k=8 slot rule for N>=12
    run_case(64, cores=8, num_heads=12)
    run_case(32, cores=8, num_heads=24)
    LOGGER.info("ALL DELIVERY CASES PASS")


if __name__ == "__main__":
    main()
