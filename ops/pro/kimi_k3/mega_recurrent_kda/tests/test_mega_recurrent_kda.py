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
"""mega_recurrent_kda PyPTO-Pro tests and latency benchmark.

Each ``test_*`` builds one case's inputs on CPU, runs the public wrapper on the
NPU and compares the output, ``conv_state`` and ``recurrent_state`` against the
independent CPU golden with the workflow's Scheme-A checker.

Run: ``python test_mega_recurrent_kda.py`` or ``pytest test_mega_recurrent_kda.py``.
Bench: ``python test_mega_recurrent_kda.py --bench``.

The benchmark uses ``torch_npu.profiler`` and reads only the
``PYPTO_mega_recurrent_kda_kernel`` kernel duration (device time).  Do **not**
switch it back to ``torch.npu.Event``: event timing includes about 750-900 us of
host launch overhead per call.
"""

import logging
import os
import sys

import pytest
import torch
import torch_npu
from pypto_pro.runtime.platform import get_platform_info

# The AIC quota defaults to the largest supported quota the device can host;
# override with MEGA_KDA_CUBE_CORES (one of 8/16/24/28/32).

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_THIS_DIR)
for _path in (_THIS_DIR, os.path.join(_THIS_DIR, "assets"), os.path.join(_OP_DIR, "op_kernel")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import mega_recurrent_kda_golden_cpu as golden  # noqa: E402
from mega_recurrent_kda import (  # noqa: E402
    _SUPPORTED_CUBE_CORES, mega_recurrent_kda_pad_input, mega_recurrent_kda_wrapper)
from mega_recurrent_kda_precision import check_precision  # noqa: E402,F401

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOG = logging.getLogger(__name__)

_SCALE = 128 ** -0.5
_LOWER = -5.0
_RMS_EPS = 1e-5
# Argument indices (of the public 20-argument signature) that must be NZ.
NZ_WEIGHT_ARGUMENTS = (1, 2, 3, 4, 5, 7)

# (batch, num_heads) — TP16 -> 6 heads, TP8 -> 12, TP4 -> 24.
_CASES = {
    "tp16_b16": (16, 6),
    "tp16_b32": (32, 6),
    "tp16_b48": (48, 6),
    "tp4_short_b4": (4, 24),
    "tp4_tail_b12": (12, 24),
    "tp4_main_b32": (32, 24),
    "tp8_short_b8": (8, 12),
    "tp8_tail_b24": (24, 12),
    "tp8_main_b64": (64, 12),
}

# 12 representative cases: (label, batch, num_heads).
_TARGET_CASES = [
    ("TP4  mb1", 4, 24),
    ("TP4  mb8", 32, 24),
    ("TP4  mb16", 64, 24),
    ("TP4  mb32", 128, 24),
    ("TP8  mb1", 8, 12),
    ("TP8  mb8", 64, 12),
    ("TP8  mb16", 128, 12),
    ("TP8  mb32", 256, 12),
    ("TP16 mb1", 16, 6),
    ("TP16 mb8", 128, 6),
    ("TP16 mb16", 256, 6),
    ("TP16 mb32", 512, 6),
]

_KERNEL_TAG = "mega_recurrent_kda_kernel"
_L2_EVICT_BYTES = 224 * 1024 * 1024


def _run_case(name):
    batch, num_heads = _CASES[name]
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", 0)))
    arguments = golden.make_cpu_arguments(batch, 8, 731 + batch, num_heads)
    weights = golden.weights_from_arguments(arguments)
    initial_conv = arguments[9].clone()
    initial_rec = arguments[10].clone()
    expected = golden.run_mega_recurrent_kda_golden(
        arguments[0], weights, initial_conv.clone(), initial_rec.clone(),
        *arguments[11:15], _SCALE, _LOWER, _RMS_EPS)

    npu_arguments = [value.to("npu") for value in arguments]
    # The A operand is handed over already padded to the MTile grid (the wrapper
    # neither pads nor copies it).
    npu_arguments[0] = mega_recurrent_kda_pad_input(npu_arguments[0])
    # The six projection weights are handed over as FRACTAL_NZ allocation-root
    # tensors in the kernel's [K, N] orientation (arguments 1..5 and 7); no
    # transpose is done on the host side.
    for index in NZ_WEIGHT_ARGUMENTS:
        npu_arguments[index] = torch_npu.npu_format_cast(
            npu_arguments[index], torch_npu.Format.FRACTAL_NZ)
    actual = mega_recurrent_kda_wrapper(*npu_arguments, _SCALE, _LOWER, _RMS_EPS)
    torch.npu.synchronize()

    # SPEC §5: elementwise atol = rtol = 5e-3 on output, conv_state and recurrent_state.
    for label, got, want in (
        ("output", actual.cpu().float(), expected.output.float()),
        ("conv_state", npu_arguments[9].cpu().float(), expected.conv_state.float()),
        ("recurrent_state", npu_arguments[10].cpu().float(), expected.recurrent_state.float()),
    ):
        max_diff = (got - want).abs().max().item()
        _LOG.info("[%s] %-16s max_diff=%.3e", name, label, max_diff)
        torch.testing.assert_close(got, want, atol=5e-3, rtol=5e-3,
                                   msg=f"{name}: {label} exceeds SPEC tolerance")

    # The immutable null slot 0 must stay bit-exact.
    assert torch.equal(npu_arguments[9].cpu()[0], initial_conv[0]), "conv_state null slot changed"
    assert torch.equal(npu_arguments[10].cpu()[0], initial_rec[0]), "recurrent_state null slot changed"


@pytest.mark.soc("950")
@pytest.mark.parametrize("name", list(_CASES))
def test_precision(name):
    _run_case(name)


def test_rejects_unpadded_hidden():
    # 12 batches x S=8 = 96 rows, which needs 128 rows on the MTile grid; the
    # wrapper only accepts the caller-padded [128, 7168] operand.
    batch = 12
    args = [None] * 20
    args[0] = torch.empty((batch * 8, 7168), dtype=torch.bfloat16)
    args[11] = torch.ones(batch, dtype=torch.int32)
    args[17], args[18], args[19] = _SCALE, _LOWER, _RMS_EPS
    with pytest.raises(ValueError, match="padded A operand"):
        mega_recurrent_kda_wrapper(*args)


@pytest.mark.soc("950")
def test_rejects_unsupported_cores():
    max_cube = int(get_platform_info().core_num)
    old = os.environ.get("MEGA_KDA_CUBE_CORES")
    try:
        for bad in (12, max_cube + 1):
            os.environ["MEGA_KDA_CUBE_CORES"] = str(bad)
            with pytest.raises(ValueError, match="not supported|operator supports"):
                _run_case("tp4_short_b4")
    finally:
        if old is None:
            os.environ.pop("MEGA_KDA_CUBE_CORES", None)
        else:
            os.environ["MEGA_KDA_CUBE_CORES"] = old


def test_public_signature():
    import inspect

    assert list(inspect.signature(mega_recurrent_kda_wrapper).parameters) == [
        "hidden_states", "qkv_projection_weight", "decay_projection_a_weight",
        "decay_projection_b_weight", "beta_projection_weight", "output_gate_projection_weight",
        "output_norm_weight", "output_projection_weight", "conv1d_weight", "conv_state",
        "recurrent_state", "conv_state_indices", "ssm_state_indices", "conv_num_accepted_tokens",
        "num_accepted_tokens", "a_log", "dt_bias", "scale", "lower_bound", "rms_norm_eps",
    ]


def _device_times(prof_dir):
    """Kernel device durations (us) collected by the last profiler window."""
    import csv
    import glob

    files = glob.glob(os.path.join(prof_dir, "**", "kernel_details.csv"), recursive=True)
    out = []
    for path in files:
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if _KERNEL_TAG not in row.get("Name", ""):
                    continue
                try:
                    duration = float(row["Duration(us)"])
                except (KeyError, TypeError, ValueError):
                    continue
                if duration > 0:
                    out.append(duration)
    out.sort()
    return out


def _bench(batch, num_heads, iters=20, warmup=3):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    arguments = golden.make_cpu_arguments(batch, 8, 731 + batch, num_heads)
    npu_args = [value.to("npu") for value in arguments]
    npu_args[0] = mega_recurrent_kda_pad_input(npu_args[0])
    for index in NZ_WEIGHT_ARGUMENTS:
        npu_args[index] = torch_npu.npu_format_cast(npu_args[index], torch_npu.Format.FRACTAL_NZ)
    initial_conv = npu_args[9].clone()
    initial_rec = npu_args[10].clone()

    def call():
        return mega_recurrent_kda_wrapper(*npu_args, _SCALE, _LOWER, _RMS_EPS)

    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    evict = torch.ones(_L2_EVICT_BYTES // 4, dtype=torch.float32, device="npu")
    prof_dir = os.path.join(_THIS_DIR, "build", f"bench_prof_b{batch}_n{num_heads}")
    if os.path.isdir(prof_dir):
        import shutil
        shutil.rmtree(prof_dir)
    profiler = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=iters, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
        profile_memory=False, with_stack=False, with_modules=False,
    )
    with profiler:
        for _ in range(iters):
            npu_args[9].copy_(initial_conv)
            npu_args[10].copy_(initial_rec)
            evict.sum()
            call()
            profiler.step()
    torch.npu.synchronize()
    times = _device_times(prof_dir)
    if not times:
        raise RuntimeError("profiler produced no mega_recurrent_kda_kernel events")
    return sum(times) / len(times)


def bench_main():
    max_cube = int(get_platform_info().core_num)
    supported = tuple(c for c in _SUPPORTED_CUBE_CORES if c <= max_cube)
    if not supported:
        raise ValueError(
            f"device exposes {max_cube} AIC; supported quotas are {_SUPPORTED_CUBE_CORES}")
    cores = int(os.environ.get("MEGA_KDA_CUBE_CORES", supported[-1]))
    if cores not in supported:
        raise ValueError(
            f"MEGA_KDA_CUBE_CORES={cores} is not supported; choose one of {supported}")
    _LOG.info("AIC quota = %d (device max %d)", cores, max_cube)
    _LOG.info("%-11s %8s %8s %10s", "case", "B", "N", "ours(us)")
    for label, batch, num_heads in _TARGET_CASES:
        us = _bench(batch, num_heads)
        _LOG.info("%-11s %8d %8d %10.1f", label, batch, num_heads, us)


def main():
    for name in _CASES:
        _run_case(name)
    test_rejects_unpadded_hidden()
    test_rejects_unsupported_cores()
    test_public_signature()


if __name__ == "__main__":
    if "--bench" in sys.argv[1:]:
        raise SystemExit(bench_main())
    raise SystemExit(main())
