#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You should not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------

"""flash_kda_cp_preprocess test: one test function per case, plus a ``main`` entry.

Run: ``python test_flash_kda_cp_preprocess.py``  or  ``pytest tests -q``
"""

import logging
import os
import sys
from typing import NamedTuple

import torch
import torch_npu  # noqa: F401

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.join(_THIS_DIR, "assets"),
           os.path.dirname(_THIS_DIR) + "/op_kernel"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from flash_kda_cp_preprocess import (  # noqa: E402
    CHUNK, D, flash_kda_cp_preprocess, flash_kda_cp_preprocess_metadata)
from flash_kda_cp_preprocess_golden_cpu import (  # noqa: E402
    flash_kda_cp_preprocess_golden_cpu)
from flash_kda_cp_preprocess_precision import check_precision  # noqa: E402

LOGGER = logging.getLogger(__name__)

pytestmark = pytest.mark.soc("950")


def _get_device():
    if not (torch.npu.is_available() and torch.npu.device_count() > 0):
        raise RuntimeError("本用例集只在 NPU 上跑：未检测到可用设备")
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    return torch.device(f"npu:{device_id}")


class _CaseInputs(NamedTuple):
    """测试输入束。"""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    cu: object


def _make_case(device, layout, n_kv, n_v, seq_len, seed=53, boundaries=None):
    gen = torch.Generator().manual_seed(seed)
    if layout == "TND":
        total = sum(boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1))
        q = torch.randn(total, n_kv, D, generator=gen).bfloat16().to(device)
        k = torch.randn_like(q)
        v = torch.randn(total, n_v, D, generator=gen).bfloat16().to(device)
        g = torch.randn_like(v)
        beta = torch.randn(total, n_v, generator=gen).bfloat16().to(device)
        cu = torch.tensor(boundaries, dtype=torch.int32, device=device)
    else:
        if layout == "BNSD":
            q = torch.randn(1, n_kv, seq_len, D, generator=gen).bfloat16().to(device)
            v = torch.randn(1, n_v, seq_len, D, generator=gen).bfloat16().to(device)
            beta = torch.randn(1, n_v, seq_len, generator=gen).bfloat16().to(device)
        else:
            q = torch.randn(1, seq_len, n_kv, D, generator=gen).bfloat16().to(device)
            v = torch.randn(1, seq_len, n_v, D, generator=gen).bfloat16().to(device)
            beta = torch.randn(1, seq_len, n_v, generator=gen).bfloat16().to(device)
        k = torch.randn_like(q)
        g = torch.randn_like(v)
        cu = None
    a_log = torch.linspace(-1.0, 0.5, n_v, dtype=torch.float32, device=device)
    dt_bias = torch.randn(n_v, D, generator=gen).to(device)
    return _CaseInputs(q, k, v, g, beta, a_log, dt_bias, cu)


def _run_case(name, layout, n_kv, n_v, seq_len, boundaries=None, seed=53):
    device = _get_device()
    if layout == "TND" and boundaries is None:
        boundaries = [0, seq_len]
    q, k, v, g, beta, a_log, dt_bias, cu = _make_case(
        device, layout, n_kv, n_v, seq_len, seed=seed, boundaries=boundaries)
    metadata = flash_kda_cp_preprocess_metadata(q, v, layout, cu)
    outputs = flash_kda_cp_preprocess(
        q, k, v, g, beta, 128 ** -0.5, a_log, dt_bias, -0.1, layout, metadata, cu)
    args = [q, k, v, g, beta, a_log, dt_bias,
            torch.zeros(16, dtype=torch.int32)]
    kwargs = {"scale": 128 ** -0.5, "lower_bound": -0.1, "layout_qkv": layout}
    valid_chunks = None
    if cu is not None:
        valid_chunks = [(boundaries[i + 1] - boundaries[i] + CHUNK - 1) // CHUNK
                        for i in range(len(boundaries) - 1)]
        kwargs["cu_seqlens"] = cu.cpu()
    _assert_precision8(outputs, args, kwargs, label=name,
                       decay_valid_chunks=valid_chunks)
    LOGGER.info("[%s] PASS", name)


def _assert_precision8(outputs, args, kwargs, label="", decay_valid_chunks=None):
    """8 输出双重判据（precision_compare + 合同 99.9 分位）。"""
    inputs_cpu = [a.cpu() if hasattr(a, "cpu") else a for a in args]
    g_outs = flash_kda_cp_preprocess_golden_cpu(*inputs_cpu, **kwargs)
    names = ["transition", "state_ext", "decay", "qg", "kg", "u", "w", "mqk"]
    actuals = list(outputs)
    goldens = list(g_outs)
    for actual, golden in zip(actuals, goldens, strict=True):
        assert actual.shape == golden.shape and actual.dtype == golden.dtype
        assert actual.is_contiguous()
    if decay_valid_chunks is not None:
        for b, valid in enumerate(decay_valid_chunks):
            assert torch.count_nonzero(outputs[2][b, :, valid:].cpu()) == 0
        actuals[2] = torch.cat([outputs[2][b, :, :n] for b, n in enumerate(decay_valid_chunks)], dim=1)
        goldens[2] = torch.cat([g_outs[2][b, :, :n] for b, n in enumerate(decay_valid_chunks)], dim=1)
        actuals[7] = torch.cat([outputs[7][b, :, :n] for b, n in enumerate(decay_valid_chunks)], dim=1)
        goldens[7] = torch.cat([g_outs[7][b, :, :n] for b, n in enumerate(decay_valid_chunks)], dim=1)
    failures = []
    for name, actual, golden in zip(names, actuals, goldens):
        actual_c = actual.cpu()
        if not torch.isfinite(actual_c.float()).all():
            failures.append(f"{name}: 输出含 NaN/Inf")
            continue
        error = (actual_c.float() - golden.float()).abs()
        tolerance = 0.005 + 0.005 * golden.float().abs()
        ratio = (error / tolerance).flatten()
        if ratio.numel() > 10_000_000:
            stride = ratio.numel() // 10_000_000 + 1
            ratio = ratio[::stride]
        q999 = ratio.quantile(0.999).item()
        if name in ("transition", "state_ext"):
            if q999 > 1.0:
                failures.append(f"{name}: contract 99.9pct={q999:.4f} > 1")
            else:
                logging.info("[%s] %s PASS (contract q999=%.4f)", label, name, q999)
        else:
            passed, summary = check_precision(actual_c, golden)
            if not passed:
                failures.append(f"{name}: {summary}")
            elif q999 > 1.0:
                failures.append(f"{name}: contract 99.9pct={q999:.4f} > 1")
            else:
                logging.info("[%s] %s PASS (%s; contract q999=%.4f)", label, name,
                             summary, q999)
    if failures:
        raise AssertionError("精度不达标: " + "; ".join(failures))


def test_aligned():
    _run_case("aligned_bnsd_1_1_64", "BNSD", 1, 1, 64)


def test_tail():
    _run_case("tail_bnsd_1_3_17", "BNSD", 1, 3, 17)


def test_tail_gqa_two_chunks():
    _run_case("tail_gqa_bnsd_2_4_65", "BNSD", 2, 4, 65)


def test_bsnd():
    _run_case("bsnd_1_5_49", "BSND", 1, 5, 49)


def test_bsnd_wide_heads():
    # nv=48>32 走 32 头跨步窗口 β 路径（末窗 16 头对齐尾）
    _run_case("bsnd_wide_1_48_65", "BSND", 1, 48, 65)


def test_single_token():
    _run_case("single_token_s1", "BNSD", 1, 2, 1, seed=59)


def test_tnd():
    _run_case("tnd_3_6_193", "TND", 3, 6, 193)


def test_tnd_wide_heads():
    # TND 多块，每个 Q/K 头对应 20 个 V 头。β 使用跨步窗口。
    _run_case("tnd_wide_2_40_193", "TND", 2, 40, 193)


def test_wide96_heads():
    # BNSD 宽头双块，第二块只有一行有效输入。
    _run_case("wide96_bnsd_3_96_65", "BNSD", 3, 96, 65)


def test_wide96_single_chunk_bnsd():
    _run_case("wide96_single_chunk_bnsd", "BNSD", 3, 96, 64)


def test_wide96_single_chunk_bsnd():
    _run_case("wide96_single_chunk_bsnd", "BSND", 3, 96, 64)


def test_wide96_single_chunk_tnd():
    _run_case("wide96_single_chunk_tnd", "TND", 3, 96, 64, boundaries=[0, 64])


def test_packed_tnd():
    _run_case("packed_tnd_17_65", "TND", 1, 3, 82, boundaries=[0, 17, 82], seed=58)


def test_perf_p0():
    _run_case("perf_bnsd_32_8192", "BNSD", 32, 32, 8192)


def test_perf_bsnd_p0():
    _run_case("perf_bsnd_32_8192", "BSND", 32, 32, 8192)


def test_perf_tnd_p0():
    _run_case("perf_tnd_32_8192", "TND", 32, 32, 8192)


def test_perf_bsnd_6_16384():
    # 少头长序，Stage 2 将每条链分成四个行片。
    _run_case("perf_bsnd_6_16384", "BSND", 6, 6, 16384)


def test_perf_bsnd_12_16384():
    # 少头长序，Stage 2 将每条链分成两个行片。
    _run_case("perf_bsnd_12_16384", "BSND", 12, 12, 16384)


def test_perf_bsnd_96_1024():
    # 覆盖 96 个头的跨步 β 读取和多块摘要。
    _run_case("perf_bsnd_96_1024", "BSND", 96, 96, 1024)


def test_readme_bnsd_5_577():
    _run_case("readme_bnsd_5_577", "BNSD", 5, 5, 577, seed=42)


def test_readme_bnsd_1_1024():
    _run_case("readme_bnsd_1_1024", "BNSD", 1, 1, 1024, seed=42)


def _repeat_summary(layout, n_kv, n_v, seq_len, repeats):
    device = _get_device()
    boundaries = [0, seq_len] if layout == "TND" else None
    seed = 42 if n_kv == n_v else 53
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        cpu = _make_case(torch.device("cpu"), layout, n_kv, n_v, seq_len,
                         seed=seed, boundaries=boundaries)
    scale = D ** -0.5
    golden = flash_kda_cp_preprocess_golden_cpu(
        *cpu[:-1], torch.zeros(16, dtype=torch.int32), scale=scale,
        lower_bound=-0.1, layout_qkv=layout, cu_seqlens=cpu.cu)
    q, k, v, g, beta, a_log, dt_bias, cu = (
        x.to(device) if x is not None else None for x in cpu)
    metadata = flash_kda_cp_preprocess_metadata(q, v, layout, cu)
    first = None
    for launch in range(repeats + 1):
        outputs = flash_kda_cp_preprocess(
            q, k, v, g, beta, scale, a_log, dt_bias, -0.1, layout, metadata, cu)
        torch.npu.synchronize()
        summaries = tuple(x.cpu() for x in outputs[:2])
        for name, actual, expected in zip(("transition", "state_ext"), summaries, golden[:2]):
            ratio = (actual - expected).abs() / (0.005 + 0.005 * expected.abs())
            assert torch.isfinite(ratio).all(), f"{name} launch {launch}: NaN/Inf"
            q999 = ratio.flatten().quantile(0.999).item()
            assert q999 <= 1, f"{name} launch {launch}: q999={q999:.4f}"
        if first is None:
            first = summaries
        else:
            assert all(torch.equal(a, b) for a, b in zip(summaries, first)), (
                f"{layout} launch {launch}: summary changed for identical inputs")


def test_repeat_bnsd_5_5_577():
    _repeat_summary('BNSD', 5, 5, 577, 12)


def test_repeat_bnsd_2_4_65():
    _repeat_summary('BNSD', 2, 4, 65, 12)


def test_repeat_bnsd_8_16_193():
    _repeat_summary('BNSD', 8, 16, 193, 12)


def test_repeat_tnd_3_6_193():
    _repeat_summary('TND', 3, 6, 193, 12)


def test_repeat_bsnd_6_6_16384():
    _repeat_summary('BSND', 6, 6, 16384, 60)


def test_repeat_bsnd_12_12_16384():
    _repeat_summary('BSND', 12, 12, 16384, 30)


def test_repeat_bnsd_32_32_8192():
    _repeat_summary('BNSD', 32, 32, 8192, 12)


def test_repeat_tnd_32_32_8192():
    _repeat_summary('TND', 32, 32, 8192, 12)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_aligned()
    test_tail()
    test_tail_gqa_two_chunks()
    test_bsnd()
    test_bsnd_wide_heads()
    test_single_token()
    test_tnd()
    test_tnd_wide_heads()
    test_wide96_heads()
    test_wide96_single_chunk_bnsd()
    test_wide96_single_chunk_bsnd()
    test_wide96_single_chunk_tnd()
    test_packed_tnd()
    test_perf_p0()
    test_perf_bsnd_p0()
    test_perf_tnd_p0()
    test_perf_bsnd_6_16384()
    test_perf_bsnd_12_16384()
    test_perf_bsnd_96_1024()
    test_readme_bnsd_5_577()
    test_readme_bnsd_1_1024()
    test_repeat_bnsd_5_5_577()
    test_repeat_bnsd_2_4_65()
    test_repeat_bnsd_8_16_193()
    test_repeat_tnd_3_6_193()
    test_repeat_bsnd_6_6_16384()
    test_repeat_bsnd_12_12_16384()
    test_repeat_bnsd_32_32_8192()
    test_repeat_tnd_32_32_8192()
    LOGGER.info("all cases PASS")


if __name__ == "__main__":
    main()
