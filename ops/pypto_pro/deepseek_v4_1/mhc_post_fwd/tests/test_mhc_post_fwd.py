#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""mhc_post_fwd（手写 PyPTO-Pro 算子）的独立测试与性能入口。

本文件是算子唯一的测试入口，把「精度矩阵校验」与「性能采集」合并在一个可执行脚本里：

* **精度**：P0 = 代表测试矩阵 ``n0 in {1,2} x n1=4096 x h in {1280,2560,5120,7168}``
  （``n0``：batch、``n1``：sequence、``h``：hidden，``mhc=4`` 固定）。输入由
  ``assets/mhc_post_fwd_golden_cpu.py`` 的确定性构造器生成，在设备上调用公开 wrapper 后
  ``torch.npu.synchronize()``，再用 ``assets/mhc_post_fwd_precision.py`` 的 BF16 容差口径
  逐元素比对；同时用哨兵检查确认输出确实被写入（防空过）。
* **回归**：``test_multi_item_per_core`` 覆盖曾定位到的一个确定性错误——单核执行多于一个
  工作项时，若四个输出复用一个累加 tile 就会算错；该用例迫使每个核处理多个工作项。
* **性能**：**冷 L2** 口径（每次计时前用 256 MiB buffer 驱逐 L2，本仓 ``attn_epilogue``
  等算子采用同一口径），``torch.npu.Event`` 包一次调用，多轮取 min/中位。
  绝对性能见 ``RESULTS.md``。

用法：
  # 精度矩阵（默认）
  python test_mhc_post_fwd.py

  # 精度 + 性能
  python test_mhc_post_fwd.py --bench

  # 指定设备 / 子集 / 输出
  TILE_FWK_DEVICE_ID=1 python test_mhc_post_fwd.py --cases 1x4096x1280 --bench --out /tmp/mhc.json

  # pytest（14 个用例：8 个 P0 逐 case parametrize + 4 个窄尾块 + 2 个回归）
  pytest tests -q
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

import torch

import torch_npu  # noqa: F401

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "assets"))
sys.path.insert(0, str(_HERE.parent / "op_kernel"))

from mhc_post_fwd_golden_cpu import (  # noqa: E402
    MHC,
    P0_CASES,
    h_blk_reference,
    make_inputs,
    mhc_post_fwd_golden_cpu,
)
from mhc_post_fwd_precision import (  # noqa: E402
    assert_matches_reference,
    assert_written,
    max_abs_diff,
)
from mhc_post_fwd import mhc_post_fwd  # noqa: E402

import pytest  # noqa: E402

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.soc("950")

DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
# 逐 case 分发的确定性输入种子；允许用环境变量覆盖以便复现同一批输入。
SEED = int(os.environ.get("PYPTO_PERF_SEED", 42))
SENTINEL = -1.0
L2_FLUSH_MB = 256
_FLUSH = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _set_device():
    torch.npu.set_device(DEVICE)


def _flush_l2():
    """冷 L2：每次计时前写满一个 >L2(128 MiB) 的 buffer 驱逐工作集。"""
    global _FLUSH
    if L2_FLUSH_MB <= 0:
        return
    if _FLUSH is None:
        _FLUSH = torch.empty(L2_FLUSH_MB * 1024 * 1024, dtype=torch.int8, device=DEVICE)
    _FLUSH.zero_()


def _device_inputs(n0, n1, h, seed=None):
    """在设备上就绪的一批输入与输出张量（供精度与性能路径复用）。"""
    x, residual, post, comb = make_inputs(n0, n1, h, seed=SEED if seed is None else seed)
    x, residual, post, comb = (t.to(DEVICE) for t in (x, residual, post, comb))
    out = torch.empty((n0, n1, MHC, h), dtype=torch.bfloat16, device=DEVICE)
    return x, residual, post, comb, out


def run_case(n0, n1, h, seed=None, check=True):
    """一次公开接口调用 + 精度校验；返回 (ok, max_abs_diff)。"""
    x, residual, post, comb, out = _device_inputs(n0, n1, h, seed=seed)
    out.fill_(SENTINEL)
    got = mhc_post_fwd(x, residual, post, comb, out)
    torch.npu.synchronize()
    if not check:
        return True, float("nan")
    assert out.data_ptr() == got.data_ptr() or torch.equal(out, got), "wrapper must fill `out`"
    assert_written(out, SENTINEL)
    ref = mhc_post_fwd_golden_cpu(x, residual, post, comb)
    assert_matches_reference(out, ref, f"{n0}x{n1}x{h}")
    return True, max_abs_diff(out, ref)


def host_cost(n0, n1, h, iters=30):
    """wrapper 的纯 host 下发耗时（us）：排空队列后计时一次调用，取 min。

    与时延口径的区别：这里**不含设备执行时间**——入队是异步的，host 把任务塞进队列
    即返回。它回答的是“host 能不能喂饱设备”，与 kernel 的 device 时间是两个量纲，
    不可相加。
    """
    x, residual, post, comb, out = _device_inputs(n0, n1, h)

    def call():
        return mhc_post_fwd(x, residual, post, comb, out)

    for _ in range(5):
        call()
    torch.npu.synchronize()
    best = float("inf")
    for _ in range(iters):
        torch.npu.synchronize()
        start = time.perf_counter()
        call()
        best = min(best, (time.perf_counter() - start) * 1e6)
    torch.npu.synchronize()
    return best


def bench_case(n0, n1, h, iters=20, warmup=5):
    """冷 L2 逐调用时延（us），返回 (min, median, io_gb)。"""
    x, residual, post, comb, out = _device_inputs(n0, n1, h)

    def call():
        return mhc_post_fwd(x, residual, post, comb, out)

    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    costs = []
    for _ in range(iters):
        _flush_l2()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.npu.synchronize()
        costs.append(start.elapsed_time(end) * 1e3)
    io_gb = (x.numel() * 2 + residual.numel() * 2 + out.numel() * 2
             + post.numel() * 4 + comb.numel() * 4) / 1e9
    return min(costs), statistics.median(costs), io_gb


# ---------------------------------------------------------------------------
# pytest cases
# ---------------------------------------------------------------------------
def _case_ids():
    return [f"{n0}x{n1}x{h}" for (n0, n1, h) in P0_CASES]


#: 64 对齐但 ``h % h_blk != 0`` 的用例：``h_blk`` 公式给出非整除切分，最后一块变窄
#: （``valid`` 仍是 64 的倍数）。常量掩码路径必须在这类形状上同样正确。
NARROW_TAIL_CASES = [(1, 64, 4160), (1, 64, 4416), (1, 64, 8256), (1, 4096, 4160)]
NARROW_TAIL_IDS = [f"{n0}x{n1}x{h}" for (n0, n1, h) in NARROW_TAIL_CASES]


@pytest.mark.parametrize("n0,n1,h", P0_CASES, ids=_case_ids())
def test_precision_p0(n0, n1, h):
    """P0 矩阵逐 case 独立用例：任一形状失败只影响自己那一行。"""
    _set_device()
    ok, mad = run_case(n0, n1, h)
    assert ok, f"{n0}x{n1}x{h} max_abs_diff={mad:.3e}"


@pytest.mark.parametrize("n0,n1,h", NARROW_TAIL_CASES, ids=NARROW_TAIL_IDS)
def test_precision_narrow_tail(n0, n1, h):
    """``h % h_blk != 0`` 的窄尾块：验证「常量掩码已足够」的不变量（见 kernel docstring）。

    这些形状是 64 对齐的合法输入，但按 ``h_blk`` 公式切分后 ``h % h_blk != 0``；
    断言先证明用例确实落在该分支上，避免测试名不副实。
    """
    _set_device()
    h_blk = h_blk_reference(h)
    assert h % 64 == 0, f"{h} 应为 64 对齐"
    assert h % h_blk != 0, f"该用例应落在 h % h_blk != 0 分支，实际 h_blk={h_blk}"
    ok, mad = run_case(n0, n1, h)
    assert ok, f"{n0}x{n1}x{h} max_abs_diff={mad:.3e}"


def test_multi_item_per_core():
    """回归：单核处理 >1 工作项时复用累加 tile 会算错（本实现用 4 个独立累加寄存器）。

    ``h=128`` 时 ``h_blk=h``、每 token 一个工作项；取 n0*n1 = 512 > 64 个 vector core，
    迫使每个核跑多个工作项。
    """
    _set_device()
    for n1 in (128, 512):
        ok, mad = run_case(1, n1, 128)
        logger.info("[multi-item] 1x%d x128: PASS  max_abs_diff=%.3e", n1, mad)
        assert ok


def test_output_argument_filled():
    """传入 out 时必须在原地填充（不新分配）。"""
    _set_device()
    n0, n1, h = 1, 64, 1280
    x, residual, post, comb, out = _device_inputs(n0, n1, h)
    out.fill_(SENTINEL)
    got = mhc_post_fwd(x, residual, post, comb, out)
    torch.npu.synchronize()
    assert got.data_ptr() == out.data_ptr()
    assert_written(out, SENTINEL)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_cases(spec):
    """解析 ``--cases``（逗号分隔的 ``n0xn1xh``）；非法 id 抛 ``ValueError``。"""
    if not spec:
        return P0_CASES
    cases = []
    for item in spec.split(","):
        item = item.strip().lower()
        parts = item.split("x")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(f"unknown case id: {item!r} (expected n0xn1xh)")
        cases.append(tuple(int(v) for v in parts))
    return cases


def _run_precision(cases, report):
    """跑精度矩阵，返回失败个数。"""
    logger.info("== 精度（P0 矩阵；BF16 容差 v.s. CPU golden）==")
    n_fail = 0
    for (n0, n1, h) in cases:
        start = time.time()
        ok, mad = run_case(n0, n1, h)
        n_fail += 0 if ok else 1
        entry = report["cases"].setdefault(f"{n0}x{n1}x{h}", {})
        entry["max_abs_diff"] = mad
        entry["compile_s"] = round(time.time() - start, 2)
        entry["h_blk"] = h_blk_reference(h)
        logger.info("  %dx%dx%-5d: %s  max_abs_diff=%.3e  (h_blk=%d)",
                    n0, n1, h, "PASS" if ok else "FAIL", mad, h_blk_reference(h))
    return n_fail


def _run_bench(cases, report, iters):
    """冷 L2 逐调用时延。"""
    logger.info("== 性能（冷 L2；min/median over %d iters）==", iters)
    logger.info("%16s %10s %10s %8s", "case", "min_us", "med_us", "GB/s")
    for (n0, n1, h) in cases:
        low, mid, io_gb = bench_case(n0, n1, h, iters=iters)
        report["cases"].setdefault(f"{n0}x{n1}x{h}", {}).update(
            {"min_us": round(low, 2), "median_us": round(mid, 2),
             "GB/s": round(io_gb / low * 1e6, 1)})
        logger.info("%16s %10.2f %10.2f %8.1f",
                    f"{n0}x{n1}x{h}", low, mid, io_gb / low * 1e6)


def _run_host_cost(cases, report):
    """wrapper 的纯 host 下发耗时（排空队列，min）。"""
    logger.info("== wrapper host 下发耗时（排空队列，min）==")
    logger.info("%16s %12s", "case", "wrapper_us")
    for (n0, n1, h) in cases:
        cost = host_cost(n0, n1, h)
        report["cases"].setdefault(f"{n0}x{n1}x{h}", {})["wrapper_host_us"] = round(cost, 2)
        logger.info("%16s %12.1f", f"{n0}x{n1}x{h}", cost)
    logger.info("注：wrapper(host) 与 kernel(device) 是不同量纲；背靠背下发时每次有效耗时"
                "≈ max(wrapper, kernel)，逐调用同步时 = wrapper + kernel。")


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default="", help="逗号分隔的 n0xn1xh，默认全部 P0")
    parser.add_argument("--bench", action="store_true", help="额外采集冷 L2 逐调用时延")
    parser.add_argument("--host", action="store_true",
                        help="额外汇总 wrapper 的纯 host 下发耗时（排空队列，min）")
    parser.add_argument("--bench-iters", type=int, default=20)
    parser.add_argument("--skip-precision", action="store_true")
    parser.add_argument("--out", default="", help="把结果写成 JSON")
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args()

    cases = _parse_cases(args.cases)
    _set_device()
    if len(cases) == 1:
        # 单 case 分发时输出唯一选择器标记，便于外部逐 case 采集脚本绑定本次运行。
        n0, n1, h = cases[0]
        logger.info("PYPTO_PERF_SELECTED_CASE=%dx%dx%d", n0, n1, h)
    report = {"device": DEVICE, "l2_flush_mb": L2_FLUSH_MB, "cases": {}}
    logger.info("device %s   cases=%d   l2_flush=%dMiB", DEVICE, len(cases), L2_FLUSH_MB)

    n_fail = _run_precision(cases, report) if not args.skip_precision else 0
    if args.bench:
        _run_bench(cases, report, args.bench_iters)
    if args.host:
        _run_host_cost(cases, report)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        logger.info("wrote %s", args.out)
    logger.info("RESULT: %s", "FAIL" if n_fail else "PASS")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
