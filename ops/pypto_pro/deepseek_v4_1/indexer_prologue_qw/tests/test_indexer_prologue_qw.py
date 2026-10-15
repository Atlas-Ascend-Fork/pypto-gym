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
"""indexer_prologue_qw（手写 PyPTO-Pro 算子）的独立测试与性能入口。

本文件是整个算子唯一的测试入口，把「精度矩阵校验」与「性能采集」合并在一个
可执行脚本里，针对 SPEC 的全部 21 个 P0 ``T``：

* **精度**：用 CPU golden（``indexer_prologue_qw_golden_cpu.py``）自带的输入构造器
  造输入，在设备上调用公开 wrapper 并 ``torch.npu.synchronize()``，再用仓库内自包含的
  比较器 ``indexer_prologue_qw_precision.py`` 逐行比对（E2M1 同号单档移动 / 有符号零；
  E8M0 在 ``6*2^k`` 边界上 ±1；``w`` 用 ``rtol=atol=2e-2``）。大 ``T`` 的 CPU
  侧按 ``--chunk-rows`` 分块，设备 kernel 仍看完整 ``T``。比较所需量化前
  ``q_fp32`` 由 CPU golden 的可选调试返回提供，不依赖任何外部文件。
* **性能**：按冻结口径采集时延——NPU-graph 逐调用（去掉 host 启动开销、含图内设备
  调度，与 SPEC ``exit_criteria`` / cannbotDSL 参考同口径），并附带 eager 逐调用
  对照。默认只测 4 个目标 case ``{72,2048,4096,262144}``；``--perf all`` 覆盖全部 21 个。

除命令行入口外，本文件同时是 pytest 用例：``test_precision_p0`` 覆盖全 21 个 P0
``T``，另有额外 ``x`` 行的死区契约、可选 ``softmax_scale`` 默认路径，以及性能 case
时延记录。

用法：
  # 全量精度矩阵 + 四个目标 case 性能（默认）
  python test_indexer_prologue_qw.py

  # 只跑精度子集
  python test_indexer_prologue_qw.py --t 1,72,2048 --skip-perf

  # 全 21 个 T 的精度 + 性能
  python test_indexer_prologue_qw.py --perf all

  # 指定设备与输出
  TILE_FWK_DEVICE_ID=1 python test_indexer_prologue_qw.py --out /tmp/ipqw.json

  # pytest
  pytest ops/pypto_pro/deepseek_v4_1/indexer_prologue_qw/tests -q

返回码：全部精度通过为 0，否则 1。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback

import pytest
import torch

import torch_npu  # noqa: F401

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_THIS_DIR)
for _path in (_THIS_DIR, os.path.join(_THIS_DIR, "assets"),
              os.path.join(_OP_DIR, "op_kernel")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import indexer_prologue_qw_golden_cpu as golden  # noqa: E402
import indexer_prologue_qw_precision as verify  # noqa: E402
from indexer_prologue_qw import SOFTMAX_SCALE, indexer_prologue_qw  # noqa: E402

P0_CASES = golden.P0_CASES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
#: 公开 wrapper 的位置参数顺序（与比较器 bundle 的字段顺序一致）。
_NAMES = verify.POSITIONAL_NAMES
#: 必须为 FRACTAL_NZ 的两个权重在 ``_NAMES`` 中的下标。
_NZ_INDICES = (2, 3)
#: torch_npu.Format.FRACTAL_NZ 的格式码。
_FRACTAL_NZ = 29
#: 默认的性能目标 case（SPEC ``optimization_target``）。
_PERF_CASES = (72, 2048, 4096, 262144)
#: case 名 -> P0 下标。
_INDEX_OF = {t: i for i, (_, t) in enumerate(P0_CASES)}
#: 精度比较的默认行分块。
_CHUNK_ROWS = 1024

#: 仓库 conftest 按 soc 标记过滤用例；本算子面向 A5（950）。
pytestmark = pytest.mark.soc("950")


# ---------------------------------------------------------------------------
# 模块/输入加载
# ---------------------------------------------------------------------------
def _device():
    """按 ``TILE_FWK_DEVICE_ID`` 返回（并设置）目标 NPU 设备。

    参数：无。

    数据流：读取环境变量，调用 ``torch.npu.set_device``，返回 ``torch.device``。
    """
    device = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}")
    torch.npu.set_device(device)
    return device


def _load_references():
    """返回（仓库内自包含的）比较器与 CPU golden 模块。

    参数：无。

    数据流：返回 ``(verify, golden)`` 两个已导入模块，供精度与性能流程复用。
    """
    return verify, golden


def _as_e4m3(args):
    """把 golden_cpu 生成的 uint8 ``qr``/``wqb`` 位型重解释为 E4M3。

    职责：``golden_cpu`` 用 uint8 字节构造 ``qr``/``wqb``，而比较器需要 E4M3
    位视图；这里只对下标 1/2 做 ``.view``。

    参数：``args`` 为 8 元素张量列表。

    数据流：返回浅拷贝列表，原列表不变。
    """
    args = list(args)
    args[1] = args[1].view(torch.float8_e4m3fn)
    args[2] = args[2].view(torch.float8_e4m3fn)
    return args


def _inputs_as_dataclass(args, kwargs, x=None):
    """把位置张量包装成比较器的 ``IndexerPrologueQwInputs``。

    职责：按 ``_NAMES`` 顺序把 8 个张量与 ``softmax_scale`` 组成 dataclass，便于
    ``verify.select_rows`` 按行取子集。

    参数：``args`` 8 个张量；``kwargs`` 含 ``softmax_scale``；``x`` 可选覆盖。

    数据流：返回 dataclass 实例。
    """
    tensors = dict(zip(_NAMES, args))
    if x is not None:
        tensors["x"] = x
    return verify.IndexerPrologueQwInputs(softmax_scale=kwargs["softmax_scale"], **tensors)


def _to_nz(tensor):
    """把设备权重转成 FRACTAL_NZ 格式。

    职责：按 SPEC，``wqb``/``ww`` 必须为 FRACTAL_NZ，这里对 contiguous 张量做
    ``npu_format_cast``（格式码 29）。

    参数：``tensor`` 设备上的 ND 张量。

    数据流：返回 NZ 张量，供算子输入。
    """
    return torch_npu.npu_format_cast(tensor.contiguous(), _FRACTAL_NZ)


def _make_case_cpu(t, index):
    """按 case index 用固定种子构造该 ``T`` 的 CPU 输入。

    职责：调 ``golden.make_case``（种子 ``index+1``）得到与 ``_make_inputs`` 一致的
    8 个张量与 ``softmax_scale``。

    参数：``t`` 本 case 的 ``T``；``index`` P0 下标。

    数据流：返回 ``(cpu_args, kwargs)``，CPU 侧只构造一次，供精度/性能复用。
    """
    return golden.make_case(t, torch.device("cpu"), seed=index + 1)


def _to_device(cpu_args, device):
    """把 CPU 输入搬到目标设备，并把两个权重转为 FRACTAL_NZ。

    职责：对每个张量 ``contiguous().to(device)``，再对 ``wqb``/``ww`` 调 ``_to_nz``。

    参数：``cpu_args`` CPU 张量列表；``device`` 目标设备。

    数据流：返回设备张量列表，顺序与 ``_NAMES`` 一致。
    """
    dev = [a.contiguous().to(device) for a in cpu_args]
    for i in _NZ_INDICES:
        dev[i] = _to_nz(dev[i])
    return dev


# ---------------------------------------------------------------------------
# 精度
# ---------------------------------------------------------------------------
def _check_case(verify_mod, golden_mod, t, index, device, chunk_rows):
    """构造输入、设备上跑一次完整 ``T``、与 CPU golden 逐块比较全部行。

    职责：按 case index 造输入并搬到设备，调用算子并同步，再按 ``chunk_rows`` 分块
    用仓库内比较器统计差异；比较步骤的数值断言由比较器抛出。

    参数：``verify_mod``/``golden_mod`` 比较器与 CPU golden；``t`` 本 case 的 ``T``；
    ``index`` P0 下标；``device`` 设备；``chunk_rows`` 每次比较的行数上限。

    数据流：返回 ``(raw_q, raw_scales, max_w)``——被豁免的 q/scale 边界字节数与
    ``w`` 的最大绝对误差。
    """
    cpu_args, kwargs = _make_case_cpu(t, index)
    e4_args = _as_e4m3(cpu_args)
    inputs = _inputs_as_dataclass(e4_args, kwargs)
    dev = _to_device(e4_args, device)
    got = indexer_prologue_qw(*dev, softmax_scale=kwargs["softmax_scale"])
    torch.npu.synchronize()
    got_cpu = tuple(o.cpu() for o in got)

    raw_q = raw_scales = 0
    max_w = 0.0
    for start in range(0, t, chunk_rows):
        stop = min(t, start + chunk_rows)
        subset = verify_mod.select_rows(inputs, slice(start, stop))
        want = verify_mod.indexer_prologue_qw_golden(subset)
        metrics = verify_mod.compare(tuple(o[start:stop] for o in got_cpu), want, subset)
        raw_q += metrics[0]
        raw_scales += metrics[1]
        max_w = max(max_w, metrics[2])
    return raw_q, raw_scales, max_w


# ---------------------------------------------------------------------------
# 性能
# ---------------------------------------------------------------------------
def _time_eager(dev, scale, repeats, rounds):
    """eager 逐调用 NPU-event 时延（含 host 启动开销）。

    职责：warmup 3 次后，每轮连续调用 ``repeats`` 次并记录整轮耗时，重复 ``rounds``
    轮后升序返回。

    参数：``dev`` 设备输入；``scale`` ``softmax_scale``；``repeats``/``rounds`` 计时参数。

    数据流：返回升序的每调用时延列表（µs）。
    """
    for _ in range(3):
        indexer_prologue_qw(*dev, softmax_scale=scale)
    torch.npu.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            indexer_prologue_qw(*dev, softmax_scale=scale)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repeats)
    samples.sort()
    return samples


def _time_graph(dev, scale, repeats, rounds, warmup=5):
    """NPU-graph 逐调用时延：SPEC ``exit_criteria`` 采用的冻结口径。

    职责：把 ``repeats`` 次算子调用捕获进一张 ``NPUGraph``，replay 后按 NPU Event
    计时；排除 host 启动开销、包含图内设备调度，与 cannbotDSL 参考同口径。

    参数：``dev`` 设备输入；``scale`` ``softmax_scale``；``repeats`` 图中的调用次数；
    ``rounds`` 计时轮数；``warmup`` replay 前的预热次数。

    数据流：返回升序的每调用时延列表（µs）。
    """
    for _ in range(warmup):
        indexer_prologue_qw(*dev, softmax_scale=scale)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(repeats):
            indexer_prologue_qw(*dev, softmax_scale=scale)
    graph.replay()
    torch.npu.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repeats)
    samples.sort()
    return samples


def _stats(samples):
    """把升序时延列表折叠成 ``min/median/max``。

    参数：``samples`` 升序列表。

    数据流：返回 ``{"min_us","median_us","max_us"}``。
    """
    return {
        "min_us": samples[0],
        "median_us": samples[len(samples) // 2],
        "max_us": samples[-1],
    }


def _measure_case(verify_mod, golden_mod, t, index, device, eager_args, graph_args):
    """采集单个 ``T`` 的 eager 与 NPU-graph 时延。

    职责：构造该 ``T`` 的设备输入，分别按 eager/graph 参数采样并折叠成统计。

    参数：``verify_mod``/``golden_mod``；``t``/``index``；``device``；``eager_args``/
    ``graph_args`` 为 ``(repeats, rounds)``。

    数据流：返回 ``{"graph": stats, "eager": stats}``。
    """
    cpu_args, kwargs = _make_case_cpu(t, index)
    dev = _to_device(_as_e4m3(cpu_args), device)
    scale = kwargs["softmax_scale"]
    graph = _stats(_time_graph(dev, scale, graph_args[0], graph_args[1]))
    eager = _stats(_time_eager(dev, scale, eager_args[0], eager_args[1]))
    return {"graph": graph, "eager": eager}


def _parse_ts(text):
    """把逗号分隔的 ``T`` 列表解析成 int 列表（容忍空格）。

    参数：``text`` 形如 ``"1, 72,2048"``。

    数据流：返回 ``[int, ...]``，供 ``main`` 过滤 P0 case；空串返回 ``[]``。
    """
    if not text:
        return []
    return [int(v) for v in text.replace(" ", "").split(",") if v]


# ---------------------------------------------------------------------------
# pytest 用例
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def npu_device():
    """会话级 NPU 设备 fixture；无可用设备时跳过整组用例。"""
    device = None
    try:
        device = _device()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"NPU unavailable: {type(exc).__name__}: {exc}")
    return device


@pytest.mark.parametrize(("case_name", "t"), P0_CASES, ids=[n for n, _ in P0_CASES])
def test_precision_p0(npu_device, case_name, t):
    """全部 21 个 P0 ``T`` 的精度矩阵（仓库内比较器，无外部路径）。"""
    del case_name
    _check_one(npu_device, t)


def _check_one(device, t):
    """单个 P0 ``T`` 的精度比较；供下面每个显式 case 入口复用。

    参数：``device`` 目标 NPU；``t`` P0 的 ``T``。

    数据流：调 ``_check_case`` 建输入→上板→与 CPU golden 比较，断言通过。
    """
    raw_q, raw_scales, max_w = _check_case(
        verify, golden, t, _INDEX_OF[t], device, _CHUNK_ROWS
    )
    assert raw_q >= 0 and raw_scales >= 0
    assert math.isfinite(max_w)


# 代表性 case 的显式入口（覆盖：最小 T / decode 小 T / split-K W 切分边界 /
# 中等 T / split-T 边界 / 最大 T），与 #800/#811 的「每 case 一个显式 test_*」一致。
@pytest.mark.soc("950")
def test_precision_t1(npu_device):
    """P0 ``T=1``（最小 T，尾块/启动开销边界）。"""
    _check_one(npu_device, 1)


@pytest.mark.soc("950")
def test_precision_t72(npu_device):
    """P0 ``T=72``（decode 小 T）。"""
    _check_one(npu_device, 72)


@pytest.mark.soc("950")
def test_precision_t192(npu_device):
    """P0 ``T=192``（split-K 的 W-K 切分窗口边界）。"""
    _check_one(npu_device, 192)


@pytest.mark.soc("950")
def test_precision_t2048(npu_device):
    """P0 ``T=2048``（中等 T）。"""
    _check_one(npu_device, 2048)


@pytest.mark.soc("950")
def test_precision_t4096(npu_device):
    """P0 ``T=4096``（split-K/split-T 模板边界，n_tiles=32）。"""
    _check_one(npu_device, 4096)


@pytest.mark.soc("950")
def test_precision_t262144(npu_device):
    """P0 ``T=262144``（最大 T，带宽受限区间）。"""
    _check_one(npu_device, 262144)


def test_extra_x_rows_ignored(npu_device):
    """``x`` 行数可大于 ``T``，多出的行为死区：输出与精确 ``T`` 逐位一致。"""
    t = 6
    index = _INDEX_OF[t]
    cpu_args, kwargs = _make_case_cpu(t, index)
    scale = kwargs["softmax_scale"]

    base = _to_device(_as_e4m3(cpu_args), npu_device)
    with_extra = list(base)
    extra_x = torch.randn((37, base[0].shape[1]), dtype=torch.float32)
    with_extra[0] = torch.cat(
        [cpu_args[0].contiguous().to(npu_device), extra_x.to(npu_device).to(torch.bfloat16)],
        dim=0,
    )

    base_out = [o.cpu() for o in indexer_prologue_qw(*base, softmax_scale=scale)]
    pad_out = [o.cpu() for o in indexer_prologue_qw(*with_extra, softmax_scale=scale)]
    torch.npu.synchronize()

    assert pad_out[0].shape == (t, base_out[0].shape[1], base_out[0].shape[2])
    for name, want, got in zip(("q", "descale_q", "w"), base_out, pad_out):
        assert torch.equal(want, got), f"extra-x-rows changed {name}"


def test_optional_softmax_scale_default(npu_device):
    """可选 ``softmax_scale``：省略时取默认 FP32 冻结值，与显式传入逐位一致。"""
    t = 6
    cpu_args, _ = _make_case_cpu(t, _INDEX_OF[t])
    dev = _to_device(_as_e4m3(cpu_args), npu_device)

    explicit = [o.cpu() for o in indexer_prologue_qw(*dev, softmax_scale=SOFTMAX_SCALE)]
    defaulted = [o.cpu() for o in indexer_prologue_qw(*dev)]
    torch.npu.synchronize()

    for name, want, got in zip(("q", "descale_q", "w"), explicit, defaulted):
        assert torch.equal(want, got), f"default softmax_scale changed {name}"


def test_golden_optional_q_fp32_return():
    """CPU golden 的可选调试返回：默认 3 输出，``return_q_fp32=True`` 时 4 输出。"""
    t = 4
    cpu_args, kwargs = _make_case_cpu(t, _INDEX_OF[t])
    args = _as_e4m3(cpu_args)
    q, descale_q, w = golden.indexer_prologue_qw_golden_cpu(*args, **kwargs)
    q2, descale_q2, w2, q_fp32 = golden.indexer_prologue_qw_golden_cpu(
        *args, **kwargs, return_q_fp32=True
    )
    assert q_fp32.shape == (t, golden.N_HEADS, golden.D)
    assert q_fp32.dtype == torch.float32
    assert torch.equal(q, q2) and torch.equal(descale_q, descale_q2) and torch.equal(w, w2)


def test_perf_cases(npu_device):
    """记录 4 个目标 case 的 NPU-graph / eager 逐调用时延。"""
    eager_args = (50, 5)
    graph_args = (100, 10)
    measured = {}
    for t in _PERF_CASES:
        perf = _measure_case(
            verify, golden, t, _INDEX_OF[t], npu_device, eager_args, graph_args
        )
        measured[t] = perf
        logger.info(
            "[perf] T=%s graph_median=%.2fus min=%.2f max=%.2f eager_median=%.2fus",
            t, perf["graph"]["median_us"], perf["graph"]["min_us"],
            perf["graph"]["max_us"], perf["eager"]["median_us"],
        )
        assert perf["graph"]["median_us"] > 0
    assert set(measured) == set(_PERF_CASES)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    """命令行入口：跑精度矩阵与（可选）性能采集，并持续写 JSON 报告。

    职责：解析 ``--t``/``--chunk-rows``/``--perf``/``--skip-perf``/``--out`` 等；加载
    比较器与 CPU golden，选定 case；逐 case 校验精度并写报告，再按 ``--perf`` 采集时延
    （``targets`` 仅四个目标 case，``all`` 覆盖全部 P0）。

    参数：无（从 ``sys.argv`` 解析）。

    数据流：返回 0 表示全部精度通过，否则 1；过程中持续写 ``args.out``。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t", default=None, help="逗号分隔的 T 子集（默认全部 21 个 P0）")
    parser.add_argument("--chunk-rows", type=int, default=_CHUNK_ROWS,
                        help="CPU golden 每次比较的行数上限")
    parser.add_argument("--perf", choices=("targets", "all", "none"), default="targets",
                        help="性能采集范围：targets(默认四个目标 case)/all(全 21)/none")
    parser.add_argument("--skip-perf", action="store_true", help="等价于 --perf none")
    parser.add_argument("--graph-repeats", type=int, default=100)
    parser.add_argument("--graph-rounds", type=int, default=10)
    parser.add_argument("--eager-repeats", type=int, default=50)
    parser.add_argument("--eager-rounds", type=int, default=5)
    parser.add_argument("--out", default=os.path.join(_THIS_DIR, "reports", "pro_pypto",
                                                      "validation.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    device = _device()
    verify_mod, golden_mod = _load_references()

    p0 = list(golden_mod.P0_CASES)
    index_of = {t: i for i, (_, t) in enumerate(p0)}
    if args.t is None:
        selected = list(p0)
    else:
        wanted = set(_parse_ts(args.t))
        selected = [(n, t) for (n, t) in p0 if t in wanted]

    report = {
        "operator": "indexer_prologue_qw",
        "device": str(torch.npu.get_device_properties(device)),
        "softmax_scale": SOFTMAX_SCALE,
        "protocol": {
            "graph": f"NPU-graph per-call event time, median of {args.graph_rounds} "
                     f"rounds x {args.graph_repeats} calls (frozen)",
            "eager": f"NPU-event per-call (eager), median of {args.eager_rounds} "
                     f"rounds x {args.eager_repeats} calls",
        },
        "precision": [],
        "performance": [],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def save():
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")

    save()

    # ---- 精度矩阵 ----
    all_pass = True
    for name, t in selected:
        record = {"case": name, "T": t}
        started = time.time()
        try:
            raw_q, raw_scales, max_w = _check_case(
                verify_mod, golden_mod, t, index_of[t], device, args.chunk_rows
            )
            record.update(passed=True, q_boundary_bytes=raw_q,
                          scale_boundary_bytes=raw_scales, w_max_abs_error=max_w)
            logger.info("[%s] T=%s PASS  q_boundary=%s scale_boundary=%s "
                        "w_max_abs_error=%.3e  (%.1fs)",
                        name, t, raw_q, raw_scales, max_w, time.time() - started)
        except AssertionError as exc:
            record.update(passed=False, error=str(exc))
            all_pass = False
            logger.info("[%s] T=%s FAIL  %s", name, t, exc)
        except Exception as exc:  # noqa: BLE001
            record.update(passed=False, error=f"{type(exc).__name__}: {exc}",
                          traceback=traceback.format_exc())
            all_pass = False
            logger.info("[%s] T=%s BLOCKED  %s: %s", name, t, type(exc).__name__, exc)
        report["precision"].append(record)
        save()

    # ---- 性能 ----
    if args.skip_perf:
        perf_mode = "none"
    else:
        perf_mode = args.perf
    if perf_mode != "none":
        perf_ts = list(_PERF_CASES) if perf_mode == "targets" else [t for _, t in p0]
        eager_args = (args.eager_repeats, args.eager_rounds)
        graph_args = (args.graph_repeats, args.graph_rounds)
        for t in perf_ts:
            if t not in index_of:
                continue
            try:
                perf = _measure_case(
                    verify_mod, golden_mod, t, index_of[t], device, eager_args, graph_args
                )
                report["performance"].append({"T": t, **perf})
                logger.info("[perf] T=%s graph_median=%.2fus min=%.2f max=%.2f "
                            "eager_median=%.2fus", t, perf["graph"]["median_us"],
                            perf["graph"]["min_us"], perf["graph"]["max_us"],
                            perf["eager"]["median_us"])
            except Exception as exc:  # noqa: BLE001
                report["performance"].append({"T": t, "error": f"{type(exc).__name__}: {exc}"})
                logger.info("[perf] T=%s BLOCKED %s: %s", t, type(exc).__name__, exc)
            save()

    passed = sum(1 for r in report["precision"] if r.get("passed"))
    logger.info("SUMMARY %s/%s passed; report=%s", passed,
                len(report["precision"]), args.out)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
