#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""全设备端形态的 D2H 探测用例(tiling_device 通道)。

数据流: AICPU planner 板上产出 ``metadata[1024] + tiling_device[16]`` ->
主算子 kernel 从这两个 GM 张量读 tiling 参数与调度; host 侧(op_host tiling 与
两个 torch binding)只消费 shape/属性, 全程不触碰张量值。

D2H 探测方法(torch_npu profiler, chrome trace):
- 绑定侧旧 D2H 通道(``HostValues``)会发出 ``aten::to -> aten::_to_copy ->
  aten::copy_``; JIT wrapper 的 ``.item()/.tolist()`` 会发出 ``aten::item ->
  aten::_local_scalar_dense``。对捕获区断言这些 cpu_op 一个都没有。
- 正控用例先证明探测器本身能看到已知 D2H(``tensor.cpu()``)。

用法:
  pytest ops/pro/deepseek_v4_1/mixed_quant_sparse_flash_mla_pto_bnn/tests/test_torch_no_d2h_pto_bnn.py -q --mode torch
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pytest
import torch

import torch_npu  # noqa: F401
from torch_npu.profiler import profile, ProfilerActivity

pytestmark = [pytest.mark.soc("950"), pytest.mark.usefixtures("device")]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (_THIS_DIR, os.path.join(_THIS_DIR, "assets")):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

import _standalone_env  # noqa: E402,F401  直跑自举: 注入 tests/common 与安装态 vendor 环境

import mixed_quant_sparse_flash_mla_pto_bnn_golden_cpu as golden  # noqa: E402
import mixed_quant_sparse_flash_mla_pto_bnn_precision as verify  # noqa: E402

try:
    from pypto_gym import (  # noqa: E402
        mixed_quant_sparse_flash_mla_metadata_pto_bnn as planner,
        mixed_quant_sparse_flash_mla_pto_bnn as mla,
    )
except ImportError as error:  # 安装态 vendor 不在(torch 绑定未随包构建)时跳过
    pytest.skip(f"installed vendor misses an operator binding: {error}",
                allow_module_level=True)

logger = logging.getLogger(__name__)

#: D2H/host-readback 的 cpu_op 特征(旧通道逐一对应: HostValues -> to/_to_copy/copy_;
#: wrapper .item()/.tolist() -> item/_local_scalar_dense)。
_D2H_SIGNATURES = (
    "aten::to",
    "aten::_to_copy",
    "aten::copy_",
    "aten::item",
    "aten::_local_scalar_dense",
)

_INPUT_KEYS = (
    "q", "ori_kv", "ori_sparse_indices", "ori_block_table", "ori_topk_length", "sinks",
    "cmp_kv", "cmp_sparse_indices", "cmp_block_table", "cmp_topk_length",
    "cu_seqlens_q",
)

_LSE_RTOL = 1e-5
_LSE_ATOL = 1e-5

_TRACE_TMP = "/tmp/opencode/d2h_trace.json"


def _device():
    device = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}")
    torch.npu.set_device(device)
    return device


def _profile_events(fn):
    """Run fn under the NPU profiler and return its cpu_op event names."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        fn()
        torch.npu.synchronize()
    prof.export_chrome_trace(_TRACE_TMP)
    raw = open(_TRACE_TMP).read()
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:  # 导出偶发截断, 修补尾部再解析
        doc = json.loads(raw + "]")
    if isinstance(doc, dict):
        doc = doc.get("traceEvents", [])
    return [str(ev.get("name", "")) for ev in doc if ev.get("ph") == "X"]


def _d2h_hits(names):
    return sorted({n for n in names if n in _D2H_SIGNATURES})


def _plan_dev(dev, case_meta, has_cmp):
    """AICPU 规划(dev 输入): 返回 (metadata, tiling_device), shape 标量在 host 量好透传。"""
    return planner(
        dev["cu_seqlens_q"], dev["ori_topk_length"],
        dev["cmp_topk_length"] if has_cmp else None,
        has_cmp=1 if has_cmp else 0,
        k_ori=int(case_meta["k_ori"]), k_cmp=int(case_meta["k_cmp"]),
        bs_ori=int(dev["ori_kv"].shape[1]),
        bs_cmp=int(dev["cmp_kv"].shape[1]) if has_cmp else 0,
        bt_ori=int(dev["ori_block_table"].shape[1]),
        bt_cmp=int(dev["cmp_block_table"].shape[1]) if has_cmp else 0,
        ori_pages=int(dev["ori_kv"].shape[0]),
        cmp_pages=int(dev["cmp_kv"].shape[0]) if has_cmp else 0,
        ori_kv_stride0=int(dev["ori_kv"].stride(0)),
        cmp_kv_stride0=int(dev["cmp_kv"].stride(0)) if has_cmp else 0,
    )


def _run_attention(dev, plan, tiling_device, case_meta, out, lse):
    """主算子调用: plan/tiling_device 设备端直喂, 输入全 NPU 放置。"""
    return mla(
        dev["q"], ori_kv=dev["ori_kv"],
        ori_sparse_indices=dev["ori_sparse_indices"],
        ori_block_table=dev["ori_block_table"],
        cu_seqlens_q=dev["cu_seqlens_q"], sinks=dev["sinks"],
        metadata=plan, tiling_device=tiling_device,
        quant_mode=1, softmax_scale=float(case_meta["softmax_scale"]),
        cmp_kv=dev["cmp_kv"], cmp_sparse_indices=dev["cmp_sparse_indices"],
        cmp_block_table=dev["cmp_block_table"],
        ori_topk_length=dev["ori_topk_length"],
        cmp_topk_length=dev["cmp_topk_length"],
        return_softmax_lse=True, out=out, lse=lse,
    )


def _assert_td_passthrough_words(td_words, inputs, case_meta, has_cmp):
    """逐字校验 12 个透传形状标量(td[0..11])。"""
    ori_kv = inputs["ori_kv"]
    cmp_kv = inputs["cmp_kv"] if has_cmp else None
    expected = [
        int(inputs["q"].shape[0]),                       # TD_T1
        int(inputs["cu_seqlens_q"].numel()),             # TD_B_NUM
        int(case_meta["k_ori"]),                         # TD_K_ORI
        int(case_meta["k_cmp"]),                         # TD_K_CMP
        int(ori_kv.shape[1]),                            # TD_BS_ORI
        int(cmp_kv.shape[1]) if has_cmp else 0,          # TD_BS_CMP
        int(inputs["ori_block_table"].shape[1]),         # TD_BT_ORI
        int(inputs["cmp_block_table"].shape[1]) if has_cmp else 0,  # TD_BT_CMP
        int(ori_kv.shape[0]),                            # TD_ORI_PAGES
        int(cmp_kv.shape[0]) if has_cmp else 0,          # TD_CMP_PAGES
        int(ori_kv.stride(0)),                           # TD_ORI_KV_STRIDE0
        int(cmp_kv.stride(0)) if has_cmp else 0,         # TD_CMP_KV_STRIDE0
    ]
    got = td_words[:12]
    assert got == expected, f"tiling_device passthrough words mismatch: {got} != {expected}"


def test_d2h_detector_positive_control():
    """探测器自证: 已知 D2H(tensor.cpu())必须被看见, 否则后续阴性断言无效。"""
    device = _device()
    t = torch.arange(16, dtype=torch.int32, device=device)
    torch.npu.synchronize()
    names = _profile_events(lambda: t.cpu())
    hits = _d2h_hits(names)
    assert "aten::to" in hits and "aten::_to_copy" in hits, (
        f"detector failed to see the known D2H via tensor.cpu(): saw {hits}")
    logger.info("detector positive control OK: %s", hits)


@pytest.mark.parametrize("case", ("mixed_decode", "netspec_decode_96"))
def test_planner_and_attention_no_d2h(case):
    """AICPU 规划 + AI Core 主算子全链路: 捕获区不得出现任何 D2H/host 回读。

    覆盖 uniform(mixed_decode)与 FD(netspec_decode_96)两条规划路径; 输入全部
    NPU 放置(长度张量也在板上), tiling_device 由 planner 在板上写出并直喂主算子。
    """
    device = _device()
    torch.manual_seed(0)
    inputs, _block_dim, case_meta = golden.build_inputs(case, torch.device("cpu"))
    for key in ("ori_kv", "cmp_kv"):
        inputs[key] = inputs[key].contiguous()
    dev = {k: inputs[k].to(device) for k in _INPUT_KEYS}
    rows = int(dev["q"].shape[0])
    out = torch.full(tuple(dev["q"].shape), float("nan"), dtype=torch.bfloat16, device=device)
    lse = torch.full((1, rows, 64), float("nan"), dtype=torch.float32, device=device)
    has_cmp = bool(case_meta["has_cmp"])

    def _chain():
        plan, tiling_device = _plan_dev(dev, case_meta, has_cmp)
        _run_attention(dev, plan, tiling_device, case_meta, out, lse)

    _chain()  # warmup(JIT 绑定编译/加载在捕获区外完成)
    torch.npu.synchronize()
    names = _profile_events(_chain)
    hits = _d2h_hits(names)
    assert not hits, f"{case}: D2H/host readback detected in the op chain: {hits}"
    logger.info("%s: planner+attention chain is D2H-free", case)


@pytest.mark.parametrize("case", ("mixed_decode", "netspec_decode_96"))
def test_tiling_device_words_match_golden(case):
    """tiling_device 字内容校验: 形状标量透传 + 规划字与 host golden planner 一致。"""
    device = _device()
    inputs, block_dim, case_meta = golden.build_inputs(case, torch.device("cpu"))
    has_cmp = bool(case_meta["has_cmp"])
    for key in ("ori_kv", "cmp_kv"):
        inputs[key] = inputs[key].contiguous()
    dev = {k: inputs[k].to(device) for k in _INPUT_KEYS}
    plan, td = _plan_dev(dev, case_meta, has_cmp)
    torch.npu.synchronize()
    td_words = td.cpu().tolist()
    plan_words = plan.cpu().tolist()

    _assert_td_passthrough_words(td_words, inputs, case_meta, has_cmp)
    # 规划字: planned_block_dim 与 host golden planner 的 block_dim 一致;
    # fd_enable 对齐 metadata 计划字 md[900] > 0。
    assert td_words[12] == int(block_dim), (
        f"TD_PLANNED_BLOCK_DIM {td_words[12]} != golden block_dim {int(block_dim)}")
    assert td_words[13] == (1 if plan_words[900] > 0 else 0), (
        "TD_FD_ENABLE must mirror metadata plan word [900]")
    assert td_words[14] == 0 and td_words[15] == 0, "reserved words must stay zero"


@pytest.mark.parametrize("case", ("mixed_decode", "netspec_decode_96"))
def test_attention_precision_with_device_tiling(case):
    """全链路精度: tiling_device 通道驱动的主算子输出对 CPU golden。"""
    device = _device()
    inputs, _block_dim, case_meta = golden.build_inputs(case, torch.device("cpu"))
    reference = golden.reference(case, inputs)
    for key in ("ori_kv", "cmp_kv"):
        inputs[key] = inputs[key].contiguous()
    dev = {k: inputs[k].to(device) for k in _INPUT_KEYS}
    has_cmp = bool(case_meta["has_cmp"])
    plan, td = _plan_dev(dev, case_meta, has_cmp)
    rows = int(dev["q"].shape[0])
    out = torch.full(tuple(dev["q"].shape), float("nan"), dtype=torch.bfloat16, device=device)
    lse = torch.full((1, rows, 64), float("nan"), dtype=torch.float32, device=device)
    result = _run_attention(dev, plan, td, case_meta, out, lse)
    torch.npu.synchronize()
    out, lse = result["out"], result["lse"]
    assert not bool(torch.isnan(out).any().item() or torch.isnan(lse).any().item())
    ok_out, s_out = verify.check_precision(out.cpu(), reference["out"])
    ok_lse, s_lse = verify.check_precision(lse.cpu(), reference["lse"])
    assert ok_out, s_out
    assert ok_lse, s_lse
    assert bool(torch.allclose(lse.cpu(), reference["lse"], rtol=_LSE_RTOL, atol=_LSE_ATOL))
    if case == "netspec_decode_96":
        assert int(plan.cpu()[900]) > 0, "expected a FlashDecode plan for netspec_decode_96"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    test_d2h_detector_positive_control()
    logging.info("positive control PASS")
    for _name in ("mixed_decode", "netspec_decode_96"):
        test_planner_and_attention_no_d2h(_name)
        logging.info("%s: no-D2H PASS", _name)
        test_tiling_device_words_match_golden(_name)
        logging.info("%s: tiling_device words PASS", _name)
        test_attention_precision_with_device_tiling(_name)
        logging.info("%s: precision PASS", _name)
    logging.info("All no-D2H suite cases passed!")
