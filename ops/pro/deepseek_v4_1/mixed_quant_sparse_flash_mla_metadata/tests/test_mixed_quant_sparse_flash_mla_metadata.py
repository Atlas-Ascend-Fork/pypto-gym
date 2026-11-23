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
"""mixed_quant_sparse_flash_mla_metadata 的测试入口（aclnn 模式）。

对拍基准是主算子 `mixed_quant_sparse_flash_mla` 的 host planner
（`tests/assets/mixed_quant_sparse_flash_mla_golden_cpu.py` 的 `build_inputs`），
两条规划路径都要覆盖：uniform 整行切分与 flat `(row, tile)` FlashDecode 切分。

* `test_metadata_matches_host_plan`：AICPU 算子写出的 `int32[1024]` 与 host planner 逐字相同；
* `test_attention_with_aicpu_metadata`：把 AICPU 产物喂给主算子，端到端精度通过（主算子未构建时跳过）。

用法：
  pytest ops/pro/deepseek_v4_1/mixed_quant_sparse_flash_mla_metadata/tests -q --mode aclnn
"""

from __future__ import annotations

import logging
import os
import sys

import pytest
import torch

import torch_npu  # noqa: F401

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_THIS_DIR)
_MAIN_TESTS = os.path.join(os.path.dirname(_OP_DIR), "mixed_quant_sparse_flash_mla", "tests")
for _path in (_THIS_DIR, _MAIN_TESTS, os.path.join(_MAIN_TESTS, "assets")):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

import _standalone_env  # noqa: E402,F401  直跑自举: 注入 tests/common、aclnn 桥与 vendor 环境

try:
    import mixed_quant_sparse_flash_mla_golden_cpu as golden  # noqa: E402
    import mixed_quant_sparse_flash_mla_precision as verify  # noqa: E402
except ImportError:  # 主算子用例未随仓部署时，只保留 metadata 的形状/契约检查
    pytest.skip("main operator golden assets are required for the metadata parity test",
                allow_module_level=True)

try:
    from op_caller import get_op_caller  # noqa: E402

    mixed_quant_sparse_flash_mla_metadata = get_op_caller(
        "mixed_quant_sparse_flash_mla_metadata")
except ImportError:  # standalone run outside pytest
    from aclnn_bridge_mixed_quant_sparse_flash_mla_metadata import (  # noqa: E402
        mixed_quant_sparse_flash_mla_metadata,
    )

pytestmark = pytest.mark.soc("950")

#: 两条规划路径各取一条：uniform（split=0）与 FlashDecode（split>0）。
_CASES = ("mixed_decode", "netspec_decode_96")
_LSE_RTOL = 1e-5
_LSE_ATOL = 1e-5
_INPUT_KEYS = (
    "q", "ori_kv", "ori_sparse_indices", "ori_block_table", "ori_topk_length", "sinks",
    "metadata", "cmp_kv", "cmp_sparse_indices", "cmp_block_table", "cmp_topk_length",
    "cu_seqlens_q",
)


def _device():
    device = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}")
    torch.npu.set_device(device)
    return device


def _run_planner(name, device):
    inputs, block_dim, case = golden.build_inputs(name, torch.device("cpu"))
    blocks = int(torch.npu.get_device_properties(device).cube_core_num)
    has_cmp = bool(case["has_cmp"])
    cmp_len = inputs["cmp_topk_length"].to(device) if has_cmp else None
    plan, tiling_device = mixed_quant_sparse_flash_mla_metadata(
        inputs["cu_seqlens_q"].to(device),
        inputs["ori_topk_length"].to(device),
        cmp_len,
        blocks=blocks,
        has_cmp=1 if has_cmp else 0,
        k_ori=int(case["k_ori"]),
        k_cmp=int(case["k_cmp"]),
        bs_ori=int(inputs["ori_kv"].shape[1]),
        bs_cmp=int(inputs["cmp_kv"].shape[1]) if has_cmp else 0,
        bt_ori=int(inputs["ori_block_table"].shape[1]),
        bt_cmp=int(inputs["cmp_block_table"].shape[1]) if has_cmp else 0,
        ori_pages=int(inputs["ori_kv"].shape[0]),
        cmp_pages=int(inputs["cmp_kv"].shape[0]) if has_cmp else 0,
        ori_kv_stride0=int(inputs["ori_kv"].stride(0)),
        cmp_kv_stride0=int(inputs["cmp_kv"].stride(0)) if has_cmp else 0,
    )
    torch.npu.synchronize()
    return inputs, block_dim, case, plan, tiling_device


@pytest.mark.parametrize("case", _CASES)
def test_metadata_matches_host_plan(case):
    device = _device()
    inputs, block_dim, _case, plan, tiling_device = _run_planner(case, device)
    assert tuple(plan.shape) == (1024,) and plan.dtype == torch.int32
    assert tuple(tiling_device.shape) == (16,) and tiling_device.dtype == torch.int32

    expect = inputs["metadata"].to(torch.int32)
    got = plan.cpu()
    assert torch.equal(got, expect), (
        f"{case}: {int((got != expect).sum())} words differ; "
        f"first={int((got != expect).nonzero()[0, 0])}"
    )
    assert int(got[900]) == int(expect[900])
    enable = got[:324].reshape(36, 9)[:, 0]
    assert int((enable > 0).sum()) == int(block_dim), (
        f"{case}: enabled cores {int((enable > 0).sum())} != block_dim {block_dim}"
    )


@pytest.mark.parametrize("case", ("mixed_decode",))
def test_attention_with_aicpu_metadata(case):
    try:
        mla = get_op_caller("mixed_quant_sparse_flash_mla")
    except Exception:  # 主算子未构建（aclnn 桥/torch 绑定都不在）时跳过端到端
        pytest.skip("mixed_quant_sparse_flash_mla entry is not available")
    device = _device()
    inputs, block_dim, _case, plan, tiling_device = _run_planner(case, device)
    reference = golden.reference(case, inputs)
    for key in ("ori_kv", "cmp_kv"):
        inputs[key] = inputs[key].contiguous()
    dev = {k: inputs[k].to(device) for k in _INPUT_KEYS}
    dev["metadata"] = plan
    rows = int(inputs["q"].shape[0])
    out = torch.full(tuple(inputs["q"].shape), float("nan"), dtype=torch.bfloat16, device=device)
    lse = torch.full((1, rows, 64), float("nan"), dtype=torch.float32, device=device)
    result = mla(
        dev["q"],
        ori_kv=dev["ori_kv"],
        ori_sparse_indices=dev["ori_sparse_indices"],
        ori_block_table=dev["ori_block_table"],
        cu_seqlens_q=dev["cu_seqlens_q"],
        sinks=dev["sinks"],
        metadata=plan,
        tiling_device=tiling_device,
        quant_mode=1,
        softmax_scale=float(_case["softmax_scale"]),
        cmp_kv=dev["cmp_kv"],
        cmp_sparse_indices=dev["cmp_sparse_indices"],
        cmp_block_table=dev["cmp_block_table"],
        ori_topk_length=dev["ori_topk_length"],
        cmp_topk_length=dev["cmp_topk_length"],
        return_softmax_lse=True,
        out=out,
        lse=lse,
        block_dim=block_dim,
    )
    torch.npu.synchronize()
    out, lse = result["out"], result["lse"]
    assert not bool(torch.isnan(out).any().item() or torch.isnan(lse).any().item())
    assert bool(torch.isfinite(out.float()).all().item() and torch.isfinite(lse).all().item())
    ok_out, s_out = verify.check_precision(out.cpu(), reference["out"])
    ok_lse, s_lse = verify.check_precision(lse.cpu(), reference["lse"])
    assert ok_out, s_out
    assert ok_lse, s_lse
    assert bool(torch.allclose(lse.cpu(), reference["lse"], rtol=_LSE_RTOL, atol=_LSE_ATOL))


def _find_vendor_root():
    """定位已安装的 pypto_gym vendor 根目录(读取交付物 json 做 SK/AICPU 断言)."""
    for part in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
        if part and os.path.isdir(os.path.join(part, "op_impl")):
            return part
    home = os.environ.get("ASCEND_HOME_PATH", "")
    cand = os.path.join(home, "opp", "vendors", "pypto_gym") if home else ""
    return cand if cand and os.path.isdir(cand) else None


def test_aicpu_aicore_linkage_fd_plan_and_sk():
    """AICPU 规划器 ↔ AI Core 执行器联动(FD 调度最深路径) + SK 交付物证明.

    联动证明(真机, netspec_decode_96):
      1. AICPU metadata 算子板上产出 plan(device 张量, 不回 host 中转);
      2. plan 与 host 参考规划器逐字一致, 且 FD_USED_VEC_NUM_WORD(900) > 0 ——
         本 case 确实走 FlashDecode 调度(AICPU 切行产生归约任务, AI Core FA 阶段
         写部分结果, AIV 按 FD 槽位归约, 三类核按同一份 plan 协作);
      3. plan 从 device 侧直接喂 AI Core 主算子, out/lse 对 CPU golden ——
         主算子正确消费 AICPU 产的 FD plan.
    SK 证明(交付物契约):
      4. 安装态 vendor 内主算子全部 kernel 二进制 json 的 supportSuperKernel == 1
         (ACLGraph superkernel 融合读取的资格字段, 由 @pl.jit(sk_cap=...) 驱动产出);
      5. vendor 的 AICPU 交付件齐套(cust_aicpu_kernel.json 含 metadata 算子条目).
    """
    try:
        mla = get_op_caller("mixed_quant_sparse_flash_mla")
    except Exception:  # 主算子未构建（aclnn 桥/torch 绑定都不在）时跳过端到端
        pytest.skip("mixed_quant_sparse_flash_mla entry is not available")
    import glob as _glob
    import json as _json

    device = _device()
    case = "netspec_decode_96"

    # 1)+2) AICPU 板上规划: 与 host 参考一致, 且确实走 FD 调度路径
    inputs, block_dim, _case, plan, tiling_device = _run_planner(case, device)
    got = plan.cpu().to(torch.int32)
    expect = inputs["metadata"].to(torch.int32)
    assert torch.equal(got, expect), (
        f"{case}: aicpu plan differs from host golden planner; "
        f"{int((got != expect).sum())} words differ")
    fd_used = int(got[900])
    assert fd_used > 0, f"{case}: expected FlashDecode plan, got FD_USED_VEC_NUM_WORD={fd_used}"

    # 3) device 侧 plan 直喂 AI Core 主算子(不经 host 中转), 输出对 CPU golden
    reference = golden.reference(case, inputs)
    for key in ("ori_kv", "cmp_kv"):
        inputs[key] = inputs[key].contiguous()
    dev = {k: inputs[k].to(device) for k in _INPUT_KEYS}
    dev["metadata"] = plan
    rows = int(inputs["q"].shape[0])
    out = torch.full(tuple(inputs["q"].shape), float("nan"), dtype=torch.bfloat16, device=device)
    lse = torch.full((1, rows, 64), float("nan"), dtype=torch.float32, device=device)
    result = mla(
        dev["q"],
        ori_kv=dev["ori_kv"],
        ori_sparse_indices=dev["ori_sparse_indices"],
        ori_block_table=dev["ori_block_table"],
        cu_seqlens_q=dev["cu_seqlens_q"],
        sinks=dev["sinks"],
        metadata=plan,
        tiling_device=tiling_device,
        quant_mode=1,
        softmax_scale=float(_case["softmax_scale"]),
        cmp_kv=dev["cmp_kv"],
        cmp_sparse_indices=dev["cmp_sparse_indices"],
        cmp_block_table=dev["cmp_block_table"],
        ori_topk_length=dev["ori_topk_length"],
        cmp_topk_length=dev["cmp_topk_length"],
        return_softmax_lse=True,
        out=out,
        lse=lse,
        block_dim=block_dim,
    )
    torch.npu.synchronize()
    out, lse = result["out"], result["lse"]
    assert not bool(torch.isnan(out).any().item() or torch.isnan(lse).any().item()), \
        "FD-plan driven attention produced NaN"
    ok_out, s_out = verify.check_precision(out.cpu(), reference["out"])
    ok_lse, s_lse = verify.check_precision(lse.cpu(), reference["lse"])
    assert ok_out, s_out
    assert ok_lse, s_lse
    assert bool(torch.allclose(lse.cpu(), reference["lse"], rtol=_LSE_RTOL, atol=_LSE_ATOL))

    # 4)+5) 交付物契约: AI Core SK 资格 + AICPU 交付件齐套
    vendor = _find_vendor_root()
    assert vendor is not None, "cannot locate installed pypto_gym vendor root"
    bin_jsons = sorted(_glob.glob(os.path.join(
        vendor, "op_impl/ai_core/tbe/kernel/*/mixed_quant_sparse_flash_mla/*.json")))
    assert bin_jsons, f"no AI Core kernel binary json under {vendor}"
    for path in bin_jsons:
        with open(path, encoding="utf-8") as fh:
            doc = _json.load(fh)
        assert int(doc.get("supportSuperKernel", 0)) == 1, \
            f"{os.path.basename(path)}: supportSuperKernel != 1 (sk_cap not in effect)"
    aicpu_cfg = os.path.join(vendor, "op_impl/cpu/config/cust_aicpu_kernel.json")
    assert os.path.isfile(aicpu_cfg), f"missing {aicpu_cfg}"
    with open(aicpu_cfg, encoding="utf-8") as fh:
        aicpu_doc = _json.load(fh)
    assert "MixedQuantSparseFlashMlaMetadata" in aicpu_doc, \
        "aicpu metadata op entry missing from cust_aicpu_kernel.json"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for _name in _CASES:
        test_metadata_matches_host_plan(_name)
        logging.info("%s: metadata parity PASS", _name)
    test_attention_with_aicpu_metadata("mixed_decode")
    logging.info("mixed_decode: e2e attention PASS")
    test_aicpu_aicore_linkage_fd_plan_and_sk()
    logging.info("netspec_decode_96: aicpu<->aicore FD linkage + SK artifact PASS")
