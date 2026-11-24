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
_MAIN_OP = os.path.join(os.path.dirname(_OP_DIR), "mixed_quant_sparse_flash_mla")
_MAIN_TESTS = os.path.join(_MAIN_OP, "tests")
for _path in (
    _THIS_DIR,
    _MAIN_TESTS,
    os.path.join(_MAIN_TESTS, "assets"),
    os.path.join(_MAIN_OP, "op_kernel"),
):
    # 本算子路径必须优先于其它算子/同名 pypto_pro 样例目录（root conftest 会注入后者）。
    if os.path.isdir(_path):
        while _path in sys.path:
            sys.path.remove(_path)
        sys.path.insert(0, _path)


def _load_asset(module_name, file_path):
    """按文件路径加载 assets，避免同名 pypto_pro 模块经 sys.path 遮蔽本算子 golden。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    golden = _load_asset(
        "mixed_quant_sparse_flash_mla_golden_cpu",
        os.path.join(_MAIN_TESTS, "assets", "mixed_quant_sparse_flash_mla_golden_cpu.py"),
    )
    verify = _load_asset(
        "mixed_quant_sparse_flash_mla_precision",
        os.path.join(_MAIN_TESTS, "assets", "mixed_quant_sparse_flash_mla_precision.py"),
    )
except OSError:  # 主算子用例未随仓部署时，只保留 metadata 的形状/契约检查
    pytest.skip("main operator golden assets are required for the metadata parity test",
                allow_module_level=True)


def _planner_entry():
    """AICPU metadata 规划入口：非 jit 模式走安装态绑定/桥；jit 模式按文件加载 host launcher。"""
    try:
        from op_caller import get_mode, get_op_caller  # noqa: E402

        if get_mode() != "jit":
            return get_op_caller("mixed_quant_sparse_flash_mla_metadata")
    except ImportError:
        pass
    kernel = _load_asset(
        "mixed_quant_sparse_flash_mla_metadata_kernel",
        os.path.join(os.path.dirname(_OP_DIR), "mixed_quant_sparse_flash_mla_metadata",
                     "op_kernel_aicpu", "mixed_quant_sparse_flash_mla_metadata.py"),
    )
    launcher = kernel.mixed_quant_sparse_flash_mla_metadata_aicpu

    def _planner(cu_seqlens_q, ori_topk_length, cmp_topk_length, *, blocks, has_cmp,
                 k_ori, k_cmp):
        return launcher(
            ori_topk_length,
            cmp_topk_length,
            cu_seqlens_q=cu_seqlens_q,
            has_cmp_kv=bool(has_cmp),
            block_num=blocks,
            k_ori=k_ori,
            k_cmp=k_cmp,
        )

    return _planner


mixed_quant_sparse_flash_mla_metadata = _planner_entry()

logger = logging.getLogger(__name__)

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
    plan = mixed_quant_sparse_flash_mla_metadata(
        inputs["cu_seqlens_q"].to(device),
        inputs["ori_topk_length"].to(device),
        cmp_len,
        blocks=blocks,
        has_cmp=1 if has_cmp else 0,
        k_ori=int(case["k_ori"]),
        k_cmp=int(case["k_cmp"]),
    )
    torch.npu.synchronize()
    return inputs, block_dim, case, plan


@pytest.mark.parametrize("case", _CASES)
def test_metadata_matches_host_plan(case):
    device = _device()
    inputs, block_dim, _case, plan = _run_planner(case, device)
    assert tuple(plan.shape) == (1024,) and plan.dtype == torch.int32

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
        from op_caller import get_op_caller

        mla = get_op_caller("mixed_quant_sparse_flash_mla")
    except Exception:  # 主算子未构建（aclnn 桥/torch 绑定都不在）时跳过端到端
        pytest.skip("mixed_quant_sparse_flash_mla entry is not available")
    device = _device()
    inputs, block_dim, _case, plan = _run_planner(case, device)
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for _name in _CASES:
        test_metadata_matches_host_plan(_name)
        logger.info("%s: metadata parity PASS", _name)
    test_attention_with_aicpu_metadata("mixed_decode")
    logger.info("mixed_decode: e2e attention PASS")
