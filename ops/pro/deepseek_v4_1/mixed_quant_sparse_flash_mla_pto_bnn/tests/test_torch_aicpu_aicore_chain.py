#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""AICPU(AICPU kernel) 与 AI Core(Aicore kernel) 的 torch 模式串联验证。

链路: 安装态 ``torch.ops.pypto_gym.*`` 入口(前置 .run 安装的 vendor)——

1. AICPU metadata 算子(torch 绑定)读长度张量, 板上产出 ``int32[1024]`` 调度表,
   与 host 参考 golden planner 逐字一致(uniform + FlashDecode 两条规划路径);
2. 该 device 侧 metadata 直喂 AI Core 主算子(torch 绑定), out/lse 对 CPU golden,
   FD case 走最深调度路径(word[900] > 0, 三类核按同一份 plan 协作)。

用例均要求安装态 vendor(.run 安装后 ASCEND_CUSTOM_OPP_PATH + PYTHONPATH 生效):
  pytest ops/pro/deepseek_v4_1/mixed_quant_sparse_flash_mla_pto_bnn/tests \
         ops/pro/deepseek_v4_1/mixed_quant_sparse_flash_mla_metadata/tests -q --mode torch
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

import torch_npu  # noqa: F401

pytestmark = [pytest.mark.soc("950"), pytest.mark.usefixtures("device")]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(_THIS_DIR, "assets")
for _path in (_THIS_DIR, _ASSETS_DIR):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

import _standalone_env  # noqa: E402,F401  直跑自举: 注入 tests/common、aclnn 桥与 vendor 环境


def _load_asset(module_name, file_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    golden = _load_asset(
        "mixed_quant_sparse_flash_mla_pto_bnn_golden_cpu",
        os.path.join(_ASSETS_DIR, "mixed_quant_sparse_flash_mla_pto_bnn_golden_cpu.py"),
    )
    verify = _load_asset(
        "mixed_quant_sparse_flash_mla_pto_bnn_precision",
        os.path.join(_ASSETS_DIR, "mixed_quant_sparse_flash_mla_pto_bnn_precision.py"),
    )
except (OSError, ImportError):  # 主算子 assets 未部署时整文件跳过
    pytest.skip("main operator golden assets are required for the torch chain test",
                allow_module_level=True)

try:
    import pypto_gym  # noqa: F401
except ImportError:
    pytest.skip("installed pypto_gym vendor is required for --mode torch "
                "(install the .run and set ASCEND_CUSTOM_OPP_PATH)",
                allow_module_level=True)


def _torch_entries():
    # vendor 缺任一算子绑定(如 metadata 算子未随包构建)时降级为 skip, 而非用例 ERROR。
    try:
        from pypto_gym import mixed_quant_sparse_flash_mla_metadata_pto_bnn as planner
        from pypto_gym import mixed_quant_sparse_flash_mla_pto_bnn as mla
    except ImportError as error:
        pytest.skip(f"installed vendor misses an operator binding: {error}")
    return planner, mla

_LSE_RTOL = 1e-5
_LSE_ATOL = 1e-5
_INPUT_KEYS = (
    "q", "ori_kv", "ori_sparse_indices", "ori_block_table", "ori_topk_length", "sinks",
    "metadata", "cmp_kv", "cmp_sparse_indices", "cmp_block_table", "cmp_topk_length",
    "cu_seqlens_q",
)


def _kv_to_device(tensor, device):
    if tensor.is_contiguous():
        return tensor.to(device)
    strides = tuple(int(s) for s in tensor.stride())
    extent = 1
    for dim, stride in zip(tensor.shape, strides):
        extent += (int(dim) - 1) * stride
    backing = torch.full((extent,), 0xA5, dtype=tensor.dtype, device=device)
    view = backing.as_strided(tuple(tensor.shape), strides)
    view.copy_(tensor.to(device))
    return view


def _plan_on_device(planner, case, inputs, case_meta, device):
    # 两个用例共用的 AICPU 板上规划调用: 长度张量上板, 读 golden 的 clamp 参数;
    # 返回 (metadata, tiling_device), 主算子消费设备端 tiling 通道。
    has_cmp = bool(case_meta["has_cmp"])
    ori_kv = inputs["ori_kv"].contiguous()
    cmp_kv = inputs["cmp_kv"].contiguous() if has_cmp else None
    plan, tiling_device = planner(
        inputs["cu_seqlens_q"].to(device),
        inputs["ori_topk_length"].to(device),
        inputs["cmp_topk_length"].to(device) if has_cmp else None,
        has_cmp=1 if has_cmp else 0,
        k_ori=int(case_meta["k_ori"]),
        k_cmp=int(case_meta["k_cmp"]),
        bs_ori=int(ori_kv.shape[1]),
        bs_cmp=int(cmp_kv.shape[1]) if has_cmp else 0,
        bt_ori=int(inputs["ori_block_table"].shape[1]),
        bt_cmp=int(inputs["cmp_block_table"].shape[1]) if has_cmp else 0,
        ori_pages=int(ori_kv.shape[0]),
        cmp_pages=int(cmp_kv.shape[0]) if has_cmp else 0,
        ori_kv_stride0=int(ori_kv.stride(0)),
        cmp_kv_stride0=int(cmp_kv.stride(0)) if has_cmp else 0,
    )
    torch.npu.synchronize()
    return plan, tiling_device


@pytest.mark.parametrize("case", ("mixed_decode", "netspec_decode_96"))
def test_torch_aicpu_plan_matches_host_golden(case):
    """AICPU kernel: torch 绑定产出与 host golden planner 逐字一致, enabled 核数与规划一致."""
    planner, _ = _torch_entries()
    device = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}")
    torch.npu.set_device(device)
    inputs, block_dim, case_meta = golden.build_inputs(case, torch.device("cpu"))
    plan, tiling_device = _plan_on_device(planner, case, inputs, case_meta, device)
    assert tuple(plan.shape) == (1024,) and plan.dtype == torch.int32
    assert tuple(tiling_device.shape) == (16,) and tiling_device.dtype == torch.int32
    expect = inputs["metadata"].to(torch.int32)
    got = plan.cpu()
    assert torch.equal(got, expect), (
        f"{case}: {int((got != expect).sum())} words differ; "
        f"first={int((got != expect).nonzero()[0, 0])}"
    )
    enable = got[:324].reshape(36, 9)[:, 0]
    # block_dim 只是 golden 期望核数; 实际 launch 走全核超集, 由 kernel 按 plan 自分流。
    assert int((enable > 0).sum()) == int(block_dim)


@pytest.mark.parametrize("case", ("mixed_decode", "netspec_decode_96"))
def test_torch_aicpu_to_aicore_chain(case):
    """串联: torch AICPU 板上规划 -> device 侧 plan/tiling_device 直喂 torch AI Core 主算子 -> 对 CPU golden.

    netspec_decode_96 的 plan word[900] > 0, 覆盖 FlashDecode 最深调度路径.
    """
    planner, mla = _torch_entries()
    device = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}")
    torch.npu.set_device(device)
    inputs, _block_dim, case_meta = golden.build_inputs(case, torch.device("cpu"))
    reference = golden.reference(case, inputs)

    plan, tiling_device = _plan_on_device(planner, case, inputs, case_meta, device)

    dev = {k: inputs[k].to(device) for k in _INPUT_KEYS if k not in ("ori_kv", "cmp_kv")}
    for key in ("ori_kv", "cmp_kv"):
        dev[key] = _kv_to_device(inputs[key], device)
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
        softmax_scale=float(case_meta["softmax_scale"]),
        cmp_kv=dev["cmp_kv"],
        cmp_sparse_indices=dev["cmp_sparse_indices"],
        cmp_block_table=dev["cmp_block_table"],
        ori_topk_length=dev["ori_topk_length"],
        cmp_topk_length=dev["cmp_topk_length"],
        return_softmax_lse=True,
        out=out,
        lse=lse,
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
    if case == "netspec_decode_96":
        assert int(plan.cpu()[900]) > 0, "expected a FlashDecode plan for netspec_decode_96"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
