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
"""torch 绑定 Meta kernel 的生效场景验证(mixed_quant_sparse_flash_mla_metadata_pto_bnn)。

Meta 注册前, meta 张量/PT2 编译 API 对该算子直接抛
``NotImplementedError: no fake impl or Meta kernel registered``。
注册后三个场景分别命中 Meta 分发:

* ``test_meta_device_call``: meta 设备张量直调, 返回 (int32[1024], int32[16])
  meta 占位, 不触碰 ACLNN/vendor; 标量校验仍在 Meta 侧生效;
* ``test_torch_export_graph``: torch.export 用 FakeTensor 追踪, Meta 提供输出
  spec, 导出图含自定义算子节点且输出 meta 形状/dtype 正确;
* ``test_torch_compile_fullgraph_parity``: torch.compile(fullgraph=True,
  backend="eager") 追踪经 Meta 走通, 运行期回落 PrivateUse1 真实 AICPU kernel,
  编译结果与 eager 一致。

依赖安装态 vendor(.run 安装后 ASCEND_CUSTOM_OPP_PATH + PYTHONPATH 生效):
  pytest ops/pro/deepseek_v4_1/mixed_quant_sparse_flash_mla_metadata/tests/ \\
      test_torch_binding_meta_pto_bnn.py -q --mode torch
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

import torch_npu  # noqa: F401

pytestmark = [pytest.mark.soc("950"), pytest.mark.usefixtures("device")]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_THIS_DIR)
_MAIN_TESTS = os.path.join(os.path.dirname(_OP_DIR), "mixed_quant_sparse_flash_mla_pto_bnn", "tests")
for _path in (_THIS_DIR, _MAIN_TESTS):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

import _standalone_env  # noqa: E402,F401  直跑自举: 注入 tests/common、aclnn 桥与 vendor 环境

try:
    from pypto_gym import mixed_quant_sparse_flash_mla_metadata_pto_bnn as planner
except Exception as error:  # 绑定源/vendor 缺失时整文件跳过
    pytest.skip(f"installed pypto_gym torch binding is required: {error}",
                allow_module_level=True)

_BLOCKS = 8  # 显式传 blocks, 绕开 wrapper 内不可追踪的 get_device_properties 分支


def _meta_inputs():
    cu = torch.tensor([0, 1, 2], dtype=torch.int32, device="meta")
    otl = torch.zeros(2, 1, dtype=torch.int32, device="meta")
    return cu, otl


def test_meta_device_call():
    cu, otl = _meta_inputs()
    metadata, tiling_device = planner(cu, otl, None, blocks=_BLOCKS, has_cmp=0)
    assert tuple(metadata.shape) == (1024,) and metadata.dtype == torch.int32
    assert tuple(tiling_device.shape) == (16,) and tiling_device.dtype == torch.int32
    assert metadata.device.type == "meta" and tiling_device.device.type == "meta"
    # 直调 torch.ops 绕开 python wrapper 的前置校验, 证明 C++ Meta 侧标量/dtype 校验生效。
    op = torch.ops.pypto_gym.mixed_quant_sparse_flash_mla_metadata_pto_bnn
    with pytest.raises(RuntimeError, match="blocks must be in"):
        op(cu, otl, None, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    bad = torch.zeros(2, 1, dtype=torch.float32, device="meta")
    with pytest.raises(RuntimeError, match="must be int32"):
        op(cu, otl, bad, _BLOCKS, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def test_torch_export_graph():
    class PlanModule(torch.nn.Module):
        def forward(self, cu, otl):
            return planner(cu, otl, None, blocks=_BLOCKS, has_cmp=0)

    cu, otl = _meta_inputs()
    exported = torch.export.export(PlanModule(), (cu, otl))
    nodes = [str(node.target) for node in exported.graph.nodes
             if node.op == "call_function"]
    assert any("mixed_quant_sparse_flash_mla_metadata_pto_bnn" in t for t in nodes), (
        f"custom op missing from exported graph: {nodes}")
    metadata, tiling_device = exported.module()(cu, otl)
    assert tuple(metadata.shape) == (1024,) and metadata.dtype == torch.int32
    assert tuple(tiling_device.shape) == (16,) and tiling_device.dtype == torch.int32
    assert metadata.device.type == "meta" and tiling_device.device.type == "meta"


def _plan_fn(cu, otl):
    # 无 cmp 输入时 has_cmp=0, 与真实调用约定一致(AICPU kernel 校验该组合)。
    return planner(cu, otl, None, blocks=_BLOCKS, has_cmp=0)


def test_torch_compile_fullgraph_parity():
    device = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}")
    torch.npu.set_device(device)
    cu = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    otl = torch.ones(2, 1, dtype=torch.int32, device=device)
    eager_md, eager_td = _plan_fn(cu, otl)
    compiled = torch.compile(_plan_fn, backend="eager", fullgraph=True)
    got_md, got_td = compiled(cu, otl)
    torch.npu.synchronize()
    # fullgraph=True 未断图即证明 Meta 让 FakeTensor 追踪走通; 数值与 eager 一致。
    assert tuple(got_md.shape) == (1024,) and got_md.dtype == torch.int32
    assert tuple(got_td.shape) == (16,) and got_td.dtype == torch.int32
    assert torch.equal(got_md.cpu(), eager_md.cpu())
    assert torch.equal(got_td.cpu(), eager_td.cpu())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
