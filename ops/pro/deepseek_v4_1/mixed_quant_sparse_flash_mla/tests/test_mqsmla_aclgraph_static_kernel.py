#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""MQSMLA 双算子(metadata AICPU 规划 + AI Core 主算子)整链 aclgraph + static kernel 用例。

把 ``mixed_quant_sparse_flash_mla_metadata`` 与 ``mixed_quant_sparse_flash_mla`` 的安装态
torch 绑定(``torch.ops.pypto_gym.*``)串进同一个 ``torch.compile(backend="npugraph_ex")`` 编译目标:

  forward = [AICPU planner(读 GM 长度 -> 板上产出 metadata/tiling_device)]
         -> [AI Core 主算子(读 GM metadata/长度, 写 out/lse)]

两个用例分层:

1. ``test_aclgraph_chain_capture_replay`` — aclgraph 整链(不开 static kernel): fullgraph 整图
   捕获 + 多次回放与 eager 链 bit 级一致; 回放前原地更新设备侧 topk 长度, 图内 AICPU 自动
   重规划 + 主算子重算, 输出仍与 eager 重算 bit 级一致(FD/uniform 切换由 kernel 运行期自分流)。
2. ``test_aclgraph_static_kernel_chain`` — 两算子联合 + ``static_kernel_compile=True`` 直接运行:
   torch_npu 的 ``_compile_static_kernel`` 会在 ``_aclop_start_dump`` 下重放整链以 dump 算子
   信息 json, 本用例即用于暴露该联合路径的运行期问题(当前 CANN 9.3.0 ``libnnopbase.so`` 的
   ``aclnnOpInfoRecord::TilingContextToJson`` 对主算子 tiling 上下文序列化段错误, 进程级崩溃,
   日志经 faulthandler/gdb 采集)。该用例同时打开 ``_super_kernel_optimize``: op_compiler 带
   ``--enable_super_kernel`` 执行 SK 融合, SK 日志(``sk_fused_nodes.log`` /
   ``sk_fusion_fail_reasons.log`` 等)由 ``libascendsk.so`` 落在构建目录。

torch 绑定加载契约: 安装态 ``pypto_gym`` 的绑定是惰性 JIT 编译/加载 —— 仅 ``import pypto_gym``
不注册任何 ``torch.ops``, 直接访问 ``torch.ops.pypto_gym.<op>`` 前必须先 ``pypto_gym.load()``。

用法:
  pytest ops/pro/deepseek_v4_1/mixed_quant_sparse_flash_mla/tests/ \\
      test_mqsmla_aclgraph_static_kernel.py -q --mode torch
"""

from __future__ import annotations

import logging
import os
import sys

import pytest
import torch

import torch_npu  # noqa: F401

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (_THIS_DIR, os.path.join(_THIS_DIR, "assets")):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

import mixed_quant_sparse_flash_mla_golden_cpu as golden  # noqa: E402
import mixed_quant_sparse_flash_mla_precision as verify  # noqa: E402

try:
    import pypto_gym
    # torch 绑定惰性 JIT 加载: 直接访问 torch.ops.pypto_gym.* 前必须显式 load。
    _LOADED_BINDINGS = list(pypto_gym.load())
except ImportError as error:
    pytest.skip(f"installed pypto_gym vendor is required for the aclgraph static-kernel "
                f"chain test: {error}", allow_module_level=True)

for _binding in ("mixed_quant_sparse_flash_mla_metadata",
                 "mixed_quant_sparse_flash_mla"):
    if _binding not in _LOADED_BINDINGS:
        pytest.skip(f"installed vendor misses the {_binding} torch binding "
                    f"(loaded: {_LOADED_BINDINGS})", allow_module_level=True)

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.soc("950")

#: 图内外共用的输入键(metadata 由图内 AICPU 算子产出, 不进输入集)。
_INPUT_KEYS = (
    "q", "ori_kv", "ori_sparse_indices", "ori_block_table", "ori_topk_length", "sinks",
    "cmp_kv", "cmp_sparse_indices", "cmp_block_table", "cmp_topk_length", "cu_seqlens_q",
)

#: aclgraph 整链基线选项(不开 static kernel)。
_CHAIN_COMPILE_OPTIONS = {
    "frozen_parameter": True,
    "clone_input": False,
}

#: static kernel 输出根目录(CompilerConfig 直连 _aclnn_static_shape_kernel_build_dir,
#: torch.compile options 未暴露该 key); 产物落 <dir>/static_kernel_compile_outputs/ts*_outputs/。
_STATIC_BUILD_DIR = os.environ.get("MQSMLA_STATIC_BUILD_DIR", "/tmp/sk_out")


def _device():
    device = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}")
    torch.npu.set_device(device)
    return device


def _build_dev_inputs(name, device):
    """golden 输入 CPU -> device(全 NPU 放置, 无任何 host 值通道); 同时给出 CPU golden 参考。"""
    inputs, _block_dim, case = golden.build_inputs(name, torch.device("cpu"))
    reference = golden.reference(name, inputs)
    for key in ("ori_kv", "cmp_kv"):
        inputs[key] = inputs[key].contiguous()
    return {k: inputs[k].to(device) for k in _INPUT_KEYS}, case, reference


class _MqsmlaChainModule(torch.nn.Module):
    """双算子整链编译目标: forward 只含两次注册算子调用, 可被 npugraph_ex 整图捕获。

    所有预处理(blocks 取设备核数、shape/stride/标量)在 __init__(编译区域之外)完成;
    metadata/tiling_device 由图内 AICPU 算子按设备侧长度张量现算, 值依赖数据流全程留在设备侧。
    """

    def __init__(self, dev, case):
        super().__init__()
        blocks = int(torch.npu.get_device_properties(dev["q"].device).cube_core_num)
        self._plan_args = (
            dev["cu_seqlens_q"], dev["ori_topk_length"], dev["cmp_topk_length"],
            blocks, 1,
            int(case["k_ori"]), int(case["k_cmp"]),
            int(dev["ori_kv"].shape[1]), int(dev["cmp_kv"].shape[1]),
            int(dev["ori_block_table"].shape[1]), int(dev["cmp_block_table"].shape[1]),
            int(dev["ori_kv"].shape[0]), int(dev["cmp_kv"].shape[0]),
            int(dev["ori_kv"].stride(0)), int(dev["cmp_kv"].stride(0)),
        )
        self._mla_args = (
            dev["q"], dev["ori_kv"], dev["ori_sparse_indices"],
            dev["ori_block_table"], dev["cu_seqlens_q"], dev["sinks"],
        )
        self._mla_kwargs = dict(
            ori_topk_length=dev["ori_topk_length"],
            cmp_kv=dev["cmp_kv"],
            cmp_sparse_indices=dev["cmp_sparse_indices"],
            cmp_block_table=dev["cmp_block_table"],
            cmp_topk_length=dev["cmp_topk_length"],
            quant_mode=1,
            softmax_scale=float(case["softmax_scale"]),
            return_softmax_lse=True,
        )

    def forward(self):
        plan, tiling_device = torch.ops.pypto_gym.mixed_quant_sparse_flash_mla_metadata(
            *self._plan_args)
        out, lse = torch.ops.pypto_gym.mixed_quant_sparse_flash_mla(
            *self._mla_args, plan, tiling_device, **self._mla_kwargs)
        return out, lse


def _skip_on_npu_patch_collision(err):
    """torchair 与 npugraph_ex 后端单进程先后初始化时的 aten::to 二次注册 -> 精确跳过。"""
    if "Trying to override a python impl" in str(err):
        pytest.skip(
            "torch_npu registers the aten::to AutogradPrivateUse1 impl in both the "
            "torchair and npugraph_ex backends without a shared idempotency guard; "
            "second backend init in one process aborts - rerun this case in a "
            "fresh process")


def _compile_chain(module, options=_CHAIN_COMPILE_OPTIONS, backend=None):
    """按给定 options/backend 编译整链并完成首次捕获, 返回 (compiled, out, lse)。"""
    if backend is None:
        backend = "npugraph_ex"
    try:
        compiled = torch.compile(module, fullgraph=True, dynamic=False,
                                 backend=backend, options=options)
        out, lse = compiled()
        torch.npu.synchronize()
    except RuntimeError as err:
        _skip_on_npu_patch_collision(err)
        raise
    return compiled, out, lse


def _build_static_kernel_backend():
    """CompilerConfig 直连构建 static kernel 后端(build_dir 等未暴露到 torch.compile options)。

    开启 _aclnn_static_shape_kernel 并固定 _aclnn_static_shape_kernel_build_dir,
    dump json / op_compiler 产物 / .run 统一落 <_STATIC_BUILD_DIR>/static_kernel_compile_outputs/。
    同时开启 _super_kernel_optimize: static 编译带 --enable_super_kernel 做 SK 融合,
    运行期 capture/replay 也会调用 graph.super_kernel_optimize。
    """
    from torch_npu.dynamo import npugraph_ex

    # safe_resolve_output_dir 用 mkdir(exist_ok=True) 不带 parents, 根目录必须预先存在。
    os.makedirs(_STATIC_BUILD_DIR, exist_ok=True)
    cfg = npugraph_ex.CompilerConfig()
    cfg.mode = "npugraph_ex"
    cfg.experimental_config.aclgraph._aclnn_static_shape_kernel = True
    # SK 总开关: 缺省 False; 打开后 libascendsk 才输出 sk_fused_nodes.log 等日志。
    cfg.experimental_config.aclgraph._super_kernel_optimize = True
    cfg.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = _STATIC_BUILD_DIR
    cfg.experimental_config.frozen_parameter = True
    # cfg.debug.aclgraph.clone_input = False
    return npugraph_ex.get_npu_backend(compiler_config=cfg)


def _assert_bitexact(tag, got, want):
    assert bool(torch.equal(got[0], want[0])), f"{tag}: out not bit-identical to eager"
    assert bool(torch.equal(got[1], want[1])), f"{tag}: lse not bit-identical to eager"


def _eager_golden_baseline(case, device):
    """构造整链模块并对 CPU golden 站住精度, 返回 (module, eager_out, eager_lse)。"""
    dev, case_meta, reference = _build_dev_inputs(case, device)
    module = _MqsmlaChainModule(dev, case_meta)
    eo, el = module()
    torch.npu.synchronize()
    ok_out, msg_out = verify.check_precision(eo.cpu(), reference["out"])
    ok_lse, msg_lse = verify.check_precision(el.cpu(), reference["lse"])
    assert ok_out, msg_out
    assert ok_lse, msg_lse
    return module, eo, el


@pytest.mark.parametrize("case", ("mixed_decode", "netspec_decode_96"))
def test_aclgraph_chain_capture_replay(case):
    """双算子链 aclgraph(不开 static kernel): 整图捕获/回放与 eager bit 级一致, 含回放期重规划。

    mixed_decode 走 uniform 计划路径, netspec_decode_96 走 FlashDecode 计划路径
    (plan word[900] > 0), 回放期原地缩短长度后再对照 eager 重算, 覆盖图内重规划。
    """
    device = _device()
    torch.manual_seed(0)
    module, eo, el = _eager_golden_baseline(case, device)
    ref = (eo.clone(), el.clone())

    compiled, co, cl = _compile_chain(module, _CHAIN_COMPILE_OPTIONS)
    _assert_bitexact(f"{case} capture", (co, cl), ref)
    for round_id in range(2):
        co, cl = compiled()
        torch.npu.synchronize()
        _assert_bitexact(f"{case} replay#{round_id}", (co, cl), ref)

    # 回放期动态更新: 原地缩短每行 topk 宽度, 图内 AICPU 重规划 + 主算子重算。
    _halve_topk_lengths(module)
    eo2, el2 = module()
    torch.npu.synchronize()
    co2, cl2 = compiled()
    torch.npu.synchronize()
    _assert_bitexact(f"{case} replay-after-update", (co2, cl2), (eo2, el2))
    logger.info("%s: 双算子链 aclgraph 捕获/回放 bit-exact, 含回放期重规划", case)


@pytest.mark.parametrize("case", ("mixed_decode", "netspec_decode_96"))
def test_aclgraph_static_kernel_chain(case):
    """两算子联合 + static_kernel_compile 直接运行: 暴露联合路径的运行期问题并采集日志。

    经 CompilerConfig 直连开启 _aclnn_static_shape_kernel 并固定 build_dir(默认
    /tmp/sk_out, 可用 MQSMLA_STATIC_BUILD_DIR 覆盖), 产物落
    <build_dir>/static_kernel_compile_outputs/ts*_outputs/。torch_npu 的
    _compile_static_kernel 会在 _aclop_start_dump 下重放整链生成算子信息 json,
    随后 op_compiler 静态编译/装包/reselect; 当前已知崩溃点见模块 docstring
    (libnnopbase TilingContextToJson 段错误, faulthandler/gdb 采集日志)。
    """
    device = _device()
    torch.manual_seed(0)
    module, eo, el = _eager_golden_baseline(case, device)
    ref = (eo.clone(), el.clone())

    logger.info("static kernel outputs -> %s/static_kernel_compile_outputs", _STATIC_BUILD_DIR)
    compiled, co, cl = _compile_chain(module, backend=_build_static_kernel_backend())
    _assert_bitexact(f"{case} static-kernel capture", (co, cl), ref)
    for round_id in range(2):
        co, cl = compiled()
        torch.npu.synchronize()
        _assert_bitexact(f"{case} static-kernel replay#{round_id}", (co, cl), ref)
    logger.info("%s: 双算子链 aclgraph(static kernel) 捕获/回放 bit-exact", case)


def _halve_topk_lengths(module):
    """原地减半两侧每行 topk 宽度(触发图内 AICPU 重规划与 FD->uniform 计划切换)。"""
    module._plan_args[1].div_(2, rounding_mode="floor")
    module._plan_args[2].div_(2, rounding_mode="floor")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    def _run(name, fn, *args):
        try:
            fn(*args)
        except pytest.skip.Exception as skip:  # 直跑时跳过项不中断后续用例
            logger.info("SKIP %s: %s", name, skip.msg)

    for _name in ("mixed_decode", "netspec_decode_96"):
        _run(f"chain[{_name}]", test_aclgraph_chain_capture_replay, _name)
        _run(f"static[{_name}]", test_aclgraph_static_kernel_chain, _name)
    logging.info("All runnable aclgraph static-kernel chain tests passed!")
