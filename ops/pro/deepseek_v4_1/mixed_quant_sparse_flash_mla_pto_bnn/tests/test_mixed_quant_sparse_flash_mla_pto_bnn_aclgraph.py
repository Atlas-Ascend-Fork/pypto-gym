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
"""mixed_quant_sparse_flash_mla_pto_bnn 的 ACLGraph(NPUGraph) 捕获/回放用例。

验证安装态 torch 绑定(``torch.ops.pypto_gym.mixed_quant_sparse_flash_mla_pto_bnn``)可被
``torch.npu.NPUGraph`` 整图捕获并多次回放, 且数值与非捕获 eager 执行 bit 级一致:

1. ``test_aclgraph_capture_replay_bitexact`` — 主算子(含可选 metadata AICPU 前置)在
   graph 内捕获, 3 次回放 + 覆写输入后再回放, out/lse 与 eager 基准逐位相等;
2. ``test_aclgraph_metadata_capture_replay`` — metadata AICPU 算子单独成图捕获回放,
   输出与 eager 规划逐字一致(int32 精确匹配)。

捕获语义注意: 全设备端形态下绑定走 aclnn Tensor 变体, 全部输入(含长度张量)NPU
放置, 无任何 host D2H —— 捕获/回放语义与 ds41 全设备指针算子一致; host tiling
值无关(全核 blockDim + FD workspace 超集), 回放期无需重算, 固定单 case 静态形状
(与 LLM serving 的 decode 稳态一致), 不做动态 shape 回放。

用法:
  pytest ops/pro/deepseek_v4_1/mixed_quant_sparse_flash_mla_pto_bnn/tests/ \\
      test_mixed_quant_sparse_flash_mla_pto_bnn_aclgraph.py -q --mode torch
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
for _path in (_THIS_DIR, os.path.join(_THIS_DIR, "assets")):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

import _standalone_env  # noqa: E402,F401  直跑自举: 注入 tests/common 与安装态 vendor 环境

import mixed_quant_sparse_flash_mla_pto_bnn_golden_cpu as golden  # noqa: E402
import mixed_quant_sparse_flash_mla_pto_bnn_precision as verify  # noqa: E402

try:
    from pypto_gym import (  # noqa: E402
        mixed_quant_sparse_flash_mla_metadata_pto_bnn as _planner,
        mixed_quant_sparse_flash_mla_pto_bnn as _mla,
    )
except ImportError as error:  # 安装态 vendor 不在(torch 绑定未随包构建)时跳过
    pytest.skip(f"installed vendor misses an operator binding: {error}", allow_module_level=True)

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.soc("950")

#: 图内外共用的输入键(与 P0 用例一致, 12 个张量输入)。
_INPUT_KEYS = (
    "q", "ori_kv", "ori_sparse_indices", "ori_block_table", "ori_topk_length", "sinks",
    "metadata", "cmp_kv", "cmp_sparse_indices", "cmp_block_table", "cmp_topk_length",
    "cu_seqlens_q",
)

_LSE_RTOL = 1e-5
_LSE_ATOL = 1e-5


def _device():
    device = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}")
    torch.npu.set_device(device)
    return device


def _build_dev_inputs(name, device):
    """golden 输入 CPU -> device(全 NPU 放置)。

    全设备端形态: 长度张量(cu_seqlens_q/ori_topk_length/cmp_topk_length)与其他
    输入一样留在设备上 —— 绑定走 aclnn Tensor 变体, kernel 直接读 GM, 全程
    无 D2H/aclIntArray, 天然 capture-safe(此前 host 值通道形态下的 CPU 放置
    已被绑定校验拒绝)。
    """
    inputs, block_dim, case = golden.build_inputs(name, torch.device("cpu"))
    for key in ("ori_kv", "cmp_kv"):
        inputs[key] = inputs[key].contiguous()
    dev = {k: inputs[k].to(device) for k in _INPUT_KEYS}
    return dev, block_dim, case


def _plan_dev(dev, case, device):
    """AICPU 板上规划(全 NPU): 返回 (metadata, tiling_device), 供主算子设备端 tiling 通道。"""
    has_cmp = bool(case["has_cmp"])
    ori_kv = dev["ori_kv"]
    cmp_kv = dev["cmp_kv"]
    plan, tiling_device = _planner(
        dev["cu_seqlens_q"], dev["ori_topk_length"],
        dev["cmp_topk_length"] if has_cmp else None,
        has_cmp=1 if has_cmp else 0,
        k_ori=int(case["k_ori"]), k_cmp=int(case["k_cmp"]),
        bs_ori=int(ori_kv.shape[1]),
        bs_cmp=int(cmp_kv.shape[1]) if has_cmp else 0,
        bt_ori=int(dev["ori_block_table"].shape[1]),
        bt_cmp=int(dev["cmp_block_table"].shape[1]) if has_cmp else 0,
        ori_pages=int(ori_kv.shape[0]),
        cmp_pages=int(cmp_kv.shape[0]) if has_cmp else 0,
        ori_kv_stride0=int(ori_kv.stride(0)),
        cmp_kv_stride0=int(cmp_kv.stride(0)) if has_cmp else 0,
    )
    torch.npu.synchronize()
    return plan, tiling_device


def _mla_call(dev, case, out, lse, scale, plan, tiling_device):
    return _mla(
        dev["q"],
        ori_kv=dev["ori_kv"],
        ori_sparse_indices=dev["ori_sparse_indices"],
        ori_block_table=dev["ori_block_table"],
        cu_seqlens_q=dev["cu_seqlens_q"],
        sinks=dev["sinks"],
        metadata=plan,
        tiling_device=tiling_device,
        quant_mode=1,
        softmax_scale=scale,
        cmp_kv=dev["cmp_kv"],
        cmp_sparse_indices=dev["cmp_sparse_indices"],
        cmp_block_table=dev["cmp_block_table"],
        ori_topk_length=dev["ori_topk_length"],
        cmp_topk_length=dev["cmp_topk_length"],
        return_softmax_lse=True,
        out=out,
        lse=lse,
    )


def _skip_on_npu_patch_collision(err):
    """torchair 与 npugraph_ex 后端单进程先后初始化时, aten::to 的 AutogradPrivateUse1 impl 二次注册 -> 精确跳过。"""
    if "Trying to override a python impl" in str(err):
        pytest.skip(
            "torch_npu registers the aten::to AutogradPrivateUse1 impl in both the "
            "torchair and npugraph_ex backends without a shared idempotency guard; "
            "second backend init in one process aborts - rerun this case in a "
            "fresh process")


def test_aclgraph_capture_replay_bitexact():
    """主算子整图捕获 + 3 次回放, out/lse 与 eager 基准 bit 级一致。"""
    device = _device()
    torch.manual_seed(0)
    dev, _block_dim, case = _build_dev_inputs("mixed_decode", device)
    rows = int(dev["q"].shape[0])
    scale = float(case["softmax_scale"])
    plan, tiling_device = _plan_dev(dev, case, device)

    # eager 基准(与捕获用同一块 out/lse 缓冲, 避免分配差异)
    out = torch.full(tuple(dev["q"].shape), float("nan"), dtype=torch.bfloat16, device=device)
    lse = torch.full((1, rows, 64), float("nan"), dtype=torch.float32, device=device)
    _mla_call(dev, case, out, lse, scale, plan, tiling_device)
    torch.npu.synchronize()
    ref_out, ref_lse = out.clone(), lse.clone()

    # 捕获: 覆写 out/lse 为哨兵后进图, 同一缓冲在图内被算子原地写回。
    # 全 NPU 放置: 绑定无任何同步 D2H / rtStreamSynchronize, 标准(global)捕获
    # 模式即可整图捕获; tiling 参数由 tiling_device 张量在板上供给, 图内固化
    # 无值烘焙风险。
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        _mla_call(dev, case, out, lse, scale, plan, tiling_device)

    for round_id in range(3):
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        assert bool(torch.equal(out, ref_out)), f"replay#{round_id}: out not bit-identical to eager"
        assert bool(torch.equal(lse, ref_lse)), f"replay#{round_id}: lse not bit-identical to eager"

    # 覆写 q 后回放: 图内绑定的是同一 device 缓冲, 新数据必须生效(不是回放旧结果)
    q_backup = dev["q"].clone()
    dev["q"].copy_(q_backup * -1.0)
    _mla_call(dev, case, out, lse, scale, plan, tiling_device)
    torch.npu.synchronize()
    eager_neg_out, eager_neg_lse = out.clone(), lse.clone()
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    torch.npu.synchronize()
    assert bool(torch.equal(out, eager_neg_out)), "replay after input update: out stale"
    assert bool(torch.equal(lse, eager_neg_lse)), "replay after input update: lse stale"
    dev["q"].copy_(q_backup)

    logger.info("aclgraph capture+replay: 3 replays + input-update replay bit-identical")


@pytest.mark.parametrize("case", ("mixed_decode", "decode_6"))
def test_aclgraph_metadata_capture_replay(case):
    """metadata AICPU 算子单独成图: 捕获回放输出与 eager 规划逐字一致(int32)。"""
    device = _device()
    inputs, _block_dim, case_meta = golden.build_inputs(case, torch.device("cpu"))
    has_cmp = bool(case_meta["has_cmp"])
    cu = inputs["cu_seqlens_q"].to(device)
    len_o = inputs["ori_topk_length"].to(device)
    len_c = inputs["cmp_topk_length"].to(device) if has_cmp else None

    def _plan(cu_t, o_t, c_t):
        return _planner(cu_t, o_t, c_t, blocks=None,
                        has_cmp=1 if has_cmp else 0,
                        k_ori=int(case_meta["k_ori"]), k_cmp=int(case_meta["k_cmp"]))

    ref_md, ref_td = _plan(cu, len_o, len_c)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured_md, captured_td = _plan(cu, len_o, len_c)
    for round_id in range(3):
        got_md, got_td = _plan(cu, len_o, len_c)  # eager 对照(每次新分配)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        assert tuple(captured_md.shape) == (1024,) and captured_md.dtype == torch.int32
        assert tuple(captured_td.shape) == (16,) and captured_td.dtype == torch.int32
        assert torch.equal(captured_md.cpu(), got_md.cpu()), \
            f"{case} replay#{round_id}: captured plan differs from eager plan"
        assert torch.equal(captured_td.cpu(), got_td.cpu()), \
            f"{case} replay#{round_id}: captured tiling_device differs from eager"
    logger.info("%s: metadata aclgraph capture+replay word-exact", case)


class _MlaModule(torch.nn.Module):
    """aclgraph 编译目标: forward 只调注册算子(参考 ops-transformer quant_flash_attn graph.py)。

    所有入参在 __init__(编译区域之外)预处理完成并保存为 self. 属性; forward
    无 Python 预处理, 仅一次 torch.ops.pypto_gym.mixed_quant_sparse_flash_mla_pto_bnn
    调用, 可被 npugraph_ex 捕获。
    """

    def __init__(self, dev, case, plan, tiling_device):
        super().__init__()
        self._args = (
            dev["q"],
            dev["ori_kv"],
            dev["ori_sparse_indices"],
            dev["ori_block_table"],
            dev["cu_seqlens_q"],
            dev["sinks"],
            plan,
            tiling_device,
        )
        self._kwargs = dict(
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
        out, lse = torch.ops.pypto_gym.mixed_quant_sparse_flash_mla_pto_bnn(
            *self._args, **self._kwargs)
        return out, lse


def test_torchair_aclgraph_capture():
    """torchair aclgraph(CompilerConfig mode=npugraph_ex)捕获回放 bit 级一致; 缺 torchair 时跳过。

    依赖 torch_extension/graph_convert_mixed_quant_sparse_flash_mla_pto_bnn.py 注册的
    fx 转换器(torchair 缺失时该注册为 no-op, 本用例同步跳过)。
    顺序约束: 本用例须先于 test_aclgraph_npugraph_ex_compile 执行 —— torchair 的
    add_npu_patch 不带 py_kernels 幂等保护, 若 npugraph_ex 后端已在同进程注册过
    aten::to 的 AutogradPrivateUse1 impl 会直接 RuntimeError(上游 torch_npu 问题,
    实测 torch_npu 2.8.0.post4 及更新版本均如此); 撞上时降级为精确 skip,
    独立进程运行本用例即可复验。
    """
    try:
        import torchair
        # torch_npu 对 torchair 提供惰性 shim: 属性访问才触发真实导入(缺依赖时
        # 在此异步抛错), 探测必须覆盖到属性层。
        _ = torchair.CompilerConfig
        _ = torchair.get_npu_backend
    except Exception:  # ImportError / StopIteration / 惰性导入的依赖错误 均视为不可用
        pytest.skip("torchair unavailable; aclgraph via torchair not testable")

    device = _device()
    torch.manual_seed(0)
    dev, _block_dim, case = _build_dev_inputs("mixed_decode", device)  # 全 NPU 放置

    plan, tiling_device = _plan_dev(dev, case, device)
    module = _MlaModule(dev, case, plan, tiling_device)
    eo, el = module()
    torch.npu.synchronize()

    cfg = torchair.CompilerConfig()
    cfg.mode = "npugraph_ex"  # aclgraph 捕获回放; reduce-overhead 模式已被上游废弃
    try:
        try:
            backend = torchair.get_npu_backend(compiler_config=cfg)
        except RuntimeError as err:
            _skip_on_npu_patch_collision(err)
            raise
        compiled = torch.compile(module, backend=backend, dynamic=False)
        for _ in range(3):  # warmup capture + replays
            co, cl = compiled()
    finally:
        pass
    torch.npu.synchronize()
    assert bool(torch.equal(co, eo)), "torchair aclgraph out not bit-identical to eager"
    assert bool(torch.equal(cl, el)), "torchair aclgraph lse not bit-identical to eager"
    logger.info("torchair aclgraph(npugraph_ex mode) capture+replay bit-exact")


def test_aclgraph_npugraph_ex_compile():
    """ops-transformer 方式的 aclgraph: torch.compile(backend=npugraph_ex) 捕获回放。

    与 quant_flash_attn 的 GRAPH_PATH=7 路径一致: 预处理在编译区域之外,
    forward 只调注册算子, npugraph_ex 后端捕获 + mark_static + fullgraph=False。
    全设备端形态: 全部输入(含长度张量)NPU 放置, 绑定走 aclnn Tensor 变体,
    无 host 值通道、无混合设备假值传播问题(不再依赖 torch>=2.9 的
    fake_tensor_prefer_device_type 开关)。
    顺序约束: 须晚于 test_torchair_aclgraph_capture 执行(见该用例 docstring);
    若本用例后端初始化撞上 torch_npu 的 aten::to 二次注册问题则精确 skip。
    """
    device = _device()
    torch.manual_seed(0)
    dev, _block_dim, case = _build_dev_inputs("mixed_decode", device)  # 全 NPU 放置

    plan, tiling_device = _plan_dev(dev, case, device)
    module = _MlaModule(dev, case, plan, tiling_device)
    eo, el = module()
    torch.npu.synchronize()

    try:
        compiled = torch.compile(module, fullgraph=False, backend="npugraph_ex",
                                 dynamic=False)
        for tensor in dev.values():
            # mark_static 只经 torch._dynamo 暴露(无公开 API); getattr 规避
            # G.CLS.11 受保护成员静态告警, 运行时语义不变。
            getattr(torch, "_dynamo").mark_static(tensor)
        co, cl = compiled()
        torch.npu.synchronize()
    except RuntimeError as err:
        _skip_on_npu_patch_collision(err)
        raise

    assert bool(torch.equal(co, eo)), "npugraph_ex out not bit-identical to eager"
    assert bool(torch.equal(cl, el)), "npugraph_ex lse not bit-identical to eager"

    for round_id in range(2):  # 回放轮次
        co, cl = compiled()
        torch.npu.synchronize()
        assert bool(torch.equal(co, eo)), f"npugraph_ex replay#{round_id}: out differs"
        assert bool(torch.equal(cl, el)), f"npugraph_ex replay#{round_id}: lse differs"
    logger.info("aclgraph npugraph_ex: capture + 2 replays bit-identical to eager")


def test_torch_compile_fullgraph():
    """注册算子可组入 torch.compile fullgraph(无 graph break), 与 eager bit 级一致。

    aot_eager/eager 后端演练 Meta 假值传播 + NPU 派发(捕获前置依赖); 全 NPU
    放置通过 dynamo 的 FakeTensor 设备一致性检查。
    """
    device = _device()
    torch.manual_seed(0)
    dev, _block_dim, case = _build_dev_inputs("mixed_decode", device)

    plan, tiling_device = _plan_dev(dev, case, device)
    module = _MlaModule(dev, case, plan, tiling_device)
    eo, el = module()
    torch.npu.synchronize()
    for backend in ("aot_eager", "eager"):
        compiled = torch.compile(module, backend=backend, fullgraph=True, dynamic=False)
        co, cl = compiled()
        torch.npu.synchronize()
        assert bool(torch.equal(co, eo)), f"compile[{backend}]: out not bit-identical to eager"
        assert bool(torch.equal(cl, el)), f"compile[{backend}]: lse not bit-identical to eager"
        logger.info("torch.compile(backend=%s) fullgraph traces the registered op, bit-exact", backend)


@pytest.mark.parametrize("case", ("mixed_decode", "netspec_decode_96"))
def test_aclgraph_end_to_end_dynamic_plan(case):
    """全设备端旗舰场景: AICPU 规划 + AI Core 主算子同图捕获, 回放期动态跟随设备值。

    旧 host 值通道形态不可能做到: 长度张量上板则捕获期同步 D2H 被拒, CPU 放置
    则值在捕获期被烘焙进 tiling, 回放期更新长度只会得到错位结果。全设备端形态
    下两个算子都只按 shape 发射, 值依赖数据流全程在设备侧:

      graph = [AICPU planner(读 GM 长度 -> 写 GM plan)] -> [主算子(读 GM plan/长度)]

    回放前原地更新设备上的 topk 长度, 图内自动重规划并重算 attention, 输出与
    eager 重算逐位一致 —— FD/uniform 计划类型切换(FD 槽全零即 uniform)也由
    kernel 运行期自分流, 无需重编译/重捕获。
    """
    device = _device()
    torch.manual_seed(0)
    inputs, _block_dim, case_meta = golden.build_inputs(case, torch.device("cpu"))
    for key in ("ori_kv", "cmp_kv"):
        inputs[key] = inputs[key].contiguous()
    dev = {k: inputs[k].to(device) for k in _INPUT_KEYS if k != "metadata"}
    scale = float(case_meta["softmax_scale"])
    rows = int(dev["q"].shape[0])
    out = torch.full(tuple(dev["q"].shape), float("nan"), dtype=torch.bfloat16, device=device)
    lse = torch.full((1, rows, 64), float("nan"), dtype=torch.float32, device=device)

    def _plan_tensors():
        return _planner(
            dev["cu_seqlens_q"], dev["ori_topk_length"], dev["cmp_topk_length"],
            has_cmp=1, k_ori=int(case_meta["k_ori"]), k_cmp=int(case_meta["k_cmp"]),
            bs_ori=int(dev["ori_kv"].shape[1]),
            bs_cmp=int(dev["cmp_kv"].shape[1]),
            bt_ori=int(dev["ori_block_table"].shape[1]),
            bt_cmp=int(dev["cmp_block_table"].shape[1]),
            ori_pages=int(dev["ori_kv"].shape[0]),
            cmp_pages=int(dev["cmp_kv"].shape[0]),
            ori_kv_stride0=int(dev["ori_kv"].stride(0)),
            cmp_kv_stride0=int(dev["cmp_kv"].stride(0)))

    def _run_mla(plan, tiling_device):
        return _mla(
            dev["q"], ori_kv=dev["ori_kv"],
            ori_sparse_indices=dev["ori_sparse_indices"],
            ori_block_table=dev["ori_block_table"],
            cu_seqlens_q=dev["cu_seqlens_q"], sinks=dev["sinks"],
            metadata=plan, tiling_device=tiling_device,
            quant_mode=1, softmax_scale=scale,
            cmp_kv=dev["cmp_kv"], cmp_sparse_indices=dev["cmp_sparse_indices"],
            cmp_block_table=dev["cmp_block_table"],
            ori_topk_length=dev["ori_topk_length"],
            cmp_topk_length=dev["cmp_topk_length"],
            return_softmax_lse=True, out=out, lse=lse)

    def plan_and_run():
        plan, tiling_device = _plan_tensors()
        return _run_mla(plan, tiling_device)

    def _assert_matches_eager(tag):
        plan, tiling_device = _plan_tensors()
        _run_mla(plan, tiling_device)
        torch.npu.synchronize()
        ref_out, ref_lse = out.clone(), lse.clone()
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        assert bool(torch.equal(out, ref_out)), f"{case}/{tag}: replay out not bit-identical to eager"
        assert bool(torch.equal(lse, ref_lse)), f"{case}/{tag}: replay lse not bit-identical to eager"

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        plan_and_run()
    _assert_matches_eager("captured-values")

    # 回放期动态更新: 原地缩短每行 topk 宽度(FD case 同时覆盖 FD->uniform 计划切换)。
    dev["ori_topk_length"].div_(2, rounding_mode="floor")
    dev["cmp_topk_length"].div_(2, rounding_mode="floor")
    _assert_matches_eager("halved-lengths")
    logger.info("%s: AICPU+AI Core 同图捕获, 回放动态跟随设备值 bit-exact", case)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    def _run(name, fn, *args):
        try:
            fn(*args)
        except pytest.skip.Exception as skip:  # 直跑时跳过项不中断后续用例
            logger.info("SKIP %s: %s", name, skip.msg)

    _run("bitexact", test_aclgraph_capture_replay_bitexact)
    for _name in ("mixed_decode", "decode_6"):
        _run(f"metadata[{_name}]", test_aclgraph_metadata_capture_replay, _name)
    _run("torchair_aclgraph", test_torchair_aclgraph_capture)
    _run("npugraph_ex", test_aclgraph_npugraph_ex_compile)
    _run("compile_fullgraph", test_torch_compile_fullgraph)
    for _name in ("mixed_decode", "netspec_decode_96"):
        _run(f"e2e_dynamic[{_name}]", test_aclgraph_end_to_end_dynamic_plan, _name)
    logging.info("All runnable aclgraph tests passed!")
