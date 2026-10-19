# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""gdr_fwd 算子精度测试入口（pytest）。

直接调用 ``gdr_fwd_impl.chunk_gated_delta_rule_wrapper``（PyPTO NPU kernel），
以 ``gdr_fwd_golden.gdr_fwd_golden``（纯 torch 参考）为基准。

测试用例：
    test_bt128                    B=2 T=512  H=4  D=128  chunk_size=128
    test_depth256_T32K_H4         B=1 T=32K  H=4  D=128  链深 256（默认 skip）
    test_varlen_T32K_H8           B=1 T=32K  H=8  D=128  64段 varlen + l2norm

精度门限（kernel bf16 vs golden emulate_bf16）：
    o            atol/rtol=2e-3  l2_rel<=3e-3
    final_state  atol/rtol=5e-3  l2_rel<=3e-3
    long case                 l2_rel<=6e-3

用法：
    export TILE_FWK_DEVICE_ID=0
    python -m pytest tests/ops/qwen3_5/gdr_fwd/test_gdr_fwd.py -v
    python tests/ops/qwen3_5/gdr_fwd/test_gdr_fwd.py
"""

import os
import sys
from dataclasses import dataclass

_test_dir = os.path.dirname(os.path.abspath(__file__))
if _test_dir not in sys.path:      # 让 gdr_fwd_golden 可直接 import
    sys.path.insert(0, _test_dir)
# 让 gdr_fwd 包（ops/pypto_tensor/qwen3_5/gdr_fwd/）可 import
_repo_root = _test_dir
for _ in range(5):
    if os.path.isdir(os.path.join(_repo_root, "ops", "pypto_tensor", "qwen3_5")):
        break
    _repo_root = os.path.dirname(_repo_root)
_ops_dir = os.path.join(_repo_root, "ops", "pypto_tensor", "qwen3_5")
if _ops_dir not in sys.path:
    sys.path.insert(0, _ops_dir)
del _repo_root, _ops_dir
# ═══════════════════════════════════════════════════════════════════════════════

import logging

import pytest
import torch
import torch_npu

from gdr_fwd_golden import gdr_fwd_golden
from gdr_fwd import gdr_fwd_impl
from gdr_fwd.gdr_fwd_impl import chunk_gated_delta_rule_wrapper

# ─────────────────────────────────────────────────────────────────────────────
# 判据门限
# ─────────────────────────────────────────────────────────────────────────────
# 主判据：kernel vs golden(emulate_bf16=True)。atol/rtol 做**逐元素**把关；
# l2_rel 门限做**整体**把关，二者必须同时满足。
MAIN_ATOL_O, MAIN_RTOL_O = 2e-3, 2e-3          # o：bf16 输出，1 ulp ≈ 4e-3·|x|
MAIN_ATOL_S, MAIN_RTOL_S = 5e-3, 5e-3          # final_state：fp32，跨 chunk 累加量级更大
MAIN_L2_GATE = 3e-3
LONG_L2_GATE = 6e-3    # 长序列（256+ chunks）状态递推累积舍入误差，门限适当放宽


@dataclass
class CaseShape:
    """输入构造的形状 + 配置参数，``_make_case`` 的入参。"""
    b: int
    t: int
    h: int
    hv: int
    d: int
    seed: int = 0
    g_range: tuple = (-0.10, -0.001)
    with_state: bool = False
    n_seq: int | None = None
    dtype: torch.dtype = torch.bfloat16


# =============================================================================
# 基础设施
# =============================================================================

def _set_device() -> None:
    """按标准环境变量初始化 NPU 设备。必须在任何张量落 NPU 之前调用。"""
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))


def _stats(a, b):
    """返回 ``(max_abs, l2_rel)``；``l2_rel = ‖a-b‖₂ / ‖b‖₂`` 是主指标。"""
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    denom = b.norm().item()
    return diff.max().item(), (diff.norm().item() / denom if denom > 0 else 0.0)


# =============================================================================
# 输入构造 —— 约束来自 SPEC §3.1（dtype）与 §5（值域）
# =============================================================================

def _make_case(shape: CaseShape):
    """构造一组语义合法的输入。

    * ``q`` / ``k``：先 L2 归一化再落 ``dtype``（``use_qk_l2norm_in_kernel=False``
      要求上游已归一化）。
    * ``v``：``dtype``，乘 0.5 收敛动态范围。
    * ``g``：fp32，**log 域**遗忘门，恒 ``<= 0``。
    * ``beta``：``dtype``，post-sigmoid 空间 ``(0, 1)``。
    * ``initial_state``：``[N, HV, K, V]`` fp32，**K 在 V 前**（要求 1）。

    ``dtype`` 的取值直接决定比较分辨率，两档各有唯一正确的用法：

    * **bf16（默认，用于所有 kernel 对比）** —— SPEC §3.1 规定的 IO dtype。
      golden 的 matmul 操作数 dtype 由调用方决定（``gdr_fwd_golden.py:244`` 原注释），
      传 fp32 会让 golden 用比 kernel 更精确的 q/k，主判据失真。
    * **fp32（仅用于 suite N 的 golden 自洽）** —— 两侧 ``o`` 的 dtype 都跟随 ``q``
      （SPEC §3.2），传 bf16 会把 ``o`` 量化到 bf16 网格，比较分辨率被截断到
      1 ulp（实测：仅 17/65536 个近零元素差 1 ulp，就已越过 SPEC §7 的 1e-5），
      根本测不出 fp32 层面的算法自洽性。SPEC §7 第一行要求的正是 fp32-vs-fp32。
    """
    dev = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}")
    gen = torch.Generator(device="cpu").manual_seed(shape.seed)

    q = torch.randn(shape.b, shape.t, shape.h, shape.d, generator=gen)
    q = (q / q.norm(dim=-1, keepdim=True)).to(shape.dtype).to(dev)
    k = torch.randn(shape.b, shape.t, shape.h, shape.d, generator=gen)
    k = (k / k.norm(dim=-1, keepdim=True)).to(shape.dtype).to(dev)
    v = (torch.randn(shape.b, shape.t, shape.hv, shape.d, generator=gen) * 0.5).to(shape.dtype).to(dev)
    g = torch.empty(shape.b, shape.t, shape.hv).uniform_(shape.g_range[0], shape.g_range[1],
                                                         generator=gen).float().to(dev)
    beta = torch.rand(shape.b, shape.t, shape.hv, generator=gen).to(shape.dtype).to(dev)

    h0 = None
    if shape.with_state:
        n = shape.n_seq if shape.n_seq is not None else shape.b
        h0 = (torch.randn(n, shape.hv, shape.d, shape.d, generator=gen) * 0.1).float().to(dev)
    return dict(q=q, k=k, v=v, g=g, beta=beta, h0=h0)


def _check_tensor(got, ref, name, atol, rtol, l2_gate):
    """逐个元素 + 整体 l2_rel 双门限检查。"""
    g, r = got.cpu().float(), ref.cpu().float()
    max_abs, l2_rel = _stats(g, r)
    diff = (g - r).abs()
    tol = atol + rtol * r.abs()
    oor = int((diff > tol).sum().item())
    ok = oor == 0 and l2_rel <= l2_gate
    logging.info("  %s: max_abs=%.3e l2_rel=%.3e oor=%d/%d %s",
                 name, max_abs, l2_rel, oor, g.numel(), "PASS" if ok else "FAIL")
    return ok


def _run_pair(inp, case, *, kernel_kwargs):
    kw = dict(kernel_kwargs or {})
    args = (inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"])
    o_ker, s_ker = chunk_gated_delta_rule_wrapper(*args, **kw)
    o_ref, s_ref = gdr_fwd_golden(*args, **kw, emulate_bf16=True)
    ok = _check_tensor(o_ker, o_ref, "o", MAIN_ATOL_O, MAIN_RTOL_O, MAIN_L2_GATE)
    if s_ker is not None:
        ok &= _check_tensor(s_ker, s_ref, "final_state", MAIN_ATOL_S, MAIN_RTOL_S, MAIN_L2_GATE)
    if not ok:
        raise AssertionError(f"Case {case} FAILED")


def _run_long_case(case, shape, bt):
    inp = _make_case(shape)
    args = (inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"])
    kw = dict(scale=shape.d ** -0.5, initial_state=None, output_final_state=True,
              use_qk_l2norm_in_kernel=True, chunk_size=bt)
    o_ker, s_ker = chunk_gated_delta_rule_wrapper(*args, **kw)
    o_ref, s_ref = gdr_fwd_golden(*args, **kw, emulate_bf16=True)
    ok = _check_tensor(o_ker, o_ref, "o", MAIN_ATOL_O, MAIN_RTOL_O, LONG_L2_GATE)
    if s_ker is not None:
        ok &= _check_tensor(s_ker, s_ref, "final_state", MAIN_ATOL_S, MAIN_RTOL_S, LONG_L2_GATE)
    if not ok:
        raise AssertionError(f"Case {case} FAILED")


# =============================================================================
# 测试辅助函数
# =============================================================================

def do_test_gdr_fwd_pair(case_name, shape, call_kwargs):
    """基于 shape 和 call_kwargs 运行 pair 类型测试用例。"""
    _set_device()
    logging.info(f"\n=== run test case: {case_name} ===")
    inp = _make_case(shape)
    kw = dict(call_kwargs)
    if shape.with_state:
        kw["initial_state"] = inp["h0"]
    if isinstance(kw.get("cu_seqlens"), list):
        kw["cu_seqlens"] = torch.tensor(kw["cu_seqlens"], dtype=torch.long, \
                device=f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}")
    _run_pair(inp, case_name, kernel_kwargs=kw)
    logging.info(f"=== {case_name}: PASS ===")


def do_test_gdr_fwd_long(case_name, shape, bt):
    """基于 shape 和 bt 运行 long 类型测试用例。"""
    _set_device()
    logging.info(f"\n=== run test case: {case_name} ===")
    _run_long_case(case_name, shape, bt)
    logging.info(f"=== {case_name}: PASS ===")


# =============================================================================
# test cases
# =============================================================================

@pytest.mark.soc("950", "910")
def test_bt128():
    shape = CaseShape(b=2, t=512, h=4, hv=4, d=128, seed=8)
    call_kwargs = dict(chunk_size=128)
    do_test_gdr_fwd_pair("bt128", shape, call_kwargs)


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large test case")
def test_depth256_t32k_h4():
    shape = CaseShape(b=1, t=32768, h=4, hv=4, d=128, seed=900)
    do_test_gdr_fwd_long("depth256_T32K_H4", shape, bt=128)


@pytest.mark.soc("950", "910")
def test_varlen_t32k_h8():
    shape = CaseShape(b=1, t=32768, h=8, hv=8, d=128, seed=10)
    cu_seqlens = [0, 796, 1560, 2262, 2914, 3535, 4137, 4734, 5319, 5893,
                  6415, 6925, 7422, 7898, 8358, 8802, 9234, 9656, 10073, 10488,
                  10878, 11226, 11550, 11860, 12162, 12462, 12758, 13047, 13333, 13613,
                  13893, 14173, 14451, 14728, 15004, 15279, 15551, 15822, 16089, 16354,
                  16616, 16876, 17135, 17394, 17647, 17899, 18151, 18401, 18650, 18896,
                  19138, 19376, 19611, 19842, 20072, 20302, 20530, 20756, 20981, 21204,
                  21419, 21633, 21844, 22041, 32768]
    call_kwargs = dict(chunk_size=128, use_qk_l2norm_in_kernel=True,
                       cu_seqlens=cu_seqlens)
    do_test_gdr_fwd_pair("varlen_T32K_H8", shape, call_kwargs)


@pytest.mark.soc("950", "910")
def test_b2_t2048():
    shape = CaseShape(b=2, t=2048, h=4, hv=4, d=128, seed=42)
    call_kwargs = dict(chunk_size=128)
    do_test_gdr_fwd_pair("b2_t2048", shape, call_kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_bt128()
