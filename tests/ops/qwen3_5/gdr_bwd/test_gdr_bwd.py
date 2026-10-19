# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""gdr_bwd 算子精度测试入口（pytest）。

直接调用 ``gdr_bwd_impl.chunk_gated_delta_rule_backward_wrapper``（PyPTO NPU kernel），
以 ``gdr_bwd_golden.chunk_gated_delta_rule_bwd_torch_golden_aligned``（纯 torch 参考）为基准。

测试用例：
    test_t256                           B=1 T=256   H=16 D=128  2段 varlen（冒烟）
    test_t1024                          B=1 T=1024  H=16 D=128  7段 varlen（P0规模）
    test_t4096                          B=1 T=4096  H=16 D=128  15段 varlen
    test_varlen64_t32k_h8_bt64          B=1 T=32K   H=8  D=128  64段 varlen（默认 skip）

精度门限（kernel bf16 vs golden bf16量化对齐）：
    dq  atol/rtol=1e-1（dS carry 累积 + bf16 量化）
    dk  atol/rtol=3e-2（dS carry 累积较轻）
    dv  atol/rtol=3e-3（无 dS carry 依赖）
    db  atol/rtol=1e-1（beta 梯度链较长）
    dg  atol/rtol=2.0（gate 梯度跨 chunk 累积最严重）

用法：
    export TILE_FWK_DEVICE_ID=0
    python -m pytest tests/ops/qwen3_5/gdr_bwd/test_gdr_bwd.py -v
    python tests/ops/qwen3_5/gdr_bwd/test_gdr_bwd.py
"""
import os
import sys
import time
import logging

import pytest
import torch
import numpy as np

DEV = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
os.environ["TILE_FWK_DEVICE_ID"] = str(DEV)

torch.npu.set_device(DEV)
DEVICE = f"npu:{DEV}"

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 让 gdr_bwd 包（ops/pypto_tensor/qwen3_5/gdr_bwd/）可 import
_repo_root = HERE
for _ in range(5):
    if os.path.isdir(os.path.join(_repo_root, "ops", "pypto_tensor")):
        break
    _repo_root = os.path.dirname(_repo_root)
_ops_dir = os.path.join(_repo_root, "ops", "pypto_tensor")
if _ops_dir not in sys.path:
    sys.path.insert(0, _ops_dir)
del _repo_root, _ops_dir

from qwen3_5.gdr_bwd import gdr_bwd_impl as K   # noqa: E402
from gdr_bwd_golden import (  # noqa: E402
    chunk_gated_delta_rule_bwd_golden as golden_bwd,
    recompute_a_from_forward as recompute_a,
    detailed_tensor_compare,
)
from common_utils import detailed_allclose_manual as compare

PYTO_LN2 = getattr(K, '_LN2', 0.6931471805599453)
pypto_bwd = K.chunk_gated_delta_rule_backward_wrapper

GRAD_NAMES = ["dq", "dk", "dv", "db", "dg", "dh0", "dA_log", "ddt_bias"]
COMPARE_NAMES = ["dq", "dk", "dv", "db", "dg", "dh0"]


def _l2norm_vjp(dy, y, rstd):
    """L2norm 的 VJP：将对归一化后 y 的梯度 dy 转换为对原始 x 的梯度 dx。

    dx = rstd * (dy - (dy·y) * y)   其中 · 是沿最后一维点积
    """
    dot = (dy * y).sum(dim=-1, keepdim=True)
    return rstd * (dy - dot * y)



def _l2norm(x, eps=1e-6):
    r = torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)
    return x * r, r


def _varlen_cumsum(x, lens, chunk):
    """chunk-local cumsum over a packed varlen token axis (B==1)."""
    out = torch.empty_like(x)
    off = 0
    for cur_len in lens:
        seg = x[:, off:off + cur_len]
        nfull = (cur_len // chunk) * chunk
        if nfull:
            batch, _, n_heads = seg[:, :nfull].shape
            out[:, off:off + nfull] = (
                seg[:, :nfull].reshape(batch, nfull // chunk, chunk, n_heads).cumsum(2).reshape(batch, nfull, n_heads))
        if cur_len > nfull:
            out[:, off + nfull:off + cur_len] = seg[:, nfull:].cumsum(1)
        off += cur_len
    return out


def make_inputs(params, seed=0):
    """构造反向输入（l2norm 在 kernel 内，输入不预归一化）。

    支持两种模式:
    - varlen=False (默认): 等长 batch 模式，cu=None，strides 由 kernel 内推导
    - varlen=True: 传入 varlen 分段列表

    Returns:
        dict with keys: q_raw, k_raw, v, g_nat(natural cumsum), beta, scale,
                        cu(or None), q_hat, k_hat, q_rstd, k_rstd, do, h0=None
    """
    batch, seq_len, n_heads, d_head, chunk_size = params["batch"], params["seq_len"], \
                            params["n_heads"], params["d_head"], params["chunk_size"]
    varlen = params.get("varlen")
    case_name = params.get("name", "unknown")

    if varlen is not None:
        lens = [varlen[i + 1] - varlen[i] for i in range(len(varlen) - 1)]
        cu = torch.tensor(varlen, dtype=torch.int32, device=DEVICE)
        num_seqs = len(lens)
    else:
        lens = [seq_len] * batch
        cu = None
        num_seqs = batch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    q_raw = (torch.randn(batch, seq_len, n_heads, d_head, generator=gen) * 0.7)
    k_raw = (torch.randn(batch, seq_len, n_heads, d_head, generator=gen) * 0.7)
    v = (torch.randn(batch, seq_len, n_heads, d_head, generator=gen) * 0.5)
    g_per_token = torch.empty(batch, seq_len, n_heads).uniform_(-0.10, -0.001, generator=gen)
    beta = torch.rand(batch, seq_len, n_heads, generator=gen)

    # natural-log chunk-local cumsum (g_is_natural_cumsum=True)
    if varlen is not None:
        g_nat = _varlen_cumsum(g_per_token, lens, chunk_size).float()
    else:
        nfull = (seq_len // chunk_size) * chunk_size
        g_raw = g_per_token.float()
        g_nat = g_raw.clone()
        if nfull:
            g_chunked = g_raw[:, :nfull].view(batch, nfull // chunk_size, chunk_size, n_heads).cumsum(2)
            g_nat[:, :nfull] = g_chunked.view(batch, nfull, n_heads)
        if seq_len > nfull:
            g_nat[:, nfull:] = g_raw[:, nfull:].cumsum(1)

    # l2norm 残差（kernel 内做 l2norm，但 wrapper 的 aligned 接口需要归一化后的 q/k + rstd）
    q_hat, q_rstd = _l2norm(q_raw.float())
    k_hat, k_rstd = _l2norm(k_raw.float())

    scale = float(d_head ** -0.5)

    # do cotangent
    do = (torch.randn(batch, seq_len, n_heads, d_head, generator=gen) * 0.5)
    # h0/dht: golden 不支持 None，用零张量（与 wrapper h0=None 等价）
    h0 = torch.zeros(num_seqs, n_heads, d_head, d_head, device=DEVICE, dtype=torch.float32)
    dht = torch.zeros(num_seqs, n_heads, d_head, d_head, device=DEVICE, dtype=torch.float32)

    # to device
    q_raw_d = q_raw.to(DEVICE)
    k_raw_d = k_raw.to(DEVICE)
    v_d = v.to(DEVICE)
    g_nat_d = g_nat.to(DEVICE)
    beta_d = beta.to(DEVICE)
    q_hat_d = q_hat.to(DEVICE)
    k_hat_d = k_hat.to(DEVICE)
    q_rstd_d = q_rstd.to(DEVICE)
    k_rstd_d = k_rstd.to(DEVICE)
    do_d = do.to(DEVICE)

    varlen_label = f"varlen{len(lens)}" if varlen is not None else f"batch{batch}"
    return dict(
        q_raw=q_raw_d, k_raw=k_raw_d, v=v_d, g_nat=g_nat_d, beta=beta_d,
        scale=scale, cu=cu, chunk_size=chunk_size, num_seqs=num_seqs,
        q_hat=q_hat_d, k_hat=k_hat_d, q_rstd=q_rstd_d, k_rstd=k_rstd_d,
        do=do_d, h0=h0, dht=dht,
        lens=lens, case=case_name,
        shape=f"batch={batch} seq_len={seq_len} n_heads={n_heads} d_head={d_head} {varlen_label}",
    )


def call_pypto(inp):
    q_hat = inp["q_hat"]
    k_hat = inp["k_hat"]
    bt = inp["chunk_size"]
    num_seqs = inp["num_seqs"]
    cu = inp.get("cu")

    if cu is not None:
        # varlen 模式：batch=1，cu 定义分段边界，沿 dim=1 拼接
        cu64 = cu.to(torch.int64)
        a_list = []
        for n in range(num_seqs):
            t0, t1 = int(cu64[n]), int(cu64[n + 1])
            a_n = recompute_a(k_hat[:, t0:t1], inp["g_nat"][:, t0:t1] / float(PYTO_LN2),
                              inp["beta"][:, t0:t1], chunk_size=bt)
            a_list.append(a_n)
        a_tensor = torch.cat(a_list, dim=1)
    else:
        # batch 模式：每条 batch 独立序列，沿 dim=0 拼接
        a_list = []
        for n in range(num_seqs):
            a_n = recompute_a(k_hat[n:n + 1], inp["g_nat"][n:n + 1] / float(PYTO_LN2),
                              inp["beta"][n:n + 1], chunk_size=bt)
            a_list.append(a_n)
        a_tensor = torch.cat(a_list, dim=0)

    return pypto_bwd(
        inp["q_hat"].to(torch.bfloat16), inp["k_hat"].to(torch.bfloat16),
        inp["v"].to(torch.bfloat16), inp["g_nat"], inp["beta"].to(torch.bfloat16),
        a_tensor,
        inp["scale"],
        inp["h0"],
        inp["do"].to(torch.bfloat16),
        inp["dht"],
        cu_seqlens=cu,
        chunk_size=inp["chunk_size"],
        g_is_natural_cumsum=True,
        q_rstd=inp["q_rstd"],
        k_rstd=inp["k_rstd"],
    )


def call_golden(inp):
    q_hat = inp["q_hat"]
    k_hat = inp["k_hat"]
    q_hat_bf = q_hat.to(torch.bfloat16).to(torch.float32)
    k_hat_bf = k_hat.to(torch.bfloat16).to(torch.float32)
    v_bf = inp["v"].to(torch.bfloat16).to(torch.float32)
    beta_bf = inp["beta"].to(torch.bfloat16).to(torch.float32)
    do_bf = inp["do"].to(torch.bfloat16).to(torch.float32)
    num_seqs = inp["num_seqs"]
    cu = inp.get("cu")
    bt = inp["chunk_size"]

    if cu is not None:
        # varlen 模式，沿 dim=1 拼接
        cu64 = cu.to(torch.int64)
        a_list = []
        for n in range(num_seqs):
            t0, t1 = int(cu64[n]), int(cu64[n + 1])
            a_n = recompute_a(k_hat_bf[:, t0:t1], inp["g_nat"][:, t0:t1] / float(PYTO_LN2),
                              inp["beta"][:, t0:t1].to(torch.bfloat16).to(torch.float32),
                              chunk_size=bt)
            a_list.append(a_n)
        a_tensor = torch.cat(a_list, dim=1)
    else:
        # batch 模式，沿 dim=0 拼接
        a_list = []
        for n in range(num_seqs):
            a_n = recompute_a(k_hat_bf[n:n + 1], inp["g_nat"][n:n + 1] / float(PYTO_LN2),
                              inp["beta"][n:n + 1].to(torch.bfloat16).to(torch.float32),
                              chunk_size=bt)
            a_list.append(a_n)
        a_tensor = torch.cat(a_list, dim=0)
    a_bf = a_tensor.to(torch.bfloat16).to(torch.float32)

    out = golden_bwd(
        q_hat_bf, k_hat_bf, v_bf, inp["g_nat"] / float(PYTO_LN2), beta_bf,
        a_bf, inp["scale"], inp["h0"], do_bf, inp["dht"],
        cu_seqlens=cu.to(torch.int64) if cu is not None else None,
        chunk_size=bt,
    )
    out = list(out)
    out[0] = _l2norm_vjp(out[0], q_hat_bf, inp["q_rstd"])
    out[1] = _l2norm_vjp(out[1], k_hat_bf, inp["k_rstd"])
    return tuple(out)


# ─────────────────────────────────────────────────────────────────────────────
# precision mode
# ─────────────────────────────────────────────────────────────────────────────

def run_precision(inp):
    case = inp["case"]
    logging.info(f"\n{'=' * 80}")
    logging.info(f"PRECISION: {case}  {inp['shape']}")
    logging.info(f"{'=' * 80}")

    logging.info(f"  golden bwd begin.....")
    t0 = time.time()
    golden_result = call_golden(inp)
    logging.info(f"  golden bwd: {time.time() - t0:.1f}s")

    t0 = time.time()
    pypto_output = call_pypto(inp)
    torch.npu.synchronize()
    logging.info(f"  pypto bwd: {time.time() - t0:.1f}s")

    pypto_dict = dict(zip(GRAD_NAMES, pypto_output))
    golden_dict = dict(zip(GRAD_NAMES, golden_result))

    ok = True

    # shape check (dtype may differ: kernel bf16 vs golden fp32)
    logging.info("\n  shape (pypto_output vs G):")
    for n in COMPARE_NAMES:
        p, g = pypto_dict.get(n), golden_dict[n]
        s_ok = tuple(p.shape) == tuple(g.shape)
        ok &= s_ok
        logging.info(f"    {n:5s}: pypto_output{tuple(p.shape)} {p.dtype}  G{tuple(g.shape)} {g.dtype}  "
              f"{'OK' if s_ok else 'MISMATCH'}")

    # detailed_tensor_compare — cast both to fp32 for fair comparison
    # 反向 kernel 的 forward recompute（w/v_new/S_i）用 bf16 输入，golden 用 fp32。
    # dS carry 跨 chunk 累积误差导致 dq/dg 有系统性偏差（chunk 0 精确对齐，后续 chunk
    # 逐步发散）。dv 不受 dS carry 影响（PASS）。这是已知的 bf16 forward recompute 限制。
    # 门限按梯度分量分别设置：dv 严格（3e-3），dq/dk/db/dg 放宽。
    tol_map = {
        "dq": (1e-3, 1e-3),   # dS carry 累积 + bf16 量化
        "dk": (1e-3, 1e-3),   # dS carry 累积较轻
        "dv": (1e-3, 1e-3),   # 无 dS carry 依赖
        "db": (1e-2, 1e-2),   # beta 梯度链较长
        "dg": (1e-2, 1e-2),    # gate 梯度跨 chunk 累积最严重
        "dh0": (1e-3, 1e-3),
    }
    logging.info(f"\n  detailed_tensor_compare (per-gradient tolerances, both cast to fp32):")
    for n in COMPARE_NAMES:
        p = pypto_dict.get(n)
        g = golden_dict.get(n)
        if p is None or g is None:
            continue
        p, g = p.float(), g.float()
        try:
            rtol, atol = tol_map[n]
        except KeyError:
            pass
        compare(
            np.array(g.cpu().flatten().tolist()),
            np.array(p.cpu().flatten().tolist()),
            f"{n}",
            rtol=rtol,
            atol=atol,
        )
        r = detailed_tensor_compare(p, g, f"pypto_output-vs-G {n}",
                                    rtol=rtol, atol=atol, verbose=False)
        c_ok = r["all_close"]
        ok &= c_ok
        logging.info(f"    {n:5s}: oor={r['out_of_tolerance_ratio'] * 100:7.4f}%  "
              f"max_diff={r['max_diff']:.3e}  rtol={rtol:.0e} atol={atol:.0e}  "
              f"all_close={c_ok}  "
              f"{'PASS' if c_ok else 'FAIL'}")

    logging.info(f"\n  VERDICT: {'PASS' if ok else 'FAIL'}")
    return ok


# =============================================================================
# 测试辅助函数
# =============================================================================

def do_test_gdr_bwd(case_name, params, seed=0):
    """基于 params 运行精度验证。"""
    logging.info(f"=== run test case: {case_name} ===")
    inp = make_inputs(params, seed=seed)
    ok = run_precision(inp)
    if not ok:
        raise AssertionError(f"Case {case_name} FAILED")
    logging.info(f"=== {case_name}: PASS ===")


# =============================================================================
# test cases
# =============================================================================

@pytest.mark.soc("950", "910")
def test_t1024():
    params = dict(
        name="t1024",
        batch=1, seq_len=1024, n_heads=16, d_head=128, chunk_size=128,
        varlen=[0, 163, 297, 421, 540, 658, 781, 1024],
    )
    do_test_gdr_bwd("t1024", params)


@pytest.mark.soc("950", "910")
def test_t256():
    params = dict(
        name="t256",
        batch=1, seq_len=256, n_heads=16, d_head=128, chunk_size=128,
        varlen=[0, 163, 256],
    )
    do_test_gdr_bwd("t256", params)


@pytest.mark.soc("950", "910")
def test_t4096():
    params = dict(
        name="t4096",
        batch=1, seq_len=4096, n_heads=16, d_head=128, chunk_size=128,
        varlen=[0, 363, 711, 1046, 1370, 1676, 1975, 2269,
                2561, 2847, 3130, 3411, 3683, 3950, 4096],
    )
    do_test_gdr_bwd("t4096", params)


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large test case")
def test_varlen64_t32k_h8_bt64():
    params = dict(
        name="varlen64_t32k_h8_bt64",
        batch=1, seq_len=32768, n_heads=8, d_head=128, chunk_size=128,
        varlen=[0, 796, 1560, 2262, 2914, 3535, 4137, 4734, 5319, 5893,
                6415, 6925, 7422, 7898, 8358, 8802, 9234, 9656, 10073, 10488,
                10878, 11226, 11550, 11860, 12162, 12462, 12758, 13047, 13333, 13613,
                13893, 14173, 14451, 14728, 15004, 15279, 15551, 15822, 16089, 16354,
                16616, 16876, 17135, 17394, 17647, 17899, 18151, 18401, 18650, 18896,
                19138, 19376, 19611, 19842, 20072, 20302, 20530, 20756, 20981, 21204,
                21419, 21633, 21844, 22041, 32768],
    )
    do_test_gdr_bwd("varlen64_t32k_h8_bt64", params)


@pytest.mark.soc("950", "910")
def test_t2048():
    params = dict(
        name="t2048",
        batch=1, seq_len=2048, n_heads=4, d_head=128, chunk_size=128,
        varlen=[0, 2048],
    )
    do_test_gdr_bwd("t2048", params)


@pytest.mark.soc("950", "910")
def test_b2_t2048():
    params = dict(
        name="b2_t2048",
        batch=2, seq_len=2048, n_heads=4, d_head=128, chunk_size=128,
    )
    do_test_gdr_bwd("b2_t2048", params)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_b2_t2048()
    test_t1024()
    test_t256()
    test_t4096()
    test_varlen64_t32k_h8_bt64()
    test_t2048()