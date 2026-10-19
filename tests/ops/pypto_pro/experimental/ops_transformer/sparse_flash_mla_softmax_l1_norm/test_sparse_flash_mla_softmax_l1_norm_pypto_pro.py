# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pytest harness for sparse_flash_mla_softmax_l1_norm JIT kernel.

NPU PyPTO-Pro JIT kernel 输出与 CPU FP32 golden 参考实现逐元素比对，覆盖：
  * layout:  TND / BSND
  * sparse:  dense / sparse
  * mask:    mask_mode 0 / 3
  * seqused_q/k、topk_length、cmp_ratio>1
  * dtype:   FP16 / BF16

Run on NPU:
    pytest test_sparse_flash_mla_softmax_l1_norm_pypto_pro.py -v
or direct:
    python test_sparse_flash_mla_softmax_l1_norm_pypto_pro.py
"""

import logging
import math
import os
import sys

import torch
import torch_npu
try:
    import pytest
    _HAS_PYTEST = True
except ImportError:  # pragma: no cover - direct跑时无需 pytest
    pytest = None
    _HAS_PYTEST = False

if not _HAS_PYTEST:  # pragma: no cover
    class _Params:
        @staticmethod
        def param(*args, **kwargs):
            name = kwargs.get("id", args[0] if args else "")
            return (name, args[1] if len(args) > 1 else {})
    pytest = _Params()

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_REPO_ROOT = _THIS_DIR
while not os.path.isdir(os.path.join(_REPO_ROOT, "src")):
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
    if _REPO_ROOT == os.path.dirname(_REPO_ROOT):
        break
_SRC_DIR = _REPO_ROOT
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sparse_flash_mla_softmax_l1_norm_golden import softmax_l1_norm_golden
from compare import _compare

from ops.pypto_pro.experimental.ops_transformer.sparse_flash_mla_softmax_l1_norm.sparse_flash_mla_softmax_l1_norm_impl import (
    sparse_flash_mla_softmax_l1_norm_wrapper,
)


DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
DEVICE = f"npu:{DEVICE_ID}"

RTOL = 1e-2
ATOL = 1e-2


class _TilingWrapper:
    """把 kernel 的 OpTiling 包一层，额外注入 sparse_mode 供参考实现读取。"""

    def __init__(self, tiling, sparse_mode):
        self._tiling = tiling
        self.sparse_mode = sparse_mode

    def __getattr__(self, name):
        return getattr(self._tiling, name)


def _make_inputs(cfg, device):
    g, d = cfg["g"], cfg["d"]
    pt_dtype = cfg["pt_dtype"]
    is_tnd = cfg["is_tnd"]
    is_sparse = cfg["is_sparse"]

    if is_tnd:
        t1, t2 = cfg["t1"], cfg["t2"]
        q = torch.rand((t1, g, d), device=device, dtype=pt_dtype)
        k = torch.rand((t2, 1, d), device=device, dtype=pt_dtype)
        lse = torch.rand((1, t1, g), device=device, dtype=torch.float32)
        cu_q = torch.tensor(cfg["cu_q"], dtype=torch.int32, device=device)
        cu_k = torch.tensor(cfg["cu_k"], dtype=torch.int32, device=device)
    else:
        b, sq, sk = cfg["b"], cfg["sq"], cfg["sk"]
        q = torch.rand((b, sq, g, d), device=device, dtype=pt_dtype)
        k = torch.rand((b, sk, 1, d), device=device, dtype=pt_dtype)
        lse = torch.rand((b, sq, 1, g), device=device, dtype=torch.float32)
        cu_q = torch.tensor([0] * (b + 1), dtype=torch.int32, device=device)
        cu_k = torch.tensor([0] * (b + 1), dtype=torch.int32, device=device)

    b = q.shape[0] if not is_tnd else (cfg.get("b", 1))
    seq_q = cfg.get("seq_q")
    seq_k = cfg.get("seq_k")
    seq_q_t = torch.tensor(seq_q, dtype=torch.int32, device=device) if seq_q is not None else None
    seq_k_t = torch.tensor(seq_k, dtype=torch.int32, device=device) if seq_k is not None else None
    cmp_res = torch.tensor(cfg.get("cmp_res", [0] * b), dtype=torch.int32, device=device)

    sparse_idx = None
    topk = None
    if is_sparse:
        if is_tnd:
            sparse_idx = torch.randint(0, cfg["t2"], (cfg["t1"], 1, cfg["k_length"]),
                                       device=device, dtype=torch.int32)
            if cfg["has_topk"]:
                topk = torch.randint(1, cfg["k_length"] + 1, (cfg["t1"], 1),
                                     device=device, dtype=torch.int32)
        else:
            sparse_idx = torch.randint(0, cfg["sk"], (cfg["b"], cfg["sq"], 1, cfg["k_length"]),
                                       device=device, dtype=torch.int32)
            if cfg["has_topk"]:
                topk = torch.randint(1, cfg["k_length"] + 1, (cfg["b"], cfg["sq"], 1),
                                     device=device, dtype=torch.int32)

    return dict(q=q, k=k, lse=lse, cu_q=cu_q, cu_k=cu_k, seq_q=seq_q, seq_k=seq_k,
                seq_q_t=seq_q_t, seq_k_t=seq_k_t, cmp_res=cmp_res,
                sparse_idx=sparse_idx, topk=topk)


def _run_case(cfg, num_cores=4, seed=0):
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.set_device(DEVICE)
    device = DEVICE

    inputs = _make_inputs(cfg, device)
    q, k, lse = inputs["q"], inputs["k"], inputs["lse"]
    is_tnd = cfg["is_tnd"]
    is_sparse = cfg["is_sparse"]
    sparse_mode = cfg["sparse_mode"]
    g = cfg["g"]

    npu_out = sparse_flash_mla_softmax_l1_norm_wrapper(
        q, k, lse,
        sparse_indices=inputs["sparse_idx"],
        cu_seqlens_q=inputs["cu_q"] if is_tnd else None,
        cu_seqlens_k=inputs["cu_k"] if is_tnd else None,
        seqused_q=inputs["seq_q_t"],
        seqused_k=inputs["seq_k_t"],
        cmp_residual_k=inputs["cmp_res"],
        topk_length=inputs["topk"],
        mask_mode=sparse_mode,
        cmp_ratio=cfg["cmp_ratio"],
        softmax_scale=cfg["scale"],
        num_cores=num_cores,
    )
    torch.npu.synchronize()

    tiling = _build_tiling(cfg, is_tnd, is_sparse, num_cores)

    golden = softmax_l1_norm_golden(
        q, k, lse, inputs["sparse_idx"],
        inputs["cu_q"].cpu(), inputs["cu_k"].cpu(),
        inputs["seq_q"], inputs["seq_k"],
        inputs["cmp_res"].cpu(), inputs["topk"],
        _TilingWrapper(tiling, sparse_mode), is_tnd, is_sparse, sparse_mode,
    ).cpu()

    passed, max_abs, max_rel = _compare(npu_out, golden, RTOL, ATOL)
    return passed, max_abs, max_rel


def _build_tiling(cfg, is_tnd, is_sparse, num_cores):
    """构造 wrapper 相同的 OpTiling（供 golden 使用）。"""
    from ops.pypto_pro.experimental.ops_transformer.sparse_flash_mla_softmax_l1_norm.sparse_flash_mla_softmax_l1_norm_impl import (
        OpTiling,
    )
    g, d = cfg["g"], cfg["d"]
    if is_tnd:
        t1, t2 = cfg["t1"], cfg["t2"]
        b = cfg.get("b", 1)
    else:
        b, sq, sk = cfg["b"], cfg["sq"], cfg["sk"]
        t1, t2 = b * sq, b * sk
    k_length = cfg.get("k_length", 0) if is_sparse else 0
    max_seqlen_k = 0 if is_sparse else (t2 if is_tnd else cfg["sk"])
    out_len = k_length if is_sparse else max_seqlen_k
    sq_tiles = int(sum(cfg["seq_q"])) if cfg.get("seq_q") else (t1 if is_tnd else b * cfg["sq"])
    init_total = sq_tiles * out_len
    init_per = ((init_total + 8192 - 1) // 8192) * 8192
    return OpTiling(
        b=b, sq=cfg.get("sq", t1), sk=cfg.get("sk", t2), g=g, d=d,
        t1=t1, t2=t2, max_seqlen_k=max_seqlen_k, k_length=k_length,
        cmp_ratio=cfg["cmp_ratio"], init_per_core_num=init_per, init_total_num=init_total,
        softmax_scale=cfg["scale"],
        has_seqused_q=cfg.get("seq_q") is not None,
        has_seqused_k=cfg.get("seq_k") is not None,
        has_topk_length=cfg.get("has_topk", False),
    )


# ═══════════════════════════════════════════════════════════════════
# Test case matrix
# ═══════════════════════════════════════════════════════════════════

def _make_cases():
    g, d = 128, 512
    scale = 1.0 / math.sqrt(d)
    cases = []
    for sm in (0, 3):
        cases.append(pytest.param("TND_dense_mask%d" % sm, dict(
            is_tnd=True, is_sparse=False, sparse_mode=sm, b=1, g=g, d=d,
            t1=8, t2=256, max_seqlen_k=256, k_length=0, cmp_ratio=1,
            scale=scale, has_topk=False, cu_q=[0, 8], cu_k=[0, 256],
            pt_dtype=torch.float16), id="TND_dense_mask%d" % sm))
        cases.append(pytest.param("TND_sparse_mask%d" % sm, dict(
            is_tnd=True, is_sparse=True, sparse_mode=sm, b=1, g=g, d=d,
            t1=8, t2=256, max_seqlen_k=0, k_length=128, cmp_ratio=1,
            scale=scale, has_topk=True, cu_q=[0, 8], cu_k=[0, 256],
            pt_dtype=torch.float16), id="TND_sparse_mask%d" % sm))
    cases.append(pytest.param("TND_dense_mask0_seqused", dict(
        is_tnd=True, is_sparse=False, sparse_mode=0, b=2, g=g, d=d,
        t1=10, t2=256, max_seqlen_k=256, k_length=0, cmp_ratio=1,
        scale=scale, has_topk=False, cu_q=[0, 6, 10], cu_k=[0, 128, 256],
        seq_q=[6, 4], seq_k=[100, 128], pt_dtype=torch.float16),
        id="TND_dense_mask0_seqused"))
    cases.append(pytest.param("TND_dense_mask3_cmp4", dict(
        is_tnd=True, is_sparse=False, sparse_mode=3, b=1, g=g, d=d,
        t1=8, t2=256, max_seqlen_k=256, k_length=0, cmp_ratio=4,
        scale=scale, has_topk=False, cu_q=[0, 8], cu_k=[0, 256], cmp_res=[2],
        pt_dtype=torch.float16), id="TND_dense_mask3_cmp4"))
    for sm in (0, 3):
        cases.append(pytest.param("BSND_dense_mask%d" % sm, dict(
            is_tnd=False, is_sparse=False, sparse_mode=sm, b=2, sq=6, sk=256,
            g=g, d=d, max_seqlen_k=0, k_length=0, cmp_ratio=1, scale=scale,
            has_topk=False, pt_dtype=torch.float16), id="BSND_dense_mask%d" % sm))
        cases.append(pytest.param("BSND_sparse_mask%d" % sm, dict(
            is_tnd=False, is_sparse=True, sparse_mode=sm, b=2, sq=6, sk=256,
            g=g, d=d, max_seqlen_k=0, k_length=128, cmp_ratio=1, scale=scale,
            has_topk=True, pt_dtype=torch.float16), id="BSND_sparse_mask%d" % sm))
    cases.append(pytest.param("BSND_dense_mask0_seqused", dict(
        is_tnd=False, is_sparse=False, sparse_mode=0, b=2, sq=6, sk=256,
        g=g, d=d, max_seqlen_k=0, k_length=0, cmp_ratio=1, scale=scale,
        has_topk=False, seq_q=[4, 6], seq_k=[200, 256], pt_dtype=torch.float16),
        id="BSND_dense_mask0_seqused"))
    cases.append(pytest.param("BSND_sparse_mask3_cmp4", dict(
        is_tnd=False, is_sparse=True, sparse_mode=3, b=2, sq=6, sk=256,
        g=g, d=d, max_seqlen_k=0, k_length=128, cmp_ratio=4, scale=scale,
        has_topk=True, cmp_res=[2, 1], pt_dtype=torch.float16),
        id="BSND_sparse_mask3_cmp4"))
    cases.append(pytest.param("TND_dense_mask0_bf16", dict(
        is_tnd=True, is_sparse=False, sparse_mode=0, b=1, g=g, d=d,
        t1=8, t2=256, max_seqlen_k=256, k_length=0, cmp_ratio=1, scale=scale,
        has_topk=False, cu_q=[0, 8], cu_k=[0, 256], pt_dtype=torch.bfloat16),
        id="TND_dense_mask0_bf16"))
    cases.append(pytest.param("TND_sparse_mask3_bf16", dict(
        is_tnd=True, is_sparse=True, sparse_mode=3, b=1, g=g, d=d,
        t1=8, t2=256, max_seqlen_k=0, k_length=128, cmp_ratio=1, scale=scale,
        has_topk=True, cu_q=[0, 8], cu_k=[0, 256], pt_dtype=torch.bfloat16),
        id="TND_sparse_mask3_bf16"))
    return cases


CASES = _make_cases()


if _HAS_PYTEST:
    @pytest.mark.soc("950")
    @pytest.mark.parametrize("name,cfg", CASES)
    def test_sparse_flash_mla_softmax_l1_norm_pypto_pro(name, cfg):
        """NPU JIT kernel 输出 vs CPU FP32 golden。"""
        passed, max_abs, max_rel = _run_case(cfg)
        log.info("  [%s] max_abs=%.3e max_rel=%.3e [%s]", name, max_abs, max_rel,
                 "PASS" if passed else "FAIL")
        assert passed, f"Precision check failed for {name}: max_abs={max_abs} max_rel={max_rel}"


def main():
    all_pass = True
    for case in CASES:
        if hasattr(case, "values"):
            name, cfg = case.values
        else:
            name, cfg = case
        try:
            passed, max_abs, max_rel = _run_case(cfg)
        except Exception as exc:
            log.info("  EXCEPTION in %s: %s", name, exc)
            passed = False
            max_abs = max_rel = float("nan")
        log.info("  %-30s max_abs=%.3e max_rel=%.3e %s", name, max_abs, max_rel,
                 "PASS" if passed else "FAIL")
        all_pass = all_pass and passed
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())