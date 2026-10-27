#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mixed_quant_sparse_flash_mla 的独立测试入口（精度矩阵 + 可选性能）。

针对合约的全部 P0 case（13 条）：

* 用 `tests/assets/mixed_quant_sparse_flash_mla_golden_cpu.py` 的 ``build_inputs``
  造输入（并按 host 规划器生成 kernel 消费的 ``metadata``），在设备上调用公开入口
  ``mixed_quant_sparse_flash_mla``，再用仓库内自包含方案 A 比较器
  `tests/assets/mixed_quant_sparse_flash_mla_precision.py` 逐 case 比对。
* 额外检查：输出污染门（torch.empty 语义，不允许残留 NaN）、有限性，以及（当 case
  提供紧致容差时）LSE 的 FP32 直接比较。

用法：
  # 全部 P0 case（默认）
  python test_mixed_quant_sparse_flash_mla.py

  # 子集
  python test_mixed_quant_sparse_flash_mla.py --cases mixed_decode,decode_1

  # pytest
  pytest ops/pypto_pro/deepseek_v4_1/mixed_quant_sparse_flash_mla/tests -q

返回码：全部精度通过为 0，否则 1。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import pytest
import torch

import torch_npu  # noqa: F401

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_THIS_DIR)
for _path in (_THIS_DIR, os.path.join(_THIS_DIR, "assets"), os.path.join(_OP_DIR, "op_kernel")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import mixed_quant_sparse_flash_mla_golden_cpu as golden  # noqa: E402
import mixed_quant_sparse_flash_mla_precision as verify  # noqa: E402
from mixed_quant_sparse_flash_mla import mixed_quant_sparse_flash_mla  # noqa: E402

#: P0 case 名（合约定义的 13 条，按合同顺序）。
P0_CASES = list(golden.load_case_params().keys())

#: 该 case 的空行/尾块附加准出要求严格 LSE：LSE 用 FP32 直接比较。
_LSE_RTOL = 1e-5
_LSE_ATOL = 1e-5

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.soc("950")


def _device():
    device = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}")
    torch.npu.set_device(device)
    return device


#: 公开入口调用所需的张量键（``build_inputs`` 返回的 12 个输入）。
_INPUT_KEYS = (
    "q",
    "ori_kv",
    "ori_sparse_indices",
    "ori_block_table",
    "ori_topk_length",
    "sinks",
    "metadata",
    "cmp_kv",
    "cmp_sparse_indices",
    "cmp_block_table",
    "cmp_topk_length",
    "cu_seqlens_q",
)


def _run_case(name, device):
    inputs, block_dim, case = golden.build_inputs(name, torch.device("cpu"))
    reference = golden.reference(name, inputs)
    for key in ("ori_kv", "cmp_kv"):
        inputs[key] = inputs[key].contiguous()
    dev = {k: inputs[k].to(device) for k in _INPUT_KEYS}
    rows = int(inputs["q"].shape[0])
    out = torch.full(tuple(inputs["q"].shape), float("nan"), dtype=torch.bfloat16, device=device)
    lse = torch.full((1, rows, 64), float("nan"), dtype=torch.float32, device=device)
    result = mixed_quant_sparse_flash_mla(
        dev["q"],
        ori_kv=dev["ori_kv"],
        ori_sparse_indices=dev["ori_sparse_indices"],
        ori_block_table=dev["ori_block_table"],
        cu_seqlens_q=dev["cu_seqlens_q"],
        sinks=dev["sinks"],
        metadata=dev["metadata"],
        quant_mode=1,
        softmax_scale=float(case["softmax_scale"]),
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

    pollution = bool(torch.isnan(out).any().item() or torch.isnan(lse).any().item())
    finite = bool(torch.isfinite(out.float()).all().item() and torch.isfinite(lse).all().item())
    ok_out, s_out = verify.check_precision(out.detach().cpu(), reference["out"])
    ok_lse, s_lse = verify.check_precision(lse.detach().cpu(), reference["lse"])
    lse_ok = bool(torch.allclose(lse.detach().cpu(), reference["lse"], rtol=_LSE_RTOL, atol=_LSE_ATOL))
    passed = (not pollution) and finite and ok_out and ok_lse and lse_ok
    detail = {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "pollution_free": not pollution,
        "finite": finite,
        "out": s_out,
        "lse": s_lse,
        "lse_strict": lse_ok,
    }
    logger.info("%s: %s", name, detail["status"])
    if not passed:
        logger.info("%s", s_out)
        logger.info("%s", s_lse)
    return detail


@pytest.mark.parametrize("case", P0_CASES)
def test_precision_p0(case):
    detail = _run_case(case, _device())
    assert detail["status"] == "PASS", detail


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="", help="comma-separated subset of the P0 cases")
    parser.add_argument(
        "--out",
        default=os.path.join(_THIS_DIR, "reports", "pro_pypto", "validation.json"),
        help="path for the per-case validation report",
    )
    args = parser.parse_args()
    device = _device()
    torch.manual_seed(0)
    wanted = [c for c in args.cases.split(",") if c]
    selected = [c for c in P0_CASES if not wanted or c in wanted]
    results = [_run_case(name, device) for name in selected]
    status = "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL"
    report = {
        "status": status,
        "case_count": len(results),
        "cases": [{"name": row["name"], "status": row["status"]} for row in results],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    logger.info("status=%s case_count=%s", status, len(results))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
