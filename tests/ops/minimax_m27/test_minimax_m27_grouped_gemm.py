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
"""MiniMax-M2.7 grouped-GEMM precision test: compares the PyPTO fused kernel against the golden."""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as functional

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("minimax_m27.test")

try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise ImportError("torch_npu not available; this test only runs on Ascend NPU.") from exc

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
_p = _DIR
while _p != "/" and not os.path.isdir(os.path.join(_p, "src")):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)

from ops.pypto_tensor.minimax.minimax_grouped_gemm_impl import (  # noqa: E402
    minimax_moe_grouped_gemm,
    convert_minimax_weights,
    MoeDims,
)


def minimax_m27_grouped_gemm_golden(sorted_tokens, gate_up_proj, down_proj, expert_cumsum):
    """Eager per-expert golden reference for the grouped-GEMM SwiGLU MoE."""
    result = torch.zeros_like(sorted_tokens)
    for expert in range(gate_up_proj.shape[0]):
        start, end = int(expert_cumsum[expert]), int(expert_cumsum[expert + 1])
        if start == end:                                   # expert received no tokens
            continue
        gate_up = functional.linear(sorted_tokens[start:end].float(), gate_up_proj[expert].float())
        half = gate_up.shape[-1] // 2
        out = functional.linear(functional.silu(gate_up[..., :half]) * gate_up[..., half:],
                                down_proj[expert].float())
        result[start:end] = out.to(sorted_tokens.dtype)
    return result


def prep_env():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)


def get_device():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    return f"npu:{device_id}"


def load_test_cases():
    with open(os.path.join(_DIR, "test_cases.json"), "r") as handle:
        return json.load(handle)


def run_single_case(case, device):
    logger.info("=" * 60)
    logger.info("Test: %s — %s", case["id"], case.get("description", ""))
    logger.info("=" * 60)
    torch.manual_seed(case.get("seed", 42))

    params = case["params"]
    num_experts = params["num_experts"]
    hidden_size = params["hidden_size"]
    intermediate_size = params["intermediate_size"]
    counts = torch.tensor(params["counts"], dtype=torch.int64)
    num_tokens = int(counts.sum())
    cumsum = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    cumsum[1:] = torch.cumsum(counts, 0).to(torch.int32).to(device)

    sorted_tokens = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    gate_up_proj = torch.randn(num_experts, 2 * intermediate_size, hidden_size,
                               dtype=torch.bfloat16, device=device) * 0.02   # [E, 2I, H]
    down_proj = torch.randn(num_experts, hidden_size, intermediate_size,
                            dtype=torch.bfloat16, device=device) * 0.02       # [E, H, I]

    golden = minimax_m27_grouped_gemm_golden(sorted_tokens, gate_up_proj, down_proj, cumsum)

    w13_flat, w2_flat = convert_minimax_weights(gate_up_proj, down_proj)
    result = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    minimax_moe_grouped_gemm(
        sorted_tokens, (w13_flat, w2_flat), cumsum, result,
        MoeDims(num_experts, hidden_size, intermediate_size),
    )

    max_diff = (result.float() - golden.float()).abs().max().item()
    logger.info("  active_experts=%d/%d  N=%d  max_diff=%.6e",
                int((counts > 0).sum()), num_experts, num_tokens, max_diff)

    rtol = case.get("rtol", 8e-3)
    atol = case.get("atol", 8e-3)
    np.testing.assert_allclose(result.float().cpu().numpy(), golden.float().cpu().numpy(),
                               rtol=rtol, atol=atol,
                               err_msg=f"{case['id']}: PyPTO grouped GEMM != golden")
    out_shape = case.get("output", {}).get("shape")
    if out_shape is not None:
        assert list(result.shape) == out_shape, f"shape {list(result.shape)} != {out_shape}"
    logger.info("  [PRECISION_PASS] rtol=%s atol=%s", rtol, atol)


def test_minimax_m27_grouped_gemm():
    """pytest entry: run every case in test_cases.json."""
    torch_npu.npu.config.allow_internal_format = True
    prep_env()
    device = get_device()
    for case in load_test_cases()["test_cases"]:
        run_single_case(case, device)


def main():
    parser = argparse.ArgumentParser(description="MiniMax-M2.7 grouped-GEMM precision test")
    parser.add_argument("case_id", nargs="?", help="run a single case id")
    parser.add_argument("--list", action="store_true", help="list all cases")
    args = parser.parse_args()

    cases = load_test_cases()["test_cases"]
    if args.list:
        for case in cases:
            logger.info("  %s — %s", case["id"], case.get("description", ""))
        return
    torch_npu.npu.config.allow_internal_format = True
    prep_env()
    device = get_device()
    for case in (cases if not args.case_id else [c for c in cases if c["id"] == args.case_id]):
        run_single_case(case, device)
    logger.info("All tests passed!")


if __name__ == "__main__":
    main()
