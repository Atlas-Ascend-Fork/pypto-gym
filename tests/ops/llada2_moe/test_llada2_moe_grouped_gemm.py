# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical test for the grouped GEMM MoE kernel (all experts in one call).

Verifies that the single-kernel grouped path matches the per-expert reference
(loop of torch matmuls) within BF16 tolerance.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("torch_npu not available; this test only runs on Ascend NPU.")
    raise RuntimeError("torch_npu not available; this test only runs on Ascend NPU.") from None

import pypto

_IMPL = Path(__file__).resolve().parents[3] / "ops" / "pypto_tensor" / "llada2_moe"
sys.path.insert(0, str(_IMPL))

from llada2_moe_grouped_gemm_impl import (
    llada2_moe_grouped_gemm,
)


def reference_per_expert(sorted_tokens, w13_list, w2_list, expert_cumsum):
    """Per-expert reference using pure PyTorch (FP32 accumulation)."""
    expert_count = len(w13_list)
    result = torch.zeros_like(sorted_tokens)

    for expert_idx in range(expert_count):
        start = expert_cumsum[expert_idx].item()
        end = expert_cumsum[expert_idx + 1].item()
        if start == end:
            continue
        x = sorted_tokens[start:end].float()
        w13 = w13_list[expert_idx].float()
        w2 = w2_list[expert_idx].float()

        gate_up = x @ w13
        intermediate_size = w13.shape[1] // 2
        gate = gate_up[..., :intermediate_size]
        up = gate_up[..., intermediate_size:]
        sw = F.silu(gate) * up
        down = sw @ w2
        result[start:end] = down.to(sorted_tokens.dtype)

    return result


def _run_grouped_gemm_test(counts, hidden_size, intermediate_size, dev):
    """Run one grouped-GEMM smoke case with mixed token counts."""
    torch.manual_seed(42)
    counts = torch.tensor(counts, dtype=torch.int64)
    expert_count = int(counts.numel())
    cumsum = torch.zeros(expert_count + 1, dtype=torch.int32, device=dev)
    cumsum[1:] = torch.cumsum(counts, 0).to(torch.int32).to(dev)
    token_count = int(counts.sum().item())
    sorted_tokens = torch.randn(token_count, hidden_size, dtype=torch.bfloat16, device=dev) * 0.02
    w13_list = [
        torch.randn(hidden_size, 2 * intermediate_size, dtype=torch.bfloat16, device=dev) * 0.02
        for _ in range(expert_count)
    ]
    w2_list = [
        torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=dev) * 0.02
        for _ in range(expert_count)
    ]
    w13_flat = torch.cat(w13_list, dim=0).contiguous()
    w2_flat = torch.cat(w2_list, dim=0).contiguous()
    result = torch.empty(token_count, hidden_size, dtype=torch.bfloat16, device=dev)
    llada2_moe_grouped_gemm(
        sorted_tokens, w13_flat, w2_flat, cumsum, result,
        num_experts=expert_count,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size)
    ref = reference_per_expert(sorted_tokens, w13_list, w2_list, cumsum)
    np.testing.assert_allclose(
        result.float().cpu().numpy(), ref.float().cpu().numpy(),
        rtol=8e-3, atol=8e-3,
        err_msg=(
            f"experts={expert_count}, tokens={token_count}: "
            f"grouped GEMM mismatch (counts={counts.tolist()})"))
    print(f"  PASS  experts={expert_count:>3d}  tokens={token_count:>4d}  counts={counts.tolist()}")


def test_llada2_moe_grouped_gemm():
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    dev = f"npu:{device_id}"
    hidden_size = 2048
    intermediate_size = 512

    _run_grouped_gemm_test([0, 1, 2, 4, 8, 16, 32, 0], hidden_size, intermediate_size, dev)


def main():
    pypto.set_host_options(compile_monitor_enable=1, compile_timeout=10,
                            compile_timeout_stage=5,
                            compile_monitor_print_interval=2)
    test_llada2_moe_grouped_gemm()
    print("\nAll grouped GEMM tests passed.")


if __name__ == "__main__":
    main()
