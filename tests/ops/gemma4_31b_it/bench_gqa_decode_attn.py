# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Fair benchmark: PyPTO kernel vs eager attention (pre-allocated).

Both paths pre-allocate all inputs and intermediates before timing.
- Kernel: pre-padded K/V/mask, pre-allocated output
- Eager:  pre-expanded K/V (FP32, transposed), pre-allocated score/attn/output buffers

Usage:
    export TILE_FWK_DEVICE_ID=0
    python3 tests/ops/gemma4_31b_it/bench_gqa_decode_attn.py --device npu:0
    python3 tests/ops/gemma4_31b_it/bench_gqa_decode_attn.py --device npu:0 --kind local
    python3 tests/ops/gemma4_31b_it/bench_gqa_decode_attn.py --device npu:0 --skv 64,512,1024
"""

import os
import sys
import time
import statistics
import argparse
from pathlib import Path

import torch
import torch.nn.functional as Fn
import numpy as np

_IMPL = Path(__file__).resolve().parents[3] / "ops" / "pypto_tensor" / "gemma4_31b_it"
sys.path.insert(0, str(_IMPL))

from gqa_decode_attn.gqa_decode_attn_impl import (
    gemma4_decode_attn_gqa, Nq, Nkv, GROUPS, D, W, S2_TILE,
)

B = 1
Sq = 1
WARMUP = 10
REPEAT = 100


def sync(device):
    if 'npu' in str(device):
        torch.npu.synchronize()
    elif 'cuda' in str(device):
        torch.cuda.synchronize()


def setup_inputs(Skv, layer_kind, device):
    """Pre-allocate all inputs and intermediates for both kernel and eager."""
    torch.manual_seed(42 + Skv)

    q = torch.randn(B, Nq, Sq, D, dtype=torch.bfloat16, device=device) * 0.3
    k = torch.randn(B, Nkv, Skv, D, dtype=torch.bfloat16, device=device) * 0.3
    v = torch.randn(B, Nkv, Skv, D, dtype=torch.bfloat16, device=device) * 0.3
    mask = torch.zeros(B, 1, Sq, Skv, dtype=torch.float32, device=device)

    # Sliding window
    eff_Skv = Skv
    if layer_kind == "local" and Skv > W:
        k = k[:, :, Skv - W:, :].contiguous()
        v = v[:, :, Skv - W:, :].contiguous()
        mask = mask[:, :, :, Skv - W:].contiguous()
        eff_Skv = W

    # ── Kernel inputs (pre-padded) ──────────────────────────────
    Skv_padded = ((eff_Skv + S2_TILE - 1) // S2_TILE) * S2_TILE
    q_kernel = q[0, :, 0, :].contiguous()                        # [Nq, D]

    if Skv_padded > eff_Skv:
        pad_len = Skv_padded - eff_Skv
        k_padded = Fn.pad(k, (0, 0, 0, pad_len))[0].contiguous()  # [Nkv, Skv_padded, D]
        v_padded = Fn.pad(v, (0, 0, 0, pad_len))[0].contiguous()
    else:
        k_padded = k[0].contiguous()
        v_padded = v[0].contiguous()

    mask_base = mask[0, 0, 0, :].float()
    if Skv_padded > eff_Skv:
        mask_full = Fn.pad(mask_base, (0, Skv_padded - eff_Skv), value=-1e30)
    else:
        mask_full = mask_base
    mask_3d = mask_full.view(1, 1, Skv_padded).expand(Nkv, GROUPS, Skv_padded).contiguous()

    out_kernel = torch.empty(Nq, D, dtype=torch.bfloat16, device=device)

    # ── Eager inputs (pre-expanded, pre-cast, pre-transposed) ──
    k_exp = (k[:, :, None, :, :]
             .expand(B, Nkv, GROUPS, eff_Skv, D)
             .reshape(B, Nq, eff_Skv, D))
    v_exp = (v[:, :, None, :, :]
             .expand(B, Nkv, GROUPS, eff_Skv, D)
             .reshape(B, Nq, eff_Skv, D))

    q_fp32 = q.float().contiguous()                               # [B, Nq, Sq, D]
    k_exp_fp32_t = k_exp.float().transpose(-2, -1).contiguous()   # [B, Nq, D, eff_Skv]
    v_exp_fp32 = v_exp.float().contiguous()                       # [B, Nq, eff_Skv, D]
    mask_fp32 = mask.float().contiguous()                          # [B, 1, Sq, eff_Skv]

    # Pre-allocated eager intermediate buffers
    scores_buf = torch.empty(B, Nq, Sq, eff_Skv, dtype=torch.float32, device=device)
    attn_buf = torch.empty_like(scores_buf)
    out_eager = torch.empty(B, Nq, Sq, D, dtype=torch.float32, device=device)

    kernel_args = (q_kernel, k_padded, v_padded, mask_3d, out_kernel)
    eager_args = (q_fp32, k_exp_fp32_t, v_exp_fp32, mask_fp32,
                  scores_buf, attn_buf, out_eager)

    return kernel_args, eager_args, eff_Skv


def bench_kernel(kernel_args, device, warmup, repeat):
    q_k, k_p, v_p, m3, out_k = kernel_args

    for _ in range(warmup):
        gemma4_decode_attn_gqa(q_k, k_p, v_p, m3, out_k)
    sync(device)

    times = []
    for _ in range(repeat):
        sync(device)
        t0 = time.perf_counter()
        gemma4_decode_attn_gqa(q_k, k_p, v_p, m3, out_k)
        sync(device)
        times.append(time.perf_counter() - t0)
    return times


def bench_eager(eager_args, device, warmup, repeat):
    q_f, k_t, v_f, m_f, s_buf, a_buf, o_buf = eager_args

    for _ in range(warmup):
        torch.matmul(q_f, k_t, out=s_buf)
        s_buf.add_(m_f)
        torch.softmax(s_buf, dim=-1, out=a_buf)
        torch.matmul(a_buf, v_f, out=o_buf)
    sync(device)

    times = []
    for _ in range(repeat):
        sync(device)
        t0 = time.perf_counter()
        torch.matmul(q_f, k_t, out=s_buf)
        s_buf.add_(m_f)
        torch.softmax(s_buf, dim=-1, out=a_buf)
        torch.matmul(a_buf, v_f, out=o_buf)
        sync(device)
        times.append(time.perf_counter() - t0)
    return times


def verify_outputs(kernel_args, eager_args, device):
    """Run once and check both paths produce similar results."""
    q_k, k_p, v_p, m3, out_k = kernel_args
    q_f, k_t, v_f, m_f, s_buf, a_buf, o_buf = eager_args

    gemma4_decode_attn_gqa(q_k, k_p, v_p, m3, out_k)
    sync(device)
    kernel_out = out_k.float().cpu().numpy()   # [Nq, D]

    torch.matmul(q_f, k_t, out=s_buf)
    s_buf.add_(m_f)
    torch.softmax(s_buf, dim=-1, out=a_buf)
    torch.matmul(a_buf, v_f, out=o_buf)
    sync(device)
    eager_out = o_buf[0, :, 0, :].cpu().numpy()  # [Nq, D]

    max_diff = float(np.abs(kernel_out - eager_out).max())
    mean_diff = float(np.abs(kernel_out - eager_out).mean())
    return max_diff, mean_diff


def main():
    parser = argparse.ArgumentParser(description="GQA decode: kernel vs eager benchmark")
    parser.add_argument("--device", type=str, default="npu")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--repeat", type=int, default=REPEAT)
    parser.add_argument("--skv", type=str, default="64,128,256,512,1024,2048",
                        help="Comma-separated Skv values")
    parser.add_argument("--kind", type=str, default="global", choices=["global", "local"])
    parser.add_argument("--no-verify", action="store_true", help="Skip output verification")
    args = parser.parse_args()

    device = args.device
    if device.startswith("npu"):
        if ":" in device:
            device_id = int(device.split(":")[1])
        else:
            device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
        import torch_npu  # noqa: F401
        torch.npu.set_device(device_id)
        device = f"npu:{device_id}"

    skv_list = [int(s) for s in args.skv.split(",")]

    print(f"Device: {device}  |  kind={args.kind}  |  warmup={args.warmup}  repeat={args.repeat}")
    print(f"Kernel compiled on NPU {os.environ.get('TILE_FWK_DEVICE_ID', '?')}")
    print()
    print(f"{'Skv':>6} {'EffSkv':>6} | "
          f"{'Kernel(ms)':>10} {'±':>6} | "
          f"{'Eager(ms)':>10} {'±':>6} | "
          f"{'Speedup':>7} | "
          f"{'MaxDiff':>9} {'MeanDiff':>9}")
    print("-" * 95)

    for Skv in skv_list:
        kernel_args, eager_args, eff_Skv = setup_inputs(Skv, args.kind, device)

        # Verify correctness
        if not args.no_verify:
            max_d, mean_d = verify_outputs(kernel_args, eager_args, device)
        else:
            max_d, mean_d = float('nan'), float('nan')

        # Benchmark
        kt = bench_kernel(kernel_args, device, args.warmup, args.repeat)
        et = bench_eager(eager_args, device, args.warmup, args.repeat)

        km = statistics.mean(kt) * 1000
        ks = statistics.stdev(kt) * 1000 if len(kt) > 1 else 0.0
        em = statistics.mean(et) * 1000
        es = statistics.stdev(et) * 1000 if len(et) > 1 else 0.0
        speedup = em / km if km > 0 else float('inf')

        print(f"{Skv:>6} {eff_Skv:>6} | "
              f"{km:>10.3f} {ks:>5.3f}  | "
              f"{em:>10.3f} {es:>5.3f}  | "
              f"{speedup:>6.2f}x | "
              f"{max_d:>9.2e} {mean_d:>9.2e}")


if __name__ == "__main__":
    main()
