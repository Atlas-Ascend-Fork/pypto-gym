# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""E2E test for flash_attention_score — integrated JIT kernel vs golden.

Precision standards (Ascend Precision Standard 2.1, L0):
  atol=1  rtol=0  MARE<10  MERE<2  RMSE<2
"""

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import torch
import torch_npu  # noqa: F401
import pypto

from experimental.ops_transformer.flash_attention_score.flash_attention_score_impl import (
    flash_attention_score_kernel_npu)
from flash_attention_score_golden import (
    flash_attention_score_golden,
    FlashAttentionInputs,
)

_MERE_EXEMPT = 2.0 ** (-8)


def _set_device() -> int:
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    return device_id


def _precision_verify(impl_out, gold_out, tag):
    gold_out = tuple(t.cpu() for t in gold_out)
    impl_out = tuple(t.cpu() for t in impl_out)

    names = ("output", "softmax_max", "softmax_sum")
    atol_rt = [(0.0001, 0.0078125), (0.000025, 0.005), (0.000025, 0.005)]

    all_ok = True

    for gold, impl, name, (atol, rtol) in zip(gold_out, impl_out, names, atol_rt):
        g = gold.float()
        im = impl.float()
        n_total = g.numel()

        abs_err = (im - g).abs()
        exceed = abs_err > (atol + rtol * g.abs())
        n_fail = exceed.sum().item()
        max_d = abs_err.max().item()

        passed = (n_fail == 0)
        print(f"  {tag}_{name}: n={n_total}  fail={n_fail}  "
              f"max_diff={max_d:.2e}  {'PASS' if passed else 'FAIL'}  "
              f"(atol={atol}, rtol={rtol})")

        if not passed:
            all_ok = False

    if not all_ok:
        return False

    g = gold_out[0].cpu().float()
    im = impl_out[0].cpu().float()
    n_tot = g.numel()

    abs_err = (im - g).abs()

    large_mask = g.abs() >= _MERE_EXEMPT
    n_small = (~large_mask).sum().item()
    n_large = large_mask.sum().item()

    if n_large > 0:
        g_large = g[large_mask]
        im_large = im[large_mask]
        abs_diff_large = abs_err[large_mask]
        relative_error = abs_diff_large / (g_large.abs() + 1e-7)
        mare = relative_error.max().item()
        mere = relative_error.mean().item()
        rmse = (im_large - g_large).pow(2).mean().sqrt().item()
    else:
        mare, mere, rmse = 0.0, 0.0, 0.0

    small_err_count = 0
    if n_small > 0:
        small_err_count = (abs_err[~large_mask] > 2**-16).sum().item()

    mare_ok = mare < 10.0
    mere_ok = mere < 2.0
    rmse_ok = rmse < 2.0

    print(f"    MARE = {mare:.6f}  {'PASS' if mare_ok else 'FAIL'}  (threshold < 10)")
    print(f"    MERE = {mere:.6f}  {'PASS' if mere_ok else 'FAIL'}  (threshold <  2)")
    print(f"    RMSE = {rmse:.6f}  {'PASS' if rmse_ok else 'FAIL'}  (threshold <  2)")
    print(f"    small (|gold|<2^-8): {n_small}/{n_tot}  "
          f"large: {n_large}/{n_tot}  small_err>{'2^-16'}: {small_err_count}")

    return mare_ok and mere_ok and rmse_ok


def _make_inputs(device, seed, shape, pse_type=1, keep_prob=1.0):
    torch.manual_seed(seed)

    B = shape["B"]
    N = shape["N"]
    N_kv = shape.get("N_kv", N)
    Sq = shape["Sq"]
    Skv = shape["Skv"]
    D_val = shape["D"]

    return FlashAttentionInputs(
        query=torch.randn(B, N, Sq, D_val,
                          dtype=torch.bfloat16, device=device) * 0.1,
        key=torch.randn(B, N_kv, Skv, D_val,
                        dtype=torch.bfloat16, device=device) * 0.1,
        value=torch.randn(B, N_kv, Skv, D_val,
                          dtype=torch.bfloat16, device=device) * 0.1,
        atten_mask=torch.zeros(Sq, Skv,
                               dtype=torch.bfloat16, device=device),
        pse=torch.randn(B, N, Sq, Skv,
                        dtype=torch.bfloat16, device=device) * 0.01,
        drop_mask=torch.ones(Sq, Skv,
                             dtype=torch.bfloat16, device=device),
        pse_type=pse_type,
        keep_prob=keep_prob,
        scale_value=1.0 / (D_val ** 0.5),
    )


def _run_kernel(inputs, shape, dev):
    B = shape["B"]
    N = shape["N"]
    Sq = shape["Sq"]
    D = shape["D"]

    total_q = B * N * Sq
    output = torch.zeros(total_q, D, dtype=torch.bfloat16, device=dev)
    softmax_max = torch.zeros(total_q, 1, dtype=torch.float32, device=dev)
    softmax_sum = torch.zeros(total_q, 1, dtype=torch.float32, device=dev)

    flash_attention_score_kernel_npu(
        inputs.query, inputs.key, inputs.value,
        inputs.atten_mask, inputs.pse, inputs.drop_mask,
        output, softmax_max, softmax_sum,
        inputs.pse_type, inputs.keep_prob, inputs.scale_value,
    )

    return (output.reshape(B, N, Sq, D),
            softmax_max.reshape(B, N, Sq, 1),
            softmax_sum.reshape(B, N, Sq, 1))


def test_l0() -> None:
    dev_id = _set_device()
    dev = torch.device(f"npu:{dev_id}")

    shape = {"B": 1, "N": 64, "Sq": 1024, "Skv": 1024, "D": 128}
    inputs = _make_inputs(dev, seed=50, shape=shape, pse_type=1, keep_prob=1.0)

    print("=" * 60)
    print("L0 — B=1 N=64 Sq=1024 Skv=1024 D=128")
    print(f"  pse_type=1, keep_prob=1.0, scale={inputs.scale_value:.6f}")

    gold = flash_attention_score_golden(inputs, npu=True)
    impl = _run_kernel(inputs, shape, dev)

    ok = _precision_verify(impl, gold, tag="L0")
    assert ok, "L0: precision check FAILED"
    print()


def test_l1() -> None:
    dev_id = _set_device()
    dev = torch.device(f"npu:{dev_id}")

    shape = {"B": 2, "N": 32, "Sq": 500, "Skv": 500, "D": 128}
    inputs = _make_inputs(dev, seed=71, shape=shape, pse_type=1, keep_prob=1.0)

    print("=" * 60)
    print("L1 — B=2 N=32 Sq=500 Skv=500 D=128 (tail blocks)")
    print(f"  pse_type=1, keep_prob=1.0, scale={inputs.scale_value:.6f}")

    gold = flash_attention_score_golden(inputs, npu=True)
    impl = _run_kernel(inputs, shape, dev)

    ok = _precision_verify(impl, gold, tag="L1")
    assert ok, "L1: precision check FAILED"
    print()


def test_l2() -> None:
    dev_id = _set_device()
    dev = torch.device(f"npu:{dev_id}")

    shape = {"B": 8, "N": 32, "Sq": 439, "Skv": 439, "D": 128}
    inputs = _make_inputs(dev, seed=99, shape=shape, pse_type=1, keep_prob=1.0)

    print("=" * 60)
    print("L2 — B=8 N=32 Sq=439 Skv=439 D=128 (multi-batch tail)")
    print(f"  pse_type=1, keep_prob=1.0, scale={inputs.scale_value:.6f}")

    gold = flash_attention_score_golden(inputs, npu=True)
    impl = _run_kernel(inputs, shape, dev)

    ok = _precision_verify(impl, gold, tag="L2")
    assert ok, "L2: precision check FAILED"
    print()


if __name__ == "__main__":
    print("flash_attention_score E2E test")
    print("=" * 60)
    print(f"  NPU device: {os.environ.get('TILE_FWK_DEVICE_ID', '0')}")
    print()

    failed = False
    for name, fn in [("L0", test_l0), ("L1", test_l1), ("L2", test_l2)]:
        try:
            fn()
            print(f"\N{white heavy check mark} {name} PASSED")
        except Exception as e:
            print(f"\N{cross mark} {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed = True

    print()
    if failed:
        print("[PRECISION_FAIL]")
        sys.exit(1)
    else:
        print("=" * 60)
        print("\N{white heavy check mark} ALL E2E TESTS PASSED")
        print("=" * 60)
        print("[PRECISION_PASS]")