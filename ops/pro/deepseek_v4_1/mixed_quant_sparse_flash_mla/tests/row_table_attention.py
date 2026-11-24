#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native GM row-table accuracy + timing for MQSMLA.

The attention kernel consumes the pre-resolved GM physical-row table directly:
``ori_sparse_indices`` / ``cmp_sparse_indices`` are the two ``[T1, 1, k*_tab]``
views of one combined ``[T1, k1_tab + k2_tab]`` table.  This script runs the
public wrapper once per call; the kernel now builds the GM table in its fused
prologue (no separate builder task).

The default case is ``spread_8192``: every block-table entry is a real page
permutation of the full 8192-page window, and the sparse indices are random
over that whole window.

Usage:
  python3 row_table_attention.py                       # spread_8192, Pro + precision
  python3 row_table_attention.py --case mixed_decode
  python3 row_table_attention.py --no-perf
"""

from __future__ import annotations

import argparse
import logging
import os
import runpy
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_THIS_DIR)
for _path in (_THIS_DIR, os.path.join(_THIS_DIR, "assets"), os.path.join(_OP_DIR, "op_kernel")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import mixed_quant_sparse_flash_mla_golden_cpu as golden  # noqa: E402
import mixed_quant_sparse_flash_mla_precision as verify  # noqa: E402
import mixed_quant_sparse_flash_mla as mq  # noqa: E402

_PROFILE_HELPER = "/workspace/g00955608/.opencode/skills/pypto-pro-kernel/scripts/profile_kernel.py"
logger = logging.getLogger(__name__)

_SCALE = 0.04419417382415922


def _device(device_id):
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    return device


def _kv_to_device(tensor, device):
    if tensor.is_contiguous():
        return tensor.to(device)
    strides = tuple(int(s) for s in tensor.stride())
    extent = 1
    for dim, stride in zip(tensor.shape, strides):
        extent += (int(dim) - 1) * stride
    backing = torch.full((extent,), 0xA5, dtype=tensor.dtype, device=device)
    view = backing.as_strided(tuple(tensor.shape), strides)
    view.copy_(tensor.to(device))
    return view


def _build_inputs(case, device):
    inputs, block_dim, _case = golden.build_inputs(case, torch.device("cpu"))
    reference = golden.reference(case, inputs)
    dev = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    for key in ("ori_kv", "cmp_kv"):
        dev[key] = _kv_to_device(inputs[key], device)
    return inputs, dev, block_dim, reference


def _run(inputs, dev, block_dim, device, scale, out=None, lse=None):
    t1 = int(inputs["q"].shape[0])
    if out is None:
        out = torch.full(tuple(inputs["q"].shape), float("nan"), dtype=torch.bfloat16, device=device)
    if lse is None:
        lse = torch.full((1, t1, 64), float("nan"), dtype=torch.float32, device=device)
    mq.mixed_quant_sparse_flash_mla(
        dev["q"],
        ori_kv=dev["ori_kv"],
        ori_sparse_indices=dev["ori_sparse_indices"],
        ori_block_table=dev["ori_block_table"],
        cu_seqlens_q=dev["cu_seqlens_q"],
        sinks=dev["sinks"],
        metadata=dev["metadata"],
        quant_mode=1,
        softmax_scale=float(scale),
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
    return out, lse


def _precision(name, reference, out, lse):
    pollution = bool(torch.isnan(out).any().item() or torch.isnan(lse).any().item())
    finite = bool(torch.isfinite(out.float()).all().item() and torch.isfinite(lse).all().item())
    ok_out, s_out = verify.check_precision(out.detach().cpu(), reference["out"])
    ok_lse, s_lse = verify.check_precision(lse.detach().cpu(), reference["lse"])
    lse_ok = bool(torch.allclose(lse.detach().cpu(), reference["lse"], rtol=1e-5, atol=1e-5))
    passed = (not pollution) and finite and ok_out and ok_lse and lse_ok
    logger.info(f"{name}: {'PASS' if passed else 'FAIL'} pollution_free={not pollution} finite={finite}")
    if not passed:
        logger.info("  out: %s", s_out)
        logger.info("  lse: %s lse_strict: %s", s_lse, lse_ok)
    return passed


def _measure(call, tasks_per_call):
    helper = runpy.run_path(_PROFILE_HELPER)
    return helper["measure_native"](
        call, warmup=3, repeats=20, tasks_per_call=tasks_per_call
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="spread_8192")
    parser.add_argument("--device", type=int, default=int(os.environ.get("TILE_FWK_DEVICE_ID", 0)))
    parser.add_argument("--no-perf", action="store_true")
    args = parser.parse_args()

    device = _device(args.device)
    inputs, dev, block_dim, reference = _build_inputs(args.case, device)
    out, lse = _run(inputs, dev, block_dim, device, _SCALE)
    passed = _precision("pro fused", reference, out, lse)
    logger.info(f"case={args.case}")
    if args.no_perf:
        return 0 if passed else 1

    _run(inputs, dev, block_dim, device, _SCALE, out=out, lse=lse)
    e2e = _measure(lambda: _run(inputs, dev, block_dim, device, _SCALE, out=out, lse=lse), 1)
    rows = [("fused builder+attention", e2e)]
    for name, stats in rows:
        logger.info(
            f"{name}: median={stats['median_us']:.4f} us "
            f"min={stats['min_us']:.4f} max={stats['max_us']:.4f} "
            f"tasks={stats['tasks_per_call']}"
        )
        for task in stats["per_task"]:
            logger.info(f"  task{task['position']}: {task['median_us']:.4f} us")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
