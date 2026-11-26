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
"""attn_prologue PyPTO-Pro 测试入口（精度 + 可选性能）。

职责
----
唯一执行入口：构造 CPU 输入（复用独立 golden 的 fixture）→ 搬到 NPU（权重 ``to_nz``、
FP8 字节重解释）→ 调用公开 wrapper ``attn_prologue(...)`` → 用 FP64 值域判据
（``attn_prologue_precision``）逐 case 比较 → 写 ``reports/pro_pypto/validation.json``。

精度判据（冻结）
--------------------------------------
* Q：全有限，且 ``|got-ref| > 0.02 + 0.02*|ref|`` 比例 ≤ 0.1%。
* QR/descale/KV 字节：FP32 首过提名整行 → 独立 FP64 复核，``hard`` 必须为 0。
* 未命中 cache slot：逐字节不变。

运行
----
    D=ops/pro/deepseek_v4_1/attn_prologue
    python $D/tests/test_attn_prologue_routes.py                 # 全 32 case 精度（默认 1 卡）
    python $D/tests/test_attn_prologue_routes.py --t 1,72,1024   # 子集
    python $D/tests/test_attn_prologue_routes.py --case-id decode-b12-s6-t72
    python $D/tests/test_attn_prologue_routes.py --list          # 列出 case（不碰 NPU）
    pytest $D/tests -q                                    # 4 个冒烟 case

CPU golden 与判据自检不需要 NPU；只有 NPU 路径才 import torch_npu / 算子。
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OP_ROOT = HERE.parent
for _p in (str(HERE), str(HERE / "assets"), str(OP_ROOT / "op_kernel")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _assert_paged_cache(ref, flat, prefix, cache_index, block, page_stride, row_stride, cols, sentinel):
    torch.npu.synchronize()
    ref_flat = ref.reshape(-1).cpu()
    paged_flat = flat[prefix:].cpu()
    written = torch.zeros(paged_flat.numel(), dtype=torch.bool)
    for slot in cache_index.cpu().tolist():
        base = (slot // block) * page_stride + (slot % block) * row_stride
        assert torch.equal(paged_flat[base:base + cols],
                           ref_flat[slot * cols:(slot + 1) * cols])
        written[base:base + cols] = True
    assert (paged_flat[~written] == sentinel).all()
    assert (flat[:prefix] == sentinel).all()


def _load_asset(module_name, filename):
    """Load one asset by path so renamed operators cannot collide on module names."""
    spec = importlib.util.spec_from_file_location(module_name, HERE / "assets" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


golden = _load_asset("attn_prologue_gym_golden_cpu", "attn_prologue_golden_cpu.py")
verify = _load_asset("attn_prologue_gym_precision", "attn_prologue_precision.py")

LOG = logging.getLogger("attn_prologue.test")

_FRACTAL_NZ = 29
_E4M3 = torch.float8_e4m3fn
_E8M0 = torch.float8_e8m0fnu
_WEIGHT_FIELDS = ("wqa", "wqb", "wkv", "descale_wqa", "descale_wqb", "descale_wkv",
                  "norm_weight_qr", "norm_weight_kv")
_PER_TOKEN_FIELDS = ("x", "descale_x", "rope_sin", "rope_cos", "cache_index")


def _default_device() -> str:
    return os.environ.get("TILE_FWK_DEVICE_ID", "0")


def _resolve_device(spec: str) -> str:
    LOG.info("============ device: %s", spec)
    if spec.startswith("npu"):
        return spec
    return f"npu:{spec}"


def _to_nz(weight: torch.Tensor) -> torch.Tensor:
    import torch_npu  # noqa: F401

    return torch_npu.npu_format_cast(weight.contiguous(), _FRACTAL_NZ)


def _device_input(name, value, device, to_nz):
    """Move one ND fixture to the public API, preserving encoded FP8 bytes.

    FP8/E8M0 carriers are reinterpreted (view), never numerically cast; only the
    three weights go through ``to_nz``. Non-FP8 fields are a plain ``.to``.
    """
    weights = ("wqa", "wqb", "wkv")
    if name in ("x",) + weights:
        dtype = _E4M3
    elif name in ("descale_x", "descale_wqa", "descale_wqb", "descale_wkv"):
        dtype = _E8M0
    else:
        return value.to(device)
    if value.dtype not in (torch.int8, torch.uint8, dtype):
        raise TypeError(f"{name} must contain encoded {dtype} bytes, got {value.dtype}")
    payload = value.contiguous().view(torch.int8).to(device)
    if name in weights:
        payload = to_nz(payload)
    return payload.view(dtype)


class DeviceWeights:
    """Cache the NZ-converted / device-moved weight fields per device."""

    def __init__(self, weights: golden.SharedWeights, device: str):
        self._fields = {
            name: _device_input(name, value, device, _to_nz)
            for name, value in weights.fields.items()
        }

    def as_dict(self):
        return dict(self._fields)


def _load_wrapper():
    """Lazily load the Pro public wrapper by path (keeps CPU collection NPU-free)."""
    import torch_npu  # noqa: F401
    spec = importlib.util.spec_from_file_location(
        "attn_prologue_gym_kernel", OP_ROOT / "op_kernel" / "attn_prologue.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["attn_prologue_gym_kernel"] = module
    spec.loader.exec_module(module)
    return module.attn_prologue


def _run_case(case, device, weights, shadow, wrapper, dev_weights, *,
              row_chunk=256, norm_eps=1e-6):
    inputs = golden.make_case_inputs(case, weights)
    cache = inputs.kv_cache.to(device)
    values = dev_weights.as_dict()
    for name in _PER_TOKEN_FIELDS:
        values[name] = _device_input(name, getattr(inputs, name), device, _to_nz)
    values["kv_cache"] = cache
    q, qr, descale_qr = wrapper(
        values["x"], values["wqa"], values["wqb"], values["wkv"],
        values["descale_x"], values["descale_wqa"], values["descale_wqb"],
        values["descale_wkv"], values["norm_weight_qr"], values["norm_weight_kv"],
        values["rope_sin"], values["rope_cos"], values["cache_index"], cache,
        norm_eps=norm_eps,
    )
    torch.npu.synchronize()
    outputs = (q, qr, descale_qr)
    t0 = time.time()
    stats = verify.validate_outputs(
        inputs, weights, outputs, cache,
        row_chunk=row_chunk, shadow=shadow, norm_eps=norm_eps,
    )
    elapsed = time.time() - t0
    record = dict(
        **case.record(),
        reference_norm_eps=norm_eps, row_chunk=row_chunk,
        compare_elapsed_s=round(elapsed, 3), **stats.output(),
    )
    # free device tensors before the next case
    del q, qr, descale_qr, outputs, cache, values
    gc.collect()
    try:
        torch.npu.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return record


def _select_cases(args):
    cases = golden.all_cases()
    if args.phase != "all":
        cases = [c for c in cases if c.phase == args.phase]
    if args.case_id:
        want = set(args.case_id)
        cases = [c for c in cases if c.case_id in want]
    if args.t:
        want_t = set(args.t)
        cases = [c for c in cases if c.token_count in want_t]
    if args.unique_t:
        seen, kept = set(), []
        for c in cases:
            if c.token_count in seen:
                continue
            seen.add(c.token_count)
            kept.append(c)
        cases = kept
    return cases


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Validate attn_prologue accuracy")
    parser.add_argument("--phase", choices=["all", "decode", "prefill"], default="all")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--t", type=int, action="append")
    parser.add_argument("--unique-t", action="store_true")
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--out", default=str(OP_ROOT / "tests" / "reports" / "pro_pypto" / "validation.json"))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=min(16, os.cpu_count() or 4))
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    # handler 指向 stdout：--list 的 JSON 契约走 stdout，且满足 G.LOG.02（统一用 logging）
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    cases = _select_cases(args)
    if args.list:
        LOG.info(json.dumps([c.record() for c in cases], indent=2))
        return 0
    if not cases:
        LOG.error("no cases selected")
        return 2

    device = _resolve_device(args.device)
    torch.set_num_threads(args.cpu_threads)
    wrapper = _load_wrapper()
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    weights = golden.SharedWeights()
    shadow = verify.FP64Shadow(weights)
    dev_weights = DeviceWeights(weights, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results, all_pass = [], True
    for idx, case in enumerate(cases, 1):
        LOG.info("[%d/%d] RUN %s (T=%d, %s)", idx, len(cases), case.case_id, case.token_count, device)
        LOG.info("%s", case)
        try:
            rec = _run_case(case, device, weights, shadow, wrapper, dev_weights,
                            row_chunk=args.row_chunk, norm_eps=args.norm_eps)
        except Exception as exc:  # noqa: BLE001
            rec = dict(**case.record(), passed=False,
                       error=f"{type(exc).__name__}: {exc}")
            LOG.exception("case %s raised", case.case_id)
        ok = bool(rec.get("passed"))
        all_pass = all_pass and ok
        results.append(rec)
        LOG.info("  -> %s  q_bad_rate=%s qr_hard=%s kv_hard=%s untouched_bad=%s",
                 "PASS" if ok else "FAIL", rec.get("q_bad_rate"), rec.get("qr_hard"),
                 rec.get("kv_hard"), rec.get("untouched_bad"))
        out_path.write_text(json.dumps(
            {"operator": "attn_prologue", "device": device,
             "case_count": len(results), "all_passed": all_pass, "cases": results},
            indent=2))
    LOG.info("=" * 60)
    LOG.info("%d/%d cases passed", sum(1 for r in results if r.get("passed")), len(results))
    LOG.info("report: %s", out_path)
    return 0 if all_pass else 1


# ---------------------------------------------------------------------------
# pytest smoke path (CPU collection stays NPU-free)
# ---------------------------------------------------------------------------
try:
    import pytest

    pytestmark = pytest.mark.soc("950")
    _SMOKE = ["decode-b1-s1-t1", "decode-b12-s6-t72", "prefill-b1-s1024-t1024",
              "prefill-b1-s2048-t2048"]

    @pytest.fixture(scope="module")
    def runtime():
        device = _resolve_device(_default_device())
        wrapper = _load_wrapper()
        import torch_npu  # noqa: F401

        torch.npu.set_device(device)
        weights = golden.SharedWeights()
        shadow = verify.FP64Shadow(weights)
        dev_weights = DeviceWeights(weights, device)
        yield device, wrapper, weights, shadow, dev_weights
        gc.collect()
        try:
            torch.npu.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    @pytest.fixture(scope="module")
    def binary_runtime():
        """Installed torch/ACLNN binary path; skipped when the package is absent."""
        try:
            import pypto_gym
            pypto_gym.load("attn_prologue")
            _ = torch.ops.pypto_gym.attn_prologue
        except Exception as exc:  # noqa: BLE001
            if os.environ.get("GYM_TEST_MODE") == "torch":
                raise
            pytest.skip(f"attn_prologue binary package unavailable: {exc}")
        device = _resolve_device(_default_device())
        import torch_npu  # noqa: F401

        torch.npu.set_device(device)
        weights = golden.SharedWeights()
        shadow = verify.FP64Shadow(weights)
        dev_weights = DeviceWeights(weights, device)
        yield device, weights, shadow, dev_weights
        gc.collect()
        try:
            torch.npu.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _binary_inputs(case, device, weights, dev_weights):
        inputs = golden.make_case_inputs(case, weights)
        values = dev_weights.as_dict()
        for name in _PER_TOKEN_FIELDS:
            values[name] = _device_input(name, getattr(inputs, name), device, _to_nz)
        return inputs, values, inputs.kv_cache.to(device)

    def _run_binary(case, device, weights, dev_weights, *, qr_norm=None):
        inputs, values, cache = _binary_inputs(case, device, weights, dev_weights)
        with torch.no_grad():
            q, qr, dqr, returned_cache = torch.ops.pypto_gym.attn_prologue(
                values["x"], values["wqa"], values["wqb"], values["wkv"],
                values["descale_x"], values["descale_wqa"], values["descale_wqb"],
                values["descale_wkv"], values["norm_weight_qr"],
                values["norm_weight_kv"], values["rope_sin"], values["rope_cos"],
                values["cache_index"], cache, qr_norm=qr_norm, norm_eps=1e-6,
            )
        assert returned_cache.data_ptr() == cache.data_ptr()
        torch.npu.synchronize()
        return inputs, (q, qr, dqr), cache

    def test_binary_schema_writes(binary_runtime):
        from torch.fx.operator_schemas import get_signature_for_torch_op

        op = torch.ops.pypto_gym.attn_prologue.default
        _, (schema,) = get_signature_for_torch_op(op, return_schemas=True)
        aliases = {arg.name: arg.alias_info for arg in schema.arguments}
        assert aliases["kv_cache"].is_write and aliases["qr_norm"].is_write
        assert schema.returns[3].alias_info.after_set == aliases["kv_cache"].after_set
        assert aliases["qr_norm"].after_set.isdisjoint(aliases["kv_cache"].after_set)

    @pytest.mark.parametrize("route", ["jit", "binary"])
    def test_noncontiguous_rope_rejected(route, request):
        if route == "jit":
            device, op, weights, _, dev_weights = request.getfixturevalue("runtime")
        else:
            device, weights, _, dev_weights = request.getfixturevalue("binary_runtime")
            op = torch.ops.pypto_gym.attn_prologue.default
        case = next(c for c in golden.all_cases() if c.token_count == 72)
        _, values, cache = _binary_inputs(case, device, weights, dev_weights)
        dense = values["rope_sin"]
        view = torch.empty((*dense.shape, 2), dtype=dense.dtype, device=device)[..., 0]
        view.copy_(dense)
        with pytest.raises((ValueError, RuntimeError), match="rope_sin.*contiguous"):
            op(**{**values, "rope_sin": view}, kv_cache=cache)

    @pytest.mark.parametrize("case_id", ["decode-b12-s6-t72", "decode-b32-s6-t192", "prefill-b1-s1024-t1024"])
    def test_binary_attn_prologue_accuracy(case_id, binary_runtime):
        """torch.ops/ACLNN path; one, two and eight M tiles exercise both cache profiles."""
        device, weights, shadow, dev_weights = binary_runtime
        case = next(c for c in golden.all_cases() if c.case_id == case_id)
        inputs, outputs, cache = _run_binary(case, device, weights, dev_weights)
        stats = verify.compare(inputs, weights, outputs, cache, shadow=shadow)
        assert stats.passed

    def test_binary_qr_norm_optional(binary_runtime):
        device, weights, shadow, dev_weights = binary_runtime
        case = next(c for c in golden.all_cases()
                    if c.case_id == "decode-b12-s6-t72")
        inputs, (q1, qr1, d1), _ = _run_binary(case, device, weights, dev_weights)
        qn = torch.full((case.token_count, 1280), float("nan"), dtype=torch.bfloat16,
                        device=device)
        _, (q2, qr2, d2), _ = _run_binary(case, device, weights, dev_weights,
                                          qr_norm=qn)
        assert torch.equal(q1, q2)
        assert torch.equal(qr1.view(torch.uint8), qr2.view(torch.uint8))
        assert torch.equal(d1.view(torch.uint8), d2.view(torch.uint8))
        assert torch.isfinite(qn.float()).all()
        qn_ref = shadow.recompute_full(
            inputs, torch.arange(case.token_count), "qr", norm_eps=1e-6
        )[3].to(torch.bfloat16)
        diff = (qn.cpu().float() - qn_ref.float()).abs()
        assert (diff <= 0.01 + 0.01 * qn_ref.float().abs()).all()
        assert (qn.cpu() != qn_ref).float().mean() <= 1e-4

    def test_binary_paged_cache_row_stride(binary_runtime):
        device, weights, shadow, dev_weights = binary_runtime
        case = next(c for c in golden.all_cases()
                    if c.case_id == "decode-b16-s6-t96")
        _, values, _ = _binary_inputs(case, device, weights, dev_weights)
        cache_index = values["cache_index"]
        page, block, cols = 2, 128, golden.kv_record_bytes(512)
        sentinel = 0xA5
        ref = torch.full((page, block, 1, cols), sentinel,
                         dtype=torch.uint8, device=device)
        page_stride, row_stride, prefix = 131072, 576, 256
        body = (page - 1) * page_stride + (block - 1) * row_stride + cols
        flat = torch.full((prefix + body,), sentinel,
                          dtype=torch.uint8, device=device)
        paged = torch.as_strided(flat[prefix:], (page, block, 1, cols),
                                 (page_stride, row_stride, row_stride, 1))
        args = (values["x"], values["wqa"], values["wqb"], values["wkv"],
                values["descale_x"], values["descale_wqa"], values["descale_wqb"],
                values["descale_wkv"], values["norm_weight_qr"],
                values["norm_weight_kv"], values["rope_sin"], values["rope_cos"],
                cache_index)
        with torch.no_grad():
            torch.ops.pypto_gym.attn_prologue(*args, ref, norm_eps=1e-6)
            torch.ops.pypto_gym.attn_prologue(*args, paged, norm_eps=1e-6)
        _assert_paged_cache(ref, flat, prefix, cache_index, block, page_stride, row_stride, cols, sentinel)

    @pytest.mark.parametrize("case_id", _SMOKE)
    def test_attn_prologue_accuracy(case_id, runtime):
        device, wrapper, weights, shadow, dev_weights = runtime
        case = next(c for c in golden.all_cases() if c.case_id == case_id)
        rec = _run_case(case, device, weights, shadow, wrapper, dev_weights)
        assert rec["case_id"] == case_id
        assert rec.get("passed"), rec

    def _check_paged_layout(runtime, page_stride, row_stride, prefix=256):
        device, wrapper, weights, shadow, dev_weights = runtime
        case = next(c for c in golden.all_cases()
                    if c.case_id == "decode-b16-s6-t96")
        inputs = golden.make_case_inputs(case, weights)
        values = dev_weights.as_dict()
        for name in _PER_TOKEN_FIELDS:
            values[name] = _device_input(name, getattr(inputs, name), device, _to_nz)
        cache_index = values["cache_index"]
        page, block, cols = 2, 128, golden.kv_record_bytes(512)
        sentinel = 0xA5
        ref = torch.full((page, block, 1, cols), sentinel,
                         dtype=torch.uint8, device=device)
        body = (page - 1) * page_stride + (block - 1) * row_stride + cols
        flat = torch.full((prefix + body,), sentinel,
                          dtype=torch.uint8, device=device)
        paged = torch.as_strided(flat[prefix:], (page, block, 1, cols),
                                 (page_stride, row_stride, row_stride, 1))
        assert paged.stride(0) == page_stride and paged.stride(1) == row_stride
        wrapper(values["x"], values["wqa"], values["wqb"], values["wkv"],
                values["descale_x"], values["descale_wqa"], values["descale_wqb"],
                values["descale_wkv"], values["norm_weight_qr"],
                values["norm_weight_kv"], values["rope_sin"], values["rope_cos"],
                cache_index, ref, norm_eps=1e-6)
        torch.npu.synchronize()
        wrapper(values["x"], values["wqa"], values["wqb"], values["wkv"],
                values["descale_x"], values["descale_wqa"], values["descale_wqb"],
                values["descale_wkv"], values["norm_weight_qr"],
                values["norm_weight_kv"], values["rope_sin"], values["rope_cos"],
                cache_index, paged, norm_eps=1e-6)
        _assert_paged_cache(ref, flat, prefix, cache_index, block, page_stride, row_stride, cols, sentinel)

    def test_paged_cache_block_stride(runtime):
        _check_paged_layout(runtime, page_stride=131072, row_stride=544)

    def test_paged_cache_row_stride(runtime):
        _check_paged_layout(runtime, page_stride=131072, row_stride=576)

    def test_qr_norm_optional(runtime):
        device, wrapper, weights, shadow, dev_weights = runtime
        case = next(c for c in golden.all_cases()
                    if c.case_id == "decode-b12-s6-t72")
        inputs = golden.make_case_inputs(case, weights)
        values = dev_weights.as_dict()
        for name in _PER_TOKEN_FIELDS:
            values[name] = _device_input(name, getattr(inputs, name), device, _to_nz)
        cache = inputs.kv_cache.to(device)
        args = (values["x"], values["wqa"], values["wqb"], values["wkv"],
                values["descale_x"], values["descale_wqa"], values["descale_wqb"],
                values["descale_wkv"], values["norm_weight_qr"],
                values["norm_weight_kv"], values["rope_sin"], values["rope_cos"],
                values["cache_index"], cache)
        q1, qr1, d1 = wrapper(*args, norm_eps=1e-6)
        qn = torch.empty((case.token_count, 1280), dtype=torch.bfloat16,
                         device=device)
        q2, qr2, d2 = wrapper(*args, qr_norm=qn, norm_eps=1e-6)
        torch.npu.synchronize()
        assert torch.equal(q1, q2)
        assert torch.equal(qr1.view(torch.uint8), qr2.view(torch.uint8))
        assert torch.equal(d1.view(torch.uint8), d2.view(torch.uint8))
        assert qn.shape == (case.token_count, 1280)
        assert qn.dtype == torch.bfloat16
        assert torch.isfinite(qn.float()).all()
        assert qn.float().abs().max() > 0
        qn_ref = shadow.recompute_full(
            inputs, torch.arange(case.token_count), "qr", norm_eps=1e-6
        )[3].to(torch.bfloat16)
        assert tuple(qn_ref.shape) == tuple(qn.shape)
        diff = (qn.cpu().float() - qn_ref.float()).abs()
        assert (diff <= 0.01 + 0.01 * qn_ref.float().abs()).all()
        assert (qn.cpu() != qn_ref).float().mean() <= 1e-4

    def test_fp8_value_criterion():
        verify.test_fp8_value_criterion()

    def test_full_output_validator():
        verify.test_full_output_validator()

except ImportError:  # pragma: no cover - pytest optional
    pass


if __name__ == "__main__":
    raise SystemExit(main())
