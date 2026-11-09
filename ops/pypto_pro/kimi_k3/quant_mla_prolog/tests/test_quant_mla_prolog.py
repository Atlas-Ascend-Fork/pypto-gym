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

"""quant_mla_prolog test: one test function per case, plus a ``main`` entry.

Run: ``python test_quant_mla_prolog.py``  or  ``pytest tests -q``
"""

import logging
import os
import sys

import torch
import torch_npu  # noqa: F401

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.join(_THIS_DIR, "assets"),
           os.path.dirname(_THIS_DIR) + "/op_kernel"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

# 其中的单下划线名是内部常量，仅用于泛化检查与选档断言
from quant_mla_prolog import (  # noqa: E402
    C0, FEATURES, N_HEADS, NORM_EPS, PAGE, _pick_m, quant_mla_prolog,
    quant_mla_prolog_kernel)
from quant_mla_prolog_golden_cpu import (  # noqa: E402
    make_inputs, quant_mla_prolog_golden_cpu)
from quant_mla_prolog_precision import check_precision  # noqa: E402

LOGGER = logging.getLogger(__name__)

pytestmark = pytest.mark.soc("950")

KV_CACHE_ARG = 11  # 公开入口的 kv_cache 位次


def _get_device():
    if not (torch.npu.is_available() and torch.npu.device_count() > 0):
        raise RuntimeError("本用例集只在 NPU 上跑：未检测到可用设备")
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    return torch.device(f"npu:{device_id}")


def _to_phys(t, quant_mode_c=1):
    """Logical rank-4 kv_cache → physical PA_NZ (reshape, same bytes)."""
    last = C0 if quant_mode_c else 16
    return t.reshape(t.shape[0], FEATURES // last, PAGE, last)


def _make_case(tokens, blocks, cache_index, qscale_kv=None, seed=None,
               quant_mode_c=1):
    """Build deterministic case inputs and run them on the device.

    Returns (cpu_args, dev_args, kwargs) where dev_args are ready for the
    public wrapper and cpu_args are ready for the CPU golden.
    """
    kw = {} if seed is None else {"seed": seed}
    args, kwargs = make_inputs(tokens=tokens, blocks=blocks,
                               quant_mode_c=quant_mode_c, **kw)
    args[12] = torch.tensor(cache_index, dtype=torch.int64)
    if qscale_kv is not None:
        args[13] = torch.full_like(args[13], qscale_kv)

    dev = _get_device()
    dev_args = [a.to(dev) if torch.is_tensor(a) else a for a in args]
    phys = dev_args[KV_CACHE_ARG]
    dev_args[KV_CACHE_ARG] = phys.view(phys.shape[0], PAGE, 1, FEATURES)
    return args, dev_args, kwargs


def run_case(name, tokens, blocks, cache_index, quant_mode_c=1, seed=None,
             qscale_kv=None):
    """Run the public wrapper and the CPU golden, then check precision."""
    cpu_args, dev_args, kwargs = _make_case(
        tokens, blocks, cache_index, qscale_kv=qscale_kv, seed=seed,
        quant_mode_c=quant_mode_c)
    out = quant_mla_prolog(*dev_args, **kwargs)
    torch.npu.synchronize()

    out_dt = torch.float8_e4m3fn if quant_mode_c else torch.bfloat16
    expect_keys = (["q", "kv_cache_out", "descale_q"] if quant_mode_c
                    else ["q", "kv_cache_out"])
    assert list(out) == expect_keys, \
        f"{name} 返回键序错: {list(out)}，应为 {expect_keys}"
    q, kv_out = out["q"], out["kv_cache_out"]
    assert kv_out.data_ptr() == dev_args[KV_CACHE_ARG].data_ptr(), \
        "kv_cache 必须原位更新：返回值要和入参是同一块内存"
    assert tuple(q.shape) == (tokens, N_HEADS, FEATURES), \
        f"{name} q shape 错: {q.shape}"
    assert q.dtype == out_dt, f"{name} q dtype 错: {q.dtype}"
    assert kv_out.dtype == out_dt, f"{name} kv_cache_out dtype 错: {kv_out.dtype}"

    expected = quant_mla_prolog_golden_cpu(*cpu_args, **kwargs)
    slots = [int(v) for v in cpu_args[12].reshape(-1).tolist()]
    qs = float(cpu_args[13].reshape(-1)[0]) if quant_mode_c else 1.0
    out_check = dict(out)
    out_check["kv_cache_out"] = _to_phys(out["kv_cache_out"], quant_mode_c)
    check_precision(out_check, expected, slots, qs, quant_mode_c, name)
    return out


# ── 主 case 集 ──────────────────────────────────────────────────────────

T_SWEEP = (1, 7, 8, 9, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256)
C0_T_SWEEP = (1, 8, 32, 64, 128, 256)
T_LARGE = (257, 511, 1024)
UNLISTED = ((3, 1), (13, 2), (45, 1), (77, 1), (100, 3), (200, 4), (333, 5))

MATH06_QSCALES = (7.0, 1.7, 1.0 / 3.0, 123.456)
MATH06_BACKGROUND = ((0.3, 1), (0.001, 1))


def test_t8():
    run_case("t8", 8, 1, list(range(8)))


def test_t1():
    run_case("t1", 1, 1, [0])


def test_t5():
    run_case("t5", 5, 1, list(range(5)))


def test_skip():
    run_case("skip", 8, 1, [0, -1, 2, -1, 4, 5, -1, 7])


def test_multipage():
    run_case("multipage", 8, 3, [0, 130, 5, 260, 131, 7, 261, 2])


def test_t_sweep():
    for t in T_SWEEP:
        run_case(f"t{t}", t, (t + PAGE - 1) // PAGE, list(range(t)))


def test_c0_t_sweep():
    for t in C0_T_SWEEP:
        run_case(f"c0_t{t}", t, (t + PAGE - 1) // PAGE, list(range(t)),
                 quant_mode_c=0)


def test_c0_skip():
    run_case("c0_skip", 8, 1, [0, -1, 2, -1, 4, 5, -1, 7], quant_mode_c=0)


def test_c0_multipage():
    run_case("c0_multipage", 8, 3, [0, 130, 5, 260, 131, 7, 261, 2],
             quant_mode_c=0)


def test_t_large():
    for t in T_LARGE:
        run_case(f"t{t}", t, (t + PAGE - 1) // PAGE, list(range(t)))


def test_multiblock_skip():
    idx = [-1 if i % 7 == 3 else (i * 37) % (3 * PAGE) for i in range(257)]
    run_case("mb_skip", 257, 3, idx)


def test_unlisted_sizes():
    for tokens, blocks in UNLISTED:
        idx = [(i * 37) % (blocks * PAGE) for i in range(tokens)]
        run_case(f"unlisted_t{tokens}", tokens, blocks, idx)


# ── 数值回归 ──────────────────────────────────────────────────────────

def _bitwise_diff(a, b):
    ua = a.cpu().contiguous().view(torch.uint8).reshape(-1)
    ub = b.cpu().contiguous().view(torch.uint8).reshape(-1)
    assert ua.numel() == ub.numel(), f"字节数不等 {ua.numel()} vs {ub.numel()}"
    return int((ua != ub).sum())


def test_math06_qscale():
    """Non-default qscale: kv_cache must be byte-identical to golden."""
    t = 128
    wanted = [(qs, 0) for qs in MATH06_QSCALES] + list(MATH06_BACKGROUND)
    bad = []
    for qs, expect in wanted:
        name = f"math06_qs{qs:g}"
        cpu_args, dev_args, kwargs = _make_case(
            t, (t + PAGE - 1) // PAGE, list(range(t)), qscale_kv=qs)
        got_kv = quant_mla_prolog(*dev_args, **kwargs)["kv_cache_out"]
        want_kv = quant_mla_prolog_golden_cpu(*cpu_args, **kwargs)["kv_cache_out"]
        nd = _bitwise_diff(got_kv, want_kv)
        if nd != expect:
            bad.append((name, nd, expect))
    assert not bad, f"除法语义回归失败（实际, 期望）：{bad}"


def test_public_contract():
    """C1 输出 dict 有 descale_q，C0 没有；键序固定。"""
    _, dev_args, kwargs = _make_case(32, 1, list(range(32)))
    out_c1 = quant_mla_prolog(*dev_args, **dict(kwargs, quant_mode_c=1))
    assert list(out_c1) == ["q", "kv_cache_out", "descale_q"]
    _, dev_args_c0, kwargs_c0 = _make_case(32, 1, list(range(32)),
                                            quant_mode_c=0)
    out_c0 = quant_mla_prolog(*dev_args_c0, **kwargs_c0)
    assert list(out_c0) == ["q", "kv_cache_out"]
    assert "descale_q" not in out_c0


def test_norm_eps_variants():
    """eps flows into RMSNorm: different eps → different output."""
    tokens, blocks = 32, 1
    idx = list(range(tokens))
    got_q = []
    for eps in (NORM_EPS, 1.0):
        _, dev_args, kwargs = _make_case(tokens, blocks, idx)
        out = quant_mla_prolog(*dev_args, **dict(kwargs, norm_eps=eps))
        got_q.append(out["q"].clone())
    d = _bitwise_diff(got_q[0], got_q[1])
    assert d > 0, "换了 norm_eps 结果没变 —— eps 没流进 kernel"

    for bad in (0.0, -1.0, float("inf"), float("nan")):
        _, dev_args, kwargs = _make_case(tokens, blocks, idx)
        try:
            quant_mla_prolog(*dev_args, **dict(kwargs, norm_eps=bad))
        except ValueError:
            continue
        raise AssertionError(f"norm_eps={bad} 应当被拒")


# ── 入参校验 ──────────────────────────────────────────────────────────

def test_input_validation():
    tokens, blocks = 8, 1
    idx = list(range(tokens))
    _, base, kw = _make_case(tokens, blocks, idx)
    dev = base[0].device

    def _expect(err, i, value, why):
        args = list(base)
        args[i] = value
        args[KV_CACHE_ARG] = base[KV_CACHE_ARG].clone()
        try:
            quant_mla_prolog(*args, **kw)
        except err:
            return
        except Exception as exc:
            raise AssertionError(
                f"{why}：期望 {err.__name__}，实际 "
                f"{type(exc).__name__}: {exc}") from exc
        raise AssertionError(f"{why}：没有报错")

    _expect(ValueError, 1, base[1].permute(0, 2, 1).contiguous(),
            "wqa 的 NZ 排布反了")
    _expect(ValueError, 4, base[4].float(), "wkb 不是 BF16")
    _expect(ValueError, 9, base[9].to(torch.bfloat16),
            "norm_weight_qa 不是 FP32")
    _expect(ValueError, 12, base[12].to(torch.int32),
            "cache_index 不是 INT64")
    _expect(ValueError, 12, base[12][:tokens - 1].clone(),
            "cache_index 短于 T")
    _expect(ValueError, 5, base[5][:, :-1].clone(), "descale_x 群数少一")
    _expect(ValueError, 5, base[5].view(torch.uint8),
            "descale_x 不是 E8M0")
    _expect(ValueError, 13, torch.tensor([0.5, 0.5], device=dev),
            "qscale_kv 形状不是 [1]")
    _expect(ValueError, 0,
            torch.empty((tokens, 7168 * 2), dtype=torch.float8_e4m3fn,
                        device=dev)[:, :7168], "x 非连续")
    _expect(RuntimeError, 9, base[9].cpu(), "norm_weight_qa 不在 NPU 上")

    _, dev_big, kw_big = _make_case(200, 1,
                                    [(i * 37) % 128 for i in range(200)])
    try:
        quant_mla_prolog(*dev_big, **kw_big)
        raise AssertionError("kv_cache 容量不足没有报错")
    except ValueError:
        pass

    _, c0_args, c0_kw = _make_case(tokens, blocks, idx, quant_mode_c=0)
    out = quant_mla_prolog(*c0_args[:13], **c0_kw)
    assert "descale_q" not in out and \
        tuple(out["q"].shape) == (tokens, N_HEADS, FEATURES), \
        f"C0 省略 qscale_kv 的调用结果不对：{out['q'].shape}"


def test_mode_validation():
    _, dev_args, kwargs = _make_case(8, 1, list(range(8)), quant_mode_c=0)

    def _expect_value_error(args, kw, why):
        try:
            quant_mla_prolog(*args, **kw)
        except ValueError:
            return
        raise AssertionError(why)

    with_scale = list(dev_args)
    with_scale[13] = torch.tensor([0.5], dtype=torch.float32,
                                  device=dev_args[0].device)
    _expect_value_error(with_scale, kwargs, "C0 传了 qscale_kv 必须报错")
    _expect_value_error(dev_args, dict(kwargs, quant_mode_c=1),
                        "C1 缺 qscale_kv 必须报错")


# ── 行独立 / pad / 重复 / workspace ──────────────────────────────

def _assert_prefix_bitwise(name, tokens_big, tokens_small, blocks):
    """Same input, two T values: small output is prefix of big (bitwise)."""
    device = _get_device()
    cpu_args, _, kwargs = _make_case(tokens_big, blocks,
                                     list(range(tokens_big)))

    def run(n):
        cpu = list(cpu_args)
        for i in (0, 5, 12):
            cpu[i] = cpu_args[i][:n].clone()
        cpu[KV_CACHE_ARG] = cpu_args[KV_CACHE_ARG].clone()
        dev = [t.to(device) if torch.is_tensor(t) else t for t in cpu]
        phys = dev[KV_CACHE_ARG]
        dev[KV_CACHE_ARG] = phys.view(phys.shape[0], PAGE, 1, FEATURES)
        return quant_mla_prolog(*dev, **kwargs)

    out_b, out_s = run(tokens_big), run(tokens_small)
    n = tokens_small
    for tag in ("q", "descale_q", "kv_cache_out"):
        x, y = out_b[tag], out_s[tag]
        if tag == "kv_cache_out":
            x = _to_phys(x.cpu())
            y = _to_phys(y.cpu())
            x = torch.stack([x[i // PAGE, :, i % PAGE, :].reshape(-1)
                             for i in range(n)])
            y = torch.stack([y[i // PAGE, :, i % PAGE, :].reshape(-1)
                             for i in range(n)])
        else:
            x, y = x[:n], y
        nd = _bitwise_diff(x, y)
        assert nd == 0, f"{name} {tag} 前 {n} 行不逐 bit 相同：{nd} 字节"


def test_row_independence():
    _assert_prefix_bitwise("row_indep", 256, 128, 2)


def test_pad_isolation():
    _assert_prefix_bitwise("pad_iso", 256, 255, 2)


def test_repeat_invariance():
    device = _get_device()
    cpu_args, _, kwargs = _make_case(129, 2, list(range(129)))

    def run():
        cpu = list(cpu_args)
        cpu[KV_CACHE_ARG] = cpu_args[KV_CACHE_ARG].clone()
        dev = [t.to(device) if torch.is_tensor(t) else t for t in cpu]
        phys = dev[KV_CACHE_ARG]
        dev[KV_CACHE_ARG] = phys.view(phys.shape[0], PAGE, 1, FEATURES)
        return quant_mla_prolog(*dev, **kwargs)

    first, second = run(), run()
    for tag in ("q", "descale_q", "kv_cache_out"):
        nd = _bitwise_diff(first[tag], second[tag])
        assert nd == 0, f"repeat {tag} 两次调用不一致：{nd} 字节"


def test_workspace_isolation():
    device = _get_device()
    small, _, kwargs = _make_case(128, 1, list(range(128)))
    big, _, _ = _make_case(1024, 8, list(range(1024)))

    def run(args, n):
        dev = [t.to(device) if torch.is_tensor(t) else t for t in args]
        phys = dev[KV_CACHE_ARG]
        dev[KV_CACHE_ARG] = phys.view(phys.shape[0], PAGE, 1, FEATURES)
        return quant_mla_prolog(*dev, **kwargs)

    cold = run(small, 128)
    run(big, 1024)
    warm = run(small, 128)
    for tag in ("q", "descale_q", "kv_cache_out"):
        nd = _bitwise_diff(cold[tag], warm[tag])
        assert nd == 0, f"ws_iso {tag} 受大 capacity 残留影响：{nd} 字节"


# ── 泛化检查 ──────────────────────────────────────────────────────────

def _variant_keys():
    cache = getattr(quant_mla_prolog_kernel, "_compiled_by_signature", None)
    return set(cache.keys()) if isinstance(cache, dict) else None


def test_variant_reuse():
    """Same tile+mode: changing runtime inputs must not add new artifacts."""
    warm = 256
    run_case("reuse_warm_t256", warm, 2, list(range(warm)))
    before = _variant_keys()
    if before is None:
        return
    for tokens, eps in ((129, NORM_EPS), (200, NORM_EPS), (255, 1e-05)):
        assert _pick_m(tokens) == _pick_m(warm), "测试选错了档"
        cpu_args, dev_args, kwargs = _make_case(
            tokens, 2, [(i * 37) % 256 for i in range(tokens)])
        out = quant_mla_prolog(*dev_args, **dict(kwargs, norm_eps=eps))
        assert out["q"].shape[0] == tokens, \
            f"输出行数必须是 T={tokens}，不是 capacity：{out['q'].shape}"
        expected = quant_mla_prolog_golden_cpu(*cpu_args,
                                               **dict(kwargs, norm_eps=eps))
        slots = [int(v) for v in cpu_args[12].reshape(-1).tolist()]
        qs = float(cpu_args[13].reshape(-1)[0])
        out_check = dict(out)
        out_check["kv_cache_out"] = _to_phys(out["kv_cache_out"])
        check_precision(out_check, expected, slots, qs, 1, f"reuse_t{tokens}")

    after = _variant_keys()
    added = after - before
    assert not added, f"档内换运行期量多编了 {len(added)} 份产物：{added}"


def test_class_transitions():
    """Cross-tile: both sides of each boundary, plus return to earlier tile."""
    cases = ((8, 16), (9, 32), (32, 32), (33, 64), (64, 64), (65, 128),
             (128, 128), (129, 128), (8, 16))
    for tokens, want_m in cases:
        blocks = (tokens + PAGE - 1) // PAGE
        got_m = _pick_m(tokens)
        assert got_m == want_m, \
            f"T={tokens} 选档错：MRows={got_m}，应为 {want_m}"
        run_case(f"trans_t{tokens}_m{got_m}", tokens, blocks,
                 list(range(tokens)))

    before = _variant_keys()
    run_case("trans_c1_warm", 32, 1, list(range(32)), quant_mode_c=1)
    run_case("trans_c0", 32, 1, list(range(32)), quant_mode_c=0)
    c0 = _variant_keys()
    run_case("trans_c1_again", 32, 1, list(range(32)), quant_mode_c=1)
    back = _variant_keys()
    if before is None:
        return
    by_sig = {}
    for sig, _dtype_hash, packed in back:
        by_sig.setdefault(sig, set()).add(packed)
    shared = sorted(len(v) for v in by_sig.values())
    assert any(n >= 2 for n in shared), \
        "同一 capacity 下 C1 与 C0 应当是两份不同的产物"
    assert not (back - c0), "切回 C1 应当命中已有产物，不该新编"


# ── 入口 ──────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for fn in (
        test_t8, test_t1, test_t5, test_skip, test_multipage,
        test_t_sweep, test_math06_qscale, test_c0_t_sweep,
        test_c0_skip, test_c0_multipage, test_mode_validation,
        test_t_large, test_multiblock_skip, test_row_independence,
        test_pad_isolation, test_repeat_invariance,
        test_workspace_isolation, test_public_contract,
        test_norm_eps_variants, test_input_validation,
        test_unlisted_sizes, test_variant_reuse, test_class_transitions,
    ):
        fn()
        LOGGER.info("  %s PASS", fn.__name__)
    LOGGER.info("quant_mla_prolog 全部 case PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
