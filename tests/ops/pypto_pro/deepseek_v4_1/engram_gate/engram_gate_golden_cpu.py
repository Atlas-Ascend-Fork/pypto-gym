#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""engram_gate CPU Golden for the exact SPEC output ABI.

Pure-torch CPU reference. All reductions and post-processing run in FP32 and the
single public output y is rounded once to the SPEC output dtype (bfloat16).
image_mask==True tokens are copied from x bit-exactly (no re-rounding).
"""

import logging

import torch

_LOG = logging.getLogger(__name__)

_SPEC_FORMULA = (
    'rstd = rsqrt(mean(x^2)+eps) * rsqrt(mean(key^2)+eps); '
    'dot = sum(x*weight*key) * rstd / sqrt(dim); '
    'gate = sigmoid(copysign(sqrt(max(abs(dot), clamp_value)), dot)); '
    'y = bf16(x + gate*value); masked token (image_mask==True) -> y = x exactly'
)

_VALUE_LIMIT = 8.0
_OUTPUT_DTYPE = torch.bfloat16
_OUTPUT_RANGE = 16.0

# P0 cases from the SPEC machine contract: name, t, hc, dim, seed, mask_kind.
_P0_CASES = (
    ("partial_image_mask", 5, 4, 5120, 20260925, "partial"),
    ("model_token_count_nomask", 72, 4, 5120, 20260992, "all_false"),
    ("chunked_one_chunk_tail_one", 3, 4, 8449, 20269372, "all_false"),
    ("model_dim_plus_one_partial_mask", 5, 4, 5121, 20266041, "partial"),
)
_P0_NAMES = tuple(row[0] for row in _P0_CASES)
_EXPECTED_OUT_SHAPE = {
    "partial_image_mask": (5, 4, 5120),
    "model_token_count_nomask": (72, 4, 5120),
    "chunked_one_chunk_tail_one": (3, 4, 8449),
    "model_dim_plus_one_partial_mask": (5, 4, 5121),
}


def _randn(shape, dtype, generator, device):
    """Deterministic FP32 normal sample, truncated to SPEC value_range, cast to dtype."""
    sample = torch.randn(shape, generator=generator, dtype=torch.float32)
    sample = sample.clamp(-_VALUE_LIMIT, _VALUE_LIMIT)
    return sample.to(dtype).to(device)


def _case_args(spec, device):
    """Build one legal input tuple (x, key, value, weight, image_mask) from a case spec."""
    t, hc, dim, seed, mask_kind = spec
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = _randn((t, hc, dim), torch.bfloat16, generator, device)
    key = _randn((t, hc, dim), torch.bfloat16, generator, device)
    value = _randn((t, dim), torch.bfloat16, generator, device)
    weight = _randn((hc, dim), torch.float32, generator, device)
    if mask_kind == "partial":
        image_mask = torch.arange(t) % 2 == 1
    elif mask_kind == "all_false":
        image_mask = torch.zeros(t, dtype=torch.bool)
    elif mask_kind == "all_true":
        image_mask = torch.ones(t, dtype=torch.bool)
    elif mask_kind == "single_true":
        image_mask = torch.arange(t) == (t // 2)
    else:
        raise ValueError(f"unknown mask kind: {mask_kind}")
    return [x, key, value, weight, image_mask.to(device)]


def _engram_gate_math(inputs, eps, clamp_value):
    """SPEC formula in FP32 (no mask handling, no output rounding)."""
    x, key, value, weight, _ = inputs
    x_f = x.to(torch.float32)
    key_f = key.to(torch.float32)
    value_f = value.to(torch.float32)
    weight_f = weight.to(torch.float32)
    dim = x_f.shape[-1]
    rstd = torch.rsqrt(x_f.square().mean(dim=-1) + eps) * torch.rsqrt(
        key_f.square().mean(dim=-1) + eps
    )
    dot = (x_f * weight_f * key_f).sum(dim=-1) * rstd * (dim ** -0.5)
    signed_root = torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot)
    gate = torch.sigmoid(signed_root)
    return x_f + gate.unsqueeze(-1) * value_f.unsqueeze(-2)


def _ref_fp64(inputs, eps, clamp_value):
    """Independent higher-precision oracle in FP64 on CPU."""
    x, key, value, weight, image_mask = inputs
    x_f = x.detach().to(device="cpu", dtype=torch.float64)
    key_f = key.detach().to(device="cpu", dtype=torch.float64)
    value_f = value.detach().to(device="cpu", dtype=torch.float64)
    weight_f = weight.detach().to(device="cpu", dtype=torch.float64)
    dim = x_f.shape[-1]
    rstd = torch.rsqrt(x_f.square().mean(dim=-1) + eps) * torch.rsqrt(
        key_f.square().mean(dim=-1) + eps
    )
    dot = (x_f * weight_f * key_f).sum(dim=-1) * rstd * (dim ** -0.5)
    signed_root = torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot)
    gate = torch.sigmoid(signed_root)
    y = x_f + gate.unsqueeze(-1) * value_f.unsqueeze(-2)
    keep = image_mask.detach().to(device="cpu").reshape(-1, 1, 1)
    return torch.where(keep, x_f, y)


def engram_gate_golden_cpu(
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    image_mask: torch.Tensor,
    eps: float = 1e-06,
    clamp_value: float = 1e-06,
):
    """Implement the SPEC formula on CPU and return its declared output dtype (bfloat16).

    Args:
        x: [t, hc, dim] bfloat16.
        key: [t, hc, dim] bfloat16.
        value: [t, dim] bfloat16.
        weight: [hc, dim] float32.
        image_mask: [t] bool; True token -> y = x bit-exact.
        eps: RMS 稳定项。
        clamp_value: |dot| 的软阈值。

    Returns:
        y: [t, hc, dim] bfloat16.
    """
    y = _engram_gate_math((x, key, value, weight, image_mask), eps, clamp_value)
    out = y.to(_OUTPUT_DTYPE)
    keep = image_mask.reshape(-1, 1, 1)
    return torch.where(keep, x, out)


def _make_inputs(device=torch.device("cpu")):
    """Return every SPEC P0 case as (case_name, positional_tensor_args, scalar_kwargs).

    Deterministic, legal inputs for all P0 cases, kept on CPU.
    """
    cases = []
    for name, t, hc, dim, seed, mask_kind in _P0_CASES:
        args = _case_args((t, hc, dim, seed, mask_kind), device)
        cases.append((name, args, {"eps": 1e-06, "clamp_value": 1e-06}))
    return cases


def make_inputs(case):
    """Delivery helper: build one case's positional inputs (CPU) from its spec dict."""
    spec = (case["t"], case["hc"], case["dim"], case["seed"], case["mask_kind"])
    return _case_args(spec, torch.device("cpu"))


def reference(inputs, eps=1e-6, clamp_value=1e-6):
    """Delivery helper: independent CPU reference for one case's input tuple."""
    return engram_gate_golden_cpu(*inputs, eps=eps, clamp_value=clamp_value)


def _get_device():
    return torch.device('cpu')


def _check_output(case_name, out, args, eps, clamp_value):
    """Compare one output against the FP64 oracle and enforce the output contract."""
    x = args[0]
    image_mask = args[4]
    ref = _ref_fp64(args, eps, clamp_value)

    if out.dtype != _OUTPUT_DTYPE:
        raise AssertionError(f"{case_name}: output dtype {out.dtype} != {_OUTPUT_DTYPE}")
    if tuple(out.shape) != tuple(ref.shape):
        raise AssertionError(
            f"{case_name}: output shape {tuple(out.shape)} != {tuple(ref.shape)}")
    if not bool(torch.isfinite(out).all()):
        raise AssertionError(f"{case_name}: output has NaN/Inf")
    peak = float(out.to(torch.float32).abs().max())
    if peak > _OUTPUT_RANGE + 0.13:
        raise AssertionError(
            f"{case_name}: output exceeds SPEC value_range [-16,16]: peak={peak}")
    max_diff = float((out.to(torch.float32) - ref.to(torch.float32)).abs().max())
    if max_diff > 0.15:
        raise AssertionError(
            f"{case_name}: FP32 output deviates from FP64 oracle by {max_diff} > 0.15")
    mask = image_mask.to(torch.bool)
    if bool(mask.any()) and not torch.equal(out[mask], x[mask]):
        raise AssertionError(f"{case_name}: masked rows are not bit-exact x")
    return max_diff, peak


def _run_property_checks():
    _LOG.info("[property checks]")
    device = torch.device("cpu")

    args = _case_args((2, 3, 64, 11, "all_false"), device)
    args[0] = torch.zeros_like(args[0])
    args[1] = torch.zeros_like(args[1])
    args[2] = torch.ones_like(args[2])
    args[3] = torch.zeros_like(args[3])
    gate_zero = float(engram_gate_golden_cpu(*args).to(torch.float32).mean())
    if abs(gate_zero - 0.5) >= 1e-3:
        raise AssertionError(f"zero-dot gate {gate_zero} not ~0.5")
    _LOG.info("  [PASS] dot=0 -> gate=%.6f ~ 0.5", gate_zero)

    pos = _case_args((2, 3, 64, 12, "all_false"), device)
    pos[0] = torch.ones_like(pos[0])
    pos[1] = torch.ones_like(pos[1])
    pos[2] = torch.ones_like(pos[2])
    pos[3] = torch.ones_like(pos[3])
    y_pos = float(engram_gate_golden_cpu(*pos).to(torch.float32).mean())
    neg = [t.clone() for t in pos]
    neg[3] = -torch.ones_like(neg[3])
    y_neg = float(engram_gate_golden_cpu(*neg).to(torch.float32).mean())
    if not y_pos > 1.5 > y_neg > 1.0:
        raise AssertionError(f"sign monotonicity failed: pos={y_pos}, neg={y_neg}")
    _LOG.info("  [PASS] sign monotonicity: y(dot>0)=%.6f, y(dot<0)=%.6f", y_pos, y_neg)

    full = _case_args((4, 2, 128, 13, "all_true"), device)
    if not torch.equal(engram_gate_golden_cpu(*full), full[0]):
        raise AssertionError("all-True mask is not bit-exact x")
    _LOG.info("  [PASS] all-True image_mask -> y == x (bit-exact)")

    args = _case_args((3, 2, 256, 14, "partial"), device)
    diff, _ = _check_output(
        "nondefault_params",
        engram_gate_golden_cpu(*args, eps=1e-3, clamp_value=1e-2), args, 1e-3, 1e-2)
    _LOG.info("  [PASS] eps=1e-3, clamp_value=1e-2 (max_diff=%.6f)", diff)


def _run_generalization_checks():
    _LOG.info("[generalization over dynamic axes]")
    device = torch.device("cpu")
    plan = (
        ("gen_all_low", 1, 1, 1, False),
        ("gen_t_high", 8192, 2, 64, False),
        ("gen_hc_high", 2, 16, 64, False),
        ("gen_dim_low", 2, 3, 1, False),
        ("gen_dim_mid_tail", 2, 3, 65, True),
        ("gen_dim_high", 1, 1, 65536, False),
        ("gen_all_high_repr", 8192, 16, 64, False),
    )
    for name, t, hc, dim, partial in plan:
        spec = (t, hc, dim, 20261001 + t + hc + dim, "partial" if partial else "all_false")
        args = _case_args(spec, device)
        out = engram_gate_golden_cpu(*args)
        diff, peak = _check_output(name, out, args, 1e-6, 1e-6)
        _LOG.info("  [PASS] %s (t=%d, hc=%d, dim=%d) max_diff=%.6f peak=%.4f",
                  name, t, hc, dim, diff, peak)


def _validate():
    """Run every contract P0 case and check the engram_gate contract end-to-end."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = _get_device()
    _LOG.info("=" * 60)
    _LOG.info("engram_gate_golden_cpu validation report")
    _LOG.info("=" * 60)
    _LOG.info("Device: %s", device)

    cases = _make_inputs(device)
    if not isinstance(cases, list) or not cases:
        raise AssertionError("_make_inputs returned no cases")
    observed_names = [case[0] for case in cases]
    if len(observed_names) != len(set(observed_names)):
        raise AssertionError("duplicate case names")
    missing = [name for name in _P0_NAMES if name not in observed_names]
    if missing:
        raise AssertionError(f"missing contract P0 cases: {missing}")
    unexpected = [name for name in observed_names if name not in _P0_NAMES]
    if unexpected:
        raise AssertionError(f"unexpected P0 cases: {unexpected}")

    _LOG.info("[contract P0 cases]")
    for case_name, args, kwargs in cases:
        if not isinstance(case_name, str) or not case_name:
            raise AssertionError(f"invalid case name: {case_name!r}")
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise AssertionError(f"{case_name}: invalid case payload types")
        _LOG.info("[case: %s]", case_name)
        out = engram_gate_golden_cpu(*args, **kwargs)
        if tuple(out.shape) != _EXPECTED_OUT_SHAPE[case_name]:
            raise AssertionError(
                f"{case_name}: shape {tuple(out.shape)} != {_EXPECTED_OUT_SHAPE[case_name]}")
        diff, peak = _check_output(case_name, out, args, kwargs["eps"], kwargs["clamp_value"])
        _LOG.info("  shape=%s dtype=%s finite=PASS max_diff=%.6f peak=%.4f ... PASS",
                  tuple(out.shape), out.dtype, diff, peak)

    _run_property_checks()
    _run_generalization_checks()

    _LOG.info("=" * 60)
    _LOG.info("validation complete")
    _LOG.info("=" * 60)
    return True


if __name__ == "__main__":
    raise SystemExit(0 if _validate() else 1)
