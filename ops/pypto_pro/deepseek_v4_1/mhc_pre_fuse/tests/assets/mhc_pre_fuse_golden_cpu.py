#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""mhc_pre_fuse CPU golden: an independent pure-torch implementation.

For every token ``t`` (m = 4 mixes, H = hidden size):

    rms         = rsqrt(sqrsum / (m * H) + rms_eps)
    mixes       = gemm_out_mul * rms
    pre         = sigmoid(mixes[:, 0:4]  * scale[0] + base[0:4])  + mhc_pre_eps
    post        = sigmoid(mixes[:, 4:8]  * scale[1] + base[4:8])  * post_mult
    comb        = sinkhorn((mixes[:, 8:24] * scale[2] + base[8:24]).view(T, 4, 4))
    layer_input = bf16(sum_m fp32(residual[:, m, :]) * pre[:, m])

All reductions and post-processing run in FP32; ``post_mix`` / ``comb_mix`` are
returned in FP32 and ``layer_input`` is rounded once to bfloat16.
"""

import logging
from dataclasses import dataclass

import torch

_LOG = logging.getLogger(__name__)

MHC = 4
MHC2 = 16
MHC3 = 24

_VALUE_LIMIT = 8.0
_OUTPUT_DTYPE = torch.bfloat16
_OUTPUT_RANGE = 32.0    # |layer_input| <= 4 * _VALUE_LIMIT (m == 4 terms, pre <= 1)


@dataclass(frozen=True)
class Knobs:
    """Scalar switches of the mHC pre pass, named after the operator contract."""

    rms_eps: float = 1e-6
    mhc_pre_eps: float = 1e-6
    mhc_sinkhorn_eps: float = 1e-6
    mhc_post_mult_value: float = 1.0
    sinkhorn_repeat: int = 10


_DEFAULT_KNOBS = Knobs()

# Validation cases: name, T, hidden, seed.
_VALIDATION_CASES = (
    ("small_default", 512, 4096, 20260925),
    ("small_wide", 512, 7168, 20260992),
    ("large_default", 8192, 4096, 20269372),
    ("large_mid", 8192, 5120, 20266041),
)


def _randn(shape, dtype, generator):
    """Deterministic FP32 normal sample, truncated to the value range above."""
    sample = torch.randn(shape, generator=generator, dtype=torch.float32)
    sample = sample.clamp(-_VALUE_LIMIT, _VALUE_LIMIT)
    return sample.to(dtype)


def _case_args(spec):
    """Build one legal input tuple from a case spec."""
    t, hidden, seed = spec
    generator = torch.Generator(device="cpu").manual_seed(seed)
    gemm_out_mul = _randn((t, MHC3), torch.float32, generator)
    mhc_scale = _randn((3,), torch.float32, generator)
    mhc_base = _randn((MHC3,), torch.float32, generator)
    residual = _randn((t, MHC, hidden), torch.bfloat16, generator)
    # sqrsum is the square sum the kernel is contractually fed: it has to agree
    # with residual, otherwise rms would not be ~1.
    gemm_out_sqrsum = residual.to(torch.float32).square().sum(dim=(1, 2))
    return [gemm_out_mul, gemm_out_sqrsum, mhc_scale, mhc_base, residual]


def _sinkhorn(m, repeat, eps):
    """m: [T, 4, 4] fp32 -> doubly-stochastic normalised, same shape."""
    # The first row normalisation carries no eps; every later one does.  That
    # asymmetry is part of the contract.
    rmax = m.max(dim=-1, keepdim=True).values
    a = torch.exp(m - rmax)
    a = a / a.sum(dim=-1, keepdim=True)
    a = a + eps
    a = a / (a.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        a = a / (a.sum(dim=-1, keepdim=True) + eps)
        a = a / (a.sum(dim=-2, keepdim=True) + eps)
    return a


def _math(inputs, knobs):
    """The formula above, in FP32, with no output rounding."""
    gemm_out_mul, gemm_out_sqrsum, mhc_scale, mhc_base, residual = inputs
    mul = gemm_out_mul.to(torch.float32)
    sq = gemm_out_sqrsum.to(torch.float32)
    res = residual.to(torch.float32)
    t, m, hidden = residual.shape

    rms = torch.rsqrt(sq / float(m * hidden) + knobs.rms_eps)
    mixes = mul * rms[:, None]

    pre = torch.sigmoid(mixes[:, 0:4] * mhc_scale[0] + mhc_base[0:4]) + knobs.mhc_pre_eps
    post = (torch.sigmoid(mixes[:, 4:8] * mhc_scale[1] + mhc_base[4:8])
            * knobs.mhc_post_mult_value)
    comb_in = (mixes[:, 8:24] * mhc_scale[2] + mhc_base[8:24]).reshape(t, MHC, MHC)

    comb_out = _sinkhorn(comb_in, knobs.sinkhorn_repeat, knobs.mhc_sinkhorn_eps)
    layer_input = (res * pre.unsqueeze(-1)).sum(dim=1)
    return post, comb_out.reshape(t, MHC2), layer_input


def _ref_fp64(inputs, knobs=_DEFAULT_KNOBS):
    """Independent higher-precision oracle in FP64 on CPU."""
    return _math([x.detach().to(torch.float64) if x.is_floating_point() else x
                  for x in inputs], knobs)


def mhc_pre_fuse_golden_cpu(inputs, knobs=_DEFAULT_KNOBS):
    """Implement the formula above on CPU and return the declared output dtypes.

    Args:
        inputs: the five positional inputs, in this order
            ``gemm_out_mul`` [T, 24] float32, ``gemm_out_sqrsum`` [T] float32,
            ``mhc_scale`` [3] float32, ``mhc_base`` [24] float32,
            ``residual`` [T, 4, hidden] bfloat16.
        knobs: the five scalar switches (RMS eps, pre eps, Sinkhorn eps and
            repeat count, post multiplier), see ``Knobs``; the defaults are the
            values the operator contract prescribes.

    Returns:
        post_mix: [T, 4] float32.
        comb_mix: [T, 16] float32.
        layer_input: [T, hidden] bfloat16.
    """
    post, comb, layer_input = _math(inputs, knobs)
    return (post.contiguous(), comb.contiguous(),
            layer_input.to(_OUTPUT_DTYPE).contiguous())


def make_inputs(case):
    """Delivery helper: build one case's positional inputs (CPU) from its spec dict."""
    return _case_args((case["t"], case["hidden"], case["seed"]))


def reference(inputs, **kwargs):
    """Delivery helper: independent CPU reference for one case's input tuple."""
    return mhc_pre_fuse_golden_cpu(inputs, **kwargs)


def _check_output(case_name, out, inputs, knobs=_DEFAULT_KNOBS):
    """Compare one golden output triple against the FP64 oracle."""
    post, comb, layer_input = out
    _, _, rli = _ref_fp64(inputs, knobs)
    residual = inputs[4]

    if post.dtype != torch.float32 or comb.dtype != torch.float32:
        raise AssertionError(f"{case_name}: post/comb must be float32")
    if layer_input.dtype != _OUTPUT_DTYPE:
        raise AssertionError(f"{case_name}: layer_input dtype {layer_input.dtype}")
    expected_shape = (residual.shape[0], residual.shape[2])
    if tuple(post.shape) != (residual.shape[0], MHC):
        raise AssertionError(f"{case_name}: post shape {tuple(post.shape)}")
    if tuple(comb.shape) != (residual.shape[0], MHC2):
        raise AssertionError(f"{case_name}: comb shape {tuple(comb.shape)}")
    if tuple(layer_input.shape) != expected_shape:
        raise AssertionError(f"{case_name}: layer_input shape {tuple(layer_input.shape)}")
    for name, tensor in (("post_mix", post), ("comb_mix", comb)):
        if not bool(torch.isfinite(tensor).all()):
            raise AssertionError(f"{case_name}: {name} has NaN/Inf")
    if not bool(torch.isfinite(layer_input).all()):
        raise AssertionError(f"{case_name}: layer_input has NaN/Inf")

    peak = float(layer_input.to(torch.float32).abs().max())
    if peak > _OUTPUT_RANGE:
        raise AssertionError(f"{case_name}: layer_input exceeds range: peak={peak}")
    li = layer_input.to(torch.float32)
    diff = float((li - rli.to(torch.float32)).abs().max())
    if diff > 0.2:
        raise AssertionError(f"{case_name}: layer_input deviates from FP64 by {diff}")
    return diff, peak


def _run_property_checks():
    _LOG.info("[property checks]")
    args = _case_args((8, 512, 101))
    _, comb, _ = mhc_pre_fuse_golden_cpu(args)
    comb4 = comb.reshape(8, MHC, MHC)
    row = comb4.sum(dim=-1)
    col = comb4.sum(dim=-2)
    # The last Sinkhorn step is a column normalisation, so the column sums are
    # exact; the row sums carry the residual of an epsilon-Sinkhorn run.
    row_err = float((row - 1.0).abs().max())
    col_err = float((col - 1.0).abs().max())
    if col_err > 1e-4 or row_err > 1e-2:
        raise AssertionError(f"comb_mix not doubly stochastic: row={row_err} col={col_err}")
    _LOG.info("  [PASS] comb_mix doubly stochastic (col_err=%.2e row_err=%.2e)",
              col_err, row_err)

    zero = [t.clone() for t in args]
    zero[4] = torch.zeros_like(zero[4])
    _, _, li = mhc_pre_fuse_golden_cpu(zero)
    if float(li.abs().max()) != 0.0:
        raise AssertionError("zero residual must give a zero layer_input")
    _LOG.info("  [PASS] zero residual -> zero layer_input")

    ones = [t.clone() for t in args]
    ones[4] = torch.ones_like(ones[4])
    _, _, li1 = mhc_pre_fuse_golden_cpu(ones)
    got = float(li1.to(torch.float32).mean())
    pre = torch.sigmoid(args[0][:, 0:4] * args[2][0] + args[3][0:4]) + 1e-6
    want = float(pre.sum(dim=-1).mean())
    if abs(got - want) > 1e-2:
        raise AssertionError(f"unit residual: layer_input {got} != sum(pre) {want}")
    _LOG.info("  [PASS] unit residual -> layer_input == sum(pre) (%.6f ~ %.6f)", got, want)


def _run_generalization_checks():
    _LOG.info("[generalization over dynamic axes]")
    plan = (
        ("gen_min", 1, 64),
        ("gen_t_not_multiple_of_8", 509, 1280),
        ("gen_hidden_small", 8, 1280),
        ("gen_hidden_4160", 16, 4160),
        ("gen_hidden_7168", 32, 7168),
    )
    for name, t, hidden in plan:
        args = _case_args((t, hidden, 20261001 + t + hidden))
        out = mhc_pre_fuse_golden_cpu(args)
        diff, peak = _check_output(name, out, args)
        _LOG.info("  [PASS] %s (T=%d, hidden=%d) max_diff=%.6f peak=%.4f",
                  name, t, hidden, diff, peak)


def _validate():
    """Run every validation case and check the mhc_pre_fuse contract."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _LOG.info("=" * 60)
    _LOG.info("mhc_pre_fuse_golden_cpu validation report")
    _LOG.info("=" * 60)

    _LOG.info("[validation cases]")
    for name, t, hidden, seed in _VALIDATION_CASES:
        args = _case_args((t, hidden, seed))
        out = mhc_pre_fuse_golden_cpu(args)
        diff, peak = _check_output(name, out, args)
        _LOG.info("  [case: %s] T=%d hidden=%d max_diff=%.6f peak=%.4f ... PASS",
                  name, t, hidden, diff, peak)

    _run_property_checks()
    _run_generalization_checks()

    _LOG.info("=" * 60)
    _LOG.info("validation complete")
    _LOG.info("=" * 60)
    return True


if __name__ == "__main__":
    raise SystemExit(0 if _validate() else 1)
