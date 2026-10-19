#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
"""engram_gate_bwd CPU Golden for the exact SPEC output ABI.

Pure-torch CPU reference.  The four per-``(t, hc)`` statistics the forward
``engram_gate`` saves for the backward (``rstd_x``, ``rstd_k``, ``raw_dot``,
``gate``) are recomputed here with the forward's own formula, so the analytic
backward below can be checked as the true adjoint by ``_ref_autograd``, which
differentiates the forward with ``torch.backward``.  Every reduction runs in
FP32.

``image_mask``==True tokens returned ``x`` unchanged in the forward, so they get
``grad_x == grad_out`` and zero ``grad_key``/``grad_value``.
"""

import logging

import torch

_LOG = logging.getLogger(__name__)

_HC = 4
_EPS = 1e-06
_CLAMP_VALUE = 1e-06
_VALUE_LIMIT = 8.0
_FORWARD_MATH = (
    'rstd_x = rsqrt(mean(x^2)+eps); rstd_k = rsqrt(mean(key^2)+eps); '
    'raw_dot = sum_d x*weight*key; dot = raw_dot*rstd_x*rstd_k/sqrt(dim); '
    'gate = sigmoid(copysign(sqrt(max(abs(dot), clamp_value)), dot)); '
    'y = bf16(x + gate*value); masked token (image_mask==True) -> y = x exactly'
)
_BWD_MATH = (
    'dgate = sum_d grad_out*value; '
    'beta = dgate*gate*(1-gate)*0.5*sqrt(rstd_x*rstd_k/sqrt(dim)/abs(raw_dot)) '
    '(0 when abs(dot) < clamp_value, 0 for masked tokens); c = -beta*raw_dot/dim; '
    'grad_x = grad_out + beta*weight*key + c*rstd_x^2*x; '
    'grad_key = beta*weight*x + c*rstd_k^2*key; '
    'grad_value = sum_hc gate*grad_out; grad_weight = sum_t beta*x*key'
)

# P0 cases from the SPEC machine contract: the 12 (num_tokens, dim) shapes the
# model asks for, without image tokens.  name, t, hc, dim, seed, mask_kind.
_TOKENS = (512, 8001)
_DIMS = (2048, 2560, 3072, 4096, 6144, 7168)
_MASK_KINDS = ("all_false",)
_P0_CASES = []
for _t in _TOKENS:
    for _dim in _DIMS:
        for _mask in _MASK_KINDS:
            _P0_CASES.append((f"t{_t}_h{_dim}" + ("_mask" if _mask != "all_false" else ""),
                              _t, _HC, _dim,
                              20261014 + len(_P0_CASES), _mask))
_P0_CASES = tuple(_P0_CASES)
_P0_NAMES = tuple(row[0] for row in _P0_CASES)


def _randn(shape, dtype, generator):
    """Deterministic FP32 normal sample, truncated to the forward's value_range."""
    sample = torch.randn(shape, generator=generator, dtype=torch.float32)
    return sample.clamp(-_VALUE_LIMIT, _VALUE_LIMIT).to(dtype)


def _case_args(spec):
    """Build one legal input tuple (grad_out, x, key, value, weight, image_mask)."""
    t, hc, dim, seed, mask_kind = spec
    generator = torch.Generator(device="cpu").manual_seed(seed)
    grad_out = _randn((t, hc, dim), torch.bfloat16, generator)
    x = _randn((t, hc, dim), torch.bfloat16, generator)
    key = _randn((t, hc, dim), torch.bfloat16, generator)
    value = _randn((t, dim), torch.bfloat16, generator)
    weight = _randn((hc, dim), torch.float32, generator)
    if mask_kind == "partial":
        image_mask = torch.arange(t) % 2 == 1
    elif mask_kind == "all_false":
        image_mask = torch.zeros(t, dtype=torch.bool)
    elif mask_kind == "all_true":
        image_mask = torch.ones(t, dtype=torch.bool)
    else:
        raise ValueError(f"unknown mask kind: {mask_kind}")
    return [grad_out, x, key, value, weight, image_mask]


def engram_gate_saved_stats(x, key, weight, eps=_EPS):
    """The four statistics the forward saves: rstd_x, rstd_k, raw_dot, gate."""
    x_f = x.to(torch.float32)
    key_f = key.to(torch.float32)
    dim = x_f.shape[-1]
    rstd_x = torch.rsqrt(x_f.square().mean(dim=-1) + eps)
    rstd_k = torch.rsqrt(key_f.square().mean(dim=-1) + eps)
    raw_dot = (x_f * weight.to(torch.float32).unsqueeze(0) * key_f).sum(dim=-1)
    dot = raw_dot * rstd_x * rstd_k * (dim ** -0.5)
    signed_root = torch.copysign(dot.abs().clamp_min(_CLAMP_VALUE).sqrt(), dot)
    return rstd_x, rstd_k, raw_dot, torch.sigmoid(signed_root)


def _split(inputs):
    """The six forward-shaped tensors out of a wrapper/reference input tuple.

    Returned as ``(grad_out, x, key, value, weight, image_mask)`` so the callers
    can keep unpacking in place.
    """
    mask_at = 9 if len(inputs) > 6 else 5
    return tuple(inputs[i] for i in (0, 1, 2, 3, 4, mask_at))


def _bwd_math(inputs, eps=_EPS, clamp_value=_CLAMP_VALUE):
    """SPEC backward in FP32; returns (grad_x, grad_key, grad_value, grad_weight)."""
    grad_out, x, key, value, weight, image_mask = _split(inputs)
    x_f = x.to(torch.float32)
    key_f = key.to(torch.float32)
    value_f = value.to(torch.float32)
    weight_f = weight.to(torch.float32)
    dy = grad_out.to(torch.float32)
    dim = x_f.shape[-1]

    rstd_x, rstd_k, raw_dot, gate = engram_gate_saved_stats(x, key, weight, eps)
    keep = (~image_mask.to(torch.bool)).reshape(-1, 1).to(torch.float32)

    dot = raw_dot * rstd_x * rstd_k * (dim ** -0.5)
    beta = (dy * value_f.unsqueeze(1)).sum(-1) * gate * (1.0 - gate) * 0.5
    beta = beta * (rstd_x * rstd_k * (dim ** -0.5) / raw_dot.abs()).sqrt()
    beta = torch.where(dot.abs() >= clamp_value, beta, torch.zeros_like(beta)) * keep

    c = -beta * raw_dot / dim
    grad_x = (dy + beta.unsqueeze(-1) * weight_f.unsqueeze(0) * key_f
              + c.unsqueeze(-1) * rstd_x.square().unsqueeze(-1) * x_f)
    grad_key = (beta.unsqueeze(-1) * weight_f.unsqueeze(0) * x_f
                + c.unsqueeze(-1) * rstd_k.square().unsqueeze(-1) * key_f)
    grad_value = ((gate * keep).unsqueeze(-1) * dy).sum(1)
    grad_weight = (beta.unsqueeze(-1) * x_f * key_f).sum(0)
    return grad_x, grad_key, grad_value, grad_weight


def _ref_autograd(inputs, eps=_EPS, clamp_value=_CLAMP_VALUE):
    """Independent oracle: the forward in FP64, differentiated by ``torch.backward``."""
    grad_out, x, key, value, weight, image_mask = _split(inputs)
    x_f = x.detach().to(torch.float64).requires_grad_(True)
    key_f = key.detach().to(torch.float64).requires_grad_(True)
    value_f = value.detach().to(torch.float64).requires_grad_(True)
    weight_f = weight.detach().to(torch.float64).requires_grad_(True)
    dy = grad_out.detach().to(torch.float64)
    dim = x_f.shape[-1]

    rstd = (torch.rsqrt(x_f.square().mean(-1) + eps)
            * torch.rsqrt(key_f.square().mean(-1) + eps))
    dot = (x_f * weight_f.unsqueeze(0) * key_f).sum(-1) * rstd * (dim ** -0.5)
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    y = x_f + gate.unsqueeze(-1) * value_f.unsqueeze(-2)
    torch.where(image_mask.detach().reshape(-1, 1, 1), x_f, y).backward(dy)
    return x_f.grad, key_f.grad, value_f.grad, weight_f.grad


def engram_gate_bwd_golden_cpu(tensors, eps=_EPS, clamp_value=_CLAMP_VALUE):
    """Implement the SPEC backward on CPU and return the declared output dtypes.

    Args:
        tensors: ``(grad_out, x, key, value, weight, image_mask)`` -- the same
            positional tuple that ``reference`` and ``_bwd_math`` take, so all
            three entry points are called the same way:
            * grad_out: [t, hc, dim] bfloat16, the gradient of the forward's y.
            * x: [t, hc, dim] bfloat16.
            * key: [t, hc, dim] bfloat16.
            * value: [t, dim] bfloat16, shared by every hc.
            * weight: [hc, dim] float32.
            * image_mask: [t] bool; True token -> grad_x = grad_out, else zero.

    Returns:
        grad_x [t, hc, dim] bfloat16, grad_key [t, hc, dim] bfloat16,
        grad_value [t, dim] bfloat16, grad_weight [hc, dim] float32.
    """
    grad_x, grad_key, grad_value, grad_weight = _bwd_math(
        list(tensors), eps, clamp_value)
    return (grad_x.to(torch.bfloat16), grad_key.to(torch.bfloat16),
            grad_value.to(torch.bfloat16), grad_weight.to(torch.float32))


def make_inputs(case):
    """Delivery helper: one case's positional wrapper inputs (CPU) from its spec dict.

    The tuple already carries the four saved statistics the op consumes, derived
    from ``x``/``key``/``weight`` with the forward's formula.
    """
    spec = (case["t"], case["hc"], case["dim"], case["seed"], case["mask_kind"])
    grad_out, x, key, value, weight, image_mask = _case_args(spec)
    rstd_x, rstd_k, raw_dot, gate = engram_gate_saved_stats(x, key, weight)
    return [grad_out, x, key, value, weight, rstd_x, rstd_k, raw_dot, gate, image_mask]


def reference(inputs, eps=_EPS, clamp_value=_CLAMP_VALUE):
    """Delivery helper: independent CPU reference for one case's positional inputs."""
    return engram_gate_bwd_golden_cpu(_split(inputs), eps=eps, clamp_value=clamp_value)


def _check_against_oracle(case_name, out, inputs, rel_limit=2e-2):
    """Compare one output tuple with the FP64 autograd oracle."""
    ref = _ref_autograd(inputs)
    names = ("grad_x", "grad_key", "grad_value", "grad_weight")
    for name, got, want in zip(names, out, ref):
        if tuple(got.shape) != tuple(want.shape):
            raise AssertionError(f"{case_name}: {name} shape {tuple(got.shape)} != {tuple(want.shape)}")
        diff = (got.to(torch.float64) - want).abs().max().item()
        scale = max(want.abs().max().item(), 1e-30)
        if diff / scale > rel_limit:
            raise AssertionError(
                f"{case_name}: {name} deviates from the FP64 oracle by "
                f"{diff / scale:.3e} relative > {rel_limit}")


def _check_contract(case_name, out, inputs):
    """Check dtypes, shapes, finiteness and the image-token contract of one output tuple."""
    grad_out, x, _, value, _, image_mask = _split(inputs)
    want = ((tuple(x.shape), torch.bfloat16), (tuple(x.shape), torch.bfloat16),
            (tuple(value.shape), torch.bfloat16),
            ((x.shape[1], x.shape[2]), torch.float32))
    for name, got, (shape, dtype) in zip(
            ("grad_x", "grad_key", "grad_value", "grad_weight"), out, want):
        if tuple(got.shape) != shape:
            raise AssertionError(f"{case_name}: {name} shape {tuple(got.shape)} != {shape}")
        if got.dtype != dtype:
            raise AssertionError(f"{case_name}: {name} dtype {got.dtype} != {dtype}")
        if not bool(torch.isfinite(got).all()):
            raise AssertionError(f"{case_name}: {name} has NaN/Inf")
    mask = image_mask.to(torch.bool)
    if bool(mask.any()):
        if not torch.equal(out[0][mask], grad_out[mask]):
            raise AssertionError(f"{case_name}: masked grad_x is not bit-exact grad_out")
        if out[1][mask].abs().max().item() != 0.0:
            raise AssertionError(f"{case_name}: masked grad_key is not zero")
        if out[2][mask].abs().max().item() != 0.0:
            raise AssertionError(f"{case_name}: masked grad_value is not zero")


def _run_property_checks():
    _LOG.info("[property checks]")

    args = _case_args((4, 4, 512, 31, "all_false"))
    args[0] = torch.zeros_like(args[0])
    out = engram_gate_bwd_golden_cpu(args)
    for name, got in zip(("grad_x", "grad_key", "grad_value", "grad_weight"), out):
        if got.abs().max().item() != 0.0:
            raise AssertionError(f"{name} is not zero when grad_out == 0")
    _LOG.info("  [PASS] grad_out == 0 -> every gradient == 0")

    args = _case_args((4, 4, 512, 33, "all_false"))
    args[3] = torch.zeros_like(args[3])
    out = engram_gate_bwd_golden_cpu(args)
    if not torch.equal(out[0], args[0]):
        raise AssertionError("grad_x should collapse to grad_out when value == 0")
    for index, name in ((1, "grad_key"), (3, "grad_weight")):
        if out[index].abs().max().item() != 0.0:
            raise AssertionError(f"{name} is not zero when value == 0")
    _LOG.info("  [PASS] value == 0 -> dgate == 0, so grad_x == grad_out and grad_key/grad_weight vanish")

    full = _case_args((4, 4, 512, 32, "all_true"))
    out = engram_gate_bwd_golden_cpu(full)
    if not torch.equal(out[0], full[0]):
        raise AssertionError("all-True mask is not bit-exact grad_out")
    for index, name in ((1, "grad_key"), (2, "grad_value"), (3, "grad_weight")):
        if out[index].abs().max().item() != 0.0:
            raise AssertionError(f"all-True mask leaves a non-zero {name}")
    _LOG.info("  [PASS] all-True image_mask -> grad_x == grad_out and the rest is zero")

    _LOG.info("[generalization over dynamic axes]")
    for name, t, hc, dim, kind in (("gen_all_low", 1, 1, 512, "all_false"),
                                   ("gen_t_high", 1024, 2, 1024, "partial"),
                                   ("gen_dim_high", 2, 2, 7168, "all_false")):
        args = _case_args((t, hc, dim, 20261014 + t + hc + dim, kind))
        saved = engram_gate_saved_stats(args[1], args[2], args[4])
        inputs = args[:5] + list(saved) + [args[5]]  # the wrapper's argument order
        out = engram_gate_bwd_golden_cpu(args)
        _check_contract(name, out, inputs)
        _check_against_oracle(name, out, inputs)
        _LOG.info("  [PASS] %s (t=%d, hc=%d, dim=%d)", name, t, hc, dim)


def _validate():
    """Run every contract P0 case and check the engram_gate_bwd contract end-to-end."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _LOG.info("=" * 60)
    _LOG.info("engram_gate_bwd_golden_cpu validation report")
    _LOG.info("=" * 60)
    _LOG.info("forward : %s", _FORWARD_MATH)
    _LOG.info("backward: %s", _BWD_MATH)

    cases = [make_inputs({"t": t, "hc": hc, "dim": dim, "seed": seed, "mask_kind": mask})
             for _, t, hc, dim, seed, mask in _P0_CASES]
    observed = [row[0] for row in _P0_CASES]
    if len(observed) != len(set(observed)):
        raise AssertionError("duplicate case names")

    _LOG.info("[contract P0 cases]  (%d)", len(cases))
    for (name, t, _, dim, _, _), inputs in zip(_P0_CASES, cases):
        out = reference(inputs)
        _check_contract(name, out, inputs)
        # The FP64 autograd oracle needs ~8 copies of the input at double
        # precision, so it runs on the 512-token half of the grid; the larger
        # half is covered by the contract checks above and by the analytic
        # reference at test time.
        if t <= _TOKENS[0]:
            _check_against_oracle(name, out, inputs)
        _LOG.info("[case: %s] dtype=%s shape=%s ... PASS", name, out[0].dtype, tuple(out[0].shape))

    _run_property_checks()

    _LOG.info("=" * 60)
    _LOG.info("validation complete")
    _LOG.info("=" * 60)
    return True


if __name__ == "__main__":
    raise SystemExit(0 if _validate() else 1)
