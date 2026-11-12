# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""block_attn_res_update_rms_norm PyPTO-Pro test: one test function per case.

The kernel and the host wrapper live under ``op_kernel/block_attn_res_update_rms_norm.py``;
the CPU golden and the numeric comparator live under ``tests/assets/``. Each ``test_xxx``
prepares one case's inputs, runs the public wrapper on the NPU and compares ``y`` and the
in-place ``p`` against the independent FP32 CPU golden (``rtol=1e-3, atol=0``, mismatch
ratio <= 1e-3).

Run from the repository root (``conftest.py`` injects ``tests/``, ``tests/assets/`` and
``op_kernel/`` into ``sys.path``)::

    TILE_FWK_DEVICE_ID=<device-id> pytest ops/pro/kimi_k3/block_attn_res_update_rms_norm/tests \
        --mode jit -q
"""

import os

import numpy as np
import pytest
import torch

from block_attn_res_update_rms_norm import (
    block_attn_res_update_rms_norm,
    select_path,
)
from block_attn_res_update_rms_norm_golden_cpu import (
    block_attn_res_update_rms_norm_golden,
    block_attn_res_update_rms_norm_golden_f64,
    make_inputs,
)
from block_attn_res_update_rms_norm_precision import check_output

D = 7168
# Free runtime axis: shapes that also cover the routing threshold and a non-divisible T.
SHAPES = (1, 2, 4, 8, 32, 64, 128, 254, 512, 1024)
# Legal sizes that neither select nor tune the specialization.
UNLISTED_SHAPES = (3, 5)
# Above this many elements a NaN/Inf check on the golden is the only sanity gate needed.
DEVICE_ID = os.environ.get("TILE_FWK_DEVICE_ID", "0")
DEVICE = f"npu:{DEVICE_ID}"


def require_a5():
    """Skip when the visible device is not the Ascend 950 family."""
    try:
        torch.npu.set_device(DEVICE)
    except RuntimeError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")


def _to_device(tensors):
    return tuple(t.to(DEVICE) for t in tensors)


def _run_case(tokens, seed=42, **mode):
    """Return (y_ok, p_ok, readonly_ok, shape_ok, message)."""
    inp = make_inputs(tokens, dim=D, seed=seed, device="cpu", **mode)
    npu = _to_device(inp)
    snapshot = tuple(t.clone() for t in npu)

    y = block_attn_res_update_rms_norm(*npu)
    torch.npu.synchronize()

    p_ref, y_ref = block_attn_res_update_rms_norm_golden(*inp)
    y_ok, y_msg = check_output(y.cpu(), y_ref)
    p_ok, p_msg = check_output(npu[0].cpu(), p_ref)

    shape_ok = (
        tuple(y.shape) == (tokens, D)
        and y.dtype == torch.bfloat16
        and y.device == npu[0].device
    )
    readonly_ok = all(
        torch.equal(snapshot[i], cur) for i, cur in enumerate(npu) if i > 0
    )
    return y_ok, p_ok, readonly_ok, shape_ok, y_msg if not y_ok else p_msg


@pytest.mark.soc("950")
@pytest.mark.parametrize("tokens", SHAPES)
def test_shape(tokens):
    """y, in-place p, output metadata and read-only inputs for every supported T."""
    require_a5()
    y_ok, p_ok, readonly_ok, shape_ok, msg = _run_case(tokens)
    assert y_ok and p_ok, f"T={tokens} numeric mismatch: {msg}"
    assert shape_ok, f"T={tokens} output shape/dtype/device violated"
    assert readonly_ok, f"T={tokens} a read-only input was mutated"


@pytest.mark.soc("950")
@pytest.mark.parametrize("tokens", UNLISTED_SHAPES)
def test_unlisted_legal_shape(tokens):
    """Sizes that neither select nor tune the tiling key still route and compute."""
    require_a5()
    y_ok, p_ok, readonly_ok, shape_ok, msg = _run_case(tokens)
    assert y_ok and p_ok, f"T={tokens} numeric mismatch: {msg}"
    assert shape_ok and readonly_ok, f"T={tokens} metadata/read-only check failed"


@pytest.mark.soc("950")
@pytest.mark.parametrize("seed", (0, 1))
def test_cancellation(seed):
    """Ill-conditioned cancellation: kernel bad count <= 2x the FP32 golden's."""
    require_a5()
    tokens = 8
    inp = make_inputs(tokens, dim=D, seed=seed, device="cpu", cancellation=True)
    npu = _to_device(inp)
    snapshot = tuple(t.clone() for t in npu)

    y = block_attn_res_update_rms_norm(*npu)
    torch.npu.synchronize()
    _, y_golden = block_attn_res_update_rms_norm_golden(*inp)
    _, y_oracle = block_attn_res_update_rms_norm_golden_f64(*inp)

    def bad_count(candidate, reference):
        a = candidate.detach().cpu().float().numpy()
        b = reference.detach().cpu().float().numpy()
        if not np.isfinite(a).all():
            return None
        return int(np.count_nonzero(~np.isclose(a, b, rtol=1e-3, atol=0.0)))

    bad_kernel = bad_count(y, y_oracle)
    bad_golden = bad_count(y_golden, y_oracle)
    assert bad_kernel is not None, f"seed={seed} kernel produced non-finite values"
    limit = 2 * max(bad_golden, 1)
    assert bad_kernel <= limit, (
        f"seed={seed} kernel_bad={bad_kernel} > limit={limit} (golden_bad={bad_golden})"
    )
    assert all(
        torch.equal(snapshot[i], cur) for i, cur in enumerate(npu) if i > 0
    ), f"seed={seed} a read-only input was mutated"


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    "tokens,mode",
    ((8, {"ftz": True}), (8, {"recip_ftz": True}), (254, {"ftz": True})),
)
def test_denormal_input_modes(tokens, mode):
    """FTZ-flavoured inputs: all-zero / subnormal p and h stay finite and correct."""
    require_a5()
    y_ok, p_ok, readonly_ok, shape_ok, msg = _run_case(tokens, **mode)
    assert y_ok and p_ok, f"T={tokens} mode={mode} numeric mismatch: {msg}"
    assert shape_ok and readonly_ok, f"T={tokens} mode={mode} metadata check failed"


@pytest.mark.soc("950")
def test_route_selection():
    """The tiling-key route follows the measured T threshold."""
    assert select_path(0) == 0
    assert select_path(1) == 0
    assert select_path(2) == 0
    assert select_path(3) == 1
    assert select_path(4) == 1
    assert select_path(1024) == 1


@pytest.mark.soc("950")
def test_class_transitions():
    """Cross the threshold in both directions on one kernel object.

    Binary reuse across a transition is not observable through the installed runtime
    (no compilation-event or cache-entry query), so it stays unverified rather than
    being inferred from equal keys.
    """
    require_a5()
    for tokens in (8, 5, 4, 3, 2, 8):
        y_ok, p_ok, readonly_ok, shape_ok, msg = _run_case(tokens)
        assert y_ok and p_ok, f"T={tokens} after transition: {msg}"
        assert shape_ok and readonly_ok, f"T={tokens} after transition: metadata check"


@pytest.mark.soc("950")
def test_empty_token():
    """T == 0 returns an empty BF16 tensor without launching."""
    require_a5()
    inp = make_inputs(4, dim=D, device="cpu")
    npu = _to_device(inp)
    y = block_attn_res_update_rms_norm(
        npu[0][:0], npu[1][:0], npu[2], npu[3][:0], npu[4][:0], npu[5][:0], npu[6]
    )
    assert tuple(y.shape) == (0, D)
    assert y.dtype == torch.bfloat16


@pytest.mark.soc("950")
def test_rejects_invalid_inputs():
    """Host-side validation rejects dtype/shape/device/epsilon violations."""
    require_a5()
    inp = _to_device(make_inputs(4, dim=D, device="cpu"))
    partial, delta, query, numer, lmax, esum, gamma = inp

    with pytest.raises(ValueError):
        block_attn_res_update_rms_norm(
            partial[:, :1024].contiguous(), delta, query, numer, lmax, esum, gamma
        )
    with pytest.raises(ValueError):
        block_attn_res_update_rms_norm(
            partial, query, query, numer, lmax, esum, gamma
        )
    with pytest.raises(ValueError):
        block_attn_res_update_rms_norm(
            partial, delta, query, numer, lmax, esum, gamma, score_eps=-1.0
        )
    with pytest.raises(ValueError):
        block_attn_res_update_rms_norm(
            partial.cpu(), delta.cpu(), query.cpu(), numer.cpu(),
            lmax.cpu(), esum.cpu(), gamma.cpu(),
        )
