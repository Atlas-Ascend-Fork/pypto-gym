# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# =============================================================================
# test_fused_recurrent_kda.py
# Operator-level E2E test for fused_recurrent_kda (L0 — single module).
#
# Compares fused_recurrent_kda_impl (the @pypto.frontend.jit kernel via the
# Layer K host wrapper) against fused_recurrent_kda_golden (operator-level
# torch reference) on 33 cases mirroring
# vllm-ascend/tests/.../test_fused_recurrent_kda_npu.py:
#   Group 1 (non-inplace varlen, 10): cu/D/dtype/H variations
#   Group 2 (inplace decode, 9):      N/H/dtype + non-16-aligned seq_len
#   Group 3 (fp32 non-inplace, 4):    isolate algorithmic error
#   Group 4 (inplace stress, 6):      max batch 16k/32k + H=4,8 generalization
#   Group 5 (inplace H=32, 4):        representative seq_len
#
# Comparison uses the strict per-element assertion `compare` (mirrors
# models/deepseek_v4/utils/compare.py). Pass criterion: per-element
# `|t-ref| <= atol + rtol*|ref|`, with error_count <= max_error_ratio*numel.
# No NaN/Inf allowed. Tolerance is dtype-driven:
#   bf16 / fp16 output -> atol=0.0001,    rtol=0.0078125
#   fp32         output -> atol=0.000025,  rtol=0.005
# max_error_ratio / max_error_count kept at compare() defaults (0.005 / 10).
#
# The impl wrapper already transposes ht back to naive [S,H,K,V]
# layout (impl L421), so ht is compared DIRECTLY against the golden (also
# naive [S,H,K,V]) — no extra .transpose(-1,-2) is applied.
# =============================================================================

import copy
import logging
import os
import sys
from pathlib import Path

import pytest
import pypto

# ═══════════════════════════════════════════════════════════════════════════════
# Path bootstrap: add ops/pypto_tensor so ling_3_0_flash package
# resolves, and this test dir so the golden module resolves.
# ═══════════════════════════════════════════════════════════════════════════════
_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent.parent.parent
_ops_dir = str(_REPO_ROOT / "ops" / "pypto_tensor")
if _ops_dir not in sys.path:
    sys.path.insert(0, _ops_dir)
_this_dir = str(_HERE)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  required for NPU device init

# Impl: public entry fused_recurrent_kda_impl (alias for fused_recurrent_kda_wrapper)
from ling_3_0_flash.fused_recurrent_kda.fused_recurrent_kda_impl import (
    fused_recurrent_kda_impl,
    fused_recurrent_kda,
)
from fused_recurrent_kda_golden import fused_recurrent_kda_golden

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Per-dtype tolerance table (user-specified).
#   bf16 / fp16 output -> atol=0.0001,    rtol=0.0078125
#   fp32         output -> atol=0.000025,  rtol=0.005
_TOL = {
    "bf16": dict(atol=1e-4,      rtol=7.8125e-3),
    "fp16": dict(atol=1e-4,      rtol=7.8125e-3),
    "fp32": dict(atol=2.5e-5,    rtol=5e-3),
}
# compare() defaults
_MAX_ERROR_RATIO = 0.005
_MAX_ERROR_COUNT = 10


# =============================================================================
# compare (mirrors models/deepseek_v4/utils/compare.py)
# =============================================================================

def compare(t: torch.Tensor, t_ref: torch.Tensor, name, atol, rtol,
            max_error_ratio=_MAX_ERROR_RATIO, max_error_count=_MAX_ERROR_COUNT):
    """Compare two tensors element-wise; assert on failure.

    Pass criterion:
      - t must contain no NaN / Inf
      - t.shape == t_ref.shape, t.dtype == t_ref.dtype, t.device == t_ref.device
      - number of points where |t - t_ref| > atol + rtol*|t_ref|
        must not exceed min(max_error_ratio * numel, max_error_count)

    Args:
        t: tensor under test (impl).
        t_ref: reference tensor (golden).
        name: tensor name (for logging).
        atol: absolute tolerance.
        rtol: relative tolerance.
        max_error_ratio: max fraction of out-of-tol points allowed.
        max_error_count: max display count AND upper bound for the threshold.
    """
    # ── NaN / Inf guard ──
    nan_mask = torch.isnan(t)
    nan_count = nan_mask.sum().item()
    inf_mask = torch.isinf(t)
    inf_count = inf_mask.sum().item()
    if nan_count > 0 or inf_count > 0:
        error_msg = f"\n========== tensor {name} contains illegal values (NaN/Inf forbidden) =========="
        if nan_count > 0:
            nan_positions = torch.nonzero(nan_mask, as_tuple=False)
            show_nan_count = min(nan_count, max_error_count)
            error_msg += f"\n- NaN count: {nan_count}, first {show_nan_count} positions:"
            for i in range(show_nan_count):
                pos_tuple = tuple(p.item() for p in nan_positions[i])
                error_msg += f"\n  position {pos_tuple}"
        if inf_count > 0:
            inf_positions = torch.nonzero(inf_mask, as_tuple=False)
            show_inf_count = min(inf_count, max_error_count)
            error_msg += f"\n- Inf count: {inf_count}, first {show_inf_count} positions (value type):"
            for i in range(show_inf_count):
                pos = inf_positions[i]
                pos_tuple = tuple(p.item() for p in pos)
                inf_val = t[pos_tuple].item()
                inf_type = "+Inf" if inf_val == float('inf') else "-Inf"
                error_msg += f"\n  position {pos_tuple}: {inf_type}"
        error_msg += "\n" + "=" * 80 + "\n"
        assert False, error_msg

    # ── basic attribute checks ──
    assert t.shape == t_ref.shape, f"shape mismatch: t.shape={t.shape}, t_ref.shape={t_ref.shape}"
    assert t.dtype == t_ref.dtype, f"dtype mismatch: t.dtype={t.dtype}, t_ref.dtype={t_ref.dtype}"
    assert t.device == t_ref.device, f"device mismatch: t.device={t.device}, t_ref.device={t_ref.device}"

    # ── error point threshold (min of ratio-based and max-count) ──
    error_count_threshold = round(max_error_ratio * t_ref.numel())

    # ── per-element diff vs tolerance ──
    diff_abs = (t - t_ref).abs()
    tolerance = atol + rtol * t_ref.abs()
    diff_mask = diff_abs > tolerance
    error_count = diff_mask.sum().item()

    # max diff and its position
    max_diff, flat_max_pos = torch.max(diff_abs.flatten(), dim=0)
    max_pos = tuple(idx.item() for idx in torch.unravel_index(flat_max_pos, t.shape))

    if error_count > 0:
        print(f"\n========== tensor {name} has {error_count} out-of-tol points (threshold: {error_count_threshold}) ==========")
        error_positions = torch.nonzero(diff_mask, as_tuple=False)
        show_count = min(error_count, max_error_count)
        print(f"showing first {show_count} out-of-tol points (position | impl | golden | abs_diff | tolerance):")
        for i in range(show_count):
            pos = error_positions[i]
            pos_tuple = tuple(p.item() for p in pos)
            t_val = t[pos_tuple].item()
            t_ref_val = t_ref[pos_tuple].item()
            diff_val = diff_abs[pos_tuple].item()
            tol_val = tolerance[pos_tuple].item()
            print(f"  pos {pos_tuple}: {t_val:.8f} vs {t_ref_val:.8f} | diff={diff_val:.8f} | tol={tol_val:.8f}")
        print(f"\nmax diff point: pos {max_pos} | diff={max_diff.item():.8f} | tol={tolerance[max_pos].item():.8f}")
        print("=" * 80 + "\n")

    assert error_count <= error_count_threshold, \
        (f"compare fail: {name}, max diff: {max_diff.item():.8f} at {max_pos}, "
         f"error_count: {error_count}, error_count_threshold: {error_count_threshold}")

    print("compare success !!!!")


def _tol_for(t):
    """Pick tolerance by dtype: bf16/fp16 use low-precision set, fp32 use fp32 set."""
    if t.dtype == torch.bfloat16:
        return _TOL["bf16"]
    if t.dtype == torch.float16:
        return _TOL["fp16"]
    return _TOL["fp32"]


def _safe_compare(t_impl, t_gold, name):
    """Run compare(); return (ok, msg). Captures assertion failures."""
    tol = _tol_for(t_gold)
    try:
        compare(t_impl.cpu(), t_gold.cpu(), name, atol=tol["atol"], rtol=tol["rtol"])
        return True, f"{name}: compare success (atol={tol['atol']}, rtol={tol['rtol']}, dtype={t_gold.dtype})"
    except AssertionError as e:
        first_line = str(e).splitlines()[0] if str(e) else ""
        return False, f"{name}: COMPARE FAIL (atol={tol['atol']}, rtol={tol['rtol']}, dtype={t_gold.dtype}) :: {first_line}"
    except Exception as e:
        return False, f"{name}: EXCEPTION :: {e}"


# =============================================================================
# Helpers
# =============================================================================

def _set_device():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    return torch.device(f"npu:{device_id}")


def _gen_qkvg(H, D, T, dtype, device):
    q = torch.randn(1, T, H, D, dtype=dtype, device=device)
    k = torch.randn(1, T, H, D, dtype=dtype, device=device)
    v = torch.randn(1, T, H, D, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float32, device=device))
    beta = torch.rand(1, T, H, dtype=torch.float32, device=device).sigmoid()
    return q, k, v, g, beta


# =============================================================================
# Test cases
# =============================================================================

def run_case(case_id, H, D, cu_seqlens, dtype, inplace):
    """Run one test case. Returns (case_pass, details_str).

    Pass criterion (via compare()): per-element |t-ref| <= atol+rtol*|ref|,
    error_count <= max_error_ratio*numel; no NaN/Inf. Tolerance is dtype-driven
    (bf16/fp16: atol=1e-4,rtol=7.8125e-3; fp32: atol=2.5e-5,rtol=5e-3).
    """
    device = _set_device()
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32).to(device)

    q, k, v, g, beta = _gen_qkvg(H, D, T, dtype, device)

    if inplace:
        N_decode = len(cu_seqlens) - 1
        # state_buf with sparse slot allocation: max_slots > N_decode
        # ssm_state_indices = random unique permutation (no duplicates)
        max_slots = min(N_decode * 2, N_decode + 256) + 1
        state_buf = torch.randn(max_slots, H, D, D, dtype=torch.float32, device=device)
        state_buf[0] = 0  # NULL slot
        ssm_state_indices = (torch.randperm(max_slots - 1, device=device)[:N_decode] + 1).to(torch.int32)
        kwargs = dict(
            scale=None, initial_state=state_buf, cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=True,
            inplace_final_state=True,
        )
        # Golden uses [K,V] (original); impl uses [V,K] — transpose for impl
        gold_state = state_buf.clone()
        impl_state = state_buf.clone().transpose(-1, -2).contiguous()
        gold_kwargs = dict(kwargs, initial_state=gold_state)
        impl_kwargs = dict(kwargs, initial_state=impl_state)
    else:
        initial_state = torch.randn(T, H, D, D, dtype=torch.float32, device=device)
        kwargs = dict(
            scale=None, initial_state=initial_state, cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
            inplace_final_state=False,
        )
        gold_kwargs = copy.deepcopy(kwargs)
        impl_kwargs = dict(
            scale=kwargs["scale"],
            initial_state=kwargs["initial_state"].clone().transpose(-1, -2).contiguous(),
            cu_seqlens=kwargs["cu_seqlens"].clone(),
            use_qk_l2norm_in_kernel=kwargs["use_qk_l2norm_in_kernel"],
            inplace_final_state=kwargs["inplace_final_state"],
        )

    # --- Run golden ---
    try:
        o_gold, ht_gold = fused_recurrent_kda_golden(q, k, v, g, beta, **gold_kwargs)
    except Exception as e:
        return False, f"GOLDEN CRASH: {e}"

    # --- Run impl ---
    try:
        o_impl, ht_impl = fused_recurrent_kda_impl(q, k, v, g, beta, **impl_kwargs)
    except Exception as e:
        return False, f"IMPL CRASH: {e}"

    details = []
    all_pass = True

    # --- Compare o (bf16 / fp16) ---
    print(f"  Comparing o: gold={tuple(o_gold.shape)} impl={tuple(o_impl.shape)} "
          f"dtype={o_gold.dtype}")
    o_ok, o_msg = _safe_compare(o_impl, o_gold, "o")
    print(f"    {o_msg}")
    if not o_ok:
        all_pass = False
    details.append(o_msg)

    # --- Compare ht (non-inplace) or state_buf slots (inplace) ---
    if inplace:
        # Inplace: gold_state [K,V] vs impl_state [V,K] — transpose impl back
        impl_state_kv = impl_state.cpu().transpose(-1, -2).contiguous()
        print(f"  Comparing state_buf (inplace): gold={tuple(gold_state.shape)} "
              f"impl={tuple(impl_state_kv.shape)} dtype={gold_state.dtype}")
        # Confirm returned final_state is the inplace-updated initial_state
        if ht_gold is not gold_state:
            print(f"    [WARNING] golden final_state is NOT the gold_state clone")
        if ht_impl is not impl_state:
            print(f"    [WARNING] impl final_state is NOT the impl_state clone")
        # NULL slot 0 integrity
        null_ok_gold = torch.all(gold_state[0] == 0).item()
        null_ok_impl = torch.all(impl_state_kv[0] == 0).item()
        print(f"    [NULL slot 0] gold_all_zero={null_ok_gold} impl_all_zero={null_ok_impl}")
        if not null_ok_impl:
            all_pass = False
            details.append("NULL slot[0] NOT zero in impl!")
        # Bulk compare all active slots [1:] as one tensor (fp32)
        s_ok, s_msg = _safe_compare(impl_state_kv[1:], gold_state[1:], "state_buf[1:]")
        print(f"    {s_msg}")
        if not s_ok:
            all_pass = False
        details.append(s_msg)
        # Check whether impl actually updated the caller's state_buf
        impl_changed = not torch.equal(impl_state, state_buf)
        print(f"    [inplace writeback] impl_state_buf modified by wrapper: {impl_changed}")
        if not impl_changed:
            print(f"    [WARNING] impl wrapper did NOT propagate inplace update to caller's tensor")
            details.append("impl_state_buf UNCHANGED (wrapper uses internal copy)")
    else:
        # Non-inplace: golden returns [T,H,K,V], impl returns [T,H,V,K]
        # Transpose impl ht back to [K,V] for comparison
        ht_impl_kv = ht_impl.transpose(-1, -2).contiguous()
        print(f"  Comparing ht: gold={tuple(ht_gold.shape)} "
              f"impl={tuple(ht_impl_kv.shape)} dtype={ht_gold.dtype}")
        ht_ok, ht_msg = _safe_compare(ht_impl_kv, ht_gold, "ht")
        print(f"    {ht_msg}")
        if not ht_ok:
            all_pass = False
        details.append(ht_msg)

    return all_pass, "; ".join(details)


# =============================================================================
# Main
# =============================================================================

# 19 parametrized cases mirroring
# vllm-ascend/tests/.../test_fused_recurrent_kda_npu.py
# Each tuple: (case_id, H, D, cu_seqlens, dtype, inplace, N_decode)
ALL_CASES = [
    # ── Group 1: varlen non-inplace (10 cases) ──
    ("varlen_cu012345678_H4_bf16",   4, 128, [0, 1, 2, 3, 4, 5, 6, 7, 8], torch.bfloat16,  False),
    ("varlen_cu04816_H4_bf16",       4, 128, [0, 4, 8, 16],                torch.bfloat16,  False),
    ("varlen_cu064_H4_bf16",         4, 128, [0, 64],                      torch.bfloat16,  False),
    # ── Group 2: inplace decode (seq=1..32) ──
    ("inplace_seq1_H4_bf16",         4, 128, [0, 1],                       torch.bfloat16,  True),
    ("inplace_seq2_H4_bf16",         4, 128, [0, 1, 2],                    torch.bfloat16,  True),
    ("inplace_seq3_H4_bf16",         4, 128, [0, 1, 2, 3],                torch.bfloat16,  True),
    ("inplace_seq4_H4_bf16",         4, 128, [0, 1, 2, 3, 4],             torch.bfloat16,  True),
    ("inplace_seq5_H4_bf16",         4, 128, list(range(6)),               torch.bfloat16,  True),
    ("inplace_seq6_H4_bf16",         4, 128, list(range(7)),               torch.bfloat16,  True),
    ("inplace_seq7_H4_bf16",         4, 128, list(range(8)),               torch.bfloat16,  True),
    ("inplace_seq8_H4_bf16",         4, 128, list(range(9)),               torch.bfloat16,  True),
    ("inplace_seq9_H4_bf16",         4, 128, list(range(10)),              torch.bfloat16,  True),
    ("inplace_seq10_H4_bf16",        4, 128, list(range(11)),              torch.bfloat16,  True),
    ("inplace_seq11_H4_bf16",        4, 128, list(range(12)),              torch.bfloat16,  True),
    ("inplace_seq12_H4_bf16",        4, 128, list(range(13)),              torch.bfloat16,  True),
    ("inplace_seq13_H4_bf16",        4, 128, list(range(14)),              torch.bfloat16,  True),
    ("inplace_seq14_H4_bf16",        4, 128, list(range(15)),              torch.bfloat16,  True),
    ("inplace_seq15_H4_bf16",        4, 128, list(range(16)),              torch.bfloat16,  True),
    ("inplace_seq16_H4_bf16",        4, 128, list(range(17)),              torch.bfloat16,  True),
    ("inplace_seq17_H4_bf16",        4, 128, list(range(18)),              torch.bfloat16,  True),
    ("inplace_seq18_H4_bf16",        4, 128, list(range(19)),              torch.bfloat16,  True),
    ("inplace_seq19_H4_bf16",        4, 128, list(range(20)),              torch.bfloat16,  True),
    ("inplace_seq20_H4_bf16",        4, 128, list(range(21)),              torch.bfloat16,  True),
    ("inplace_seq21_H4_bf16",        4, 128, list(range(22)),              torch.bfloat16,  True),
    ("inplace_seq22_H4_bf16",        4, 128, list(range(23)),              torch.bfloat16,  True),
    ("inplace_seq23_H4_bf16",        4, 128, list(range(24)),              torch.bfloat16,  True),
    ("inplace_seq24_H4_bf16",        4, 128, list(range(25)),              torch.bfloat16,  True),
    ("inplace_seq25_H4_bf16",        4, 128, list(range(26)),              torch.bfloat16,  True),
    ("inplace_seq26_H4_bf16",        4, 128, list(range(27)),              torch.bfloat16,  True),
    ("inplace_seq27_H4_bf16",        4, 128, list(range(28)),              torch.bfloat16,  True),
    ("inplace_seq28_H4_bf16",        4, 128, list(range(29)),              torch.bfloat16,  True),
    ("inplace_seq29_H4_bf16",        4, 128, list(range(30)),              torch.bfloat16,  True),
    ("inplace_seq30_H4_bf16",        4, 128, list(range(31)),              torch.bfloat16,  True),
    ("inplace_seq31_H4_bf16",        4, 128, list(range(32)),              torch.bfloat16,  True),
    ("inplace_seq32_H4_bf16",        4, 128, list(range(33)),              torch.bfloat16,  True),
    # ── Group 4: inplace stress (max batch + H generalization) ──
    ("inplace_stress_N1024_bf16",    4, 128, list(range(1025)),           torch.bfloat16,  True),
    ("inplace_stress_N4096_bf16",    4, 128, list(range(4097)),           torch.bfloat16,  True),
    ("inplace_stress_H4_N16_bf16",    4, 128, list(range(17)),             torch.bfloat16,  True),
    ("inplace_stress_H8_N16_bf16",    8, 128, list(range(17)),             torch.bfloat16,  True),
    ("inplace_stress_N16384_bf16",    4, 128, list(range(16385)),          torch.bfloat16,  True),
    # ── Group 5: inplace H=32 at representative seq_len ──
    ("inplace_H32_N1_bf16",          32, 128, [0, 1],                       torch.bfloat16,  True),
    ("inplace_H32_N4_bf16",          32, 128, [0, 1, 2, 3, 4],             torch.bfloat16,  True),
    ("inplace_H32_N16_bf16",         32, 128, list(range(17)),              torch.bfloat16,  True),
    ("inplace_H32_N64_bf16",         32, 128, list(range(65)),              torch.bfloat16,  True),
    ("inplace_H32_N3412_bf16",         32, 128, list(range(3413)),              torch.bfloat16,  True),
    # ── Group 6: D=64 generalization ──
    ("inplace_seq32_H4_D64_bf16",    4, 64,  list(range(33)),               torch.bfloat16,  True),
]


# =============================================================================
# Spec decoding test cases (num_accepted_tokens + 2D ssm_state_indices)
# =============================================================================
# Mirrors vllm-ascend production usage in the target model's KDA integration L276-291:
#   fused_recurrent_kda(q, k, v, g, beta, initial_state=recurrent_state_active,
#       cu_seqlens=spec_query_start_loc, ssm_state_indices=spec_state_indices,
#       num_accepted_tokens=num_accepted_tokens, inplace_final_state=True)
#
# ssm_state_indices is 2D [N, mtp], num_accepted_tokens is [N] with values in [1, mtp].

def run_spec_case(case_id, H, D, N, mtp, dtype):
    """Run one spec decoding test case.

    N = number of spec decode sequences
    mtp = multi-token prediction (tokens per sequence)

    Pass criterion: same compare() as run_case.
    """
    device = _set_device()
    torch.manual_seed(42)

    T = N * mtp  # total tokens
    cu_seqlens = list(range(0, T + 1, mtp))
    cu_t = torch.tensor(cu_seqlens, dtype=torch.int32).to(device)

    q, k, v, g, beta = _gen_qkvg(H, D, T, dtype, device)

    # 2D ssm_state_indices: [N, mtp] with unique slots (slot 0 = NULL)
    max_slots = N * mtp + 1
    ssm_state_indices = torch.arange(1, max_slots, dtype=torch.int32, device=device).reshape(N, mtp)

    # num_accepted_tokens: [N] with values in [1, mtp]
    num_accepted_tokens = torch.randint(1, mtp + 1, (N,), dtype=torch.int32, device=device)

    state_buf = torch.randn(max_slots, H, D, D, dtype=torch.float32, device=device)
    state_buf[0] = 0  # NULL slot

    gold_state = state_buf.clone()
    impl_state = state_buf.clone().transpose(-1, -2).contiguous()

    # --- Run golden ---
    try:
        o_gold, ht_gold = fused_recurrent_kda_golden(
            q, k, v, g, beta, scale=None, initial_state=gold_state,
            cu_seqlens=cu_t, ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=True, inplace_final_state=True,
        )
    except Exception as e:
        return False, f"GOLDEN CRASH: {e}"

    # --- Run impl ---
    try:
        o_impl, ht_impl = fused_recurrent_kda_impl(
            q, k, v, g, beta, scale=None, initial_state=impl_state,
            cu_seqlens=cu_t, ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=True, inplace_final_state=True,
        )
    except Exception as e:
        return False, f"IMPL CRASH: {e}"

    details = []
    all_pass = True

    # --- Compare o ---
    print(f"  Comparing o: gold={tuple(o_gold.shape)} impl={tuple(o_impl.shape)} "
          f"dtype={o_gold.dtype}")
    o_ok, o_msg = _safe_compare(o_impl, o_gold, "o")
    print(f"    {o_msg}")
    if not o_ok:
        all_pass = False
    details.append(o_msg)

    # --- Compare state_buf slots (inplace) ---
    impl_state_kv = impl_state.cpu().transpose(-1, -2).contiguous()
    print(f"  Comparing state_buf (spec decode inplace): gold={tuple(gold_state.shape)} "
          f"impl={tuple(impl_state_kv.shape)} dtype={gold_state.dtype}")
    # NULL slot 0 integrity
    null_ok_gold = torch.all(gold_state[0] == 0).item()
    null_ok_impl = torch.all(impl_state_kv[0] == 0).item()
    print(f"    [NULL slot 0] gold_all_zero={null_ok_gold} impl_all_zero={null_ok_impl}")
    if not null_ok_impl:
        all_pass = False
        details.append("NULL slot[0] NOT zero in impl!")
    # Bulk compare all active slots [1:] as one tensor (fp32)
    s_ok, s_msg = _safe_compare(impl_state_kv[1:], gold_state[1:], "state_buf[1:]")
    print(f"    {s_msg}")
    if not s_ok:
        all_pass = False
    details.append(s_msg)

    # --- Print num_accepted_tokens for debugging ---
    print(f"    [spec] N={N}, mtp={mtp}, num_accepted={num_accepted_tokens.cpu().tolist()}")

    return all_pass, "; ".join(details)


SPEC_CASES = [
    # Each tuple: (case_id, H, D, N, mtp, dtype)
    # N = num spec decode sequences, mtp = multi-token prediction (tokens per seq)
    ("spec_mtp1_N4_bf16",    4, 128, 4, 1, torch.bfloat16),   # mtp=1: same as decode
    ("spec_mtp1_N8_bf16",    4, 128, 8, 1, torch.bfloat16),
    ("spec_mtp2_N1_bf16",    4, 128, 1, 2, torch.bfloat16),   # single seq, 2 tokens
    ("spec_mtp2_N4_bf16",    4, 128, 4, 2, torch.bfloat16),
    ("spec_mtp2_N8_bf16",    4, 128, 8, 2, torch.bfloat16),
    ("spec_mtp3_N4_bf16",    4, 128, 4, 3, torch.bfloat16),   # 3 tokens per seq
    ("spec_mtp2_N16_bf16",   4, 128, 16, 2, torch.bfloat16),  # larger batch
    ("spec_mtp2_H32_N4_bf16", 32, 128, 4, 2, torch.bfloat16), # H=32
    ("spec_mtp2_N4_fp16",    4, 128, 4, 2, torch.float16),    # fp16
]


# =============================================================================
# aclgraph mode test
# =============================================================================

class _KDAWrapper(torch.nn.Module):
    """Module wrapper for torch.compile + NPUGraph capture."""

    def __init__(self, scale, initial_state, cu_seqlens, ssm_state_indices,
                 inplace_final_state=False):
        super().__init__()
        self.scale = scale
        self.initial_state = initial_state
        self.cu_seqlens = cu_seqlens
        self.ssm_state_indices = ssm_state_indices
        self.inplace_final_state = inplace_final_state

    def forward(self, q, k, v, g, beta):
        # 100 add ops to let PyPTO kernel run ahead (steal execution)
        q = q + 0.0
        for _ in range(100):
            q = q + 1e-12
        o, ht = fused_recurrent_kda(
            q, k, v, g, beta,
            scale=self.scale,
            initial_state=self.initial_state,
            cu_seqlens=self.cu_seqlens,
            ssm_state_indices=self.ssm_state_indices,
            use_qk_l2norm_in_kernel=True,
            inplace_final_state=self.inplace_final_state,
        )
        return o, ht


def run_aclgraph_case(case_id, H, D, cu_seqlens, dtype, inplace=False):
    """Run one aclgraph capture & replay test case.

    inplace=False: non-inplace mode (returns separate ht).
    inplace=True: inplace mode (state_buf updated in place, returns state slots).
    """
    device = _set_device()
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32).to(device)

    q, k, v, g, beta = _gen_qkvg(H, D, T, dtype, device)
    scale = D ** -0.5

    if inplace:
        N_decode = len(cu_seqlens) - 1
        max_slots = min(N_decode * 2, N_decode + 256) + 1
        state_buf = torch.randn(max_slots, H, D, D, dtype=torch.float32, device=device)
        state_buf[0] = 0  # NULL slot
        ssm_state_indices = (torch.randperm(max_slots - 1, device=device)[:N_decode] + 1).to(torch.int32)
        gold_state = state_buf.clone()
        impl_state = state_buf.clone().transpose(-1, -2).contiguous()

        # golden (inplace)
        o_gold, ht_gold = fused_recurrent_kda_golden(
            q, k, v, g, beta,
            scale=None, initial_state=gold_state, cu_seqlens=cu_seqlens_t,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=True,
            inplace_final_state=True,
        )

        # aclgraph capture & replay — impl uses [V,K] layout
        model = torch.compile(
            _KDAWrapper(None, impl_state, cu_seqlens_t, ssm_state_indices,
                        inplace_final_state=True),
            backend="eager", dynamic=True)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            o_impl, ht_impl = model(q, k, v, g, beta)
        graph.replay()
        torch.npu.synchronize()

        details = []
        all_pass = True

        # compare o
        print(f"  Comparing o: gold={tuple(o_gold.shape)} impl={tuple(o_impl.shape)} dtype={o_gold.dtype}")
        o_ok, o_msg = _safe_compare(o_impl, o_gold, "o")
        print(f"    {o_msg}")
        if not o_ok:
            all_pass = False
        details.append(o_msg)

        # compare state_buf slots (inplace) — impl [V,K] transpose back to [K,V]
        impl_state_kv = impl_state.transpose(-1, -2).contiguous()
        null_ok_impl = torch.all(impl_state_kv[0] == 0).item()
        print(f"    [NULL slot 0] impl_all_zero={null_ok_impl}")
        if not null_ok_impl:
            all_pass = False
            details.append("NULL slot[0] NOT zero in impl!")
        print(f"  Comparing state_buf (inplace): gold={tuple(gold_state.shape)} dtype={gold_state.dtype}")
        s_ok, s_msg = _safe_compare(impl_state_kv[1:], gold_state[1:], "state_buf[1:]")
        print(f"    {s_msg}")
        if not s_ok:
            all_pass = False
        details.append(s_msg)
        return all_pass, "; ".join(details)

    else:
        initial_state = torch.randn(T, H, D, D, dtype=torch.float32, device=device)
        ssm_state_indices = torch.empty(0, dtype=torch.int32, device=device)

        # golden (non-inplace)
        o_gold, ht_gold = fused_recurrent_kda_golden(
            q, k, v, g, beta,
            scale=scale,
            initial_state=initial_state.clone(),
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens_t,
        )

        # aclgraph capture & replay — impl uses [V,K] layout
        impl_state = initial_state.clone().transpose(-1, -2).contiguous()
        model = torch.compile(
            _KDAWrapper(scale, impl_state, cu_seqlens_t, ssm_state_indices,
                        inplace_final_state=False),
            backend="eager", dynamic=True)

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            o_impl, ht_impl = model(q, k, v, g, beta)

        graph.replay()
        torch.npu.synchronize()

        details = []
        all_pass = True

        # compare
        print(f"  Comparing o: gold={tuple(o_gold.shape)} impl={tuple(o_impl.shape)} dtype={o_gold.dtype}")
        o_ok, o_msg = _safe_compare(o_impl, o_gold, "o")
        print(f"    {o_msg}")
        if not o_ok:
            all_pass = False
        details.append(o_msg)

        # impl ht is [V,K] — transpose back to [K,V] for golden comparison
        ht_impl_kv = ht_impl.transpose(-1, -2).contiguous()
        print(f"  Comparing ht: gold={tuple(ht_gold.shape)} impl={tuple(ht_impl_kv.shape)} dtype={ht_gold.dtype}")
        ht_ok, ht_msg = _safe_compare(ht_impl_kv, ht_gold, "ht")
        print(f"    {ht_msg}")
        if not ht_ok:
            all_pass = False
        details.append(ht_msg)
        return all_pass, "; ".join(details)


ACLGRAPH_CASES = [
    # Each tuple: (case_id, H, D, cu_seqlens, dtype, inplace)
    ("aclgraph_cu01_H4_fp16",       4, 128, [0, 1],                       torch.float16, True),
    ("aclgraph_cu01234_H4_fp16",    4, 128, [0, 1, 2, 3, 4],             torch.float16, True),
    ("aclgraph_seq32_H4_fp16",      4, 128, list(range(33)),              torch.float16, True),
]

ENABLE_ACLGRAPH_TEST = False


# =============================================================================
# torch.library.impl verification
# =============================================================================
# Pattern mirrors models/deepseek_v4/test_hc_pre.py:
#   - FusedRecurrentKDA(nn.Module) whose forward calls torch.ops.pypto.<op>
#   - test_fused_recurrent_kda_inmodel: torch.compile + torchair backend
#   - test_fused_recurrent_kda_ops: direct torch.ops.pypto call
# Both compare against fused_recurrent_kda_golden.

class FusedRecurrentKDA(torch.nn.Module):
    """nn.Module whose forward dispatches through the registered custom op
    (torch.ops.pypto.fused_recurrent_kda). Mirrors HC_PRE in test_hc_pre.py.
    """

    def __init__(self, scale, initial_state, cu_seqlens, ssm_state_indices,
                 inplace_final_state, num_accepted_tokens=None):
        super().__init__()
        self.scale = scale
        self.initial_state = initial_state
        self.cu_seqlens = cu_seqlens
        self.ssm_state_indices = ssm_state_indices
        self.inplace_final_state = inplace_final_state
        self.num_accepted_tokens = num_accepted_tokens

    def forward(self, q, k, v, g, beta):
        nat = self.num_accepted_tokens
        if nat is None:
            nat = torch.empty(0, dtype=torch.int32, device=q.device)
        return torch.ops.pypto.fused_recurrent_kda(
            q, k, v, g, beta, self.initial_state, self.cu_seqlens,
            self.ssm_state_indices, nat, self.scale,
            self.inplace_final_state, True,
        )


@pytest.mark.skip(reason="torch.library verification — not a precision test")
def test_fused_recurrent_kda_inmodel(H=4, D=128, cu_seqlens=None, dtype=torch.bfloat16,
                                     inplace=True):
    """torch.compile + torchair backend path (mirrors test_hc_pre_inmodel).

    Skips gracefully if torchair is not installed.
    """
    if cu_seqlens is None:
        cu_seqlens = list(range(33))

    try:
        import torchair as tng
        from torchair.configs.compiler_config import CompilerConfig
    except Exception:
        print("Skip: torchair not installed, skip torch.compile path for "
              "test_fused_recurrent_kda_inmodel")
        return True, "skipped (torchair unavailable)"

    device = _set_device()
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_t = torch.tensor(cu_seqlens, dtype=torch.int32).to(device)
    q, k, v, g, beta = _gen_qkvg(H, D, T, dtype, device)
    scale = D ** -0.5

    if inplace:
        N_decode = len(cu_seqlens) - 1
        max_slots = min(N_decode * 2, N_decode + 256) + 1
        state_buf = torch.randn(max_slots, H, D, D, dtype=torch.float32, device=device)
        state_buf[0] = 0
        ssm_state_indices = (torch.randperm(max_slots - 1, device=device)[:N_decode] + 1).to(torch.int32)
        gold_state = state_buf.clone()
        impl_state = state_buf.clone().transpose(-1, -2).contiguous()
        gold_kwargs = dict(scale=None, initial_state=gold_state, cu_seqlens=cu_t,
                           ssm_state_indices=ssm_state_indices,
                           use_qk_l2norm_in_kernel=True, inplace_final_state=True)
    else:
        initial_state = torch.randn(T, H, D, D, dtype=torch.float32, device=device)
        ssm_state_indices = torch.empty(0, dtype=torch.int32, device=device)
        gold_state = initial_state.clone()
        impl_state = initial_state.clone().transpose(-1, -2).contiguous()
        gold_kwargs = dict(scale=None, initial_state=gold_state, cu_seqlens=cu_t,
                           ssm_state_indices=ssm_state_indices,
                           use_qk_l2norm_in_kernel=True, inplace_final_state=False)

    o_gold, ht_gold = fused_recurrent_kda_golden(q, k, v, g, beta, **gold_kwargs)

    try:
        compiler_config = CompilerConfig()
        compiler_config.mode = "reduce-overhead"
        npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
        model = torch.compile(
            FusedRecurrentKDA(scale, impl_state, cu_t, ssm_state_indices, inplace),
            dynamic=False, fullgraph=True, backend=npu_backend,
        )
        o_impl, ht_impl = model(q, k, v, g, beta)
        pypto.runtime._device_synchronize()
    except Exception as e:
        print(f"Skip: torchair backend unavailable ({type(e).__name__}), skip "
              "torch.compile path for test_fused_recurrent_kda_inmodel")
        return True, "skipped (torchair backend unavailable)"

    details = []
    all_pass = True

    o_ok, o_msg = _safe_compare(o_impl, o_gold, "o")
    print(f"    {o_msg}")
    if not o_ok:
        all_pass = False
    details.append(o_msg)

    if inplace:
        impl_kv = impl_state.cpu().transpose(-1, -2).contiguous()
        s_ok, s_msg = _safe_compare(impl_kv[1:], gold_state[1:], "state[1:]")
        print(f"    {s_msg}")
        if not s_ok:
            all_pass = False
        details.append(s_msg)
    else:
        ht_kv = ht_impl.transpose(-1, -2).contiguous()
        ht_ok, ht_msg = _safe_compare(ht_kv, ht_gold, "ht")
        print(f"    {ht_msg}")
        if not ht_ok:
            all_pass = False
        details.append(ht_msg)

    return all_pass, "; ".join(details)


@pytest.mark.skip(reason="torch.library verification — not a precision test")
def test_fused_recurrent_kda_ops(H=4, D=128, cu_seqlens=None, dtype=torch.bfloat16,
                                 inplace=True):
    """Direct torch.ops.pypto.fused_recurrent_kda call (mirrors test_hc_pre)."""
    if cu_seqlens is None:
        cu_seqlens = list(range(33))

    device = _set_device()
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_t = torch.tensor(cu_seqlens, dtype=torch.int32).to(device)
    q, k, v, g, beta = _gen_qkvg(H, D, T, dtype, device)
    scale = D ** -0.5

    if inplace:
        N_decode = len(cu_seqlens) - 1
        max_slots = min(N_decode * 2, N_decode + 256) + 1
        state_buf = torch.randn(max_slots, H, D, D, dtype=torch.float32, device=device)
        state_buf[0] = 0
        ssm_state_indices = (torch.randperm(max_slots - 1, device=device)[:N_decode] + 1).to(torch.int32)
        gold_state = state_buf.clone()
        impl_state = state_buf.clone().transpose(-1, -2).contiguous()
        gold_kwargs = dict(scale=None, initial_state=gold_state, cu_seqlens=cu_t,
                           ssm_state_indices=ssm_state_indices,
                           use_qk_l2norm_in_kernel=True, inplace_final_state=True)
    else:
        initial_state = torch.randn(T, H, D, D, dtype=torch.float32, device=device)
        ssm_state_indices = torch.empty(0, dtype=torch.int32, device=device)
        gold_state = initial_state.clone()
        impl_state = initial_state.clone().transpose(-1, -2).contiguous()
        gold_kwargs = dict(scale=None, initial_state=gold_state, cu_seqlens=cu_t,
                           ssm_state_indices=ssm_state_indices,
                           use_qk_l2norm_in_kernel=True, inplace_final_state=False)

    o_gold, ht_gold = fused_recurrent_kda_golden(q, k, v, g, beta, **gold_kwargs)

    nat = torch.empty(0, dtype=torch.int32, device=device)
    o_impl, ht_impl = torch.ops.pypto.fused_recurrent_kda(
        q, k, v, g, beta, impl_state, cu_t, ssm_state_indices,
        nat, scale, inplace, True,
    )
    pypto.runtime._device_synchronize()

    details = []
    all_pass = True

    o_ok, o_msg = _safe_compare(o_impl, o_gold, "o")
    print(f"    {o_msg}")
    if not o_ok:
        all_pass = False
    details.append(o_msg)

    if inplace:
        impl_kv = impl_state.cpu().transpose(-1, -2).contiguous()
        s_ok, s_msg = _safe_compare(impl_kv[1:], gold_state[1:], "state[1:]")
        print(f"    {s_msg}")
        if not s_ok:
            all_pass = False
        details.append(s_msg)
    else:
        ht_kv = ht_impl.transpose(-1, -2).contiguous()
        ht_ok, ht_msg = _safe_compare(ht_kv, ht_gold, "ht")
        print(f"    {ht_msg}")
        if not ht_ok:
            all_pass = False
        details.append(ht_msg)

    return all_pass, "; ".join(details)


ENABLE_TORCH_LIBRARY_INMODEL = False


def run_torch_library_tests():
    print(f"\n{'=' * 70}")
    print("torch.library.impl verification (torch.ops + torch.compile)")
    print(f"{'=' * 70}")

    tlb_cases = [
        ("tlb_inplace_seq32_H4_bf16", 4, 128, list(range(33)),   torch.bfloat16, True),
        ("tlb_varlen_H4_bf16",         4, 128, [0, 4, 8, 16],    torch.bfloat16, False),
    ]
    results = []
    for idx, (case_id, H, D, cu_seqlens, dtype, inplace) in enumerate(tlb_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"TLB CASE [{idx}/{len(tlb_cases)}]: {case_id}")
        print(f"{'─' * 70}")

        print(f"  [ops] test_fused_recurrent_kda_ops")
        ok_ops, detail_ops = test_fused_recurrent_kda_ops(H, D, cu_seqlens, dtype, inplace)
        print(f"  >> [ops] {case_id}: {'PASS' if ok_ops else 'FAIL'} | {detail_ops}")
        results.append((f"{case_id}/ops", ok_ops, detail_ops))

        if ENABLE_TORCH_LIBRARY_INMODEL:
            print(f"  [inmodel] test_fused_recurrent_kda_inmodel")
            ok_model, detail_model = test_fused_recurrent_kda_inmodel(H, D, cu_seqlens, dtype, inplace)
            print(f"  >> [inmodel] {case_id}: {'PASS' if ok_model else 'FAIL'} | {detail_model}")
            results.append((f"{case_id}/inmodel", ok_model, detail_model))

    print(f"\n{'=' * 70}")
    print("TORCH.LIBRARY.IMPL SUMMARY")
    print(f"{'=' * 70}")
    tlb_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print(f"  {name:<40} {'PASS' if ok else 'FAIL':<6}")
    print(f"\n{tlb_pass}/{len(results)} torch.library checks passed")
    print(f"{'=' * 70}")
    if all(ok for _, ok, _ in results):
        print(f"[TORCH_LIBRARY_PASS] torch.ops + torch.compile verified for {tlb_pass} checks.")
    else:
        print("[TORCH_LIBRARY_FAIL] one or more torch.library checks failed.")
        sys.exit(1)


# =============================================================================
# pytest-style precision tests (typical cases)
# =============================================================================

_PRECISION_CASES = [
    ("inplace_seq32_H4_bf16", 4, 128, list(range(33)), torch.bfloat16, True),
    ("varlen_cu04816_H4_bf16", 4, 128, [0, 4, 8, 16], torch.bfloat16, False),
]


@pytest.mark.parametrize("case_id,H,D,cu_seqlens,dtype,inplace", _PRECISION_CASES,
                         ids=lambda x: x if isinstance(x, str) else "")
def test_fused_recurrent_kda_precision(case_id, H, D, cu_seqlens, dtype, inplace):
    """Precision: PyPTO impl vs torch golden on typical shapes."""
    ok, detail = run_case(case_id, H, D, cu_seqlens, dtype, inplace=inplace)
    assert ok, f"{case_id}: {detail}"


@pytest.mark.parametrize("case_id,H,D,N,mtp,dtype", [
    ("spec_mtp3_N4_bf16", 4, 128, 4, 3, torch.bfloat16),
], ids=lambda x: x if isinstance(x, str) else "")
def test_fused_recurrent_kda_spec_decode(case_id, H, D, N, mtp, dtype):
    """Spec decoding: PyPTO impl vs torch golden with num_accepted_tokens."""
    ok, detail = run_spec_case(case_id, H, D, N, mtp, dtype)
    assert ok, f"{case_id}: {detail}"


if __name__ == "__main__":
    print("=" * 70)
    print("fused_recurrent_kda operator-level E2E verification (strict compare)")
    print("Tolerance (dtype-driven):")
    print(f"  bf16/fp16 -> atol={_TOL['bf16']['atol']}, rtol={_TOL['bf16']['rtol']}")
    print(f"  fp32      -> atol={_TOL['fp32']['atol']}, rtol={_TOL['fp32']['rtol']}")
    print(f"  max_error_ratio={_MAX_ERROR_RATIO} (default), max_error_count={_MAX_ERROR_COUNT} (default)")
    print("NPU device:", os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    print("=" * 70)

    results = []
    import time
    perf_results = []
    for idx, (case_id, H, D, cu_seqlens, dtype, inplace) in enumerate(ALL_CASES, 1):
        print(f"\n{'─' * 70}")
        cu_str = str(cu_seqlens) if len(cu_seqlens) <= 10 else f"[0,..,{cu_seqlens[-1]}](len={len(cu_seqlens)})"
        print(f"CASE [{idx}/{len(ALL_CASES)}]: {case_id}  (H={H}, D={D}, cu={cu_str}, "
              f"dtype={dtype}, inplace={inplace})")
        print(f"{'─' * 70}")
        try:
            ok, detail = run_case(case_id, H, D, cu_seqlens, dtype,
                                  inplace=inplace)
        except Exception as e:
            import traceback
            traceback.print_exc()
            ok, detail = False, f"UNHANDLED EXCEPTION: {e}"
        status = "PASS" if ok else "FAIL"
        print(f"  >> [{idx}/{len(ALL_CASES)}] {case_id}: {status} | {detail}")
        results.append((idx, case_id, ok, detail))

        # ── performance timing (impl only) ──
        if ok:
            print(f"\n  [PERF] timing impl kernel (warmup=5, iters=20)...")
            device = _set_device()
            torch.manual_seed(42)
            T = cu_seqlens[-1]
            cu_t = torch.tensor(cu_seqlens, dtype=torch.int32).to(device)
            q, k, v, g, beta = _gen_qkvg(H, D, T, dtype, device)
            if inplace:
                N_decode = len(cu_seqlens) - 1
                max_slots = min(N_decode * 2, N_decode + 256) + 1
                state_buf = torch.randn(max_slots, H, D, D, dtype=torch.float32, device=device)
                state_buf[0] = 0
                ssm_state_indices = (torch.randperm(max_slots - 1, device=device)[:N_decode] + 1).to(torch.int32)
                impl_state = state_buf.clone().transpose(-1, -2).contiguous()
                impl_kwargs = dict(
                    scale=None, initial_state=impl_state, cu_seqlens=cu_t,
                    ssm_state_indices=ssm_state_indices,
                    use_qk_l2norm_in_kernel=True, inplace_final_state=True,
                )
            else:
                initial_state = torch.randn(T, H, D, D, dtype=torch.float32, device=device)
                impl_state = initial_state.clone().transpose(-1, -2).contiguous()
                impl_kwargs = dict(
                    scale=None, initial_state=impl_state, cu_seqlens=cu_t,
                    use_qk_l2norm_in_kernel=True, inplace_final_state=False,
                )

            # warmup
            for _ in range(5):
                fused_recurrent_kda_impl(q, k, v, g, beta, **impl_kwargs)
            torch.npu.synchronize()

            # timed iters
            times = []
            for _ in range(20):
                t0 = time.perf_counter()
                fused_recurrent_kda_impl(q, k, v, g, beta, **impl_kwargs)
                torch.npu.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

            avg_ms = sum(times) / len(times)
            min_ms = min(times)
            max_ms = max(times)
            print(f"  [PERF] avg={avg_ms:.3f} ms  min={min_ms:.3f} ms  max={max_ms:.3f} ms  (20 iters)")
            perf_results.append((case_id, avg_ms, min_ms, max_ms))

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    total_pass = sum(1 for _, _, ok, _ in results if ok)
    print(f"{'#':>3}  {'case_id':<28} {'status':<6} detail")
    print(f"{'-' * 3}  {'-' * 28} {'-' * 6} {'-' * 40}")
    for idx, case_id, ok, detail in results:
        print(f"{idx:>3}  {case_id:<28} {'PASS' if ok else 'FAIL':<6} {detail}")
    print(f"\n{total_pass}/{len(results)} cases passed")
    print(f"{'=' * 70}")
    if all(ok for _, _, ok, _ in results):
        print(f"[PRECISION_PASS] compare() success on o and ht/state for all {total_pass} cases.")
    else:
        print("[PRECISION_FAIL] one or more cases failed.")
        sys.exit(1)

    # ── performance summary ──
    if perf_results:
        print(f"\n{'=' * 70}")
        print("PERFORMANCE SUMMARY (impl kernel, 20 iters after 5 warmup)")
        print(f"{'=' * 70}")
        print(f"{'case_id':<28} {'avg(ms)':>10} {'min(ms)':>10} {'max(ms)':>10}")
        print(f"{'-' * 28} {'-' * 10} {'-' * 10} {'-' * 10}")
        for case_id, avg_ms, min_ms, max_ms in perf_results:
            print(f"{case_id:<28} {avg_ms:>10.3f} {min_ms:>10.3f} {max_ms:>10.3f}")
        print(f"{'=' * 70}")

    # ── spec decoding tests (num_accepted_tokens + 2D ssm_state_indices) ──
    print(f"\n{'=' * 70}")
    print("SPEC DECODING verification (num_accepted_tokens + 2D ssm_state_indices)")
    print(f"{'=' * 70}")

    spec_results = []
    for idx, (case_id, H, D, N, mtp, dtype) in enumerate(SPEC_CASES, 1):
        print(f"\n{'─' * 70}")
        print(f"SPEC CASE [{idx}/{len(SPEC_CASES)}]: {case_id}  (H={H}, D={D}, N={N}, mtp={mtp}, dtype={dtype})")
        print(f"{'─' * 70}")
        try:
            ok, detail = run_spec_case(case_id, H, D, N, mtp, dtype)
        except Exception as e:
            import traceback
            traceback.print_exc()
            ok, detail = False, f"UNHANDLED EXCEPTION: {e}"
        status = "PASS" if ok else "FAIL"
        print(f"  >> [{idx}/{len(SPEC_CASES)}] {case_id}: {status} | {detail}")
        spec_results.append((idx, case_id, ok, detail))

    print(f"\n{'=' * 70}")
    print("SPEC DECODING SUMMARY")
    print(f"{'=' * 70}")
    spec_pass = sum(1 for _, _, ok, _ in spec_results if ok)
    print(f"{'#':>3}  {'case_id':<28} {'status':<6} detail")
    print(f"{'-' * 3}  {'-' * 28} {'-' * 6} {'-' * 40}")
    for idx, case_id, ok, detail in spec_results:
        print(f"{idx:>3}  {case_id:<28} {'PASS' if ok else 'FAIL':<6} {detail}")
    print(f"\n{spec_pass}/{len(spec_results)} spec decode cases passed")
    print(f"{'=' * 70}")
    if all(ok for _, _, ok, _ in spec_results):
        print(f"[SPEC_PASS] compare() success on o and state for all {spec_pass} spec decode cases.")
    else:
        print("[SPEC_FAIL] one or more spec decode cases failed.")
        sys.exit(1)

    # ── aclgraph mode tests ──
    if not ENABLE_ACLGRAPH_TEST:
        print(f"\n{'=' * 70}")
        print("aclgraph mode tests skipped (ENABLE_ACLGRAPH_TEST=False)")
        print(f"{'=' * 70}")
    else:
        print(f"\n{'=' * 70}")
        print("aclgraph mode verification (capture & replay, strict compare)")
        print(f"{'=' * 70}")

        acl_results = []
        for idx, (case_id, H, D, cu_seqlens, dtype, inplace) in enumerate(ACLGRAPH_CASES, 1):
            print(f"\n{'─' * 70}")
            cu_str = str(cu_seqlens) if len(cu_seqlens) <= 10 else f"[0,..,{cu_seqlens[-1]}](len={len(cu_seqlens)})"
            print(f"ACLGRAPH CASE [{idx}/{len(ACLGRAPH_CASES)}]: {case_id}  "
                  f"(H={H}, D={D}, cu={cu_str}, dtype={dtype}, inplace={inplace})")
            print(f"{'─' * 70}")
            try:
                ok, detail = run_aclgraph_case(case_id, H, D, cu_seqlens, dtype,
                                               inplace=inplace)
            except Exception as e:
                import traceback
                traceback.print_exc()
                ok, detail = False, f"UNHANDLED EXCEPTION: {e}"
            status = "PASS" if ok else "FAIL"
            print(f"  >> [{idx}/{len(ACLGRAPH_CASES)}] {case_id}: {status} | {detail}")
            acl_results.append((idx, case_id, ok, detail))

        print(f"\n{'=' * 70}")
        print("ACLGRAPH SUMMARY")
        print(f"{'=' * 70}")
        acl_pass = sum(1 for _, _, ok, _ in acl_results if ok)
        print(f"{'#':>3}  {'case_id':<28} {'status':<6} detail")
        print(f"{'-' * 3}  {'-' * 28} {'-' * 6} {'-' * 40}")
        for idx, case_id, ok, detail in acl_results:
            print(f"{idx:>3}  {case_id:<28} {'PASS' if ok else 'FAIL':<6} {detail}")
        print(f"\n{acl_pass}/{len(acl_results)} aclgraph cases passed")
        print(f"{'=' * 70}")
        if all(ok for _, _, ok, _ in acl_results):
            print(f"[ACLGRAPH_PASS] compare() success on o and ht/state for all {acl_pass} aclgraph cases.")
        else:
            print("[ACLGRAPH_FAIL] one or more aclgraph cases failed.")
            sys.exit(1)

    # ── torch.library.impl custom-op path tests ──
    # Mirrors models/deepseek_v4/test_*_v4.py pattern: an nn.Module whose
    # forward calls the registered custom op (torch.ops.pypto.fused_recurrent_kda)
    # via the *_graph convenience entry, exercised under torch.compile +
    # NPUGraph capture. Validates the Meta (shape inference) + NPU dispatch
    # registration added to fused_recurrent_kda_impl.py.
    run_torch_library_tests()
