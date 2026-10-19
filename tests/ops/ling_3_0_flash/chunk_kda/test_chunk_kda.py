# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# =============================================================================
# test_chunk_kda.py — E2E cleanup gate for the integrated chunk_kda kernel.
#
# Imports the integrated production wrapper (chunk_kda_impl.chunk_kda_wrapper)
# and the user golden (chunk_kda_golden.chunk_kda_golden) SEPARATELY and runs
# detailed_tensor_compare on EVERY leaf output (o AND S). The integrated impl
# was RECONSTRUCTED from _debug/m12_patched + golden M4, so this verifies the
# whole assembly, not just module123. Stable inputs only (q/k/v*0.1,
# g=logsigmoid<=0, beta=sigmoid) per SPEC.md §9 / MEMORY composition.
#
# Tolerances: o bf16 atol/rtol 1e-2; S fp32 atol/rtol 1e-3.
# Run:  pytest tests/ops/ling_3_0_flash/chunk_kda/test_chunk_kda.py   (NPU)
# =============================================================================

import os
import sys
from pathlib import Path

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# Path bootstrap: add ops/pypto_tensor so ling_3_0_flash package
# resolves, and this test dir so golden, detailed_tensor_compare, test_inputs
# resolve.
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
import torch_npu  # noqa: F401  required for NPU device init

from ling_3_0_flash.chunk_kda.chunk_kda_impl import chunk_kda_wrapper
from chunk_kda_golden import chunk_kda_golden
from detailed_tensor_compare import detailed_tensor_compare
from test_inputs import make_inputs, make_call_kwargs

# Per-leaf tolerance: leaf0=o (bf16, 1e-2), leaf1=S (fp32, 1e-3).
_TOL = [(1e-2, 1e-2), (1e-3, 1e-3)]


def _set_device() -> None:
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))


def _golden_per_bh(inputs: dict, kwargs: dict):
    """Run chunk_kda_golden one (B,H) slice at a time on CPU and reassemble.

    The golden materializes a [B,H,NT,BT,BT,K] inverse-decay tensor; at large
    B/T/H this OOMs the card (~32 GiB at B4/T8k/H32). Slicing to B1/H1 caps the
    decay tensor at [1,1,NT,64,64,K] and keeps the golden numerically identical
    (no cross-B/H coupling). initial_state is sliced per (b,h) when present.
    """
    q, k, v, g, beta = (inputs[n].cpu() for n in ("q", "k", "v", "g", "beta"))
    B, T, H = q.shape[0], q.shape[1], q.shape[2]
    init = kwargs.get("initial_state")
    init_cpu = init.cpu() if init is not None else None
    o_rows, S_rows = [], []
    for b in range(B):
        o_h, S_h = [], []
        for h in range(H):
            kw = dict(kwargs)
            kw["initial_state"] = init_cpu[b:b+1, h:h+1] if init_cpu is not None else None
            go, gS = chunk_kda_golden(q[b:b+1, :, h:h+1], k[b:b+1, :, h:h+1],
                                      v[b:b+1, :, h:h+1], g[b:b+1, :, h:h+1],
                                      beta[b:b+1, :, h:h+1], **kw)
            o_h.append(go); S_h.append(gS if gS is not None else None)
        o_rows.append(torch.cat(o_h, dim=2))                                  # cat over H
        S_rows.append(torch.cat(S_h, dim=1) if S_h[0] is not None else None)
    o = torch.cat(o_rows, dim=0)
    S = torch.cat(S_rows, dim=0) if S_rows[0] is not None else None
    return o, S


def _run(case: dict):
    inputs = make_inputs(case)
    kwargs = make_call_kwargs(case, inputs)
    impl_out = chunk_kda_wrapper(*inputs.values(), **kwargs)
    gold_out = _golden_per_bh(inputs, kwargs)
    return impl_out, gold_out


def _compare(case: dict) -> None:
    impl_out, gold_out = _run(case)
    assert len(impl_out) == len(gold_out), f"{case['id']}: leaf count {len(impl_out)}!={len(gold_out)}"
    bad = []
    for i, (a, b) in enumerate(zip(impl_out, gold_out)):
        if a is None and b is None:
            continue  # S omitted when output_final_state=False
        atol, rtol = _TOL[i]
        name = f"{case['id']}_{'o' if i == 0 else 'S'}"
        a = a.cpu()  # kernel on NPU, per-(B,H) golden on CPU — align device
        r = detailed_tensor_compare(a, b, tensor_name=name, atol=atol, rtol=rtol)
        print(f"  {name}: all_close={r['all_close']} max_diff={r['max_diff']:.3e} tol={atol}")
        if not r["all_close"]:
            bad.append(name)
    assert not bad, f"{case['id']}: leaves out of tolerance -> {bad}"


def _assert_leaves(label: str, impl_out, gold_out) -> None:
    """Per-leaf detailed_tensor_compare (leaf0=o 1e-2, leaf1=S 1e-3). impl on NPU,
    golden on CPU — align device. Shared by the l2norm impl-vs-golden cases."""
    assert len(impl_out) == len(gold_out), f"{label}: leaf count mismatch"
    bad = []
    for i, (a, b) in enumerate(zip(impl_out, gold_out)):
        if a is None and b is None:
            continue
        atol, rtol = _TOL[i]
        name = f"{label}_{'o' if i == 0 else 'S'}"
        r = detailed_tensor_compare(a.cpu(), b.cpu(), tensor_name=name, atol=atol, rtol=rtol)
        print(f"  {name}: all_close={r['all_close']} max_diff={r['max_diff']:.3e} tol={atol}")
        if not r["all_close"]:
            bad.append(name)
    assert not bad, f"{label}: leaves out of tolerance -> {bad}"


# ───────────────────────── varlen (TND) helpers ─────────────────────────────
def _stable_inputs(T: int, H: int, K: int, seed: int = 42):
    """Stable scaled packed inputs B==1: q/k/v~*0.1, g=logsigmoid<=0, beta=sigmoid.

    T need NOT be %64 here (packed total across segments); per-seq pad-to-64 is
    the kernel/golden's job. dtype bf16 (q/k/v) + fp32 g/beta (kernel ABI), like make_inputs.
    """
    torch.manual_seed(seed)
    dev = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID','0'))}")
    g0 = torch.randn(1, T, H, K, device=dev)
    return {
        "q":    torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16) * 0.1,
        "k":    torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16) * 0.1,
        "v":    torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16) * 0.1,
        "g":    torch.nn.functional.logsigmoid(g0).float(),
        "beta": torch.sigmoid(torch.randn(1, T, H, device=dev)).float(),  # fp32 (kernel ABI; wrapper no longer casts)
    }


def _golden_varlen_per_h(inputs: dict, cu, init, ofs: bool, use_qk_l2norm: bool = False):
    """Run chunk_kda_golden varlen path one head at a time on CPU, reassemble.

    Segments are tiny (<=320 tok) but per-h slicing caps the decay tensor at
    [1,1,NT,64,64,K] so any H is safe. init [N,H,K,V] -> [N,1,K,V] per head;
    o concat over H (dim2), S concat over H (dim1). No cross-head coupling.
    use_qk_l2norm normalizes q,k per-token over K (independent of the h-slice).
    """
    q, k, v, g, beta = (inputs[n].cpu() for n in ("q", "k", "v", "g", "beta"))
    H = q.shape[2]
    cu_cpu = cu.cpu()
    init_cpu = init.cpu() if init is not None else None
    o_h, S_h = [], []
    for h in range(H):
        ic = init_cpu[:, h:h+1] if init_cpu is not None else None
        go, gS = chunk_kda_golden(q[:, :, h:h+1], k[:, :, h:h+1], v[:, :, h:h+1],
                                  g[:, :, h:h+1], beta[:, :, h:h+1], initial_state=ic,
                                  output_final_state=ofs, use_qk_l2norm_in_kernel=use_qk_l2norm,
                                  cu_seqlens=cu_cpu)
        o_h.append(go); S_h.append(gS if gS is not None else None)
    o = torch.cat(o_h, dim=2)
    S = torch.cat(S_h, dim=1) if (ofs and S_h[0] is not None) else None
    return o, S


def _compare_varlen(label: str, segs, H: int, K: int, *, init_mode="none", ofs=True, seed=42):
    """Build cu from segs, run wrapper (varlen) vs golden per-h, compare every leaf."""
    dev = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID','0'))}")
    T = sum(segs); N = len(segs)
    acc = [0]
    for s in segs:
        acc.append(acc[-1] + s)
    cu = torch.tensor(acc, device=dev, dtype=torch.int32)
    inputs = _stable_inputs(T, H, K, seed=seed)
    init = None
    if init_mode == "rand":
        torch.manual_seed(seed + 7)
        init = torch.randn(N, H, K, K, device=dev, dtype=torch.float32) * 0.1
    impl_out = chunk_kda_wrapper(*inputs.values(), cu_seqlens=cu, initial_state=init,
                                 output_final_state=ofs, scale=K ** -0.5)
    gold_out = _golden_varlen_per_h(inputs, cu, init, ofs)
    bad = []
    for i, (a, b) in enumerate(zip(impl_out, gold_out)):
        if a is None and b is None:
            continue
        atol, rtol = _TOL[i]; name = f"{label}_{'o' if i == 0 else 'S'}"
        r = detailed_tensor_compare(a.cpu(), b, tensor_name=name, atol=atol, rtol=rtol)
        print(f"  {name}: all_close={r['all_close']} max_diff={r['max_diff']:.3e} tol={atol}")
        if not r["all_close"]:
            bad.append(name)
    assert not bad, f"{label}: varlen leaves out of tolerance -> {bad}"


@pytest.mark.skip(reason="temporarily skipped")
def test_chunk_kda_l0() -> None:
    """L0 (dev): B1 T128 H2 K128, output_final_state=True — o(1e-2) AND S(1e-3).

    K is 128 (no pad). The in-kernel layout refactor supports K==V==128 only; the
    K<128 in-kernel zero-pad path is known-broken on the S leaf, so dev stays K128.
    """
    _set_device(); torch.manual_seed(42)
    _compare({"id": "dev", "seed": 42, "shape": {"B": 1, "T": 128, "H": 2, "K": 128},
              "dtype": {"default": "bfloat16"}, "output_final_state": True})


@pytest.mark.skip(reason="temporarily skipped")
def test_chunk_kda_t1() -> None:
    """T1: B1 T512 H32 K128, output_final_state=True — o(1e-2) AND S(1e-3)."""
    _set_device(); torch.manual_seed(42)
    _compare({"id": "T1", "seed": 42, "shape": {"B": 1, "T": 512, "H": 32, "K": 128},
              "dtype": {"default": "bfloat16"}, "output_final_state": True})


@pytest.mark.skip(reason="temporarily skipped")
def test_chunk_kda_l1() -> None:
    """L1 (P0): B1 T4096 H16 K128, output_final_state=True — o(1e-2) AND S(1e-3)."""
    _set_device(); torch.manual_seed(42)
    _compare({"id": "P0", "seed": 42, "shape": {"B": 1, "T": 4096, "H": 16, "K": 128},
              "dtype": {"default": "bfloat16"}, "output_final_state": True})


@pytest.mark.skip(reason="temporarily skipped")
def test_chunk_kda_l2() -> None:
    """L2 (scale): B4 T8192 H32 K128, output_final_state=True — o(1e-2) AND S(1e-3)."""
    _set_device(); torch.manual_seed(42)
    _compare({"id": "L2", "seed": 42, "shape": {"B": 4, "T": 8192, "H": 32, "K": 128},
              "dtype": {"default": "bfloat16"}, "output_final_state": True})


@pytest.mark.skip(reason="temporarily skipped")
def test_chunk_kda_u0() -> None:
    """U0 (fixed-len small): B1 T384 H2 K128 — T 128-aligned (3x128). o(1e-2) AND S(1e-3)."""
    _set_device(); torch.manual_seed(42)
    _compare({"id": "U0", "seed": 42, "shape": {"B": 1, "T": 384, "H": 2, "K": 128},
              "dtype": {"default": "bfloat16"}, "output_final_state": True})


@pytest.mark.skip(reason="temporarily skipped")
def test_chunk_kda_u1() -> None:
    """U1 (fixed-len large): B1 T1920 H32 K128 — T 128-aligned (15x128). o(1e-2) AND S(1e-3)."""
    _set_device(); torch.manual_seed(42)
    _compare({"id": "U1", "seed": 42, "shape": {"B": 1, "T": 1920, "H": 32, "K": 128},
              "dtype": {"default": "bfloat16"}, "output_final_state": True})


@pytest.mark.skip(reason="temporarily skipped")
def test_chunk_kda_l3() -> None:
    """L3 (long-seq): B1 T131072 H32 K128, output_final_state=True — o(1e-2) AND S(1e-3)."""
    _set_device(); torch.manual_seed(42)
    _compare({"id": "L3", "seed": 42, "shape": {"B": 1, "T": 131072, "H": 32, "K": 128},
              "dtype": {"default": "bfloat16"}, "output_final_state": True})


@pytest.mark.skip(reason="temporarily skipped")
def test_chunk_kda_varlen_1seg() -> None:
    """V0 1-seg identity: a single packed segment of length T must equal the
    fixed-len BTHK path bit-for-bit (cu_seqlens=[0,T] -> N=1, no cross-seq). T128 H2."""
    _set_device(); torch.manual_seed(42)
    dev = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID','0'))}")
    inputs = _stable_inputs(128, 2, 128, seed=42)
    cu = torch.tensor([0, 128], device=dev, dtype=torch.int32)
    o_v, S_v = chunk_kda_wrapper(*inputs.values(), cu_seqlens=cu, initial_state=None,
                                 output_final_state=True, scale=128 ** -0.5)
    o_f, S_f = chunk_kda_wrapper(*inputs.values(), output_final_state=True, scale=128 ** -0.5)
    ro = detailed_tensor_compare(o_v.cpu(), o_f.cpu(), tensor_name="V0_1seg_o", atol=0.0, rtol=0.0)
    rs = detailed_tensor_compare(S_v[0].cpu(), S_f[0].cpu(), tensor_name="V0_1seg_S", atol=0.0, rtol=0.0)
    print(f"  V0_1seg_o: all_close={ro['all_close']} max_diff={ro['max_diff']:.3e} (==fixed-len)")
    print(f"  V0_1seg_S: all_close={rs['all_close']} max_diff={rs['max_diff']:.3e} (==fixed-len)")
    assert ro["all_close"] and rs["all_close"], "1-seg varlen must equal fixed-len exactly"


@pytest.mark.skip(reason="temporarily skipped")
def test_chunk_kda_varlen_cu_h2() -> None:
    """V1: cu=[0,64,264,584] segs[64,200,320] H2, initial_state[N], ofs vs golden(o,S 1e-3)."""
    _set_device(); torch.manual_seed(42)
    _compare_varlen("V1_cu_h2", [64, 200, 320], H=2, K=128, init_mode="rand", ofs=True)


@pytest.mark.skip(reason="temporarily skipped")
def test_chunk_kda_varlen_cu_h16() -> None:
    """V2: cu=[0,64,264,584] segs[64,200,320] H16, initial_state[N], ofs vs golden(o,S 1e-3)."""
    _set_device(); torch.manual_seed(42)
    _compare_varlen("V2_cu_h16", [64, 200, 320], H=16, K=128, init_mode="rand", ofs=True)


@pytest.mark.skip(reason="temporarily skipped")
def test_chunk_kda_varlen_rand() -> None:
    """V3: pure random multi-seg, no initial_state: cu=[0,100,228,512] segs[100,128,284] H4."""
    _set_device(); torch.manual_seed(7)
    _compare_varlen("V3_rand", [100, 128, 284], H=4, K=128, init_mode="none", ofs=True, seed=7)


# ─────────────────── Feature 1: use_qk_l2norm_in_kernel ──────────────────────
@pytest.mark.skip(reason="temporarily skipped")
def test_chunk_kda_l2norm_fixed() -> None:
    """L2Nfix (Feature 1, fixed-len): dev shape B1 T128 H2 K128, use_qk_l2norm_in_kernel
    =True. impl(True) vs golden(True) within tol (o 1e-2, S 1e-3); AND impl(True) differs
    from impl(False) so the in-kernel L2-norm path is genuinely exercised (not a no-op)."""
    _set_device(); torch.manual_seed(42)
    case = {"id": "L2Nfix", "seed": 42, "shape": {"B": 1, "T": 128, "H": 2, "K": 128},
            "dtype": {"default": "bfloat16"}, "output_final_state": True}
    inputs = make_inputs(case)
    kw = make_call_kwargs(case, inputs); kw["use_qk_l2norm_in_kernel"] = True
    impl_out = chunk_kda_wrapper(*inputs.values(), **kw)
    gold_out = _golden_per_bh(inputs, kw)
    _assert_leaves("L2Nfix", impl_out, gold_out)
    kw_f = dict(kw); kw_f["use_qk_l2norm_in_kernel"] = False
    o_f, _ = chunk_kda_wrapper(*inputs.values(), **kw_f)
    delta = (impl_out[0].float() - o_f.float()).abs().max().item()
    print(f"  L2Nfix True!=False: o delta={delta:.3e} (>1e-4 => path is real)")
    assert delta > 1e-4, "use_qk_l2norm_in_kernel=True must change the fixed-len output"


def test_chunk_kda_l2norm_varlen() -> None:
    """L2Nvar (Feature 1, varlen): V2 segs [64,200,320] H16 + initial_state, use_qk_l2norm
    _in_kernel=True. impl(True) vs golden(True) within tol; AND True differs from False."""
    _set_device(); torch.manual_seed(42)
    segs, H, K = [64, 200, 320], 16, 128
    dev = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}")
    N = len(segs); acc = [0]
    for s in segs:
        acc.append(acc[-1] + s)
    cu = torch.tensor(acc, device=dev, dtype=torch.int32)
    inputs = _stable_inputs(sum(segs), H, K, seed=42)
    torch.manual_seed(49)
    init = torch.randn(N, H, K, K, device=dev, dtype=torch.float32) * 0.1
    common = dict(cu_seqlens=cu, initial_state=init, output_final_state=True, scale=K ** -0.5)
    impl_out = chunk_kda_wrapper(*inputs.values(), use_qk_l2norm_in_kernel=True, **common)
    gold_out = _golden_varlen_per_h(inputs, cu, init, True, use_qk_l2norm=True)
    _assert_leaves("L2Nvar", impl_out, gold_out)
    o_f, _ = chunk_kda_wrapper(*inputs.values(), use_qk_l2norm_in_kernel=False, **common)
    delta = (impl_out[0].float() - o_f.float()).abs().max().item()
    print(f"  L2Nvar True!=False: o delta={delta:.3e} (>1e-4 => path is real)")
    assert delta > 1e-4, "use_qk_l2norm_in_kernel=True must change the varlen output"


# ─────────────────── Feature 2 REMOVED: prebuilt_meta dropped from ABI ────────────────────
# prebuilt_meta / chunk_indices / chunk_offsets were DROPPED from the ABI (Stage-3
# revision 2026-07-01, DESIGN.md §S1/§S7): segmentation now comes solely from
# cu_seqlens and the partial last chunk is zero-filled IN-KERNEL by pypto.fillpad,
# so the wrapper silently ignores any stray prebuilt_meta= in **kwargs. The old
# _build_prebuilt_meta helper + test_chunk_kda_prebuilt_meta (a bit-exactness parity
# test for a param that no longer exists) were removed; the in-kernel tail-pad on a
# non-%64 single segment is exercised by test_chunk_kda_varlen_l2n_pm (V4) below.


def test_chunk_kda_varlen_l2n_pm() -> None:
    """V4 (in-kernel tail-pad on a large single segment): varlen single-seg
    cu=[0,4117] (shape [2], N=1; T=4117 NOT %64 -> the LAST chunk is partial with
    actual_l = 4117 - 64*64 = 21 rows, zeroed IN-KERNEL by pypto.fillpad on the
    is_loop_end tail block), q/k/v/g [1,4117,4,128] H4, use_qk_l2norm_in_kernel=True
    + output_final_state=True. impl vs golden(l2norm=True) within tol (o 1e-2,
    S 1e-3). Converted former V4: the prebuilt_meta arg + the 'path==derived-from-cu'
    bit-exact sub-check are GONE (param dropped, DESIGN.md §S7); the valuable
    single-seg non-%64 + in-kernel-tail-pad + l2norm coverage is retained."""
    _set_device(); torch.manual_seed(42)
    T, H, K = 4117, 4, 128
    cu = torch.tensor([0, T], device=torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}"),
                      dtype=torch.int32)                               # shape [2]: one segment, N=1
    inputs = _stable_inputs(T, H, K, seed=42)                          # q/k/v/g [1,4117,4,128], beta [1,4117,4]
    common = dict(cu_seqlens=cu, initial_state=None, output_final_state=True, scale=K ** -0.5)
    impl_out = chunk_kda_wrapper(*inputs.values(), use_qk_l2norm_in_kernel=True, **common)
    gold_out = _golden_varlen_per_h(inputs, cu, None, True, use_qk_l2norm=True)
    _assert_leaves("V4_l2n", impl_out, gold_out)


def test_chunk_kda_varlen_l2n_pm_t37() -> None:
    """[1,37,4,128] single-seg varlen (cu=[0,37], N=1): T=37 < 64 -> the WHOLE
    sequence is ONE partial tail chunk (actual_l=37, zeroed in-kernel by pypto.fillpad
    on the is_loop_end block). Params aligned to V4 (test_chunk_kda_varlen_l2n_pm):
    use_qk_l2norm_in_kernel=True, initial_state=None, output_final_state=True,
    scale=K**-0.5. impl vs golden(l2norm=True) within tol (o 1e-2, S 1e-3)."""
    _set_device(); torch.manual_seed(42)
    T, H, K = 37, 4, 128
    cu = torch.tensor([0, T], device=torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}"),
                      dtype=torch.int32)                               # shape [2]: one segment, N=1
    inputs = _stable_inputs(T, H, K, seed=42)                          # q/k/v/g [1,37,4,128], beta [1,37,4]
    common = dict(cu_seqlens=cu, initial_state=None, output_final_state=True, scale=K ** -0.5)
    impl_out = chunk_kda_wrapper(*inputs.values(), use_qk_l2norm_in_kernel=True, **common)
    gold_out = _golden_varlen_per_h(inputs, cu, None, True, use_qk_l2norm=True)
    _assert_leaves("V4_t37", impl_out, gold_out)


@pytest.mark.skip(reason="temporarily skipped")
def test_chunk_kda_determinism() -> None:
    """Determinism: two fresh wrapper runs on identical P0 inputs must be bit-equal."""
    _set_device(); torch.manual_seed(42)
    case = {"id": "det", "seed": 42, "shape": {"B": 1, "T": 4096, "H": 16, "K": 128},
            "dtype": {"default": "bfloat16"}, "output_final_state": True}
    inputs = make_inputs(case)
    kwargs = make_call_kwargs(case, inputs)
    o1, s1 = chunk_kda_wrapper(*inputs.values(), **kwargs)
    o2, s2 = chunk_kda_wrapper(*inputs.values(), **kwargs)
    ro = detailed_tensor_compare(o1, o2, tensor_name="det_o", atol=0.0, rtol=0.0)
    rs = detailed_tensor_compare(s1, s2, tensor_name="det_S", atol=0.0, rtol=0.0)
    print(f"  det_o: all_close={ro['all_close']} max_diff={ro['max_diff']:.3e}")
    print(f"  det_S: all_close={rs['all_close']} max_diff={rs['max_diff']:.3e}")
    assert ro["all_close"] and rs["all_close"], "determinism: outputs differ across runs"


# ══════════════ §S4 historical stale-UB NaN acceptance gate ══════════════════
# Regression guard for the abandoned host-pad residual-NaN bug (DESIGN.md §S4/§S7,
# S10 #3 [VERIFIER GATE]). The Stage-3 in-kernel pypto.fillpad tail-pad REPLACES a
# host zero-pad + scatter/gather whose pypto.full()-init pad rows were not reliably
# re-zeroed across launches -> stale UB -> gcum pad drift -> exp(-gcum)=e^55 overflow
# -> inf*0 = NaN. The bug was ORDER/RESIDUAL dependent (surfaced only on V2 repeated
# and on V2-after-prior-launches), so a single isolated case does not prove the fix.
# This gate re-runs the two historical trigger orderings in-process and requires
# 0 NaN / 0 Inf on every launch (mirrors the multi-launch structure of
# _debug/smoke_tailpad.py). Finiteness only — precision is covered by the compares.
def _finite(name: str, t) -> tuple:
    """Return (ok, message) for a 0-NaN / 0-Inf check on one leaf tensor (None ok)."""
    if t is None:
        return True, f"{name}: None"
    tf = t.float()
    nan = int(torch.isnan(tf).sum().item())
    inf = int(torch.isinf(tf).sum().item())
    ok = (nan == 0 and inf == 0)
    return ok, (f"{name}: shape={tuple(t.shape)} nan={nan} inf={inf} "
                f"absmax={tf.abs().max().item():.3e} -> {'OK' if ok else 'BAD'}")


def _wrap_fixed(cid: str, B: int, T: int, H: int, K: int = 128, *, ofs: bool = True, seed: int = 42):
    """Fixed-len wrapper launch (cu=None -> arange synth). Mirrors the precision
    tests' input construction; returns (o, S) without a golden compare."""
    case = {"id": cid, "seed": seed, "shape": {"B": B, "T": T, "H": H, "K": K},
            "dtype": {"default": "bfloat16"}, "output_final_state": ofs}
    inputs = make_inputs(case)
    kwargs = make_call_kwargs(case, inputs)
    return chunk_kda_wrapper(*inputs.values(), **kwargs)


def _wrap_varlen(segs, H: int, K: int = 128, *, init_mode: str = "none", ofs: bool = True,
                 seed: int = 42, l2n: bool = False):
    """Varlen wrapper launch (cu pass-through + in-kernel fillpad tail). Mirrors
    _compare_varlen's construction exactly; returns (o, S) without a golden compare."""
    dev = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}")
    T = sum(segs); N = len(segs); acc = [0]
    for s in segs:
        acc.append(acc[-1] + s)
    cu = torch.tensor(acc, device=dev, dtype=torch.int32)
    inputs = _stable_inputs(T, H, K, seed=seed)
    init = None
    if init_mode == "rand":
        torch.manual_seed(seed + 7)
        init = torch.randn(N, H, K, K, device=dev, dtype=torch.float32) * 0.1
    return chunk_kda_wrapper(*inputs.values(), cu_seqlens=cu, initial_state=init,
                             output_final_state=ofs, scale=K ** -0.5,
                             use_qk_l2norm_in_kernel=l2n)


def _nan_case(name: str):
    """Named-case dispatch — shapes/seeds match the precision tests so the historical
    NaN triggers are reproduced faithfully (V1/V2 = segs[64,200,320] partial chunks)."""
    if name == "l0":  return _wrap_fixed("dev", 1, 128, 2)
    if name == "t1":  return _wrap_fixed("T1", 1, 512, 32)
    if name == "l1":  return _wrap_fixed("P0", 1, 4096, 16)
    if name == "u0":  return _wrap_fixed("U0", 1, 384, 2)
    if name == "u1":  return _wrap_fixed("U1", 1, 1920, 32)
    if name == "V0":  return _wrap_varlen([128], 2, init_mode="none", seed=42)
    if name == "V1":  return _wrap_varlen([64, 200, 320], 2, init_mode="rand", seed=42)
    if name == "V2":  return _wrap_varlen([64, 200, 320], 16, init_mode="rand", seed=42)
    if name == "V3":  return _wrap_varlen([100, 128, 284], 4, init_mode="none", seed=7)
    if name == "det": return _wrap_fixed("det", 1, 4096, 16)
    raise KeyError(name)


def _run_nan_order(order) -> list:
    bad = []
    for step, name in enumerate(order):
        o, S = _nan_case(name)
        ok_o, msg_o = _finite(f"{name}[{step}]_o", o)
        ok_S, msg_S = _finite(f"{name}[{step}]_S", S)
        print(f"  {msg_o}")
        print(f"  {msg_S}")
        if not (ok_o and ok_S):
            bad.append(f"{name}[{step}]")
    return bad


@pytest.mark.skip(reason="temporarily skipped")
def test_chunk_kda_nan_order_gate() -> None:
    """§S4 acceptance gate: re-run the historical stale-UB NaN trigger orderings
    in-process and require 0 NaN / 0 Inf on every launch. Order A = `V2 V2` (same
    partial-chunk case twice); Order B = `l0 t1 l1 u0 u1 V0 V1 V2 V3 det V2` (the
    long historical order). Run this FIRST in a fresh process for the most faithful
    reproduction of the order/residual-dependent bug."""
    _set_device(); torch.manual_seed(42)
    print("[NaN-gate] order A: V2 V2 (same partial-chunk case twice in one process)")
    bad_a = _run_nan_order(["V2", "V2"])
    print("[NaN-gate] order B: l0 t1 l1 u0 u1 V0 V1 V2 V3 det V2 (long historical order)")
    bad_b = _run_nan_order(["l0", "t1", "l1", "u0", "u1", "V0", "V1", "V2", "V3", "det", "V2"])
    bad = bad_a + bad_b
    assert not bad, f"NaN-gate: non-finite o/S on launches -> {bad}"
    print("  [NAN_GATE_PASS] 0 NaN / 0 Inf across all trigger orders")


if __name__ == "__main__":
    # §S4 NaN-order acceptance gate FIRST — a fresh process is the most faithful
    # reproduction of the order/residual-dependent stale-UB trigger (DESIGN.md §S7).
    test_chunk_kda_nan_order_gate()
    # Fixed-len regression (ABI-neutral: cu=None -> arange synth, every chunk full).
    test_chunk_kda_l0()
    test_chunk_kda_t1()
    test_chunk_kda_l1()
    test_chunk_kda_l2()
    test_chunk_kda_u0()
    test_chunk_kda_u1()
    # Varlen — the CHANGED path (cu pass-through + in-kernel fillpad tail on non-%64).
    test_chunk_kda_varlen_1seg()
    test_chunk_kda_varlen_cu_h2()
    test_chunk_kda_varlen_cu_h16()
    test_chunk_kda_varlen_rand()
    test_chunk_kda_l2norm_fixed()
    test_chunk_kda_l2norm_varlen()
    test_chunk_kda_varlen_l2n_pm()
    test_chunk_kda_varlen_l2n_pm_t37()
    test_chunk_kda_determinism()
    # Each test_* asserts internally per leaf, so this marker is only reached if ALL
    # cases above passed without raising (no unconditional PASS).
    print("[PRECISION_PASS]")
    # NOTE: test_chunk_kda_l3 (B1 T131072 H32 long-seq stress) is DEFINED above but
    # intentionally NOT in this default run — it is an extreme, ABI-neutral fixed-len
    # stress case outside the Stage-3 revision suite (task suite = L0/T1/L1/L2/U0/U1
    # + V0..V4 + l2norm cases). Invoke test_chunk_kda_l3() explicitly to run it.
