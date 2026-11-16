# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression tests for the source-tree ops.pypto_pro.kimi_k3.flash_kda.op_kernel API (Ascend 950).

Run from the repository root after configuring CANN and PyPTO-Pro. Examples:
  timeout --kill-after=3s 280s python -m pytest -q -s \
    ops/pypto_pro/kimi_k3/flash_kda/tests/test_flash_kda.py::test_wrappers_threeway
  timeout --kill-after=3s 280s python -m pytest -q -s \
    'ops/pypto_pro/kimi_k3/flash_kda/tests/test_flash_kda.py::test_autograd[dense]'
  timeout --kill-after=3s 280s python -m pytest -q -s \
    'ops/pypto_pro/kimi_k3/flash_kda/tests/test_flash_kda.py::test_cp_summary[cp2]'

Use --collect-only to list all parameter IDs; run serially on one NPU, with a
280-second bound per invocation. TILE_FWK_DEVICE_ID selects the device. Build
artifacts go to pytest's temporary directory. All per-output metrics are printed
and attached to JUnit XML through record_property.

The frozen relative RMS/L2 threshold is 0.008. CPU FP32 is the acceptance Golden;
BF16-rounded-input, FP32-matmul CPU results are a third-party diagnostic reference.
CP2/CP4 tests emulate AllGather and SUM on one NPU, not HCCL communication. These
small regression cases do not replace long-sequence precision/performance tests.
The packed autograd case uses lower_bound=-5; other autograd cases use -0.1.
"""

import gc
import json
import os
from pathlib import Path
import sys

import pytest
import torch
import torch_npu  # noqa: F401: registers torch.npu

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "assets"))

from kda_golden_cpu import activation_reference, golden, stable
from kda_precision import metric
from ops.pypto_pro.kimi_k3.flash_kda.op_kernel import (
    chunk_kda,
    flash_kda_stage1_wrapper as f1,
    flash_kda_stage2_wrapper as f2,
    flash_kda_bwd_stage1_wrapper as b1,
    flash_kda_bwd_stage2_wrapper as b2,
)
from ops.pypto_pro.kimi_k3.flash_kda.op_kernel.kda_cp import all_gather_summary, sum_parameter_gradients

pytestmark = [pytest.mark.soc("950"), pytest.mark.world_size(1), pytest.mark.jit_only]


@pytest.fixture(autouse=True)
def npu_workspace(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    monkeypatch.chdir(tmp_path)
    yield
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()
    torch.set_num_threads(previous_threads)


def record_metrics(record_property, report):
    text = json.dumps(report, sort_keys=True)
    print(text, flush=True)
    record_property("precision_metrics", text)


def test_wrappers_threeway(record_property):
    torch.manual_seed(292)
    batch_size, seq_len, num_heads = 1, 128, 2
    scale = 128**-0.5
    lower = -0.1

    def rand(shape, dtype=torch.bfloat16):
        return (torch.randn(shape) * 0.2).to(dtype)

    raw = [rand((batch_size, seq_len, num_heads, 128)) for _ in range(4)]
    beta = rand((batch_size, seq_len, num_heads))
    a = rand((num_heads,), torch.float32)
    bias = rand((num_heads, 128), torch.float32)
    state = rand((batch_size, num_heads, 128, 128), torch.float32) * 0.1
    seed = rand((batch_size, num_heads, 128, 128), torch.float32) * 0.05
    do = rand((batch_size, seq_len, num_heads, 128))
    cp = dict(cp_rank=0, cp_size=1)
    dev = [x.npu() for x in (*raw, beta)]
    p, packet = f1(*dev, scale, a.npu(), bias.npu(), lower, state.npu(), cp_meta=cp, num_cores=2)
    out, final, cache = f2(p, packet, cp_meta=cp, num_cores=2)
    summary, params = b1(cache, do.npu(), scale, seed.npu(), cp_meta=cp, num_cores=2)
    actual = b2(
        dev[0],
        dev[1],
        dev[3],
        a.npu(),
        bias.npu(),
        cache,
        do.npu(),
        summary,
        params,
        scale,
        lower,
        seed.npu(),
        cp_meta=cp,
        num_cores=2,
    )
    torch.npu.synchronize()
    print("device_ready", flush=True)
    s = {k: v.cpu() for k, v in cache.items() if isinstance(v, torch.Tensor)}

    def recurrence(fp32):
        def mm(x, y):
            return (
                torch.matmul(x.float(), y.float())
                if fp32
                else torch.matmul(x.bfloat16().float(), y.bfloat16().float())
            )

        q, k, g = s["q_c"].float(), s["k_c"].float(), s["gk_c"].float()
        nc = q.shape[1]
        dh = torch.empty(batch_size, nc + 1, num_heads, 128, 128)
        du = torch.empty(batch_size, nc, num_heads, 64, 128)
        dh[:, -1] = seed
        for c in reversed(range(nc)):
            dy = do[:, c * 64: (c + 1) * 64].transpose(1, 2).float()
            w = s["w"][:, c * 64: (c + 1) * 64].transpose(1, 2).float()
            kg = k[:, c] * (g[:, c, :, -1:] - g[:, c]).exp()
            u = mm(s["mqk"][:, :, c].transpose(-1, -2), dy) + mm(kg, dh[:, c + 1].transpose(-1, -2))
            du[:, c] = u
            dh[:, c] = (
                dh[:, c + 1] * g[:, c, :, -1].exp().unsqueeze(-2)
                + mm(dy.transpose(-1, -2), q[:, c] * g[:, c].exp()) * scale
                + mm(u.transpose(-1, -2), w)
            )
        return du, dh

    refs = []
    for fp32 in (True, False):
        du, dh = recurrence(fp32)
        core = stable(s, do, du, dh, scale, fp32=fp32)
        if not fp32:
            core = tuple(x.bfloat16().float() for x in core)
        result = activation_reference(core, (raw[0], raw[1], raw[3], beta), a, bias, lower, seq_len)
        if not fp32:
            result = tuple(x.bfloat16().float() for x in result[:5]) + result[5:]
        refs.append((*result, dh[:, 0]))
    got = tuple(x.cpu().float() for x in actual[:5]) + (
        params[:num_heads].cpu(),
        params[num_heads:].view(num_heads, 128).cpu(),
        actual[5].cpu(),
    )
    report = {}
    for name, x, g, t in zip(("dq", "dk", "dv", "dbeta", "dg", "dA", "dbias", "dh0"), got, *refs):
        e = (x - g).double()
        te = (t - g).double()
        ratio = float(e.norm() / (g.double().norm() + 1e-8))
        report[name] = dict(
            relative_l2=ratio,
            max_abs=float(e.abs().max()),
            npu_squared_error=float(e.square().sum()),
            third_squared_error=float(te.square().sum()),
            error_ratio_vs_third=float(e.square().sum() / (te.square().sum() + 1e-30)),
            finite=bool(torch.isfinite(x).all()),
            pass_=bool(torch.isfinite(x).all()) and ratio < 0.008,
        )
    record_metrics(record_property, report)
    assert all(x["pass_"] for x in report.values()), report
    print("THREEWAY_PASS", flush=True)
    assert out.dtype == torch.bfloat16 and final.dtype == torch.float32
    assert all(x.dtype == torch.bfloat16 for x in actual[:5])
    assert actual[5].dtype == params.dtype == torch.float32
    assert cache["meta"]["cache_version"] == "flash_kda_cp_train_v3_bf16_state"
    assert cache["meta"]["state_v_first"] is True
    for key in ("q_c", "k_c", "v_c", "akk", "h", "v_new"):
        assert cache[key].dtype == torch.bfloat16, key
    for key in ("gk_c", "beta_c"):
        assert cache[key].dtype == torch.float32, key
    assert cache["h"].shape == (batch_size, seq_len // 64 + 1, num_heads, 128, 128)
    assert cache["v_new"].shape == (batch_size, seq_len // 64, num_heads, 128, 64)

    # CP1 must preserve the packet object: private stage1 caches travel with it.
    assert all_gather_summary(summary, cp, None, "hccl") is summary
    assert sum_parameter_gradients(params, cp, None, "hccl", "internal") is params

    bad = dict(cache, meta=dict(cache["meta"], cache_version="old_cache"))
    with pytest.raises(ValueError, match="flash_kda_cp_train_v3_bf16_state"):
        b1(bad, do.npu(), scale, seed.npu(), cp_meta=cp, num_cores=2)
    bad = dict(cache, h=cache["h"].float())
    with pytest.raises(TypeError, match="h must be"):
        b1(bad, do.npu(), scale, seed.npu(), cp_meta=cp, num_cores=2)

    # Inference shares the forward result but does not expose a training cache.
    ip, packet = f1(
        *dev,
        scale,
        a.npu(),
        bias.npu(),
        lower,
        state.npu(),
        cp_meta=cp,
        is_training=False,
        num_cores=2
    )
    io, ih, ic = f2(ip, packet, cp_meta=cp, num_cores=2)
    assert ic is None
    assert metric(out.cpu(), io.cpu())["pass"]
    assert metric(final.cpu(), ih.cpu())["pass"]


@pytest.mark.parametrize(
    "case",
    [
        "dense",
        "bnsd",
        "tail",
        "single",
        "gva",
        "gva_tail",
        "bucket",
        "bucket_bnsd",
        "bucket_packed",
        "recompute",
        "defaults",
        "packed",
        "packed_gva",
        "final_only",
    ],
)
def test_autograd(case, record_property):
    cases = {
        "dense": (2, 128, 2, 2, "BSND", None),
        "bnsd": (2, 128, 2, 2, "BNSD", None),
        "tail": (2, 65, 2, 2, "BSND", None),
        "single": (2, 1, 2, 2, "BSND", None),
        "gva": (1, 128, 1, 4, "BSND", None),
        "gva_tail": (1, 65, 1, 4, "BSND", None),
        "bucket": (2, 65, 3, 12, "BSND", None),
        "bucket_bnsd": (2, 65, 3, 12, "BNSD", None),
        "bucket_packed": (1, 193, 3, 12, "TND", [0, 65, 192, 193]),
        "recompute": (1, 128, 2, 4, "BSND", None),
        "defaults": (1, 65, 2, 2, "BSND", None),
        "packed": (1, 193, 2, 2, "TND", [0, 65, 192, 193]),
        "packed_gva": (1, 193, 1, 4, "TND", [0, 65, 192, 193]),
        "final_only": (1, 65, 2, 2, "BSND", None),
    }
    batch_size, seq_len, num_key_heads, num_value_heads, layout, bounds = cases[case]
    lower_bound = -5.0 if case == "packed" else -0.1
    record_property("lower_bound", lower_bound)
    torch.manual_seed(42)

    def rand(shape, dtype=torch.bfloat16, scale=0.2):
        return (torch.randn(shape) * scale).to(dtype)

    num_sequences = len(bounds) - 1 if bounds else batch_size
    data = [
        rand((batch_size, seq_len, num_key_heads, 128)),
        rand((batch_size, seq_len, num_key_heads, 128)),
        rand((batch_size, seq_len, num_value_heads, 128)),
        rand((batch_size, seq_len, num_value_heads, 128)),
        rand((batch_size, seq_len, num_value_heads)),
        rand((num_value_heads,), torch.float32),
        rand((num_value_heads, 128), torch.float32),
        rand((num_sequences, num_value_heads, 128, 128), torch.float32, scale=0.02),
    ]
    if case == "defaults":
        data[-1].zero_()
    cpu = [x.float().detach().clone().requires_grad_() for x in data]
    device = []
    for i, x in enumerate(data):
        if i < 5:
            if layout == "BNSD":
                x = x.transpose(1, 2).contiguous()
            elif layout == "TND":
                x = x[0]
        device.append(x.npu().detach().requires_grad_())
    cfg = dict(
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
        scale=128**-0.5,
        lower_bound=lower_bound,
    )
    out, final = chunk_kda(
        *device[:5],
        initial_state=None if case == "defaults" else device[7],
        A_log=device[5],
        dt_bias=device[6],
        output_final_state=case != "defaults",
        lower_bound=lower_bound,
        layout_qkv=layout,
        head_bucket_size=1 if case.startswith("bucket") or case == "recompute" else None,
        disable_recompute=case != "recompute",
        cu_seqlens=None if bounds is None else torch.tensor(bounds, dtype=torch.int64),
        num_cores=2
    )
    torch.npu.synchronize()
    print("forward_ready", case, flush=True)
    do = rand((batch_size, seq_len, num_value_heads, 128))
    dht = rand((num_sequences, num_value_heads, 128, 128), torch.float32, 0.01)
    if case == "final_only":
        torch.autograd.backward(final, dht.npu())
    elif case == "defaults":
        torch.autograd.backward(out, do.npu())
    else:
        do_dev = (
            do.transpose(1, 2).contiguous()
            if layout == "BNSD"
            else do[0] if layout == "TND" else do
        )
        torch.autograd.backward((out, final), (do_dev.npu(), dht.npu()))
    torch.npu.synchronize()
    print("backward_ready", case, flush=True)
    ro, rh = golden(*cpu, cfg, bounds)
    if case == "final_only":
        torch.autograd.backward(rh, dht)
    elif case == "defaults":
        torch.autograd.backward(ro, do.float())
    else:
        torch.autograd.backward((ro, rh), (do.float(), dht))
    actual = [out.detach().cpu(), None if final is None else final.detach().cpu()] + [
        None if x.grad is None else x.grad.cpu() for x in device
    ]
    if layout == "BNSD":
        for i in (0, 2, 3, 4, 5, 6):
            actual[i] = actual[i].transpose(1, 2)
    elif layout == "TND":
        for i in (0, 2, 3, 4, 5, 6):
            actual[i] = actual[i].unsqueeze(0)
    ref = [ro.detach(), rh.detach()] + [x.grad for x in cpu]
    names = ["out", "final", "dq", "dk", "dv", "dg", "dbeta", "dA", "dbias", "dh0"]
    report = {
        n: metric(r, a)
        for n, r, a in zip(names, ref, actual)
        if not (case == "defaults" and n in ("final", "dh0"))
    }
    record_metrics(record_property, report)
    failed = {name: value for name, value in report.items() if not value["pass"]}
    assert not failed, failed
    print("CPU_FP32_AUTOGRAD_PASS", case, flush=True)


@pytest.mark.parametrize("cp_size,lower", [(2, -0.1), (4, 0.0)], ids=["cp2", "cp4_zero_gate"])
def test_cp_summary(cp_size, lower, record_property):
    torch.manual_seed(791)
    batch_size = 1
    seq_len = 128
    num_key_heads = 1
    num_value_heads = 4
    scale = 128**-0.5

    def rand(shape, dtype=torch.bfloat16, factor=0.2):
        return (torch.randn(shape) * factor).to(dtype)

    a = rand((num_value_heads,), torch.float32).requires_grad_()
    bias = rand((num_value_heads, 128), torch.float32).requires_grad_()
    initial = rand((batch_size, num_value_heads, 128, 128), torch.float32, 0.02).requires_grad_()
    ad, bd = a.detach().npu(), bias.detach().npu()
    xs = []
    dev = []
    prepared = []
    packets = []
    meta = []
    dos = []
    seeds = []
    for rank in range(cp_size):
        raw = [
            rand((batch_size, seq_len, num_key_heads, 128)),
            rand((batch_size, seq_len, num_key_heads, 128)),
            rand((batch_size, seq_len, num_value_heads, 128)),
            rand((batch_size, seq_len, num_value_heads, 128)),
            rand((batch_size, seq_len, num_value_heads)),
        ]
        xs.append([x.float().requires_grad_() for x in raw])
        dev.append([x.npu() for x in raw])
        dos.append(rand((batch_size, seq_len, num_value_heads, 128)))
        seeds.append(rand((batch_size, num_value_heads, 128, 128), torch.float32, 0.01))
        meta.append(
            dict(cp_rank=rank, cp_size=cp_size, first_continues=rank > 0, last_continues=rank + 1 < cp_size)
        )
        p, packet = f1(
            *dev[-1],
            scale,
            ad,
            bd,
            lower,
            initial.detach().npu() if rank == 0 else None,
            cp_meta=meta[-1],
            num_cores=2
        )
        prepared.append(p)
        packets.append(packet)
    all_forward = torch.cat(packets, 0)  # communication emulation in test only
    outs = []
    finals = []
    caches = []
    bp = []
    params = []
    for rank in range(cp_size):
        o, ht, c = f2(prepared[rank], all_forward, cp_meta=meta[rank], num_cores=2)
        outs.append(o)
        finals.append(ht)
        caches.append(c)
        packet, param = b1(
            c, dos[rank].npu(), scale, seeds[rank].npu(), cp_meta=meta[rank], num_cores=2
        )
        bp.append(packet)
        params.append(param)
    torch.npu.synchronize()
    print("all_summaries_ready", cp_size, flush=True)
    all_backward = torch.cat(bp, 0)
    grads = []
    for rank in range(cp_size):
        q, k, v, g, beta = dev[rank]
        grads.append(
            b2(
                q,
                k,
                g,
                ad,
                bd,
                caches[rank],
                dos[rank].npu(),
                all_backward,
                params[rank],
                scale,
                lower,
                seeds[rank].npu(),
                cp_meta=meta[rank],
                num_cores=2,
            )
        )
    torch.npu.synchronize()
    print("backward_ready", cp_size, flush=True)
    cfg = dict(
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
        scale=scale,
        lower_bound=lower,
    )
    reference = []
    state = initial
    for rank in range(cp_size):
        o, state = golden(*xs[rank], a, bias, state, cfg, None)
        reference.append((o, state))
    targets = [x for pair in reference for x in pair]
    upstream = [x.float() for pair in zip(dos, seeds) for x in pair]
    torch.autograd.backward(targets, upstream)
    report = {}
    for rank in range(cp_size):
        q, k, v, g, beta = xs[rank]
        ref = (
            *reference[rank],
            q.grad,
            k.grad,
            v.grad,
            beta.grad,
            g.grad,
            initial.grad if rank == 0 else torch.zeros_like(initial),
        )
        got = (outs[rank], finals[rank], *grads[rank])
        report[str(rank)] = {
            name: metric(x.detach(), y.cpu())
            for name, x, y in zip(
                ("out", "final", "dq", "dk", "dv", "dbeta", "dg", "dh0"), ref, got
            )
        }
    total = sum(x.cpu() for x in params)
    report["parameters"] = {
        "dA": metric(a.grad, total[:num_value_heads]),
        "dbias": metric(bias.grad, total[num_value_heads:].view(num_value_heads, 128)),
    }
    record_metrics(record_property, report)
    assert all(x["pass"] for r in report.values() for x in r.values()), report
    print("CPU_FP32_CP_INTEGRATION_PASS", cp_size, flush=True)
