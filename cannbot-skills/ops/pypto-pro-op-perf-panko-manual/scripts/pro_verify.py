# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Acquire and commit the next verification arm using the native ABBA state machine."""

import hashlib
import json
import math
import os
import signal
import statistics
from argparse import Namespace
from pathlib import Path
from panko import delivery, state
from pro_persistence import atomic_write
from panko.config import call_transition
from panko.config import VERIFY_ALPHA, VERIFY_BOOT, Refusal, out
from panko.measurement import bootstrap_speedup as native_interval, median_of as median
from evaluation.measure_latency import measure
from source_analysis.runtime_files import check_files, file_digest
from source_analysis import program_bundle
from measure import _profile_files, bind_timing_target, campaign_contract

READING_SIGMA_LOG = 0.02 / math.sqrt(2)


def bootstrap_speedup(pre, dele, n_boot, alpha):
    result = native_interval(pre, dele, n_boot=n_boot, alpha=alpha)
    if result is not None:
        result.update(unit="reading", n_pre=len(pre), n_dele=len(dele))
    return result


def matches(directory, filename, wanted):
    import pro_backend

    try:
        if pro_backend.ACTIVE_PROGRAM:
            program_bundle.require_current(pro_backend.ACTIVE_PROGRAM)
        else:
            program_bundle.runtime_spec(filename)
        return file_digest(filename) == wanted
    except (OSError, KeyError, ValueError):
        return False


def policy():
    return delivery.VerificationPolicy(
        warmup=2, readings=lambda arm: arm.get("reads") or arm["samples"],
        interval=reading_interval, matches=matches,
        strict_restore=True, acquisition_reads=True,
    )


def transition(args):
    try:
        return call_transition(delivery.cmd_verify, args)
    except Refusal as exc:
        out(exc.reply)
        raise


def auto(args):
    try:
        reply = call_transition(run_auto, args)
    except Refusal as exc:
        out(exc.reply)
        raise
    out(reply)


def _measure_arm(st, prompt, root):
    """Acquire one staged arm and bind its receipt to the verified runtime and profile files."""
    filename = prompt["op_file"]
    c = campaign_contract(st)
    device = str(c["device"])
    if c["env"].get("TILE_FWK_DEVICE_ID", os.environ.get("TILE_FWK_DEVICE_ID")) != device:
        raise ValueError("activate the frozen campaign device before verification")

    def check_inputs():
        if file_digest(filename) != prompt["code_hash"]:
            raise ValueError("staged program identity changed")
        check_files(c["test_files"], "frozen test input changed")
        return program_bundle.runtime_spec(filename)

    program = check_inputs()
    metrics = measure(
        c["argv"],
        warmup=c["warmup"],
        runs=prompt["runs"],
        op_dir=c["cwd"],
        env=c["env"],
        op_name=c["op_name"],
        impl_path=filename,
        budget=c["memory_capacity"],
        address_limits=c.get("address_limits"),
        timeout_s=st["config"]["eval_timeout_s"],
        keep_prof=True,
        prof_root=str(root),
        joint_correctness=True,
    )
    if check_inputs() != program:
        raise ValueError("runtime program changed during verification acquisition")
    samples = metrics.get("samples") or []
    correct = metrics.get("correctness_passed") is True and bool(metrics.get("impl_modules_seen"))
    latency = not metrics.get("p_error") and metrics.get("p") is not None
    valid = len(samples) == prompt["runs"] and all(math.isfinite(x) and x > 0 for x in samples)
    if not (correct and latency and valid):
        out(dict(error="verification_refused", reason=metrics.get("p_error") or "incomplete acquisition"))
        raise Refusal(3)
    bind_timing_target(c, metrics)
    metrics.update(schema="panko-pro-verification-v1", step=prompt["stage"], arm=prompt["arm"],
                   code_hash=prompt["code_hash"], point=st["verification"]["arms"]["delivered"]["code_hash"],
                   contract=dict(c, runs=prompt["runs"]), profile_files=_profile_files(metrics),
                   program_files=program["files"] if program else {str(Path(filename).resolve()): prompt["code_hash"]})
    return metrics


def _receipt_digest(receipt):
    return hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()


def _resume_acquisition(op_dir, st):
    v = st.get("verification") or {}
    if v.get("status") == "pending" and len(v.get("receipts", [])) != v["step"]:
        # Older manual submissions advanced the cursor without keeping their acquisitions.
        st.setdefault("superseded_verifications", []).append(v)
        arms = {name: dict(code_hash=arm["code_hash"], samples=[], reads=[]) for name, arm in v["arms"].items()}
        st["verification"] = dict(v, step=0, arms=arms, receipts=[], pending_receipt_sha256=None)
        state.save(op_dir, st)


def acquire_stage(args):
    """Measure the pending arm now and leave its submission to the caller."""
    st = state.load(args.op_dir)
    _resume_acquisition(args.op_dir, st)
    prompt = transition(args)
    if "stage" not in prompt:
        out(prompt)
        return
    root = Path(args.op_dir).resolve() / "optimization/verification"
    root.mkdir(exist_ok=True)
    reading = _measure_arm(st, prompt, root)
    v = st["verification"]
    path = Path(reading["msprof_dir"]) / "verification-receipt.json"
    atomic_write(path, json.dumps(reading))
    v["pending_receipt_sha256"] = _receipt_digest(reading)
    state.save(args.op_dir, st)
    out(dict(prompt, measurement_file=str(path)))


def check_pending_receipt(st, receipt):
    v = st.get("verification") or {}
    if not isinstance(receipt, dict) or v.get("status") != "pending":
        raise ValueError("a pending verification stage and its acquisition receipt are required")
    step = v["step"]
    arm = v["schedule"][step]
    expected = dict(step=step, arm=arm, code_hash=v["arms"][arm]["code_hash"],
                    point=v["arms"]["delivered"]["code_hash"])
    matches_stage = all(receipt.get(key) == value for key, value in expected.items())
    if not matches_stage or _receipt_digest(receipt) != v.get("pending_receipt_sha256"):
        raise ValueError("expected a fresh receipt from verify --acquire for this stage")
    return v["runs_per_step"]


def run_auto(args):
    st = state.load(args.op_dir)
    if getattr(args, "samples", None) is not None:
        raise ValueError("automatic verification acquires its own samples")
    if not st["progress"].get("stop_reason") or not st.get("verification") or getattr(args, "skip", None):
        delivery.cmd_verify(args)
        return
    _resume_acquisition(args.op_dir, st)
    filename = state.op_file_path(st, args.op_dir)
    best = delivery.delivered_hash(st)
    baseline = st["nodes"]["closed"][0]["code_hash"]
    v = st["verification"]
    if v["status"] == "no_change" and delivery.front_to_verify(st, baseline, baseline):
        del st["verification"]
        delivery.open_verification(st)
        state.save(args.op_dir, st)
    elif v["status"] in ("confirmed", "regressed", "unconfirmed", "insufficient"):
        seen = delivery.verdicts_by_point(st)
        missing = [h for h in delivery.front_to_verify(st, baseline, best) if h not in seen]
        if missing:
            st["verify_queue"] = missing
            delivery.next_verification(st, args.op_dir)
            state.save(args.op_dir, st)
    root = Path(args.op_dir) / "optimization/verification"
    root.mkdir(exist_ok=True)

    def terminate(signum, frame):
        raise Refusal(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        while True:
            prompt = transition(args)
            if "stage" not in prompt:
                out(dict(prompt, points=delivery.verdicts_by_point(state.load(args.op_dir))))
                return
            metrics = _measure_arm(state.load(args.op_dir), prompt, root)
            transition(Namespace(op_dir=args.op_dir, samples=",".join(map(str, metrics["samples"])), receipt=metrics))
    finally:
        try:
            delivery.restore_verification_program(args, filename, "delivered", best, policy=policy())
        finally:
            signal.signal(signal.SIGTERM, previous)


def assumed_speedup(pre, dele, alpha=VERIFY_ALPHA, sigma=READING_SIGMA_LOG):
    """Estimate the log-ratio interval from per-reading sigma and both sample counts."""
    if not pre or not dele:
        return None
    ratio = median(pre) / median(dele)
    se = sigma * math.sqrt(1.0 / len(pre) + 1.0 / len(dele))
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return {
        "speedup": round(ratio, 4),
        "ci_low": round(ratio * math.exp(-z * se), 4),
        "ci_high": round(ratio * math.exp(z * se), 4),
        "level": 1 - alpha,
        "unit": "reading",
        "basis": "assumed-sigma",
        "sigma_log": sigma,
        "n_pre": len(pre),
        "n_dele": len(dele),
    }


def reading_interval(pre, dele, alpha=VERIFY_ALPHA, n_boot=VERIFY_BOOT):
    """Use the wider endpoints of the empirical and assumed-noise intervals and report their basis."""
    emp = bootstrap_speedup(pre, dele, n_boot=n_boot, alpha=alpha)
    asm = assumed_speedup(pre, dele, alpha=alpha)
    if emp is None or asm is None:
        return emp or asm
    interval = dict(
        emp,
        sigma_log=asm["sigma_log"],
        ci_low=min(emp["ci_low"], asm["ci_low"]),
        ci_high=max(emp["ci_high"], asm["ci_high"]),
    )
    interval["basis"] = (
        "bootstrap"
        if (interval["ci_low"], interval["ci_high"]) == (emp["ci_low"], emp["ci_high"])
        else "bootstrap+floor"
    )
    return interval
