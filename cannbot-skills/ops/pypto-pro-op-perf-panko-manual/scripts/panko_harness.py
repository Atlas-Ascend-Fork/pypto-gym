#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Base Pro entrypoint; the canonical PyPTO core owns search and delivery."""

import fcntl
import json
import math
import statistics
import sys
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path

from bootstrap import activate

activate()
from panko.backend import register
import pro_backend
from bayesian_optimization import normalize

register(pro_backend)
from panko.config import out
from panko.state import load, file_hash
from source_analysis.runtime_files import check_files
from source_analysis import program_bundle
from evaluation.measure_latency import LATENCY_DECIMALS
from evaluation.repeat_runner import split_script_command
from evaluation.workspace_probe import profile_workspace, validate_capacity
from measure import bind_timing_target, campaign_contract


def _receipt_values(command, receipt):
    if command == "init":
        values = {"--preopt-p": receipt.get("p"), "--preopt-s": 1, "--preopt-hash": receipt.get("code_hash"),
                  "--preopt-measurement": json.dumps(receipt)}
    elif command == "record":
        values = {"--p": receipt.get("p") or 0, "--s": int(receipt.get("correctness_passed") is True),
                  "--code-hash": receipt.get("code_hash"), "--measurement": json.dumps(receipt)}
    elif command == "verify":
        values = {"--samples": ",".join(str(x) for x in receipt.get("samples", []))}
    else:
        raise ValueError("measurement-file is for init, record or verify")
    return values


def _merge_receipt_values(argv, values):
    fields = ("--samples", "--metrics", "--preopt-samples", "--preopt-metrics", "--measurement", "--preopt-measurement")
    if any(flag in argv for flag in fields):
        raise ValueError("measurement fields must come only from the receipt")
    for flag, value in values.items():
        if flag in argv:
            pos = argv.index(flag) + 1
            if pos >= len(argv) or argv.count(flag) != 1:
                raise ValueError("expected one value for " + flag)
            parsed = float(argv[pos]) if isinstance(value, (int, float)) else argv[pos]
            if parsed != value:
                raise ValueError("argument differs from measurement: " + flag)
            continue
        argv.extend((flag, str(value)))


def _receipt_args(argv):
    argv = list(argv)
    receipt = None
    legacy_init = argv[:1] == ["init"] and "--preopt-measurement-file" in argv
    if legacy_init:
        if "--measurement-file" in argv:
            raise ValueError("choose one baseline receipt")
        argv[argv.index("--preopt-measurement-file")] = "--measurement-file"
    if "--measurement-file" in argv:
        index = argv.index("--measurement-file")
        if index + 1 >= len(argv):
            raise ValueError("--measurement-file needs a JSON receipt")
        receipt = json.loads(Path(argv[index + 1]).read_text())
        if not isinstance(receipt, dict):
            raise ValueError("expected a measurement object")
        if legacy_init and receipt.get("schema") != "panko-pro-npu-v1":
            raise ValueError(
                "baseline receipt lacks source, test and profile hashes; supply the original bound measurement"
            )
        if "_validated_confirmed_us" in receipt:
            raise ValueError("internal measurement field")
        if receipt.get("schema") == "panko-confirmed-winner-v1":
            if argv[0] != "record":
                raise ValueError("a confirmed winner belongs to the record command")
            from pro_joint import confirmed_receipt

            receipt = confirmed_receipt(receipt, _validate_reading)
        del argv[index:index + 2]
        _merge_receipt_values(argv, _receipt_values(argv[0] if argv else "", receipt))
    return argv, receipt


def _check_latency(receipt, passed, runs):
    samples = receipt.get("samples", [])
    valid = len(samples) == runs and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples)
    if not (passed and valid) or not receipt.get("p") or receipt.get("p_error"):
        raise ValueError(f"successful precision and {runs} finite latency samples required")
    median = statistics.median(samples)
    expected = (median, round(median, LATENCY_DECIMALS))
    if "_validated_confirmed_us" in receipt:
        expected = (receipt["_validated_confirmed_us"],)
    if receipt["p"] not in expected:
        raise ValueError("latency does not match the recorded median")


def _check_workspace(receipt, impl, contract, allow_partial):
    capacity = contract.get("memory_capacity", {})
    validate_capacity(capacity)
    workspace = profile_workspace(impl, capacity, address_limits=contract.get("address_limits"))
    fields = ("workspace_b", "workspace_state", "workspace_by_space")
    if (not allow_partial and workspace.get("workspace_state") not in ("measured", "zero")) or any(
        receipt.get(k) != workspace.get(k) for k in fields
    ):
        raise ValueError("on-chip memory reading is missing or differs from the measured source")


def _bind_recorded_timing(contract, receipt):
    profile = _timing_profile(receipt)
    if contract["op_name"] == "*":
        tasks = profile.tasks_per_call
        if receipt.get("tasks_per_call", tasks) != tasks:
            raise ValueError("package timing target differs from the timing CSV")
        return bind_timing_target(contract, {"tasks_per_call": tasks})
    return bind_timing_target(contract, receipt)


def _validate_reading(receipt, impl, require_success, *, timing_only=False, runs=9, allow_partial_workspace=False):
    schema = "panko-pro-verification-v1" if timing_only else "panko-pro-npu-v1"
    if not isinstance(receipt, dict) or receipt.get("schema") != schema:
        raise ValueError("a fresh NPU measurement-file is required")
    if receipt.get("code_hash") != file_hash(impl):
        raise ValueError("measurement does not belong to the working source")
    contract = receipt.get("contract")
    if not isinstance(contract, dict) or not contract.get("test_files"):
        raise ValueError("measurement has no frozen precision-test manifest")
    check_files(contract["test_files"], "frozen precision input changed or missing")
    passed = receipt.get("correctness_passed") is True
    spec = program_bundle.runtime_spec(impl)
    expected_files = spec["files"] if spec else {str(Path(impl).resolve()): file_hash(impl)}
    if receipt.get("program_files") != expected_files:
        raise ValueError("receipt does not cover the complete runtime program")
    if spec and contract.get("op_name") != "*":
        raise ValueError("package measurements must include all AI-Core tasks")
    if require_success or passed:
        _check_latency(receipt, passed, runs)
        _check_workspace(receipt, impl, contract, timing_only or allow_partial_workspace)
        profiles = receipt.get("profile_files")
        if not profiles or not any(name.endswith(".csv") for name in profiles):
            raise ValueError("raw profile evidence is required")
        check_files(profiles, "profile evidence changed or missing")
        contract = _bind_recorded_timing(contract, receipt)
    elif not receipt.get("test_failed") or receipt.get("p") is not None:
        raise ValueError("acquisition failure is not an incorrect candidate; keep the evidence and retry later")
    return contract


def _bind_repair(args, config):
    import pro_init_repair as init_repair

    args.pro_context = normalize.read_context(args.case_context)
    try:
        args.pro_repair = init_repair.bind_retry(args.op_dir, args.op_file, args, config)
        if args.pro_repair:
            init_repair.check_context(args.op_dir, args.op_file, args.pro_context)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        from panko.config import Refusal

        out({"initialized": False, "refused": True, "initialization_blocked": True, "phase": "init_repair",
             "error": "init_repair_blocked", "reason": str(exc)})
        raise Refusal(4) from exc


def _check_manifest(args):
    if args.program_manifest:
        manifest = json.loads(Path(args.program_manifest).read_text())
        program = pro_backend.ACTIVE_PROGRAM
        if not program or any(manifest.get(k) != v for k, v in program.items()):
            raise ValueError("program manifest differs from the staged runtime package")
        if manifest.get("view_sha256") != file_hash(args.op_file) or any(
            manifest.get("sha256", {}).get(rel) != file_hash(Path(program["root"]) / rel) for rel in program["files"]
        ):
            raise ValueError("program manifest hashes differ from the staged sources")


def _check_original_command(args, contract):
    import pro_init_repair as init_repair
    from evaluation.measure_latency import check_campaign_command, test_argv

    command = args.test_command
    if args.pro_repair:
        original = init_repair.load(args.op_dir, args.op_file)["request"]["test_command"]
        if command and test_argv(command, args.op_dir) != test_argv(original, args.op_dir):
            raise ValueError("test command differs from original repair request")
        command = original
    check_campaign_command(command, args.op_dir, contract)


def _before_init(args, receipt):
    if not args.op_file:
        raise ValueError("--op-file is required")
    args.op_file = str(Path(args.op_file).resolve())
    if (Path(args.op_dir) / "optimization" / "search_state.json").exists():
        raise ValueError("campaign already exists; use a new directory, never reset its budget")
    config = json.loads(args.config or "{}")
    _bind_repair(args, config)
    unsupported = {key for key, value in pro_backend.defaults.items() if config.get(key, value) != value}
    if unsupported:
        raise ValueError("base-release settings cannot enable deferred features: " + ", ".join(sorted(unsupported)))
    args.pro_measurement_contract = _validate_reading(receipt, args.op_file, True)
    split_script_command(args.pro_measurement_contract["argv"])
    pro_backend.ACTIVE_PROGRAM = program_bundle.attached(args.op_file)
    capacity = args.pro_measurement_contract["memory_capacity"]
    contract = args.pro_measurement_contract
    _check_manifest(args)
    if args.preopt_s != 1 or args.preopt_p != receipt["p"] or args.preopt_hash != receipt["code_hash"]:
        raise ValueError("baseline arguments differ from measurement")
    if args.op_name and args.op_name != contract["op_name"]:
        raise ValueError("INIT op name differs from the baseline measurement")
    if str(args.device) != contract["device"]:
        raise ValueError("INIT device differs from the baseline measurement")
    if args.pro_repair or args.test_command:
        _check_original_command(args, contract)
    normalize.check_init(args, contract)
    config["objectives"] = ["aicore_e2e",
                            {"name": "workspace", "from": "m.workspace_b", "direction": "min",
                             "reference": sum(capacity.values())}]
    args.config = json.dumps(config)


_EXTRA_READINGS = ("samples", "metrics", "util", "bubble", "aic_util", "aiv_util",
                   "pred_stall", "aic_bubble", "aiv_bubble", "workspace_b", "workspace_state")


def _timing_csv(receipt):
    from evaluation.measure_latency import _op_summary_in

    profile = Path(_op_summary_in(receipt["msprof_dir"])).resolve()
    if str(profile) not in receipt["profile_files"]:
        raise ValueError("timing CSV is missing from the receipt")
    return profile


def _timing_profile(receipt):
    from evaluation.measure_latency import summarise

    profile = _timing_csv(receipt)
    contract = receipt["contract"]
    warmup = contract["warmup"]
    reading = summarise(profile, contract["op_name"], warmup, calls=warmup + contract["runs"])
    if list(reading.samples) != receipt["samples"]:
        raise ValueError("measurement samples differ from the timing CSV")
    return reading


def _remeasurement_profile(receipt, entry):
    profile = _timing_csv(receipt)
    initial = (entry.get("m") or {}).get("msprof_dir")
    original_profile = initial and profile.is_relative_to(Path(initial).resolve())
    if original_profile or str(profile) in entry.get("remeasurement_profiles", ()):
        raise ValueError("measurement profile already counted for this program")
    return str(profile)


def _cached_record(args, state):
    cached = state["eval_cache"].get(args.code_hash)
    return bool(
        cached
        and args.measurement is None
        and all(getattr(args, key) is None for key in _EXTRA_READINGS)
        and args.code_hash == file_hash(state["op_file"])
        and (args.s, args.p) == (cached["s"], cached.get("p") or 0)
    )


def _saved_winner(args, state, submitted=None):
    winner = state.get("pending_block_receipt")
    if winner and winner["parent"] == args.parent and winner["code_hash"] == args.code_hash:
        path = winner["path"]
        if file_hash(path) != winner["sha256"]:
            raise ValueError("saved block receipt changed")
        _, receipt = _receipt_args(["record", "--measurement-file", path])
        if submitted is not None:
            args.evaluation_already_charged = bool(winner.get("charged") and submitted == receipt)
            return submitted
        args.evaluation_already_charged = bool(winner.get("charged"))
        if any(x is not None for x in (args.measurement, args.samples, args.metrics)):
            raise ValueError("record fields differ from saved block measurement")
        args.measurement = json.dumps(receipt)
        return receipt
    return None


def _check_record(args, receipt):
    expected = (receipt["code_hash"], receipt.get("p") or 0, int(receipt.get("correctness_passed") is True))
    fields = ("workspace_b", "workspace_state")
    if (args.code_hash, args.p, args.s) != expected or any(
        getattr(args, k) is not None and getattr(args, k) != receipt.get(k) for k in fields
    ):
        raise ValueError("record arguments differ from measurement")


def _campaign_state(args):
    state = load(args.op_dir)
    if state.get("backend") != "pypto-pro-base-v2":
        raise ValueError("this entrypoint requires a base-release Pro campaign")
    pro_backend.ACTIVE_PROGRAM = state.get("program")
    if pro_backend.ACTIVE_PROGRAM and args.cmd not in ("step", "stop"):
        program_bundle.require_current(pro_backend.ACTIVE_PROGRAM)
    for name, expected in state["measurement_contract"]["test_files"].items():
        if file_hash(name) != expected:
            raise ValueError("frozen precision input changed: " + name)
    return state


def _before_verify(args, state, receipt):
    from pro_verify import acquire_stage, auto, check_pending_receipt

    if args.acquire:
        if args.auto or args.skip or args.samples is not None:
            raise ValueError("use --acquire on its own to measure one verification stage")
        args.fn = acquire_stage
    elif args.samples is not None:
        if args.skip or args.auto:
            raise ValueError("submit a stage receipt separately from --skip or --auto")
        runs = check_pending_receipt(state, receipt)
        contract = _validate_reading(receipt, state["op_file"], True, timing_only=True, runs=runs)
        if contract != dict(campaign_contract(state), runs=runs):
            raise ValueError("verification acquisition differs from the frozen contract")
        args.receipt = receipt
        args.samples = ",".join(map(str, receipt["samples"]))
    elif args.auto:
        args.fn = auto


def _before_record(args, state, receipt):
    remeasure = args.remeasure
    if remeasure:
        extra = args.unevaluated or any(getattr(args, key) is not None for key in _EXTRA_READINGS)
        if not receipt or extra or "_validated_confirmed_us" in receipt:
            raise ValueError("remeasure requires a fresh acquisition and only its recorded readings")
        # The core pools samples and metrics from the validated acquisition.
        args.measurement = json.dumps(receipt)
    if args.unevaluated:
        return
    if receipt is None:
        if _cached_record(args, state):
            return
        receipt = _saved_winner(args, state)
    elif not remeasure:
        _saved_winner(args, state, receipt)
    if receipt is not None:
        _check_record(args, receipt)
    contract = _validate_reading(receipt, state["op_file"], remeasure)
    expected = campaign_contract(state)
    if receipt.get("p") is None:
        expected = {k: v for k, v in expected.items() if k != "timed_target"}
    if contract.get("timed_target") != expected.get("timed_target"):
        raise ValueError("timing target differs from the baseline")
    if contract != expected:
        raise ValueError("measurement command, device, profiler settings or precision inputs changed")
    if remeasure:
        entry = state["eval_cache"].get(args.code_hash, {})
        args.remeasurement_profile = _remeasurement_profile(receipt, entry)


def _before(args, receipt, *, nested=False):
    if args.cmd == "init-repair":
        return
    if args.cmd == "normalize":
        args.fn = normalize.command
        return
    if getattr(args, "config_key", None):
        raise ValueError("a config_key is not supported by this campaign")
    if args.cmd == "init":
        _before_init(args, receipt)
        return
    state = _campaign_state(args)
    if (not nested and state.get("active_tile_block") and
            args.cmd not in ("block", "step", "probe", "pareto", "report")):
        raise ValueError("resume the active tile block before changing the campaign")
    if args.cmd == "verify":
        _before_verify(args, state, receipt)
        return
    if args.cmd in ("block", "step"):
        args.validate_reading = _validate_reading
    if args.cmd == "block":
        from pro_joint import direct

        args.fn = direct
    if args.cmd == "step":
        from pro_step import step

        args.fn = partial(step, _before=_before, _receipt_args=_receipt_args, configure=_configure)
    if args.cmd == "record":
        _before_record(args, state, receipt)


@contextmanager
def _lock(op_dir):
    folder = Path(op_dir) / "optimization"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "panko.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _configure(sub):
    step = sub.add_parser("step")
    for name in ("op-dir", "op-file", "parent", "test-command", "dispatch-id"):
        step.add_argument("--" + name, required=True)
    sub.choices["select"].add_argument("--dispatch-id")
    sub.choices["verify"].add_argument("--auto", action="store_true")
    normalize.arguments(sub)
    sub.choices["verify"].add_argument("--acquire", action="store_true",
                                       help="measure the pending stage and return its receipt for manual submission")


def main(argv=None):
    try:
        argv, receipt = _receipt_args(sys.argv[1:] if argv is None else argv)
        from panko_core import panko_harness as native

        with ExitStack() as locks:
            def prepare(args):
                if getattr(args, "op_dir", None):
                    locks.enter_context(_lock(args.op_dir))
                _before(args, receipt)

            native.main(argv, before_command=prepare, configure=_configure)
    except (ValueError, OSError, KeyError) as exc:
        out({"refused": True, "error": str(exc)})
        raise SystemExit(3) from exc


if __name__ == "__main__":
    main()
