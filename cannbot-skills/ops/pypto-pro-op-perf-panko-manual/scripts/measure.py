#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Run the frozen precision command and retain its NPU profile as a receipt."""
import argparse
import json
from pathlib import Path

from bootstrap import activate
activate()
from evaluation.measure_latency import campaign_command, measure
from evaluation.repeat_runner import loopable
from evaluation.workspace_probe import SPACES, validate_capacity
from source_analysis.runtime_files import check_files, file_digest as sha
from source_analysis.program_bundle import runtime_spec
import jsonio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("op-file", "op-dir", "test-command", "device", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--test-files", required=True, help="JSON array of all frozen test/reference/input files")
    parser.add_argument("--op-name")
    parser.add_argument("--memory-capacity", required=True,
                        help=f"JSON of verified per-core {'/'.join(SPACES)} byte capacities")
    parser.add_argument("--address-limits",
                        help="JSON of independently verified per-space tile address limits in bytes")
    parser.add_argument("--timeout", type=float, default=600)
    jsonio.emit(acquire(parser.parse_args()))


def _test_manifest(args, argv, cwd, impl):
    script, why = loopable(argv, launches=11)
    if script is None:
        raise ValueError(str(why))
    script = str((Path(cwd) / script).resolve())
    paths = json.loads(args.test_files)
    if not isinstance(paths, list) or not paths or not all(isinstance(x, str) for x in paths):
        raise ValueError("test-files must be a nonempty JSON array of file paths")
    manifest = {str(Path(x).resolve()): sha(x) for x in paths}
    if script not in manifest or impl in manifest:
        raise ValueError("test-files must include the precision entrypoint and exclude the tunable implementation")
    return manifest


def _profile_files(reading):
    paths = Path(reading["msprof_dir"]).rglob("*") if reading.get("msprof_dir") else ()
    return {str(path.resolve()): sha(path) for path in paths if path.is_file()}


def bind_timing_target(contract, reading):
    if contract["op_name"] == "*":
        tasks = reading.get("tasks_per_call")
        if not tasks or not isinstance(tasks, dict) or any(type(n) is not int or n < 1 for n in tasks.values()):
            raise ValueError("package timing target needs the measured task counts per call")
        target = {"tasks_per_call": tasks}
    elif contract["op_name"] is None:
        name, count = reading.get("op_name"), reading.get("rows_per_call")
        valid_name = isinstance(name, str) and name.strip() and name != "*"
        if not valid_name or type(count) is not int or count < 1:
            raise ValueError("auto-detected timing target needs an operator name and a positive task count per call")
        target = {"op_name": name, "rows_per_call": count}
    else:
        return contract
    if "timed_target" in contract and contract["timed_target"] != target:
        raise ValueError("timing target differs from the frozen measurement")
    return dict(contract, timed_target=target)


def campaign_contract(state):
    contract = state["measurement_contract"]
    if contract["op_name"] == "*" and "timed_target" not in contract:
        from evaluation.measure_latency import summarise

        baseline = state["nodes"]["closed"][0]["eval"]["m"]
        profile = baseline.get("msprof_dir")
        runs = contract["runs"] + contract["warmup"]
        observed = summarise(profile, "*", contract["warmup"], calls=runs) if profile else None
        if not observed or observed.error or observed.rows_per_call != baseline.get("rows_per_call"):
            raise ValueError("cannot recover the package timing target from the original baseline profile")
        return bind_timing_target(contract, {"tasks_per_call": observed.tasks_per_call})
    if contract["op_name"] is None and "timed_target" not in contract:
        # Older campaigns retained both observed fields in the baseline node.
        return bind_timing_target(contract, state["nodes"]["closed"][0]["eval"]["m"])
    return contract


def acquire_candidate(args, contract, path, timeout):
    reading = acquire(argparse.Namespace(op_file=args.op_file, op_dir=args.op_dir, test_command=args.test_command,
                                         out=str(path), device=contract["device"], op_name=contract["op_name"],
                                         timeout=timeout, test_files=json.dumps(list(contract["test_files"])),
                                         memory_capacity=json.dumps(contract["memory_capacity"]),
                                         address_limits=json.dumps(contract.get("address_limits"))))
    if reading.get("p") is not None:
        bind_timing_target(contract, reading)
    return reading


def acquire(args):
    """Run the frozen precision command under msprof and return the receipt dict.

    Write the receipt to args.out and retain profile artifacts under optimization/profiles.
    A failed measurement can yield a receipt with no latency. Invalid inputs, an
    existing receipt, or a change to the source or frozen test files raise before writing it.
    """
    capacity = json.loads(args.memory_capacity)
    validate_capacity(capacity)
    address_limits = json.loads(args.address_limits) if getattr(args, "address_limits", None) else None
    output = Path(args.out)
    if output.exists():
        raise ValueError("receipt already exists; use a fresh output path")
    impl = str(Path(args.op_file).resolve())
    argv, cwd, environment = campaign_command(args.test_command, str(Path(args.op_dir).resolve()), args.device)
    manifest = _test_manifest(args, argv, cwd, impl)
    initial_hash = sha(impl)
    program = runtime_spec(impl)
    if program:
        if args.op_name not in (None, "*"):
            raise ValueError("multi-file programs must measure all AI-Core task names, not one member")
        args.op_name = "*"
    profile_root = Path(args.op_dir).resolve() / "optimization" / "profiles"
    profile_root.mkdir(parents=True, exist_ok=True)
    reading = measure(argv, warmup=2, runs=9, agg="median", op_dir=cwd, op_name=args.op_name, timeout_s=args.timeout,
                      impl_path=impl, keep_prof=True, prof_root=str(profile_root), env=environment,
                      joint_correctness=True, budget=capacity, address_limits=address_limits)
    check_files({**manifest, impl: initial_hash}, "source or frozen precision files changed during acquisition")
    if runtime_spec(impl) != program:
        raise RuntimeError("runtime program changed during acquisition")
    reading.update(profile_files=_profile_files(reading), schema="panko-pro-npu-v1", code_hash=initial_hash,
                   program_files=program["files"] if program else {impl: initial_hash},
                   contract={"test_files": manifest, "argv": argv, "cwd": cwd,
                             "env": environment, "device": str(args.device),
                             "op_name": args.op_name, "warmup": 2, "runs": 9,
                             "memory_capacity": capacity,
                             "program": ({k: program[k] for k in ("root", "required")} | {
                                 "files": sorted(program["files"])}) if program else None,
                             "metric": "msprof Task Duration(us)"})
    if address_limits is not None:
        reading["contract"]["address_limits"] = address_limits
    if reading.get("p") is not None:
        reading["contract"] = bind_timing_target(reading["contract"], reading)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reading, indent=2) + "\n")
    return reading


if __name__ == "__main__":
    main()
