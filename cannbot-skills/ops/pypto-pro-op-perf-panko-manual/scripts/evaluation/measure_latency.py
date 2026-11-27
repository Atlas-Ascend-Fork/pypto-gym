# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Measure PyPTO-Pro latency from msprof op_summary Task Duration(us)."""

from __future__ import annotations

import csv
import glob
import hashlib
import logging
import math
import os
import runpy
import shutil
import statistics
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

from bootstrap import KERNEL_PROFILER, activate
from panko_core import measure_latency as core_profile
from panko_core.panko.measurement import sigma_log as log_sample_sigma
activate()
from evaluation.process import DEFAULT_TIMEOUT_S, run_captured
from evaluation import repeat_runner
from evaluation.workspace_probe import profile_workspace

logger = logging.getLogger(__name__)

DURATION = "Task Duration(us)"
LATENCY_DECIMALS = 4
NAME = "Op Name"
TASK_TYPE = "Task Type"
START = "Task Start Time(us)"

# Block Num is the launch grid. Mix Block Num and block_id measure different quantities.
BLOCK_COLUMNS = ("Block Num", "Block Dim")

# Only device compute tasks contribute to the timed launch.
AI_CORE_TASK_TYPES = ("AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV")

# Optional counter domains also detect corrupt or shifted duration rows.
RATIO_MAX = 1.05
UTILIZATION_MAX = 100.5
call_duration_sums = runpy.run_path(str(KERNEL_PROFILER))["call_duration_sums"]
CANONICAL_METRICS = "PipeUtilization"


@dataclass(frozen=True)
class Reading:
    """One profiled run, read. `p is None` iff `error` is set."""

    warmup: int
    error: Optional[str] = None
    rows_considered: int = 0
    op_name: Optional[str] = None
    p: Optional[float] = None
    samples: Tuple[float, ...] = ()
    sigma_log: Optional[float] = None
    block_num: Optional[int] = None
    # Each latency sample sums the device rows belonging to one call.
    calls: Optional[int] = None
    rows_per_call: Optional[int] = None


def prior_cell(row: Mapping, column: str) -> Optional[float]:
    """`row[column]` as a float, tolerating the header's unit suffix and tabs."""
    value = core_profile.profile_cell(row, column)
    return value if value is not None and math.isfinite(value) else None


# Drop the compile call and one cold launch before collecting timing samples.
DEFAULT_WARMUP = 2


def _bound_of(name: str) -> Optional[float]:
    """The domain ceiling for a column, by what the column IS, or None."""
    low = name.split("(")[0].strip().lower()
    if low.endswith("_ratio"):
        return RATIO_MAX
    if "utilization" in low or "utilisation" in low:
        return UTILIZATION_MAX
    return None


def _check_ranges(rows):
    """Refuse the first non-finite numeric cell or out-of-domain ratio/utilization in `rows`."""
    columns = (DURATION, START, "aicore_time(us)", "aiv_time(us)", *BLOCK_COLUMNS)
    numeric_names = {c.split("(")[0].strip().lower() for c in columns}
    for index, row in enumerate(rows):
        _check_row_ranges(row, index, numeric_names)


def _nonfinite_cell(value):
    try:
        return not math.isfinite(float(str(value).strip()))
    except (TypeError, ValueError):
        return False  # Missing/unreadable fields follow their existing rules.


def _check_row_ranges(row, index, numeric_names):
    for key in row.keys():
        name = str(key)
        bound = _bound_of(name)
        base = name.split("(")[0].strip()
        if (bound is not None or base.lower() in numeric_names) and _nonfinite_cell(row[key]):
            raise ValueError(f"non-finite {name} in launch row {index}: {row[key]}")
        if bound is None:
            continue
        value = prior_cell(row, base)
        if value is not None and not 0.0 <= value <= bound:
            raise ValueError(f"column misalignment: {name}={value} outside [0, {bound}] in launch row {index}")


def _group_by_call(rows, calls):
    """`rows` as one list per CALL, or raise `ValueError` saying why not."""
    if calls is None:
        return [[r] for r in rows]
    try:
        calls = int(calls)
    except (TypeError, ValueError):
        raise ValueError("calls must be an integer, not %r" % (calls,)) from None
    if calls < 1:
        raise ValueError("calls must be at least 1, not %d" % (calls,))
    if len(rows) % calls:
        raise ValueError(
            f"{len(rows)} launch row(s) for {calls} call(s): row count is not a multiple of the call count")
    k = len(rows) // calls
    return [rows[i * k:(i + 1) * k] for i in range(calls)]


def _call_durations(groups, chosen):
    if chosen == "*" and any(sorted(r[NAME] for r in group) != sorted(r[NAME] for r in groups[0]) for group in groups):
        raise ValueError("program task composition differs between repeated calls")
    durations = [[prior_cell(row, DURATION) for row in group] for group in groups]
    if any(None in cells for cells in durations):
        raise ValueError(f"missing or invalid {DURATION} for {chosen!r}")
    samples = call_duration_sums(durations)
    if any(not math.isfinite(value) for value in samples):
        raise ValueError(f"non-finite per-call sum for {DURATION}")
    return samples


def _op_summary_in(root: str) -> str:
    """The one op_summary CSV under `root`, or raise `ValueError` naming them."""
    found = sorted(glob.glob(os.path.join(root, "**", "op_summary*.csv"), recursive=True))
    if not found:
        raise ValueError("no op_summary*.csv under %r" % (root,))
    canonical = [p for p in found if CANONICAL_METRICS in os.path.basename(p)]
    if len(canonical) == 1:
        return canonical[0]
    if len(found) == 1:
        return found[0]
    raise ValueError(
        "%d op_summary CSVs under %r and none is the canonical %s: %s. They are "
        "separate acquisitions; name the file instead of letting this pick one"
        % (len(found), root, CANONICAL_METRICS,
           ", ".join(os.path.basename(p) for p in found)))


def read_op_summary(path) -> List[dict]:
    """Every task row of one op_summary CSV, in file order (= launch order)."""
    target = str(path)
    if os.path.isdir(target):
        target = _op_summary_in(target)
    return [r for r in core_profile.read_profile_rows(target) if str(r.get(NAME, "") or "").strip()]


def _ordered(rows: List[dict]) -> List[dict]:
    """`rows` in launch order, or raise `ValueError` saying why not."""
    starts = [prior_cell(r, START) for r in rows]
    if all(s is not None for s in starts):
        order = sorted(range(len(rows)), key=lambda i: starts[i])
        if order != list(range(len(rows))):
            logger.warning("Sorted %d op_summary rows by %s before dropping warm-up calls", len(rows), START)
        return [rows[i] for i in order]
    if all(s is None for s in starts):
        return list(rows)
    raise ValueError(
        "%d of %d AI-Core rows carry %s: launch order cannot be established for "
        "the rest, and the warm-up drop depends on it"
        % (sum(1 for s in starts if s is not None), len(rows), START))


def block_dim_disagreement(requested, reported, op_name=None):
    """Return the reason the profile cannot confirm the requested launch grid."""
    if requested is None:
        return ""
    where = f" for {op_name!r}" if op_name else ""
    if reported is None:
        columns = ", ".join(BLOCK_COLUMNS)
        return f"block_dim={requested} was requested but this op_summary carries no [{columns}] column{where}"
    if int(reported) != int(requested):
        return (f"block_dim={requested} was requested but the profile reports a block count of {int(reported)}"
                f"{where}. Check that the test honors {repeat_runner.BLOCK_DIM_ENV}.")
    return ""


def summarise(path, op_name: Optional[str] = None, warmup: int = DEFAULT_WARMUP, calls: Optional[int] = None,
              block_dim: Optional[int] = None) -> Reading:
    """The primary objective from one profiled run. Never raises on bad input."""
    warmup = max(0, int(warmup))
    try:
        rows = read_op_summary(path)
    except (OSError, ValueError, csv.Error) as exc:
        return Reading(warmup, "cannot read op_summary: %s" % (exc,))
    profile = _select_profile_rows(rows, path, op_name, warmup)
    if isinstance(profile, Reading):
        return profile
    try:
        return _summarise_launches(profile, _ordered(profile.selected), warmup, calls, block_dim)
    except ValueError as exc:
        return Reading(warmup, str(exc), profile.core_count, profile.chosen)


@dataclass(frozen=True)
class _ProfileRows:
    selected: list
    chosen: str
    core_count: int


def _select_profile_rows(rows, path, op_name, warmup):
    task_types = sorted({str(row.get(TASK_TYPE, "") or "").strip() for row in rows})
    allowed = {task_type.lower() for task_type in AI_CORE_TASK_TYPES}
    core = [row for row in rows if str(row.get(TASK_TYPE, "") or "").strip().lower() in allowed]
    if not core:
        return Reading(warmup,
                       "no AI-Core row in %s: task types seen are [%s], and an "
                       "AI-Core launch is one of [%s]"
                       % (os.path.basename(str(path)),
                          ", ".join(value or "<blank>" for value in task_types),
                          ", ".join(AI_CORE_TASK_TYPES)))
    names = list(dict.fromkeys(str(row.get(NAME, "") or "").strip() for row in core))
    if op_name is None and len(names) > 1:
        return Reading(warmup,
                       f"{len(names)} AI-Core op names in one profile and no op_name given: {', '.join(names)}. "
                       "Specify the kernel to measure.", len(core))
    chosen = names[0] if op_name is None else op_name
    if chosen not in ("*", *names):
        return Reading(warmup, "no AI-Core row for op %r; this profile carries [%s]" % (chosen, ", ".join(names)),
                       len(core))
    selected = [row for row in core if chosen in ("*", str(row.get(NAME, "") or "").strip())]
    return _ProfileRows(selected, chosen, len(core))


def _median_launch_row(groups, warm, median, warmup):
    # Ties follow launch order, including float-rounding ties at an even count.
    distances = [abs(duration - median) for duration in warm]
    closest = min(distances)
    index = next(index for index, distance in enumerate(distances) if distance <= closest + 1e-9)
    return groups[warmup + index][0]


def _reported_block(row):
    values = (prior_cell(row, column) for column in BLOCK_COLUMNS)
    return next((int(value) for value in values if value is not None), None)


def _summarise_launches(profile, ordered, warmup, calls, block_dim):
    chosen = profile.chosen
    _check_ranges(ordered)
    groups = _group_by_call(ordered, calls)
    values = _call_durations(groups, chosen)
    if len(values) < warmup + 1:
        raise ValueError("%d launch row(s) for %r at warmup=%d: dropping the "
                         "warm-up leaves nothing to aggregate. There is no fallback "
                         "to all rows -- the first call is a cold JIT call and reads "
                         "20-25%% slow, so the fallback would invert the reading"
                         % (len(values), chosen, warmup))
    warm = values[warmup:]
    median = statistics.median(warm)
    mid = _median_launch_row(groups, warm, median, warmup)
    block = _reported_block(mid)
    disagreement = block_dim_disagreement(block_dim, block, chosen)
    if disagreement:
        raise ValueError(disagreement)
    return Reading(p=round(median, LATENCY_DECIMALS), samples=tuple(warm), warmup=warmup, op_name=chosen,
                   sigma_log=log_sample_sigma(warm), block_num=block, rows_considered=profile.core_count, error=None,
                   calls=calls, rows_per_call=len(groups[0]))


_agg = core_profile.aggregate


def reading_to_metrics(reading, workspace, compile_s=None):
    samples = list(reading.samples)
    out = {
        "p": reading.p, "p_source": "msprof " + DURATION,
        "samples": samples, "sigma_log": reading.sigma_log,
        "warmup": reading.warmup, "runs": len(samples) or None,
        "p_agg": "median" if reading.p is not None else None,
        "raw_count": len(samples) + reading.warmup if samples else 0,
        "compile_s": compile_s, "op_name": reading.op_name,
        "block_num": reading.block_num, "rows_considered": reading.rows_considered,
        # Whether `p` is one of the warm readings rather than a synthetic midpoint.
        "p_is_sample": bool(samples) and reading.p is not None and statistics.median(samples) in samples,
    }
    optional = {"calls": reading.calls, "rows_per_call": reading.rows_per_call, "p_error": reading.error or None}
    out.update({key: value for key, value in optional.items() if value is not None})
    out.update(workspace)
    return out


def imported_impls(output):
    """Read absolute module paths and short source hashes from the test import report."""
    prefix = repeat_runner.IMPL_MARKER.split("%s", 1)[0]
    out_m = {}
    for line in (output or "").splitlines():
        parts = line.partition(prefix)[2].split(None, 1)
        if len(parts) == 2:
            out_m[parts[1].strip()] = parts[0].strip()
    return out_m


def impl_mismatch(output, impl_path):
    """The reason the test measured a DIFFERENT file, or None."""
    seen = imported_impls(output)
    if not impl_path:
        return "no staged implementation path"
    if not seen:
        return "the profiled precision script did not import the staged implementation"
    try:
        want = hashlib.sha256(Path(impl_path).read_bytes()).hexdigest()[:8]
    except OSError:
        return None
    if any(os.path.realpath(name) == os.path.realpath(impl_path) and value == want for name, value in seen.items()):
        # Count duplicate module names even when the staged file was imported.
        return None
    return ("the test imported %s, not the staged %s (sha %s): the measurement "
            "would be of a program this campaign is not tuning"
            % (", ".join("%s (sha %s)" % (p, h) for p, h in sorted(seen.items())),
               os.path.abspath(impl_path), want))


def completed_launches(output):
    """How many launches `repeat_runner` reported finishing, from its markers."""
    marker = repeat_runner.LAUNCH_MARKER.split("%d", 1)[0]
    return sum(1 for line in (output or "").splitlines() if marker in line)


def short_profile_cause(metrics, output, launches):
    """Why this acquisition is short of rows: "loop", "profile", or None."""
    measured = metrics.get("p") is not None and not metrics.get("p_error")
    # Retrying the same mismatched program would spend another device run.
    if measured or metrics.get("correctness_passed") is False or metrics.get("impl_module_mismatch"):
        return None
    return "profile" if completed_launches(output) >= int(launches or 0) else "loop"


def measure(run_once, warmup=DEFAULT_WARMUP, runs=9, agg="median", *, options=None, _reacquire=True, **profile_options):
    """One profiled acquisition: `{p, samples, sigma_log, ...}`. Never raises."""
    # Keep named keyword callers compatible; dataclass construction rejects unknown fields.
    warmup, runs = max(0, int(warmup)), max(1, int(runs))
    options = replace(options or AcquisitionOptions(), warmup=warmup, runs=runs, agg=agg, **profile_options)
    prepared = prepare_launch(run_once, options)
    if isinstance(prepared, dict):
        return prepared
    with profile_directory(options) as out_dir:
        metrics, output, returncode = _profile_once(prepared, out_dir)
        if options.keep_prof:
            Path(out_dir, "profiler.log").write_text(output or "", encoding="utf-8")
        if returncode is None:
            return metrics
        cause = short_profile_cause(metrics, output, warmup + runs)
        if cause == "profile" and _reacquire:
            again = measure(run_once, warmup=warmup, runs=runs, agg=agg, options=options, _reacquire=False)
            again.update(profile_reacquired=1, profile_reacquired_after=metrics.get("p_error"))
            return again
        return _finish_profile(metrics, output, returncode, prepared, out_dir)


@dataclass(frozen=True)
class AcquisitionOptions:
    """Schedule, source, environment, and profiling settings for one acquisition."""

    warmup: int = DEFAULT_WARMUP
    runs: int = 9
    agg: str = "median"
    op_dir: Optional[str] = None
    op_name: Optional[str] = None
    timeout_s: Optional[float] = DEFAULT_TIMEOUT_S
    keep_prof: bool = False
    prof_root: Optional[str] = None
    impl_path: Optional[str] = None
    budget: Optional[Mapping] = None
    block_dim: Optional[int] = None
    aic_metrics: Optional[str] = None
    block_dim_flag: Optional[str] = None
    env: Optional[Mapping] = None
    python: Optional[str] = None
    extra_consts: Optional[Mapping] = None
    joint_correctness: bool = False
    address_limits: Optional[Mapping] = None


@dataclass(frozen=True)
class _PreparedProfile:
    options: AcquisitionOptions
    workspace: dict
    looped: list
    child_env: dict


def prepare_launch(run_once, options):
    launches = options.warmup + options.runs
    workspace = profile_workspace(options.impl_path, options.budget, options.extra_consts,
                                  address_limits=options.address_limits)
    if not isinstance(run_once, (list, tuple)):
        reason = ("nothing to profile: `measure` takes the argv of the op's test "
                  "command. pypto_pro prints no in-process profiler summary -- the "
                  "canonical Task Duration is in an op_summary CSV msprof writes "
                  "after it exits -- so a callable cannot be measured from inside "
                  "this process and is not called.")
    elif shutil.which("msprof") is None:
        reason = "msprof not found on PATH, so nothing was measured"
    else:
        looped, why = repeat_runner.looped_argv(run_once, launches, python=options.python, block_dim=options.block_dim,
                                                block_dim_flag=options.block_dim_flag)
        if looped is not None:
            child_env = _profile_environment(options.env, options.impl_path)
            return _PreparedProfile(options, workspace, looped, child_env)
        reason = ("the test command cannot be looped in-process, so this "
                  "acquisition would have measured ONE launch and called it %d: "
                  "%s" % (launches, why))
    return reading_to_metrics(Reading(options.warmup, reason), workspace=workspace)


def _profile_environment(env, impl_path):
    child_env = dict(os.environ)
    child_env.update({str(key): str(value) for key, value in (env or {}).items()})
    if impl_path:
        child_env[repeat_runner.IMPL_ENV] = os.path.abspath(impl_path)
    return child_env


@contextmanager
def profile_directory(options):
    directory = tempfile.mkdtemp(prefix="panko_msprof_", dir=options.prof_root)
    try:
        yield directory
    finally:
        if not options.keep_prof:
            shutil.rmtree(directory, ignore_errors=True)


def _relay_profile_output(out_dir, output):
    Path(out_dir, "command.log").write_text(output)
    if output:
        logging.getLogger(__name__).info(output.rstrip("\n"))


def capture_profile(prepared, out_dir):
    options, child_env = prepared.options, prepared.child_env
    # Fresh bytecode namespace: equal-size edits within one second must execute.
    child_env = dict(child_env, PYTHONPYCACHEPREFIX=os.path.join(out_dir, "bytecode"), PYTHONDONTWRITEBYTECODE="1")
    from evaluation import acquisition_receipt
    receipt = None
    if options.joint_correctness:
        path, token, child_env = acquisition_receipt.prepare(out_dir, child_env)
        receipt = (path, token)
    cmd = core_profile.msprof_command("msprof", out_dir, prepared.looped, aic_metrics=options.aic_metrics)
    started = time.perf_counter()
    try:
        proc = run_captured(cmd, cwd=options.op_dir or None, timeout=options.timeout_s, env=child_env)
    except subprocess.TimeoutExpired as exc:
        output = exc.output or ""
        _relay_profile_output(out_dir, output)
        return Reading(options.warmup, "the profiled run exceeded %ss" % exc.timeout), output, None, {}
    output = proc.stdout or ""
    _relay_profile_output(out_dir, output)
    profile_s = time.perf_counter() - started
    parse_started = time.perf_counter()
    reading = summarise(out_dir, op_name=options.op_name, warmup=options.warmup,
                        calls=options.warmup + options.runs, block_dim=options.block_dim)
    diagnostics = (
        acquisition_receipt.validate(*receipt, options.warmup + options.runs, proc.returncode) if receipt else {}
    )
    diagnostics.setdefault("timing_s", {}).update(
        profile_process=profile_s, profile_parse=time.perf_counter() - parse_started,
        acquisition=time.perf_counter() - started)
    return reading, output, proc.returncode, diagnostics


def _profile_once(prepared, out_dir):
    started = time.perf_counter()
    reading, output, rc, diagnostics = capture_profile(prepared, out_dir)
    metrics = reading_to_metrics(reading, compile_s=round(time.perf_counter() - started, 3),
                                 workspace=prepared.workspace)
    if rc is None:
        return metrics, output, rc
    _validate_profile_metrics(metrics, reading, output, rc, prepared.options)
    metrics.update(diagnostics)
    from panko_core.bayesian_optimization.evaluator import ENV_FAULT_MARKERS
    fault = next((marker for marker in ENV_FAULT_MARKERS if marker in output), None)
    if fault or (metrics.get("impl_module_mismatch") and metrics.get("impl_modules_seen")):
        metrics.update(test_failed=False, correctness_passed=False, p=None, samples=[],
                       p_error="acquisition refused: " + (fault or "wrong implementation imported"))
    return metrics, output, rc


def _validate_profile_metrics(metrics, reading, output, returncode, options):
    if reading.error and returncode != 0:
        metrics["p_error"] = "msprof exited %d; %s" % (returncode, reading.error)
    wrong = impl_mismatch(output, options.impl_path)
    seen_impls = imported_impls(output)
    if seen_impls:
        metrics["impl_modules_seen"] = len(seen_impls)
    if wrong:
        metrics.update(p=None, p_error=wrong, impl_module_mismatch=True)


def _finish_profile(metrics, output, returncode, prepared, out_dir):
    options = prepared.options
    launches = options.warmup + options.runs
    if short_profile_cause(metrics, output, launches) == "loop":
        metrics["launches_completed"] = completed_launches(output)
        metrics["p_error"] = (
            "%s | only %d of %d launches reported done "
            "(repeat_runner's launch markers), so the test stopped running: "
            "this is a property of the candidate, not of the acquisition, "
            "and re-profiling it would repeat it"
            % (metrics.get("p_error") or "short profile",
               metrics["launches_completed"], launches))
    metrics.update(msprof_rc=returncode, test_env=",".join(sorted(options.env or {})), python_used=prepared.looped[0])
    if options.block_dim is not None:
        metrics["block_dim_requested"] = int(options.block_dim)
    if options.keep_prof:
        metrics["msprof_dir"] = out_dir
    if options.agg == "min" and metrics.get("p") is not None and metrics["samples"]:
        metrics["p"] = _agg(metrics["samples"], "min")
        metrics["p_agg"] = "min"
    else:
        metrics["p_agg"] = "median"
    return metrics


def test_argv(test_command, op_dir=None):
    """The op's single-step test command as an argv. Raises on several steps."""
    if isinstance(test_command, (list, tuple)):
        return [str(a) for a in test_command], op_dir, {}
    from panko_core.bayesian_optimization.evaluator import command_steps
    steps, cwd = command_steps(test_command)
    if len(steps) != 1:
        raise ValueError(
            "the test command runs %d steps; one profiling process may hold one "
            "measured launch, so the profiled entry has to be a single command"
            % (len(steps),))
    argv, env = steps[0]
    return argv, cwd or op_dir, dict(env)


def campaign_command(command, op_dir, device):
    """Bind a test command to the campaign device before measuring or validating it."""
    argv, cwd, env = test_argv(command, op_dir)
    selected = env.setdefault("TILE_FWK_DEVICE_ID", str(device))
    if selected != str(device):
        raise ValueError("test command device differs from the campaign device")
    return argv, cwd, env


def check_campaign_command(command, op_dir, contract):
    """Compare a command with the frozen baseline after applying the campaign device."""
    from panko_core.bayesian_optimization.evaluator import UnsupportedCommand
    try:
        argv, cwd, env = campaign_command(command, op_dir, contract["device"])
    except UnsupportedCommand as exc:
        raise ValueError(str(exc)) from exc
    if (argv, Path(cwd).resolve(), env) != (contract["argv"], Path(contract["cwd"]).resolve(), contract["env"]):
        raise ValueError("test command differs from the baseline measurement")
