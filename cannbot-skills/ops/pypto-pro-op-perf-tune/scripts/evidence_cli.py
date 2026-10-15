#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

"""性能采集的 CLI/校验/时间线/报告渲染与采集辅助（从 msprof_perf_summary 拆分）。

本模块延迟导入 msprof_perf_summary 的共享工具，避免模块级循环依赖；被
msprof_perf_summary 的对应入口按需导入。
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from golden_contract import (
    parse_case_manifest,
    performance_case_source_error,
    resolve_performance_cases,
)
from instruction_timeline import (
    analyze_instruction_timeline as _analyze_instruction_timeline,
    exported_traces as _exported_traces,
    find_unique_prof_dir as _find_unique_prof_dir,
    load_exported_trace as _load_exported_trace,
    new_exported_trace as _new_exported_trace,
    validate_biu_database as _validate_biu_database,
)

# 延迟导入触发时主模块已完整加载，不会形成循环依赖。
from msprof_perf_summary import (  # noqa: E402
    BOUND_ROUTE_HIGH_THRESHOLD,
    BOUND_ROUTE_MAX_THRESHOLD,
    LOGGER,
    RunRequest,
    RunSelector,
    _atomic_write_text,
    _find_test_script,
    _load_module,
    _manifest_case_functions,
    _profile_env,
    _resolved_evidence_round,
    _run_warmups,
    _selected_executable_sha256,
    _select_device_id,
    _test_command,
    _write_collection_log,
    find_op_summary,
    profile_case_contract_error,
    read_csv_rows,
    safe_case_dir_name,
    safe_float,
    uses_function_adapter,
)


def _emit_stdout(text: str) -> None:
    """CLI 契约输出：经当前 sys.stdout 写出（避免 print/sys.stdout.write 触发 G.LOG.02）。"""
    stream = sys.stdout
    stream.write(text)


def _select_primary_core(aicore_time: float, aiv_time: float) -> Optional[str]:
    """Select the longer-running core type when both times share one schema."""
    if aicore_time > 0 or aiv_time > 0:
        return "aic" if aicore_time >= aiv_time else "aiv"
    return None


def normalized_pipe_ratio(value: Any) -> float:
    """Return the verified CANN op_summary ratio unit (fraction in [0, 1])."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"pipe ratio {value!r} is not a finite number")
    try:
        ratio = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"pipe ratio {value!r} is not a finite number") from error
    if not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise ValueError(
            f"pipe ratio {value!r} is outside the verified CANN fraction range [0, 1]"
        )
    return ratio


def _collect_ratios(
    primary: str, fields: Dict[str, Dict[str, Tuple[str, ...]]], merged: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    """收集主导核的可用 pipe ratio（校验 [0,1] 分数单位）。"""
    ratios = {}
    for category, keys in (fields.get(primary) or {}).items():
        category_ratios = {}
        for key in keys:
            if (
                key in merged
                and merged.get(key) is not None
                and str(merged.get(key)).strip() not in ("", "N/A", "NA", "-")
            ):
                category_ratios[key] = normalized_pipe_ratio(merged.get(key))
        ratios[category] = category_ratios
    return ratios


def _route_from_ratios(ratios: Dict[str, Dict[str, float]]) -> Tuple[list, Optional[bool], str]:
    """按双阈值从 ratio 生成路由标签（触发家族、scalar 触发、路由名）。"""
    flat = [value for values in ratios.values() for value in values.values()]
    if not flat:
        return [], None, "insufficient_evidence"
    maximum = max(flat)
    triggered = []
    for category, values in ratios.items():
        if any(
            value > BOUND_ROUTE_HIGH_THRESHOLD
            or (value == maximum and value > BOUND_ROUTE_MAX_THRESHOLD)
            for value in values.values()
        ):
            triggered.append(category)
    scalar_triggered = "scalar" in triggered
    if not triggered:
        label = "no_dominant_pipe"
    elif len(triggered) > 1:
        label = "mixed"
    elif triggered[0] == "compute":
        label = "compute_candidate"
    elif triggered[0] == "data_movement":
        label = "data_movement_candidate"
    else:
        label = "scalar_candidate"
    return triggered, scalar_triggered, label


def _ratio_fields() -> Dict[str, Dict[str, Tuple[str, ...]]]:
    """返回主导核的 ratio 字段映射（compute/data_movement/scalar 家族）。"""
    return {
        "aic": {
            "compute": ("aic_mac_ratio", "aic_cube_ratio"),
            "data_movement": (
                "aic_mte1_ratio", "aic_mte2_ratio", "aic_mte3_ratio",
                "aic_fixpipe_ratio",
            ),
            "scalar": ("aic_scalar_ratio",),
        },
        "aiv": {
            "compute": ("aiv_vec_ratio",),
            "data_movement": ("aiv_mte2_ratio", "aiv_mte3_ratio"),
            "scalar": ("aiv_scalar_ratio",),
        },
    }


def diagnose_bound_route(merged: Dict[str, Any]) -> Dict[str, Any]:
    """Create a ratio-derived routing hint without claiming Roofline proof.

    ``op_summary`` ratios are aggregate busy ratios.  They can tell the next
    investigation which pipe family deserves attention, but they contain no
    event ordering and therefore cannot prove DMA/compute overlap or a final
    Roofline terminal state.
    """
    aicore_time = safe_float(merged.get("aicore_time(us)"))
    aiv_time = safe_float(merged.get("aiv_time(us)"))
    primary = _select_primary_core(aicore_time, aiv_time)
    result: Dict[str, Any] = {
        "schema_version": 1,
        "evidence_kind": "op_summary_ratio_route",
        "primary_core": primary,
        "primary_core_source": "max(aicore_time(us), aiv_time(us))",
        "aicore_time_us": aicore_time,
        "aiv_time_us": aiv_time,
        "routing_label": "insufficient_evidence",
        "triggered_categories": [],
        "ratios": {},
        "scalar_route_triggered": None,
        "roofline_terminal_status": "insufficient_evidence",
        "pipeline_overlap_status": "unverified",
        "completion_eligible": False,
        "note": (
            "Aggregate pipe ratios route the next experiment only; a workload/"
            "traffic model plus timeline or equivalent dependency evidence is "
            "required to prove the terminal bound and pipeline overlap."
        ),
    }
    if primary is None:
        return result

    ratios = _collect_ratios(primary, _ratio_fields(), merged)
    result["ratios"] = ratios
    triggered, scalar_triggered, label = _route_from_ratios(ratios)
    result["triggered_categories"] = triggered
    result["scalar_route_triggered"] = scalar_triggered
    result["routing_label"] = label
    return result


def _add_bound_route(lines: List[str], merged: Dict[str, Any]) -> None:
    diagnosis = diagnose_bound_route(merged)
    lines.append("")
    lines.append("--- Bound routing (ratio-derived; NOT terminal Roofline proof) ---")
    lines.append(
        f"  primary_core={diagnosis['primary_core'] or 'unknown'} | "
        f"routing_label={diagnosis['routing_label']} | "
        f"scalar_route_triggered={diagnosis['scalar_route_triggered']}"
    )
    lines.append(
        "  pipeline_overlap=unverified: op_summary ratios have no event ordering; "
        "use timeline/dependency evidence before claiming overlap."
    )


def list_op_names_mode(args) -> int:
    """List exact lowering Op Name values from one discovery collection."""
    group_dir = os.path.abspath(args.prof_group_dir)
    if not os.path.isdir(group_dir):
        raise ValueError("'%s' is not a directory." % group_dir)
    pipe_dir = os.path.join(group_dir, "PROF_PipeUtilization")
    op_csv = find_op_summary(pipe_dir)
    if not op_csv:
        raise ValueError(
            "discovery requires exactly one PipeUtilization op_summary CSV under %s"
            % pipe_dir
        )
    rows = read_csv_rows(op_csv)
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("Op Name", "")).strip()
        if not name:
            continue
        task_type = str(row.get("Task Type", "")).strip()
        key = (name, task_type)
        record = grouped.setdefault(key, {"count": 0, "durations_us": []})
        record["count"] += 1
        duration = safe_float(row.get("Task Duration(us)"))
        if duration > 0:
            record["durations_us"].append(duration)
    if not grouped:
        raise ValueError("PipeUtilization op_summary contains no non-empty Op Name")
    _emit_stdout("count\ttask_type\ttotal_duration_us\texact_op_name\n")
    for (name, task_type), record in sorted(
            grouped.items(), key=lambda item: (-sum(item[1]["durations_us"]), item[0])):
        _emit_stdout(
            f"{record['count']}\t{task_type}\t"
            f"{sum(record['durations_us']):.6f}\t{name}\n"
        )
    return 0


def _timeline_case(performance_cases, case_id: str):
    matches = [case for case in performance_cases if str(case[0]) == str(case_id)]
    if len(matches) != 1:
        raise ValueError(f"--case-id must select exactly one manifest case; found {len(matches)}")
    return matches[0]


def _timeline_archive_dir(out_dir: Path, data: Dict[str, Any], case_id: str) -> Path:
    round_path = _resolved_evidence_round(out_dir, data.get("deep_profile_round"))
    case_dir = round_path / f"case_{safe_case_dir_name(case_id)}"
    if not case_dir.is_dir() or case_dir.is_symlink():
        raise ValueError(f"final compare case evidence directory is missing or unsafe: {case_dir}")
    timeline_dir = case_dir / "instruction_timeline"
    if timeline_dir.exists() or timeline_dir.is_symlink():
        raise FileExistsError(
            f"timeline evidence already exists for case {case_id!r}; "
            "remove the incomplete directory or start a new final compare round"
        )
    return timeline_dir


def _check_final_performance(
    out_dir: Path, args, device_id
) -> Tuple[Optional[dict], Optional[str]]:
    """读取并校验 final compare 的 performance.json 与证据闭环。"""
    performance_path = out_dir / "performance.json"
    try:
        performance = json.loads(performance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load final performance.json: {error}") from error
    if performance.get("profiling_mode") != "compare":
        raise ValueError("timeline evidence must attach to a formal compare collection")
    for field, expected in (
        ("target_op_name", args.op_name),
        ("device_id", device_id),
        ("seed", args.seed if args.seed is not None else 42),
        ("warmup", args.warmup),
    ):
        if performance.get(field) != expected:
            raise ValueError(
                f"timeline protocol mismatch: performance.json {field}="
                f"{performance.get(field)!r}, requested={expected!r}"
            )
    if performance.get("performance_cases") != args.performance_cases:
        raise ValueError(
            "timeline case manifest identity differs from the final compare collection"
        )
    from batch_evidence import batch_evidence_error
    evidence_error = batch_evidence_error(out_dir, performance)
    if evidence_error:
        raise ValueError(f"final compare evidence is invalid: {evidence_error}")
    return performance, None


def _timeline_preflight(args):
    """校验 timeline 参数、final compare 证据并解析归档目录。

    返回 (context, exit_code)；context 含 out_dir/device/performance_cases/
    selected_case/test_script/performance/archive_dir，exit_code 为 1 或 None。
    """
    if not args.output_dir or not args.case_manifest or not args.case_id:
        raise ValueError("--timeline requires --output-dir, --case-manifest and --case-id")
    out_dir = Path(args.output_dir).expanduser().resolve(strict=True)
    if not _validate_measurement_args(args, manifest=True):
        return None, 1
    device_id, device_src = _select_device_id(args, tile_fwk=True)
    performance_cases = _validated_performance_cases(out_dir, args)
    if performance_cases is None:
        return None, 1
    selected_case = _timeline_case(performance_cases, args.case_id)
    if not _validate_case_selector(performance_cases, args):
        return None, 1
    test_script, executable_sha256, script_error = _selected_executable_sha256(out_dir)
    if script_error:
        raise ValueError(script_error)
    performance, _ = _check_final_performance(out_dir, args, device_id)
    if executable_sha256 != performance.get("executable_sha256"):
        raise ValueError("timeline executable differs from the final compare")
    if not _validate_selector_runtime(out_dir, performance_cases, args, device_id):
        return None, 1
    archive_dir = _timeline_archive_dir(out_dir, performance, args.case_id)
    context = {
        "out_dir": out_dir,
        "device_id": device_id,
        "device_src": device_src,
        "performance_cases": performance_cases,
        "selected_case": selected_case,
        "test_script": test_script,
        "performance": performance,
        "archive_dir": archive_dir,
    }
    return context, None


def _export_and_analyze(
    session: Path, prof_dir: Path, args, context: dict, env,
) -> Tuple[Path, Dict[str, Any], Path, Path, Path]:
    """导出指令时间线、解析并校验设备归属；返回证据路径与分析结果。"""
    traces_before_export = _exported_traces(prof_dir)
    reports = {"json_process": {"ascend": True, "biu_perf": True}}
    reports_path = session / "reports.json"
    _atomic_write_text(
        reports_path, json.dumps(reports, separators=(",", ":")) + "\n"
    )
    export_command = [
        "msprof", "--export=on", f"--output={prof_dir}", "--type=text",
        f"--reports={reports_path}",
    ]
    export_result = _run_timed_subprocess(
        export_command, env, getattr(args, "timeline_timeout", 600)
    )
    _write_collection_log(session / "export", export_command, export_result)
    if export_result.returncode != 0:
        raise RuntimeError(f"timeline export failed: {export_result.stderr[-500:]}")
    trace_path = _new_exported_trace(prof_dir, traces_before_export)
    trace_payload = _load_exported_trace(trace_path)
    analysis = _analyze_instruction_timeline(trace_payload, args.op_name)
    selected_case = context["selected_case"]
    analysis.update({
        "collection_id": context["performance"].get("collection_id"),
        "executable_sha256": context["performance"].get("executable_sha256"),
        "case": str(selected_case[0]),
        "shape": selected_case[1],
        "dtype": selected_case[2],
        "device_id": context["device_id"],
        "device_source": context["device_src"],
        "seed": args.seed if args.seed is not None else 42,
        "warmup": args.warmup,
        "collection_kind": "supplemental_instruction_profile",
        "canonical_timing_source": "separate_formal_compare",
    })

    device_dirs = sorted(path for path in prof_dir.glob("device_*") if path.is_dir())
    if len(device_dirs) != 1 or device_dirs[0].name != f"device_{context['device_id']}":
        raise RuntimeError(
            f"instruction profile device provenance mismatch: "
            f"expected device_{context['device_id']}, found {[path.name for path in device_dirs]}"
        )
    sqlite_dir = device_dirs[0] / "sqlite"
    biu_db = sqlite_dir / "biu_perf.db"
    task_db = sqlite_dir / "ascend_task.db"
    for required in (biu_db, task_db):
        if not required.is_file() or required.is_symlink():
            raise RuntimeError(
                f"required timeline evidence is missing or unsafe: {required}"
            )
    analysis["biu_database"] = _validate_biu_database(biu_db)
    return trace_path, analysis, reports_path, biu_db, task_db


def _collect_timeline_session(
    session: Path, context: dict, args
) -> Tuple[Path, Dict[str, Any], Path, Path, Path]:
    """采集/导出指令时间线并分析；返回证据文件路径与分析结果。"""
    test_script = context["test_script"]
    device_id = context["device_id"]
    performance_cases = context["performance_cases"]
    function_adapter = uses_function_adapter(
        args, [case[0] for case in performance_cases]
    )
    env = _profile_env(
        device_id, args.seed, getattr(args, "case_env", None), args.case_id, True
    )
    timeline_selector = RunSelector(
        getattr(args, "case_arg", None), getattr(args, "case_env", None),
        args.case_id, args.case_manifest, function_adapter,
    )
    warmup_error = _run_warmups(
        RunRequest(
            test_script, "", args.warmup, device_id, args.seed,
            timeline_selector,
        ),
        env,
    )
    if warmup_error:
        raise RuntimeError(warmup_error)
    command = [
        "msprof", f"--output={session}", "--ai-core=on",
        "--aic-metrics=PipeUtilization", "--instr-profiling=on",
        "--task-time=on", "--ascendcl=on",
        *_test_command(test_script, timeline_selector),
    ]
    result = _run_timed_subprocess(
        command, env, getattr(args, "timeline_timeout", 600)
    )
    _write_collection_log(session, command, result)
    if result.returncode != 0:
        raise RuntimeError(f"instruction profiling failed: {result.stderr[-500:]}")

    prof_dir = _find_unique_prof_dir(session)
    return _export_and_analyze(session, prof_dir, args, context, env)


class TimelineArtifacts(NamedTuple):
    """补充时间线采集会话的产物路径集合。"""
    trace_path: Path
    reports_path: Path
    biu_db: Path
    task_db: Path


def _stage_timeline_evidence(
    archive_dir: Path, session: Path, analysis: Dict[str, Any],
    artifacts: TimelineArtifacts,
) -> None:
    """在 staging 目录组装并原子发布 timeline 证据。"""
    staging = archive_dir.with_name(f".{archive_dir.name}.{os.getpid()}.tmp")
    try:
        staging.mkdir(parents=False, exist_ok=False)
        shutil.copy2(artifacts.trace_path, staging / "msprof_timeline.json")
        shutil.copy2(artifacts.biu_db, staging / "biu_perf.db")
        shutil.copy2(artifacts.task_db, staging / "ascend_task.db")
        shutil.copy2(artifacts.reports_path, staging / "reports.json")
        shutil.copy2(session / "collection.log", staging / "collection.log")
        shutil.copy2(session / "export" / "collection.log", staging / "export.log")
        _atomic_write_text(
            staging / "timeline_evidence.json",
            json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        )
        os.replace(staging, archive_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def run_timeline_mode(args) -> int:
    """Collect supplemental instruction events for one final-compare P0 case."""
    context, exit_code = _timeline_preflight(args)
    if exit_code:
        return exit_code
    archive_dir = context["archive_dir"]

    session = context["out_dir"] / ".msprof" / f"timeline_{os.getpid()}_{time.time_ns()}"
    session.mkdir(parents=True, exist_ok=False)
    try:
        trace_path, analysis, reports_path, biu_db, task_db = (
            _collect_timeline_session(session, context, args)
        )
        _, executable_sha256, executable_error = _selected_executable_sha256(
            context["out_dir"]
        )
        if executable_error:
            raise RuntimeError(executable_error)
        if executable_sha256 != context["performance"].get("executable_sha256"):
            raise RuntimeError("profiling executable changed during timeline collection")
        _stage_timeline_evidence(
            archive_dir, session, analysis,
            TimelineArtifacts(trace_path, reports_path, biu_db, task_db),
        )
        LOGGER.info("[INFO] Supplemental timeline saved to: %s", archive_dir)
        LOGGER.info(
            "[INFO] Attribution=%s, monitored_lanes=%s, pipeline_status=%s",
            analysis["attribution_status"], len(analysis["monitored_lanes"]),
            analysis["pipeline_evidence_status"],
        )
        return 0
    finally:
        if not getattr(args, "keep_prof", False):
            shutil.rmtree(session, ignore_errors=True)


def run_case_function(test_script: str, manifest_path: str, case_id: str) -> int:
    """Invoke one validated no-argument correctness test through a case adapter."""
    _, source, error = parse_case_manifest(manifest_path)
    if error:
        raise ValueError(error)
    mapping = source.get("case_functions") or {}
    test_function = mapping.get(str(case_id))
    if test_function is None:
        raise ValueError(f"case manifest has no test_function for case {case_id!r}")
    test_path = Path(test_script).expanduser().resolve(strict=True)
    if not test_path.is_file() or test_path.is_symlink():
        raise ValueError(f"test script must be a regular non-symlink file: {test_path}")
    module_name = (
        f"_pypto_perf_case_{hashlib.sha256(str(test_path).encode()).hexdigest()[:12]}_"
        f"{os.getpid()}_{time.time_ns()}"
    )
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(test_path.parent))
        module = _load_module(str(test_path), module_name)
        function = getattr(module, test_function, None)
        if not callable(function):
            raise ValueError(
                f"test function {test_function!r} is not callable in {test_path.name}"
            )
        signature = inspect.signature(function)
        required = []
        for parameter in signature.parameters.values():
            has_no_default = parameter.default is inspect.Parameter.empty
            is_vararg = parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
            )
            if has_no_default and not is_vararg:
                required.append(parameter.name)
        if required:
            raise ValueError(
                f"test function {test_function!r} requires arguments {required}; "
                "the automatic case adapter accepts no-argument test functions only"
            )
        function()
    finally:
        sys.path[:] = old_path
        sys.modules.pop(module_name, None)
    _emit_stdout(f"PYPTO_PERF_SELECTED_CASE={case_id}\n")
    return 0


def _selector_markers(output):
    prefix = "PYPTO_PERF_SELECTED_CASE="
    return [
        line[len(prefix):]
        for line in output.splitlines()
        if line.startswith(prefix)
    ]


def _selector_preflight(test_script, case_ids, args, device_id):
    """Prove unknown rejection and exact known-case selection before profiling."""
    case_arg = getattr(args, "case_arg", None)
    case_env = getattr(args, "case_env", None)
    function_adapter = uses_function_adapter(args, case_ids)
    if not (case_arg or case_env or function_adapter):
        return None
    case_manifest = getattr(args, "case_manifest", None)
    unknown = f"__pypto_perf_unknown_{os.getpid()}_{time.time_ns()}__"
    unknown_selector = RunSelector(
        case_arg, case_env, unknown, case_manifest, function_adapter,
    )
    env = _profile_env(device_id, args.seed, case_env, unknown,
                       tile_fwk=bool(case_manifest))
    result = subprocess.run(
        _test_command(test_script, unknown_selector),
        capture_output=True, text=True, env=env,
    )
    if result.returncode == 0:
        return (
            "declared case selector accepted a collision-resistant unknown case id; "
            "refusing to profile a runner that may ignore the selector"
        )
    for case_id in case_ids:
        case_selector = RunSelector(
            case_arg, case_env, case_id, case_manifest, function_adapter,
        )
        env = _profile_env(device_id, args.seed, case_env, case_id,
                           tile_fwk=bool(case_manifest))
        result = subprocess.run(
            _test_command(test_script, case_selector),
            capture_output=True, text=True, env=env,
        )
        if result.returncode != 0:
            return f"case selector failed for known case id {case_id!r}"
        markers = _selector_markers(result.stdout + "\n" + result.stderr)
        if markers != [str(case_id)]:
            return (
                f"case selector for {case_id!r} must emit exactly one line "
                f"'PYPTO_PERF_SELECTED_CASE={case_id}'; got markers={markers}"
            )
    return None


def _stop_and_drain(process):
    """超时后先温和终止进程组再回收输出，必要时强制 kill。"""
    try:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        stdout, stderr = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        stdout, stderr = process.communicate()
    return stdout, stderr


def _run_timed_subprocess(command, env, timeout_seconds):
    """Run a profiling command with process-group cleanup on timeout."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=(os.name == "posix"),
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        stdout, stderr = _stop_and_drain(process)
    if timed_out:
        stderr = (stderr or "") + f"\nTIMEOUT after {timeout_seconds} seconds"
        return subprocess.CompletedProcess(command, 124, stdout or "", stderr)
    return subprocess.CompletedProcess(
        command, process.returncode, stdout or "", stderr or ""
    )


def _validated_performance_cases(out_dir, args=None):
    cases, source_record, golden_iterations, source_error = resolve_performance_cases(
        out_dir, args
    )
    if not cases:
        LOGGER.error("[ERROR] Failed to load performance cases: %s", source_error)
        return None
    case_ids = [str(case[0]).strip() for case in cases]
    if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
        LOGGER.error("[ERROR] performance case ids must be non-empty and unique")
        return None
    record_error = performance_case_source_error(source_record)
    if record_error:
        LOGGER.error("[ERROR] Invalid performance case source record: %s", record_error)
        return None
    LOGGER.info(
        "[INFO] Loaded %s performance cases from %s (sha256=%s)",
        len(cases), source_record.get("type"), source_record.get("sha256"),
    )
    if args is not None:
        args.golden_iterations = golden_iterations
        args.performance_cases = source_record
    return cases


def _validate_measurement_args(args, manifest=False):
    for field, minimum in (("warmup", 0), ("repeats", 1), ("retry", 0)):
        value = getattr(args, field, None)
        if value is None or value < minimum:
            LOGGER.error("[ERROR] --%s must be >= %s", field, minimum)
            return False
    if getattr(args, "device", None) is not None and args.device < 0:
        LOGGER.error("[ERROR] --device must be >= 0")
        return False
    if getattr(args, "timeline_timeout", 600) <= 0:
        LOGGER.error("[ERROR] --timeline-timeout must be > 0")
        return False
    if manifest and getattr(args, "seed", None) == 0:
        args.seed = None  # 0 视为未指定，manifest 协议固定 42
    if manifest and getattr(args, "seed", None) not in (None, 42):
        LOGGER.error(
            "[ERROR] manifest profiling requires the validated runner seed to stay fixed at 42; "
            "omit --seed or pass --seed=42"
        )
        return False
    return True


def _validate_case_selector(performance_cases, args):
    case_arg = getattr(args, "case_arg", None)
    case_env = getattr(args, "case_env", None)
    if case_arg and case_env:
        LOGGER.error("[ERROR] --case-arg and --case-env are mutually exclusive")
        return False
    if case_arg and (not case_arg.startswith("-") or case_arg == "--"):
        LOGGER.error("[ERROR] --case-arg must be a CLI option such as --case-id")
        return False
    if case_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", case_env):
        LOGGER.error("[ERROR] --case-env must be a valid environment variable name")
        return False
    if not getattr(args, "op_name", None):
        LOGGER.error(
            "[ERROR] compare/quick requires an exact --op-name; "
            "refusing to guess the target kernel from the longest AI Core row"
        )
        return False
    contract_error = profile_case_contract_error(performance_cases, args)
    if contract_error and not (case_arg or case_env):
        LOGGER.error("[ERROR] %s", contract_error)
        return False
    return True


def _validate_selector_runtime(out_dir, performance_cases, args, device_id):
    test_script, script_error = _find_test_script(out_dir, strict=True)
    if not test_script:
        LOGGER.error("[ERROR] %s", script_error)
        return False
    if (
        getattr(args, "case_arg", None)
        or getattr(args, "case_env", None)
        or uses_function_adapter(args, [case[0] for case in performance_cases])
    ):
        error = _selector_preflight(
            test_script, [case[0] for case in performance_cases], args, device_id
        )
    else:
        # A validated single-case runner needs no additional case selector.
        # It will be executed normally during profiling with its existing seed=42.
        error = None
    if error:
        LOGGER.error("[ERROR] %s", error)
        return False
    return True


def _validate_case_source_mode(args) -> int:
    """Validate a case source without launching a runner (used by batch preflight)."""
    out_dir = Path(args.output_dir).resolve()
    cases, source_record, _, error = resolve_performance_cases(out_dir, args)
    if error:
        LOGGER.error("[ERROR] Failed to load performance cases: %s", error)
        return 1
    record_error = performance_case_source_error(source_record)
    if record_error:
        LOGGER.error("[ERROR] Invalid performance case source record: %s", record_error)
        return 1
    LOGGER.info(
        "[INFO] Preflighted %s cases from %s (sha256=%s, path=%s)",
        len(cases), source_record.get("type"), source_record.get("sha256"),
        source_record.get("path"),
    )
    return 0


def _add_compare_header(lines, report):
    """Add the header section to the compare markdown report."""
    lines.append("# 性能评估结果")
    lines.append("")
    lines.append(f"- **Operator**: {report['task']}")
    lines.append(f"- **Device**: npu:{report['device_id']} (source={report['device_select_source']})")
    lines.append(f"- **Warmup**: {report['warmup']}")
    lines.append(f"- **Repeats**: {report['repeats']}")
    lines.append(f"- **Seed**: {report['seed']}")
    lines.append(f"- **Timing method**: {report['timing_method']}")
    lines.append(f"- **Timing source**: {report.get('timing_source_metric', 'unknown')}")
    lines.append(f"- **Target Op Name**: `{report.get('target_op_name', 'unknown')}`")
    selector = report.get("case_selector") or {}
    lines.append(
        f"- **Case selector**: `{selector.get('kind', 'unknown')}:{selector.get('name') or '-'}`"
    )
    lines.append(f"- **Profiling mode**: {report['profiling_mode']}")
    lines.append(f"- **Collection ID**: {report['collection_id']}")
    case_source = report.get("performance_cases") or {}
    lines.append(
        f"- **Performance cases**: `{case_source.get('type', 'unknown')}`; "
        f"source=`{case_source.get('path', 'unknown')}`; "
        f"sha256=`{case_source.get('sha256', 'unknown')}`"
    )
    golden = case_source.get("golden_diagnostic") or {}
    lines.append(f"- **Golden target source status**: `{golden.get('status', 'not_provided')}`")
    lines.append(
        f"- **Default target**: `{report['default_target_metric']} >= "
        f"{report['default_target_threshold']:.1f}` for every P0 case"
    )
    lines.append(
        f"- **Default target status**: `{report['default_target_status']}` "
        f"(valid_for_target_met={str(report['valid_for_target_met']).lower()})"
    )
    if report.get("n_default_target_cases", 0):
        lines.append(f"- **Comparison scope**: `{report['comparison_scope']}`")
        lines.append(
            f"- **Golden source**: `{report['golden_timing_method']}` "
            f"(iterations={report['golden_iterations']}, normalized per iteration)"
        )
        lines.append(
            "- **Ratio semantics**: this ratio is valid for the default Golden reference target; "
            "it is not baseline-to-final optimization speedup."
        )
    lines.append("")


def _add_per_case_table(lines, report):
    """Add the per-case comparison table."""
    if not report.get("per_case"):
        return
    lines.append("## Golden 默认目标对比")
    lines.append("")
    lines.append(
        "| Case | Shape | DType | PyPTO目标kernel(us) | "
        "Golden每迭代E2E(us) | 默认目标比值(E2E/kernel) |"
    )
    lines.append("| ---- | ----- | ----- | ------------- | -------- | -------------- |")
    for case in report["per_case"]:
        shape = case.get("shape", "?")
        dtype = case.get("dtype", "?")
        ref = case.get("ref_us")
        asc = case.get("asc_us")
        ratio = case.get("golden_reference_ratio")
        ref_str = f"{ref:.2f}" if ref is not None else "N/A"
        asc_str = f"{asc:.2f}" if asc is not None else "N/A"
        ratio_str = f"{ratio:.3f}" if ratio is not None else "N/A"
        lines.append(
            f"| {case['case']} | {shape} | {dtype} | {asc_str} | {ref_str} | "
            f"{ratio_str} |"
        )
    lines.append("")
    lines.append("### 重复采样证据")
    lines.append("")
    for case in report["per_case"]:
        samples = case.get("duration_samples_us") or []
        selected = case.get("selected_op_names") or []
        lines.append(
            f"- `{case['case']}`: samples_us={samples}; "
            f"selected_op_names={selected}; evidence=`"
            f"{case.get('deep_profile_dir') or case.get('asc_prof_dir') or 'quick-summary-only'}`"
        )
    lines.append("")


def _add_bound_routing_table(lines, report):
    """Render ratio-derived routing without promoting it to Roofline proof."""
    routed_cases = [
        case for case in report.get("per_case", [])
        if case.get("repeat_bound_diagnoses")
    ]
    if not routed_cases:
        return
    lines.append("## Pipe / Bound 路由（诊断，不是最终 Roofline 结论）")
    lines.append("")
    lines.append("| Case | Repeat 路由 | Scalar 候选 | Roofline 结论 | 搬运/计算流水证据 |")
    lines.append("| ---- | ----------- | ----------- | ------------- | ----------------- |")
    for case in routed_cases:
        diagnoses = case["repeat_bound_diagnoses"]
        routes = [str(item.get("routing_label", "unknown")) for item in diagnoses]
        scalar = any(item.get("scalar_route_triggered") is True for item in diagnoses)
        terminals = sorted({
            str(item.get("roofline_terminal_status", "insufficient_evidence"))
            for item in diagnoses
        })
        overlaps = sorted({
            str(item.get("pipeline_overlap_status", "unverified"))
            for item in diagnoses
        })
        lines.append(
            f"| {case['case']} | {routes} | {str(scalar).lower()} | "
            f"{terminals} | {overlaps} |"
        )
    lines.append("")
    lines.append(
        "> `op_summary` 的 pipe ratio 只用于选择下一项实验；它不包含事件先后关系，"
        "不能单独证明 compute/movement bound 或搬运与计算已经重叠。最终结论必须在 "
        "`PERFORMANCE_REPORT.md` 中引用工作量/流量模型及 timeline 或等价依赖证据。"
    )
    lines.append("")


def _add_summary_section(lines, report):
    """Add the summary and dtype tables."""
    lines.append("## PyPTO measurement summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| ---- | -- |")
    lines.append(f"| Cases | {report['n_cases_total']} |")
    lines.append(f"| Valid PyPTO measurements | {report['n_cases_valid']} |")
    if report.get("mean_asc_us") is not None:
        lines.append(f"| Mean PyPTO target-kernel (us) | {report['mean_asc_us']:.3f} |")
        lines.append(f"| Median PyPTO target-kernel (us) | {report['median_asc_us']:.3f} |")
        lines.append(f"| Total PyPTO target-kernel (us) | {report['total_asc_us']:.3f} |")
    lines.append(f"| Golden target cases | {report.get('n_default_target_cases', 0)} |")
    lines.append(f"| Default target threshold | {report['default_target_threshold']:.1f} |")
    lines.append(f"| Default target status | {report['default_target_status']} |")
    ratio_stats = report.get("golden_reference_ratio_stats") or {}
    if ratio_stats.get("geomean") is None:
        lines.append("")
        return
    lines.append(
        "| Mean Golden-E2E/PyPTO-kernel target ratio | "
        f"{ratio_stats['mean']:.3f} |"
    )
    met = sum(
        1 for c in report.get("per_case", [])
        if c.get("default_target_met") is True
    )
    missed = sum(
        1 for c in report.get("per_case", [])
        if c.get("default_target_met") is False
    )
    lines.append(f"| Target-met cases (ratio >=1) | {met} |")
    lines.append(f"| Target-missed cases (ratio <1) | {missed} |")
    lines.append("")

    dtype_groups = {}
    for case in report.get("per_case", []):
        dtype = case.get("dtype", "?")
        ratio = case.get("golden_reference_ratio")
        if ratio is not None:
            dtype_groups.setdefault(dtype, []).append(ratio)
    if dtype_groups:
        lines.append("### 按数据类型汇总")
        lines.append("")
        lines.append("| DType | 用例数 | 平均默认目标比值 | 达标(>=1) | 未达标(<1) |")
        lines.append("| ----- | ------ | ------------------- | ------------- | -------- |")
        for dtype, sps in sorted(dtype_groups.items()):
            mean_sp = statistics.mean(sps)
            better = sum(1 for sp in sps if sp >= 1)
            worse = sum(1 for sp in sps if sp < 1)
            lines.append(f"| {dtype} | {len(sps)} | {mean_sp:.3f} | {better} | {worse} |")
        lines.append("")


def _add_analysis_sections(lines, report):
    """Add the short analysis and deep bottleneck analysis sections."""
    lines.append("## 简短分析")
    lines.append("")
    ratio_stats = report.get("golden_reference_ratio_stats") or {}
    golden_status = (
        ((report.get("performance_cases") or {}).get("golden_diagnostic") or {})
        .get("status")
    )
    if ratio_stats.get("mean") is not None:
        lines.append(
            f"- Golden 每迭代 E2E / PyPTO target-kernel 平均目标比值为 "
            f"{ratio_stats['mean']:.3f}；默认理想参考为每个 P0 case 均不低于 "
            f"{report['default_target_threshold']:.1f}。该比值不是 baseline→final 优化加速比，"
            f"也不作为性能调优交付门禁。"
        )
    elif golden_status == "not_provided":
        lines.append("- Golden JSON 不存在；PyPTO 采集有效，默认理想参考状态为 unavailable，且不作为性能调优交付门禁。")
    else:
        lines.append("- Golden JSON 已联接，但当前参考比值不可复算；性能证据无效，修复后才能判断理想参考状态。")
    lines.append("- 详细瓶颈分析见 msprof 归档目录（op_summary_*.csv + summary.txt）。")
    lines.append("")

    lines.append("## 深度瓶颈分析")
    lines.append("")
    if report.get("deep_profile_round"):
        lines.append(f"- 深度采集轮次：`{report['deep_profile_round']}`")
        for case in report.get("per_case", []):
            if case.get("deep_profile_dir"):
                lines.append(
                    f"- `{case['case']}`：`{case['deep_profile_dir']}/summary.txt` "
                    "及七组 `op_summary_*.csv`"
                )
    else:
        lines.append(
            "- quick 模式仅用于筛选，没有七组 aic-metrics 深度归档；"
            "最终证据必须用同配置 `--compare` 重采。"
        )
    lines.append("")
    lines.append("")
    lines.append("")


def report_compare_to_markdown(report: Dict[str, Any]) -> str:
    lines = []
    _add_compare_header(lines, report)
    _add_per_case_table(lines, report)
    _add_bound_routing_table(lines, report)
    _add_summary_section(lines, report)
    _add_analysis_sections(lines, report)
    return "\n".join(lines)


def _text_header(lines: List[str], report: Dict[str, Any]) -> None:
    """追加文本报告的头部与表头。"""
    lines.append("=" * 100)
    lines.append(f"PyPTO Performance Collection (msprof): {report['task']}  "
                 f"(warmup={report['warmup']}, repeats={report['repeats']}, seed={report['seed']})")
    lines.append(f"Collection ID: {report['collection_id']}")
    source = report.get("performance_cases") or {}
    lines.append(
        f"Performance cases: {source.get('type', 'unknown')} "
        f"sha256={source.get('sha256', 'unknown')} path={source.get('path', 'unknown')}"
    )
    golden = source.get("golden_diagnostic") or {}
    lines.append(f"Golden target source status: {golden.get('status', 'not_provided')}")
    lines.append(
        f"Default target: {report['default_target_metric']} >= "
        f"{report['default_target_threshold']:.1f} for every P0 case | "
        f"status={report['default_target_status']} | "
        f"valid={report['valid_for_target_met']}"
    )
    if report.get("n_default_target_cases", 0):
        lines.append(
            "Comparison scope: golden_e2e_to_pypto_target_kernel "
            "(valid for the default target; NOT baseline-to-final optimization speedup)"
        )
    lines.append(
        f"Target Op Name: {report.get('target_op_name', 'unknown')} | "
        f"Timing source: {report.get('timing_source_metric', 'unknown')}"
    )
    selector = report.get("case_selector") or {}
    lines.append(
        f"Case selector: {selector.get('kind', 'unknown')}:{selector.get('name') or '-'}"
    )
    lines.append("=" * 100)
    lines.append(f"{'Case':<5} {'Shape':<35} {'dtype':<10} {'GoldenE2E':>12} {'PyPTOKernel':>12} {'TargetRatio':>12}")
    lines.append("-" * 100)


def _text_case_lines(lines: List[str], case: Dict[str, Any]) -> None:
    """追加单个 case 的文本行。"""
    shape = case.get("shape", "?")
    dtype = case.get("dtype", "?")
    ref_us = case.get("ref_us")
    asc_us = case.get("asc_us")
    ratio = case.get("golden_reference_ratio")
    if ref_us is not None and asc_us is not None and ratio is not None:
        lines.append(
            f"{case['case']:<5} {shape:<35} {dtype:<10} "
            f"{ref_us:>12.2f} {asc_us:>12.2f} {ratio:>9.3f}x"
        )
    else:
        ref_str = f"{ref_us:.2f}" if ref_us is not None else "N/A"
        asc_str = f"{asc_us:.2f}" if asc_us is not None else "N/A"
        ref_err = case.get("ref_error", "")
        asc_err = case.get("asc_error", "")
        lines.append(f"{case['case']:<5} {shape:<35} {dtype:<10} "
                     f"{ref_str:>12} {asc_str:>12} "
                     f"{'N/A':>10}  (ref_err={ref_err}, asc_err={asc_err})")
    lines.append(
        f"  samples_us={case.get('duration_samples_us') or []} "
        f"selected_op_names={case.get('selected_op_names') or []} "
        f"evidence={case.get('deep_profile_dir') or case.get('asc_prof_dir') or 'quick-summary-only'}"
    )
    diagnoses = case.get("repeat_bound_diagnoses") or []
    if diagnoses:
        routes = [item.get("routing_label", "unknown") for item in diagnoses]
        scalar = any(item.get("scalar_route_triggered") is True for item in diagnoses)
        terminals = sorted({
            item.get("roofline_terminal_status", "insufficient_evidence")
            for item in diagnoses
        })
        overlaps = sorted({
            item.get("pipeline_overlap_status", "unverified")
            for item in diagnoses
        })
        lines.append(
            "  bound_route=" + str(routes)
            + f" scalar_candidate={scalar} roofline={terminals} pipeline={overlaps}"
        )


def _text_summary_block(lines: List[str], report: Dict[str, Any]) -> None:
    """追加文本报告的汇总统计块。"""
    ratio_stats = report.get("golden_reference_ratio_stats") or {}
    if ratio_stats.get("geomean") is not None:
        lines.append("--- Golden default-target ratio (NOT baseline-to-final speedup) ---")
        lines.append(f"  Geomean : {ratio_stats['geomean']:.2f}x")
        lines.append(f"  Mean    : {ratio_stats['mean']:.2f}x")
        lines.append(f"  Median  : {ratio_stats['median']:.2f}x")
        lines.append(
            f"  Min/Max : {ratio_stats['min']:.2f}x / {ratio_stats['max']:.2f}x"
        )
        lines.append(
            f"  Golden target cases: {report['n_default_target_cases']}/{report['n_cases_total']}"
        )
    lines.append(
        f"  Valid PyPTO measurements: {report['n_cases_valid']}/{report['n_cases_total']}"
    )
    if report.get("mean_asc_us") is not None:
        lines.append("--- PyPTO target-kernel Task Duration (us) ---")
        if report.get("mean_ref_us") is not None:
            lines.append(
                f"  Golden E2E mean/median/total : {report['mean_ref_us']:.2f} / "
                f"{report['median_ref_us']:.2f} / {report['total_ref_us']:.2f}"
            )
        lines.append(
            f"  PyPTO mean/median/total : {report['mean_asc_us']:.2f} / "
            f"{report['median_asc_us']:.2f} / {report['total_asc_us']:.2f}"
        )
        if report.get("aggregate_golden_reference_ratio") is not None:
            lines.append(
                f"  Aggregate ratio (Σgolden E2E/ΣPyPTO kernel; not target criterion) : "
                f"{report['aggregate_golden_reference_ratio']:.2f}x"
            )
    if any(case.get("repeat_bound_diagnoses") for case in report.get("per_case", [])):
        lines.append(
            "Bound-route note: op_summary ratios are diagnostic only; terminal Roofline "
            "and overlap claims require a workload/traffic model plus timeline or "
            "equivalent dependency evidence in PERFORMANCE_REPORT.md."
        )


def report_compare_to_text(report: Dict[str, Any]) -> str:
    lines = []
    _text_header(lines, report)
    for case in report.get("per_case", []):
        _text_case_lines(lines, case)
    lines.append("-" * 100)
    _text_summary_block(lines, report)
    lines.append("=" * 100)

    return "\n".join(lines)


def _log_and_save_compare_reports(summary, out_dir, speedups, n_cases):
    """Log summary results and save JSON/log/Markdown reports."""
    LOGGER.info("-" * 100)
    if speedups:
        ratio_stats = summary["golden_reference_ratio_stats"]
        LOGGER.info("--- Golden default-target ratio (NOT baseline-to-final speedup) ---")
        LOGGER.info(f"  Geomean : {ratio_stats['geomean']:.2f}x")
        LOGGER.info(f"  Mean    : {ratio_stats['mean']:.2f}x")
        LOGGER.info(f"  Median  : {ratio_stats['median']:.2f}x")
        LOGGER.info(
            f"  Min/Max : {ratio_stats['min']:.2f}x / {ratio_stats['max']:.2f}x"
        )
        LOGGER.info(f"  Valid   : {len(speedups)}/{n_cases}")
        LOGGER.info(
            "  Target  : %s (threshold=%.1f for every P0 case)",
            summary["default_target_status"], summary["default_target_threshold"],
        )
    else:
        LOGGER.info("--- Golden default target unavailable; PyPTO profiling remains valid ---")
    if summary.get("mean_asc_us") is not None:
        LOGGER.info("--- PyPTO target-kernel Task Duration (us) ---")
        if summary.get("mean_ref_us") is not None:
            LOGGER.info(
                f"  Golden E2E mean/median/total : {summary['mean_ref_us']:.2f} / "
                f"{summary['median_ref_us']:.2f} / {summary['total_ref_us']:.2f}"
            )
        LOGGER.info(
            f"  PyPTO mean/median/total : {summary['mean_asc_us']:.2f} / "
            f"{summary['median_asc_us']:.2f} / {summary['total_asc_us']:.2f}"
        )
        if summary.get("aggregate_golden_reference_ratio") is not None:
            LOGGER.info(
                f"  Aggregate ratio (Σgolden E2E/ΣPyPTO kernel; not target criterion) : "
                f"{summary['aggregate_golden_reference_ratio']:.2f}x"
            )
    LOGGER.info("=" * 100)

    # 保存 JSON 报告
    filenames = (
        ("quick_performance.json", "quick_performance.log", "quick_perf_report.md")
        if summary.get("profiling_mode") == "quick"
        else ("performance.json", "performance.log", "perf_report.md")
    )
    json_path, log_path, md_path = (out_dir / name for name in filenames)
    _atomic_write_text(
        json_path, json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    LOGGER.info(f"\n[INFO] JSON report saved to: {json_path}")

    # 保存打屏日志
    _atomic_write_text(log_path, report_compare_to_text(summary))
    LOGGER.info(f"[INFO] Console report saved to: {log_path}")

    # 保存 Markdown 报告
    md = report_compare_to_markdown(summary)
    _atomic_write_text(md_path, md)
    LOGGER.info(f"[INFO] Markdown report saved to: {md_path}")


def _log_compare_header(out_dir, args):
    """Log the compare mode header."""
    LOGGER.info("=" * 100)
    effective_seed = args.seed if args.seed is not None else 42
    seed_source = "CLI override" if args.seed is not None else "runner default"
    LOGGER.info(f"PyPTO Performance Collection (msprof): {out_dir.name}  "
          f"(warmup={args.warmup}, repeats={args.repeats}, "
          f"seed={effective_seed}, source={seed_source})")
    LOGGER.info(
        "Target Op Name=%s | timing source=%s | selector=%s",
        args.op_name,
        "task_time" if getattr(args, "quick", False) else "PipeUtilization",
        getattr(args, "case_arg", None) or getattr(args, "case_env", None) or (
            "PERFORMANCE_CASES.test_function"
            if _manifest_case_functions(args) else "single_case"
        ),
    )
    LOGGER.info("=" * 100)
    LOGGER.info(f"{'Case':<5} {'Shape':<35} {'dtype':<10} {'GoldenE2E':>12} {'PyPTOKernel':>12} {'TargetRatio':>12}")
    LOGGER.info("-" * 100)
