# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Evidence gate between declaration normalization and optimization.

Resolving declarations without editing source has no runtime effect. A changed
source needs a separate correctness and latency comparison, bound to both exact
sources and the selected case context. Static admissibility alone is not that
evidence. This module consumes measured evidence; it never invents measurements.
"""
import hashlib
import json
import math
import statistics
from pathlib import Path
from pro_persistence import atomic_write


def original_source(op_dir, source=None):
    """Pin the first preparation input; retries never replace its bytes."""
    if not op_dir:
        return None
    path = Path(op_dir) / "optimization/normalization_original.py"
    if source is not None and not path.exists():
        atomic_write(path, source)
    return path.read_bytes().decode("utf-8") if path.exists() else None


def source_hash(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def context_hash(context):
    return hashlib.sha256(json.dumps(context or {}, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def check(before, after, context=None, evidence=None, tolerance=0.03, evaluation=None):
    """
    Require paired measured equivalence when normalization changes bytes.

    The 95% confidence interval of paired log latency ratios must fit wholly within
    the declared relative tolerance, rather than merely failing to detect a change.
    At least five pairs are required. Correctness must cover the same named cases.
    """
    report = {"ok": False, "before_sha256": source_hash(before),
              "after_sha256": source_hash(after),
              "case_context_sha256": context_hash(context), "reasons": []}
    if before == after:
        return dict(report, ok=True, mode="unchanged_source",
                    evidence="Declaration inspection only; runtime source bytes are identical.")
    report["mode"] = "measured_normalization"
    if not isinstance(evidence, dict):
        report["reasons"].append("Changed normalization requires correctness and paired latency evidence.")
        return report
    try:
        _validate_evidence_identity(evidence, report, tolerance, evaluation)
        rows = [evidence.get("before", {}), evidence.get("after", {})]
        hashes = [report["before_sha256"], report["after_sha256"]]
        cases = _correctness_cases(rows, hashes, context)
        a, b = _paired_latencies(rows, cases)
        report.update(relative_tolerance=tolerance, correctness_case_ids=cases,
                      device=evidence["device"], latency_by_case={})
        for case in cases:
            result = latency_equivalence(a[case], b[case], tolerance)
            report.get("latency_by_case")[case] = result
            if not result["ok"]:
                raise ValueError("%s: latency equivalence is not established within the declared tolerance" % case)
        if len(cases) == 1:
            report.update({key: value for key, value in report.get("latency_by_case")[cases[0]].items()
                           if key != "ok"})
    except (ValueError, TypeError, OverflowError) as exc:
        report["reasons"].append(str(exc))
        return report
    report["ok"] = True
    return report


def _validate_evidence_identity(evidence, report, tolerance, evaluation=None):
    if not (0 < tolerance <= 0.05):
        raise ValueError("normalization tolerance must be positive and at most 0.05")
    kind = "device"
    if evidence.get("schema_version") != 1 or evidence.get("measurement_kind") != kind:
        raise ValueError("normalization evidence must use schema_version 1 and the campaign instrument")
    if evidence.get("case_context_sha256") != report["case_context_sha256"]:
        raise ValueError("normalization evidence belongs to a different case context")
    device = evidence.get("device")
    command = evidence.get("test_command")
    has_command = isinstance(command, str) and command.strip()
    if device is None or not str(device).strip() or not has_command:
        raise ValueError("normalization evidence must identify the device and test command")
    if not isinstance(evaluation, dict) or evaluation.get("mode", "npu") != "npu":
        raise ValueError("changed normalization requires the campaign NPU measurement contract")
    if str(device) != str(evaluation.get("device", "")):
        raise ValueError("normalization evidence device differs from the campaign device")
    if "argv" in evaluation:
        from evaluation.measure_latency import check_campaign_command
        if not {"op_dir", "cwd", "env"} <= evaluation.keys():
            raise ValueError("incomplete campaign command contract")
        check_campaign_command(command, evaluation["op_dir"], evaluation)
    elif command != evaluation.get("test_command"):
        raise ValueError("normalization evidence test command differs from the campaign command")


def _source_correctness_cases(row, expected):
    if not isinstance(row, dict):
        raise ValueError("normalization evidence source entries must be objects")
    if row.get("source_sha256") != expected:
        raise ValueError("normalization evidence source hash does not match")
    correctness = row.get("correctness", {})
    if not isinstance(correctness, dict):
        raise ValueError("normalization correctness must be an object")
    cases = correctness.get("case_ids")
    if correctness.get("all_passed") is not True or not isinstance(cases, list) or not cases:
        raise ValueError("both sources need passing correctness for named cases")
    if not all(isinstance(case, str) and case for case in cases) or len(cases) != len(set(cases)):
        raise ValueError("both sources need passing correctness for named cases")
    return cases


def _correctness_cases(rows, hashes, context):
    before = _source_correctness_cases(rows[0], hashes[0])
    after = _source_correctness_cases(rows[1], hashes[1])
    if set(before) != set(after):
        raise ValueError("normalization correctness case coverage differs")
    selected_case = (context or {}).get("__panko_case_id__")
    if selected_case and selected_case not in before:
        raise ValueError("normalization correctness does not cover the selected case")
    return before


def _paired_latencies(rows, cases):
    before, after = [row.get("latency_samples_us") for row in rows]
    if len(cases) == 1 and isinstance(before, list) and isinstance(after, list):
        before, after = {cases[0]: before}, {cases[0]: after}
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("latency evidence must provide paired samples separately for every correctness case")
    if set(before) != set(cases) or set(after) != set(cases):
        raise ValueError("latency evidence must provide paired samples separately for every correctness case")
    return before, after


def latency_equivalence(before, after, tolerance):
    if not isinstance(before, list) or not isinstance(after, list) or len(before) != len(after):
        raise ValueError('normalization needs at least five paired latency samples per case')
    if len(before) < 5:
        raise ValueError('normalization needs at least five paired latency samples per case')
    if not all(_positive_latency(x) for x in before + after):
        raise ValueError("latency samples must be positive finite numbers")
    ratios = [math.log(y) - math.log(x) for x, y in zip(before, after)]
    # Conservative Student t bounds, rounded upward, using lower-DF buckets.
    df = len(ratios) - 1

    critical = next(_critical_values_for_df(iter([(120, 1.98), (60, 2.001), (30, 2.043),
                    (20, 2.086), (10, 2.229), (5, 2.571), (4, 2.777)]), df))
    center = statistics.mean(ratios)
    radius = critical * statistics.stdev(ratios) / math.sqrt(len(ratios))
    interval = [math.exp(center - radius), math.exp(center + radius)]
    return {"ok": interval[0] >= 1 - tolerance and interval[1] <= 1 + tolerance,
            "ratio_confidence_interval_95": interval, "paired_samples": len(ratios)}


def _critical_values_for_df(iterable, df):
    for (lower, v) in iterable:
        if not (df >= lower):
            continue
        yield v




def _positive_latency(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value > 0
