#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Extract agent-friendly Timeline / Core Occupancy / Roofline digests from an
msopprof-visualization collection directory.

Reuses the original skill's parsers (visualize_common / visualize_parsers) via
sys.path injection; does not reimplement trace/bin parsing.

Usage:
    python3 extract_visual_analysis.py <collection_dir> [--output-dir <dir>] \
        [--max-events N] [--vis-skill-root <path>] [--expect-iterations N]

Outputs VISUAL_ANALYSIS.json (schema pypto-pro-visual-analysis/v1; field dictionary in
references/perf-analysis/enhanced-metrics-reference.md, normative constraints in
references/perf-analysis/evidence-protocol.md) and VISUAL_ANALYSIS.md (agent-readable
summary) into --output-dir (default: the collection dir).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

_LOGGER = logging.getLogger(__name__)

SCHEMA = "pypto-pro-visual-analysis/v1"

TID_MAP = {"CUBE": "MAC", "VECTOR": "VEC", "FIXP": "FIXPIPE"}

LOW_UTIL_GAP_PCT = 30.0
IMBALANCE_RATIO_GATE_PCT = 10.0


def _resolve_vis_scripts(explicit: Optional[str]) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    for env in ("VIS_SKILL_ROOT", "CANNBOT_CONFIG_ROOT"):
        value = os.environ.get(env)
        if not value:
            continue
        base = Path(value)
        candidates.append(base if (base / "scripts").is_dir() else base / "ops-profiling" / "msopprof-visualization")
        candidates.append(base / "skills" / "ops-profiling" / "msopprof-visualization")
    for cand in candidates:
        scripts = cand / "scripts"
        if (scripts / "visualize_parsers.py").is_file():
            return scripts.resolve()
    raise RuntimeError(
        "ERROR: msopprof-visualization scripts not found; pass --vis-skill-root "
        "or set CANNBOT_CONFIG_ROOT."
    )


def _import_parsers(scripts_dir: Path):
    sys.path.insert(0, str(scripts_dir))
    import visualize_common  # type: ignore
    import visualize_parsers  # type: ignore

    return visualize_common, visualize_parsers


def _round(value, digits=4):
    return round(value, digits) if isinstance(value, (int, float)) else value


def _median(values: list):
    ordered = sorted(values)
    if len(ordered) % 2:
        return ordered[len(ordered) // 2]
    return (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2


def _coefficient_of_variation(values: list) -> Optional[float]:
    if not values:
        return None
    mean = sum(values) / len(values)
    if mean == 0:
        return None
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return _round((variance ** 0.5) / mean, 3)


def _events_by_core(lanes: list) -> dict:
    core_events: dict = {}
    for lane in lanes:
        core = str(lane.get("pid", "")).split(".")[0]
        by_tid = core_events.setdefault(core, {})
        by_tid.setdefault(str(lane.get("tid")), []).extend(lane.get("events", []))
    return core_events


def _dominant_tid(events_by_tid: dict) -> Optional[str]:
    compute = [t for t in ("MAC", "VEC") if events_by_tid.get(t)]
    pool = compute or [t for t, ev in events_by_tid.items() if ev]
    if not pool:
        return None
    return max(pool, key=lambda t: sum(e.get("duration", 0.0) for e in events_by_tid[t]))


def _reference_lane(core_events: dict):
    best = None
    for core, by_tid in core_events.items():
        tid = _dominant_tid(by_tid)
        if tid is None:
            continue
        events = sorted(by_tid.get(tid, []), key=lambda e: e.get("start", 0.0))
        if best is None or len(events) > len(best[2]):
            best = (core, tid, events, by_tid)
    return best


def _split_bursts(events: list, gap_factor: float = 3.0):
    starts = [e.get("start", 0.0) for e in events]
    intervals = sorted(starts[i + 1] - starts[i] for i in range(len(starts) - 1))
    if not intervals:
        return None, starts, 0.0
    med_interval = intervals[len(intervals) // 2]
    if med_interval <= 0:
        return None, starts, med_interval
    gap_cut = med_interval * gap_factor
    bursts = [[events[0]]]
    for i in range(1, len(events)):
        if starts[i] - starts[i - 1] > gap_cut:
            bursts.append([])
        bursts[-1].append(events[i])
    return bursts, starts, med_interval


def _overlap_busy(events: list, w0: float, w1: float) -> float:
    busy = 0.0
    for e in events:
        s, d = e.get("start", 0.0), e.get("duration", 0.0)
        ov = min(s + d, w1) - max(s, w0)
        if ov > 0:
            busy += ov
    return busy


def _per_iter_pipe_busy(by_tid: dict, burst_starts: list) -> dict:
    per_iter: dict = {}
    for i in range(len(burst_starts) - 1):
        w0, w1 = burst_starts[i], burst_starts[i + 1]
        for pipe, evs in by_tid.items():
            per_iter.setdefault(pipe, []).append(_overlap_busy(evs, w0, w1))
    return {p: _round(_median(v)) for p, v in sorted(per_iter.items())}


def _burst_end(burst: list) -> float:
    return max(e.get("start", 0.0) + e.get("duration", 0.0) for e in burst)


def _periodicity_result(reference_lane, bursts, by_tid, core_events, expect_iterations) -> dict:
    burst_busy = [sum(e.get("duration", 0.0) for e in b) for b in bursts]
    burst_starts = [b[0].get("start", 0.0) for b in bursts]
    periods = [burst_starts[i + 1] - burst_starts[i] for i in range(len(burst_starts) - 1)]
    med_period = sorted(periods)[len(periods) // 2]
    period_cv = _coefficient_of_variation(periods)
    burst_cv = _coefficient_of_variation(burst_busy)
    stable = (period_cv if period_cv is not None else 9) <= 0.5
    stable = stable and (burst_cv if burst_cv is not None else 9) <= 0.5
    source_check = {"expected_iterations": expect_iterations, "match": None}
    if expect_iterations is not None:
        source_check["match"] = abs(len(bursts) - expect_iterations) <= 1
    return {
        "status": "periodic" if stable else "bursty_irregular",
        "reference_lane": reference_lane,
        "sampled_cores": sorted(core_events),
        "iteration_count": len(bursts),
        "period_us": _round(med_period),
        "period_cv": period_cv,
        "burst_busy_cv": burst_cv,
        "steady_window": {
            "start": _round(burst_starts[0]),
            "end": _round(_burst_end(bursts[-1])),
        },
        "per_iter_pipe_busy_us": _per_iter_pipe_busy(by_tid, burst_starts),
        "source_check": source_check,
        "notes": [
            "burst = events separated by >3x median interval on the reference lane; one burst ~ one loop iteration",
            "sample-based trace: at most 6 cores, do not extrapolate; periodicity does not prove pipe overlap",
        ],
    }


def _detect_periodicity(lanes: list, expect_iterations: Optional[int]) -> dict:
    """Detect steady loop iterations from burst structure on a reference pipe lane.

    Bursts are dense event clusters separated by gaps > 3x the median interval;
    each burst approximates one tile iteration on the reference pipe. Source-derived
    loop trip count (--expect-iterations) cross-checks the detected burst count.
    """
    core_events = _events_by_core(lanes)
    if not core_events:
        return {"status": "insufficient_events", "reason": "no lanes"}
    best = _reference_lane(core_events)
    if best is None:
        return {"status": "insufficient_events", "reason": "no reference pipe events"}
    core, tid, events, by_tid = best
    if len(events) < 6:
        return {"status": "insufficient_events", "reason": f"reference lane {core}/{tid} has only {len(events)} events"}
    bursts, _starts, med_interval = _split_bursts(events)
    if med_interval <= 0:
        return {"status": "insufficient_events", "reason": "degenerate event timestamps"}
    if bursts is None or len(bursts) < 3:
        count = 0 if bursts is None else len(bursts)
        return {
            "status": "insufficient_events",
            "reason": f"only {count} burst(s) detected",
            "reference_lane": f"{core}/{tid}",
        }
    return _periodicity_result(f"{core}/{tid}", bursts, by_tid, core_events, expect_iterations)


def _union_intervals(events: list) -> list:
    spans = sorted((e.get("start", 0.0), e.get("start", 0.0) + e.get("duration", 0.0)) for e in events)
    merged: list = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def _cluster_durations(items: list, ratio: float = 1.3) -> list:
    """Split same-kind segments into clusters where durations stay within `ratio`."""
    clusters: list = []
    for seg in sorted(items, key=lambda s: s["dur"]):
        if clusters and seg["dur"] <= clusters[-1][-1]["dur"] * ratio:
            clusters[-1].append(seg)
        else:
            clusters.append([seg])
    return clusters


def _pipe_util_in_window(lanes_by_tid: dict, w0: float, w1: float) -> dict:
    """Per-pipe busy/utilization within [w0, w1).

    lanes_by_tid maps tid -> list of per-lane event lists (e.g. veccore0 and
    veccore1 each contribute a VECTOR lane). Utilization is the mean across
    lanes of that tid, so multi-subcore sides do not inflate past 100%.
    """
    dur = w1 - w0
    out: dict = {}
    for tid, lane_events in lanes_by_tid.items():
        per_lane = [_overlap_busy(evs, w0, w1) for evs in lane_events]
        busy_mean = sum(per_lane) / len(per_lane) if per_lane else 0.0
        if busy_mean > 0:
            util_pct = _round(busy_mean / dur * 100, 2) if dur > 0 else None
            out[tid] = {"busy_us": _round(busy_mean), "util_pct": util_pct}
    return out


def _merge_lane_events(lanes_by_tid: dict) -> list:
    events: list = []
    for lane_events in lanes_by_tid.values():
        for evs in lane_events:
            events.extend(evs)
    return events


def _interval_bounds(intervals_list) -> list:
    bounds = set()
    for intervals in intervals_list:
        for seg in intervals:
            bounds.update(seg)
    return sorted(bounds)


def _covers(intervals: list, a: float, b: float) -> bool:
    for s, e in intervals:
        if s < b and a < e:
            return True
    return False


def _segment_kind(in_cube: bool, in_vector: bool) -> str:
    if in_cube and in_vector:
        return "overlapped"
    if in_cube:
        return "cube_exposed"
    if in_vector:
        return "vector_exposed"
    return "seam"


def _merge_runs(raw: list) -> list:
    merged: list = []
    for s in raw:
        if merged and merged[-1]["kind"] == s["kind"]:
            merged[-1]["dur"] += s["dur"]
        else:
            merged.append(dict(s))
    return merged


def _segment_pipe_tids(kind: str, sides: dict) -> dict:
    if kind == "cube_exposed":
        return {t: list(lane_events) for t, lane_events in sides.get("cube", {}).items()}
    if kind == "vector_exposed":
        return {t: list(lane_events) for t, lane_events in sides.get("vector", {}).items()}
    tids: dict = {}
    for side in ("cube", "vector"):
        for t, lane_events in sides.get(side, {}).items():
            tids.setdefault(f"{side}:{t}", []).extend(lane_events)
    return tids


def _core_segments(core: str, sides: dict) -> list:
    cu = _union_intervals(_merge_lane_events(sides.get("cube", {})))
    ve = _union_intervals(_merge_lane_events(sides.get("vector", {})))
    bounds = _interval_bounds((cu, ve))
    raw: list = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a <= 0:
            continue
        kind = _segment_kind(_covers(cu, a, b), _covers(ve, a, b))
        raw.append({"core": core, "kind": kind, "start": a, "dur": b - a})
    merged = _merge_runs(raw)
    for s in merged:
        if s["kind"] in ("seam",):
            s["pipes"] = {}
        else:
            s["pipes"] = _pipe_util_in_window(_segment_pipe_tids(s["kind"], sides), s["start"], s["start"] + s["dur"])
    return merged


def _cube_vector_cores(lanes: list) -> dict:
    cores: dict = {}
    for lane in lanes:
        pid = str(lane.get("pid", ""))
        side = _lane_side(pid)
        if side is None:
            continue
        core = pid.split(".")[0]
        by_side = cores.setdefault(core, {}).setdefault(side, {})
        by_side.setdefault(str(lane.get("tid")), []).append(lane.get("events", []))
    return cores


def _lane_side(pid: str) -> Optional[str]:
    if pid.endswith("cubecore"):
        return "cube"
    if ".veccore" in pid:
        return "vector"
    return None


def _cluster_pipe_median(cluster: list) -> dict:
    pipe_utils: dict = {}
    for c in cluster:
        for t, p in (c.get("pipes") or {}).items():
            pipe_utils.setdefault(t, []).append(p["util_pct"])
    return {t: _round(_median(v), 2) for t, v in sorted(pipe_utils.items())}


def _steady_cluster_row(kind: str, cluster: list, total_window: Optional[float]) -> dict:
    med = _median([c["dur"] for c in cluster])
    load_rate = _round(med * len(cluster) / total_window, 4) if total_window else None
    if kind in ("cube_exposed", "vector_exposed"):
        side, row_kind, uncoverable = kind.replace("_exposed", ""), "exposed", "requires_dag_evidence"
    else:
        side, row_kind, uncoverable = kind, kind, "n/a"
    return {
        "side": side,
        "kind": row_kind,
        "region": "steady_loop",
        "load_rate": load_rate,
        "per_iter_us": _round(med),
        "occurrences": len(cluster),
        "pipe_util_pct": _cluster_pipe_median(cluster),
        "uncoverable": uncoverable,
    }


def _prologue_row(s: dict, total_window: Optional[float]) -> dict:
    kind = s["kind"]
    if kind == "seam":
        row_kind = "seam"
    elif "exposed" in kind:
        row_kind = "exposed"
    else:
        row_kind = "overlapped"
    return {
        "side": kind.split("_")[0],
        "kind": row_kind,
        "region": "prologue_or_drain",
        "load_rate": _round(s["dur"] / total_window, 4) if total_window else None,
        "per_iter_us": _round(s["dur"]),
        "occurrences": 1,
        "pipe_util_pct": {t: p["util_pct"] for t, p in sorted((s.get("pipes") or {}).items())},
        "uncoverable": "requires_dag_evidence" if "exposed" in kind else "n/a",
    }


def _steady_load_segments(segments: list, win: dict, total_window: Optional[float]) -> list:
    in_loop: list = []
    others: list = []
    for s in segments:
        if win["start"] <= s["start"] < win["end"]:
            in_loop.append(s)
        else:
            others.append(s)
    out: list = []
    for kind in ("cube_exposed", "vector_exposed", "overlapped", "seam"):
        for cluster in _cluster_durations([s for s in in_loop if s["kind"] == kind]):
            out.append(_steady_cluster_row(kind, cluster, total_window))
    for s in others:
        out.append(_prologue_row(s, total_window))
    return out


def _core_pipe_usage(segs: list) -> dict:
    tot = sum(x["dur"] for x in segs)
    if not tot:
        return {}
    acc: dict = {}
    for x in segs:
        for t, p in (x.get("pipes") or {}).items():
            acc[t] = acc.get(t, 0.0) + p["util_pct"] * x["dur"]
    return {t: v / tot for t, v in acc.items()}


def _average_core_pipes(kind: str, per_core: dict) -> dict:
    per_core_pipes = [_core_pipe_usage(segs.get(kind, [])) for segs in per_core.values()]
    tids = set()
    for pc in per_core_pipes:
        tids.update(pc)
    pipes_avg: dict = {}
    for t in sorted(tids):
        total = sum(pc.get(t, 0.0) for pc in per_core_pipes)
        pipes_avg[t] = _round(total / len(per_core_pipes), 2) if per_core_pipes else 0.0
    return pipes_avg


def _whole_window_row(kind: str, per_core: dict, total_window: Optional[float]) -> dict:
    core_totals = [sum(x["dur"] for x in segs.get(kind, [])) for segs in per_core.values()]
    avg_us = sum(core_totals) / len(core_totals) if core_totals else 0.0
    occurrences = 0
    for segs in per_core.values():
        occurrences += len(segs.get(kind, []))
    return {
        "side": kind.split("_")[0],
        "kind": kind if kind in ("seam", "overlapped") else "exposed",
        "region": "whole_window",
        "load_rate": _round(avg_us / total_window, 4) if total_window else None,
        "per_iter_us": _round(avg_us),
        "occurrences": occurrences,
        "pipe_util_pct": {t: v for t, v in _average_core_pipes(kind, per_core).items() if v > 0},
        "uncoverable": "requires_dag_evidence" if "exposed" in kind else "n/a",
    }


def _whole_window_load_segments(segments: list, total_window: Optional[float]) -> list:
    per_core: dict = {}
    for s in segments:
        per_core.setdefault(s["core"], {}).setdefault(s["kind"], []).append(s)
    out: list = []
    for kind in sorted({s["kind"] for s in segments}):
        out.append(_whole_window_row(kind, per_core, total_window))
    return out


def _exposed_instance(s: dict, period: Optional[float], total_window: Optional[float]) -> dict:
    if period:
        load_rate = _round(s["dur"] / period, 4)
    elif total_window:
        load_rate = _round(s["dur"] / total_window, 4)
    else:
        load_rate = None
    return {
        "core": s["core"],
        "side": s["kind"].split("_")[0],
        "start": _round(s["start"]),
        "dur_us": _round(s["dur"]),
        "load_rate": load_rate,
        "pipe_util_pct": {t: p["util_pct"] for t, p in sorted((s.get("pipes") or {}).items())},
        "uncoverable": "requires_dag_evidence",
    }


def _top_exposed_instances(segments: list, period: Optional[float], total_window: Optional[float]) -> list:
    exposed = sorted(
        (s for s in segments if s["kind"] in ("cube_exposed", "vector_exposed")),
        key=lambda s: -s["dur"],
    )[:3]
    return [_exposed_instance(s, period, total_window) for s in exposed]


def _segment_load_result(out, top_instances, cadence, total_window) -> dict:
    exposed = [s for s in out if s["kind"] == "exposed" and s.get("load_rate") is not None]
    largest = max(exposed, key=lambda s: s["load_rate"]) if exposed else None
    largest_instance = max(top_instances, key=lambda s: s["dur_us"]) if top_instances else None
    return {
        "available": True,
        "cadence": cadence,
        "total_window_us": _round(total_window),
        "segments": out,
        "top_exposed_instances": top_instances,
        "largest_exposed_uncovered": largest_instance,
        "largest_exposed_cluster": largest,
        "notes": [
            "trace proves only 'not covered by the other side'; uncoverability requires "
            "Tile DAG evidence (pipeline_evidence rules)",
            "steady-loop segments (exposed/overlapped/seam) grouped by (kind, "
            "duration-cluster) signature; load_rate = per_iter_us x occurrences / "
            "total_window (cumulative share: a short but most-frequent segment is "
            "the optimization focus)",
            "segments rows are per-signature aggregates; top_exposed_instances ranks concrete segment intervals",
        ],
    }


def _segment_load(lanes: list, periodicity: dict, total_window: Optional[float]) -> dict:
    """Partition each core's cube/vector coverage into exposed/overlapped/seam segments.

    Steady-loop rows aggregate repeated same-signature segments:
    load_rate = per_iter_us x occurrences / total_window (cumulative share over
    the whole window, matching the field dictionary and guide definition).
    Prologue/drain and aperiodic rows are single segments with
    load_rate = duration / total_window. The trace proves only "not covered by
    the other side"; uncoverability needs Tile DAG evidence (evidence-protocol
    pipeline_evidence rules).
    """
    cores = _cube_vector_cores(lanes)
    if not cores:
        return {"available": False, "reason": "no cube/vector lanes in timeline"}
    segments: list = []
    for core, sides in cores.items():
        segments.extend(_core_segments(core, sides))
    per = periodicity or {}
    win = per.get("steady_window") or {}
    period = per.get("period_us")
    steady = bool(per.get("status") == "periodic" and period and win.get("start") is not None)
    if steady:
        out = _steady_load_segments(segments, win, total_window)
    else:
        out = _whole_window_load_segments(segments, total_window)
    top_instances = _top_exposed_instances(segments, period if steady else None, total_window)
    cadence = {"period_us": period, "steady_window": win} if steady else None
    return _segment_load_result(out, top_instances, cadence, total_window)


def _lane_digest_row(lane: dict, events: list, busy: float, window) -> dict:
    return {
        "pid": lane.get("pid"),
        "tid": lane.get("tid"),
        "events": len(events),
        "busy": _round(busy),
        "busy_pct_of_window": _round(busy / window * 100, 2) if window else None,
    }


def _accumulate_pipe(pipes: dict, tid, events: list, busy: float) -> None:
    agg = pipes.setdefault(tid, {"events": 0, "busy": 0.0, "longest_event": 0.0, "lane_count": 0})
    agg["events"] += len(events)
    agg["busy"] += busy
    agg["lane_count"] += 1
    longest = max((e.get("duration", 0.0) for e in events), default=0.0)
    agg["longest_event"] = max(agg["longest_event"], longest)


def _finalize_pipe_agg(agg: dict, window) -> None:
    agg["busy"] = _round(agg["busy"])
    agg["longest_event"] = _round(agg["longest_event"])
    denom = window * agg["lane_count"] if window else None
    agg["avg_busy_pct_per_lane"] = _round(agg["busy"] / denom * 100, 2) if denom else None
    del agg["lane_count"]


def _build_lanes_and_pipes(raw_lanes: list, window):
    pipes: dict = {}
    lanes: list = []
    for lane in raw_lanes:
        events = lane.get("events", [])
        busy = sum(e.get("duration", 0.0) for e in events)
        lanes.append(_lane_digest_row(lane, events, busy, window))
        _accumulate_pipe(pipes, lane.get("tid"), events, busy)
    for agg in pipes.values():
        _finalize_pipe_agg(agg, window)
    return lanes, pipes


def _timeline_digest(parsed: dict, expect_iterations: Optional[int] = None) -> dict:
    if not parsed.get("available"):
        return {"available": False, "reason": "no trace events parsed from trace.json / visualize_data.bin"}
    raw_lanes = [
        {**lane, "tid": TID_MAP.get(str(lane.get("tid")), lane.get("tid"))}
        for lane in parsed.get("lanes", [])
    ]
    window = _round(parsed.get("duration"))
    lanes, pipes = _build_lanes_and_pipes(raw_lanes, window)
    periodicity = _detect_periodicity(raw_lanes, expect_iterations)
    return {
        "available": True,
        "timeline_type": parsed.get("timeline_type"),
        "title": parsed.get("title"),
        "window": {"start": _round(parsed.get("start")), "end": _round(parsed.get("end")), "duration": window},
        "event_count": parsed.get("event_count"),
        "embedded_event_count": parsed.get("embedded_event_count"),
        "pipes": pipes,
        "lanes": lanes,
        "periodicity": periodicity,
        "segment_load": _segment_load(raw_lanes, periodicity, window),
        "notes": [
            "Pipe timeline is sample-based and shows at most 6 cores regardless of launched cores.",
            "ts/dur units follow the trace export (microseconds on current 950PR exports).",
            "tid renamed from trace natives: CUBE->MAC, VECTOR->VEC, FIXP->FIXPIPE.",
        ],
    }


def _group_balance(lanes: list, busy_key: str) -> dict:
    """Rank lanes of one subcore type by the busy metric and locate under-utilized cores.

    Within one subcore type, utilization = busy / max(busy). A lane whose
    utilization lags the busiest lane by more than 30% is a low-utilization
    core. The group metric is 核间负载不均衡率 = share of low-utilization
    lanes; above 10% the group is imbalanced.
    """
    usable = [lane for lane in lanes if isinstance((lane.get("metrics") or {}).get(busy_key), (int, float))]
    if not usable:
        return {"busy_metric": busy_key, "lane_count": len(lanes)}
    values = [lane["metrics"][busy_key] for lane in usable]
    lo, hi = min(values), max(values)
    med = _median(values)
    diff_pct = (hi - lo) / hi * 100 if hi else None
    ranked = sorted(usable, key=lambda lane: -lane["metrics"][busy_key])
    low_cut = hi * (1 - LOW_UTIL_GAP_PCT / 100) if hi else None
    low_lanes: list = []
    for lane in usable:
        if low_cut is not None and lane["metrics"][busy_key] < low_cut:
            low_lanes.append(_low_lane_entry(lane, busy_key, hi))
    imbalance_ratio_pct = _round(len(low_lanes) / len(usable) * 100, 2) if usable else None
    imbalanced = imbalance_ratio_pct is not None and imbalance_ratio_pct > IMBALANCE_RATIO_GATE_PCT
    return {
        "busy_metric": busy_key,
        "lane_count": len(usable),
        "min": _round(lo),
        "median": _round(med),
        "max": _round(hi),
        "diff_pct_of_max": _round(diff_pct, 2),
        "low_utilization": {
            "definition": f"utilization lags the busiest lane by >{LOW_UTIL_GAP_PCT:.0f}%",
            "count": len(low_lanes),
            "lanes": low_lanes,
        },
        "imbalance_ratio_pct": imbalance_ratio_pct,
        "verdict": "imbalanced" if imbalanced else "balanced",
        "verdict_rule": f"imbalance_ratio_pct (share of low-utilization lanes) > {IMBALANCE_RATIO_GATE_PCT:.0f}%",
        "busiest": [_rank_entry(lane, busy_key) for lane in ranked[:3]],
        "idlemost": [_rank_entry(lane, busy_key) for lane in ranked[-3:]],
    }


def _low_lane_entry(lane: dict, busy_key: str, hi: float) -> dict:
    return {
        "core_id": lane["core_id"],
        "subcore_id": lane["subcore_id"],
        busy_key: _round(lane["metrics"][busy_key]),
        "utilization_pct": _round(lane["metrics"][busy_key] / hi * 100, 2) if hi else None,
    }


def _rank_entry(lane: dict, busy_key: str) -> dict:
    return {
        "core_id": lane["core_id"],
        "subcore_id": lane["subcore_id"],
        busy_key: _round(lane["metrics"][busy_key]),
    }


def _occupancy_lanes(cores: list) -> list:
    lanes: list = []
    for c in cores:
        lanes.append({
            "core_id": c.get("core_id"),
            "subcore_id": c.get("subcore_id"),
            "subcore_type": str(c.get("subcore_type") or ""),
            "metrics": {k: _round(v) for k, v in (c.get("metrics") or {}).items()},
        })
    return lanes


def _core_rank_entry(core: dict) -> dict:
    entry = {"core_id": core["core_id"], "busy_sum": _round(core["busy_sum"])}
    for k, v in core["by_type"].items():
        entry[k] = _round(v)
    return entry


def _physical_core_ranking(lanes: list, busy_key: str) -> dict:
    per_core: dict = {}
    for lane in lanes:
        value = (lane.get("metrics") or {}).get(busy_key)
        if not isinstance(value, (int, float)):
            continue
        subcore = lane["subcore_type"] or "unknown"
        slot = per_core.setdefault(str(lane["core_id"]), {"core_id": lane["core_id"], "busy_sum": 0.0, "by_type": {}})
        slot["busy_sum"] += value
        slot["by_type"][subcore] = slot["by_type"].get(subcore, 0.0) + value
    ranked = sorted(per_core.values(), key=lambda c: -c["busy_sum"])
    return {
        "busy_metric": f"sum of {busy_key} across subcores",
        "caveat": "cube/vector 主频可能不同，跨子核 cycles 求和是工作量代理而非时间口径，排名仅作定性参考",
        "busiest": [_core_rank_entry(c) for c in ranked[:3]],
        "idlemost": [_core_rank_entry(c) for c in ranked[-3:]],
    }


def _metrics_by_type(lanes: list) -> dict:
    metrics_by_type: dict = {}
    for t in sorted({lane["subcore_type"] for lane in lanes}):
        type_lanes = [lane for lane in lanes if lane["subcore_type"] == t]
        metrics_by_type[t or "unknown"] = _type_metric_entries(type_lanes)
    return metrics_by_type


def _type_metric_entries(type_lanes: list) -> list:
    keys = set()
    for lane in type_lanes:
        keys.update(lane.get("metrics") or {})
    entries: list = []
    for k in sorted(keys):
        vals = [
            lane["metrics"].get(k) for lane in type_lanes
            if isinstance((lane.get("metrics") or {}).get(k), (int, float))
        ]
        if not vals:
            continue
        entries.append({"key": k, "min": _round(min(vals)), "median": _round(_median(vals)), "max": _round(max(vals))})
    return entries


def _occupancy_digest(parsed: dict) -> dict:
    if not parsed.get("available"):
        return {"available": False, "reason": "no op_detail/core_detail occupancy records parsed"}
    layout = parsed.get("layout") or {}
    lanes = _occupancy_lanes(parsed.get("cores", []))
    busy_key = "cycles" if any("cycles" in (lane["metrics"] or {}) for lane in lanes) else None
    if busy_key:
        groups: dict = {}
        for t in sorted({lane["subcore_type"] for lane in lanes}):
            type_lanes = [lane for lane in lanes if lane["subcore_type"] == t]
            groups[t or "unknown"] = _group_balance(type_lanes, busy_key)
        groups["_physical_core"] = _physical_core_ranking(lanes, busy_key)
    else:
        groups = {}
    return {
        "available": True,
        "soc": layout.get("soc"),
        "adapter": layout.get("adapter"),
        "raw_core_count": layout.get("raw_core_count"),
        "logical_core_count": layout.get("logical_core_count"),
        "lane_count": len(lanes),
        "balance_by_type": groups,
        "metrics_by_type": _metrics_by_type(lanes),
        "lanes": lanes,
    }


def _roofline_series(s: dict) -> dict:
    return {
        "bandwidth_name": s.get("bandwidth_name"),
        "compute_name": s.get("compute_name"),
        "arithmetic_intensity_ops_per_byte": _round(s.get("arithmetic_intensity")),
        "measured_performance_tops": _round(s.get("measured_performance")),
        "bandwidth": _round(s.get("bandwidth")),
        "compute_ceiling_tops": _round(s.get("compute_ceiling")),
        "knee_intensity": _round(s.get("knee_intensity")),
        "ratio_percent": _round(s.get("ratio_percent"), 2),
        "bound_region": s.get("bound_region"),
    }


def _roofline_digest(parsed: dict) -> dict:
    if not parsed.get("available"):
        return {"available": False, "reason": "no multiple_rooflines records parsed"}
    charts: list = []
    for chart in parsed.get("charts", []):
        charts.append({"title": chart.get("title"), "series": [_roofline_series(s) for s in chart.get("series", [])]})
    return {
        "available": True,
        "plot_available": parsed.get("plot_available"),
        "advice": parsed.get("advice") or [],
        "charts": charts,
        "notes": [
            "bound_region is geometric (AI vs knee); low ratio_percent far below the roof means "
            "pipeline gaps dominate, cross-check with the timeline digest.",
        ],
    }


def _segment_row(s: dict) -> str:
    pu = " ".join(f"{t} {v}" for t, v in sorted((s.get("pipe_util_pct") or {}).items(), key=lambda kv: -kv[1]))
    return (f"| {s['side']} | {s['kind']} | {s['region']} | {s.get('load_rate')} | {s.get('per_iter_us')} "
            f"| {s.get('occurrences')} | {pu} | {s.get('uncoverable')} |")


def _render_periodicity(lines: list, per: dict) -> None:
    if not per.get("status"):
        return
    lines.append("")
    lines.append(f"### Periodicity: {per['status']}")
    if per.get("status") not in {"periodic", "bursty_irregular"}:
        lines.append(f"- {per.get('reason')}")
        return
    sc = per.get("source_check") or {}
    match = {True: "match", False: "MISMATCH", None: "n/a"}[sc.get("match")]
    lines.append(f"- reference lane: {per['reference_lane']} | iterations: {per['iteration_count']} "
                 f"| period: {per['period_us']} us (cv={per['period_cv']}) "
                 f"| steady window: [{per['steady_window']['start']}, {per['steady_window']['end']}]")
    lines.append(f"- source check: expected={sc.get('expected_iterations')} -> {match}")
    lines.append("")
    lines.append("| pipe | per-iteration busy (us, median) |")
    lines.append("|---|---|")
    for p, v in sorted((per.get("per_iter_pipe_busy_us") or {}).items(), key=lambda kv: -kv[1]):
        lines.append(f"| {p} | {v} |")


def _render_segment_load(lines: list, seg: dict) -> None:
    if not seg.get("available"):
        return
    lines.append("")
    lines.append("### Segment load（分段负载率）")
    lines.append(
        "| side | kind | region | load_rate | per_iter_us | occurrences "
        "| pipe util% (段内流水) | uncoverable |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in sorted(seg.get("segments", []), key=lambda x: -(x.get("load_rate") or 0)):
        lines.append(_segment_row(s))
    lg = seg.get("largest_exposed_uncovered")
    if lg:
        lines.append("")
        lines.append(
            f"- largest exposed uncovered（单段）: core{lg['core']} {lg['side']} "
            f"start={lg['start']} dur={lg['dur_us']}us load_rate={lg['load_rate']}"
            f"（uncoverable 需 Tile DAG 证据）")


def _render_timeline_section(lines: list, tl: dict) -> None:
    lines.append("## Timeline (Pipe)")
    if not tl.get("available"):
        lines.append(f"- unavailable: {tl.get('reason')}")
        lines.append("")
        return
    lines.append(
        f"- type: {tl['timeline_type']} | window: {tl['window']['duration']} us "
        f"| events: {tl['event_count']}")
    lines.append("")
    lines.append("| pipe | events | busy | avg busy% per lane | longest event |")
    lines.append("|---|---|---|---|---|")
    for name, agg in sorted(tl["pipes"].items(), key=lambda kv: -(kv[1]["busy"] or 0)):
        lines.append(
            f"| {name} | {agg['events']} | {agg['busy']} | "
            f"{agg['avg_busy_pct_per_lane']} | {agg['longest_event']} |")
    _render_periodicity(lines, tl.get("periodicity") or {})
    _render_segment_load(lines, tl.get("segment_load") or {})
    lines.append("")


def _render_rank_rows(lines: list, g: dict, subcore: bool) -> None:
    busiest = g.get("busiest", [])
    idlemost = g.get("idlemost", [])
    hk = g["busy_metric"]
    for i in range(max(len(busiest), len(idlemost))):
        hot = busiest[i] if i < len(busiest) else {}
        idle = idlemost[i] if i < len(idlemost) else {}
        if subcore:
            lines.append(
                f"| {i + 1} | core{hot.get('core_id', '-')}/sub{hot.get('subcore_id', '-')} "
                f"{hk}={hot.get(hk, '-')} "
                f"| core{idle.get('core_id', '-')}/sub{idle.get('subcore_id', '-')} "
                f"{hk}={idle.get(hk, '-')} |")
        else:
            lines.append(
                f"| {i + 1} | core{hot.get('core_id', '-')} busy_sum={hot.get('busy_sum', '-')} "
                f"| core{idle.get('core_id', '-')} busy_sum={idle.get('busy_sum', '-')} |")


def _render_physical_core(lines: list, g: dict) -> None:
    lines.append(f"### physical core ranking ({g['busy_metric']})")
    if g.get("caveat"):
        lines.append(f"注：{g['caveat']}")
    lines.append("| rank | busiest | idlemost |")
    lines.append("|---|---|---|")
    _render_rank_rows(lines, g, subcore=False)
    lines.append("")


def _render_balance_group(lines: list, gname: str, g: dict) -> None:
    if gname == "_physical_core":
        _render_physical_core(lines, g)
        return
    lines.append(f"### {gname} lanes (busy metric: {g['busy_metric']})")
    lines.append(f"- 核间负载不均衡率: **{g.get('imbalance_ratio_pct')}%** -> verdict: **{g.get('verdict')}**"
                 f"（{g.get('verdict_rule')}）")
    lines.append(f"- diff_of_max: {g.get('diff_pct_of_max')}% "
                 f"(min {g.get('min')} / median {g.get('median')} / max {g.get('max')})")
    low = g.get("low_utilization") or {}
    lines.append(f"- low-utilization lanes: {low.get('count')}/{g.get('lane_count')}（{low.get('definition')}）")
    if low.get("lanes"):
        lines.append("")
        lines.append("| low-util lane | utilization% | busy |")
        lines.append("|---|---|---|")
        for lane in low["lanes"]:
            lines.append(f"| core{lane.get('core_id')}/sub{lane.get('subcore_id')} "
                         f"| {lane.get('utilization_pct')} | {lane.get(g['busy_metric'])} |")
    lines.append("")
    lines.append("| | busiest | idlemost |")
    lines.append("|---|---|---|")
    _render_rank_rows(lines, g, subcore=True)
    lines.append("")


def _render_occupancy_section(lines: list, oc: dict) -> None:
    lines.append("## Core Occupancy")
    if not oc.get("available"):
        lines.append(f"- unavailable: {oc.get('reason')}")
        lines.append("")
        return
    lines.append(f"- soc: {oc['soc']} | adapter: {oc['adapter']} | lanes: {oc['lane_count']} "
                 f"(raw cores: {oc['raw_core_count']}, logical: {oc['logical_core_count']})")
    lines.append("")
    for gname, g in oc.get("balance_by_type", {}).items():
        _render_balance_group(lines, gname, g)
    lines.append("")


def _roofline_row(s: dict) -> str:
    return (f"| {s['bandwidth_name']} / {s['compute_name']} | {s['arithmetic_intensity_ops_per_byte']} "
            f"| {s['measured_performance_tops']} | {s['bandwidth']} | {s['compute_ceiling_tops']} "
            f"| {s['knee_intensity']} | {s['ratio_percent']} | {s['bound_region']} |")


def _render_roofline_section(lines: list, rf: dict) -> None:
    lines.append("## Roofline")
    if not rf.get("available"):
        lines.append(f"- unavailable: {rf.get('reason')}")
        lines.append("")
        return
    if rf.get("advice"):
        lines.append(f"- profiler advice: {'; '.join(rf['advice'])}")
    lines.append("")
    for chart in rf["charts"]:
        lines.append(f"### {chart['title']}")
        lines.append("| series | AI (Ops/B) | perf (TOPS) | bw | ceiling (TOPS) | knee | ratio% | bound |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for s in chart["series"]:
            lines.append(_roofline_row(s))
        lines.append("")
    lines.append("")


def _render_markdown(digest: dict) -> str:
    lines = ["# Visual Analysis Digest", ""]
    lines.append(f"- collection: `{digest['collection']}`")
    lines.append(f"- parsers: `{digest['parsers_source']}`")
    lines.append(f"- extracted_at: {digest['extracted_at']}")
    lines.append("")
    _render_timeline_section(lines, digest["timeline"])
    _render_occupancy_section(lines, digest["core_occupancy"])
    _render_roofline_section(lines, digest["roofline"])
    return "\n".join(lines)


def _block_reason(block_meta: dict, block_id: str, not_collected: str) -> Optional[str]:
    meta = block_meta.get(block_id)
    if not meta:
        return not_collected
    status = str(meta.get("status") or "")
    if status != "ok":
        return f"block status={status or 'unknown'}: {meta.get('reason') or 'no reason recorded'}"
    return None


def _timeline_result(blocks: dict, block_meta: dict, visualize_parsers, args) -> dict:
    reason = _block_reason(block_meta, "timeline", "timeline block not collected")
    if reason:
        return {"available": False, "reason": reason}
    artifacts = blocks.get("timeline")
    if artifacts is None:
        return {"available": False, "reason": "timeline block not collected"}
    parsed = visualize_parsers.parse_timeline(artifacts, args.max_events)
    return _timeline_digest(parsed, args.expect_iterations)


def _occupancy_result(blocks: dict, block_meta: dict, visualize_parsers) -> dict:
    reason = _block_reason(block_meta, "details", "details block not collected")
    if reason:
        return {"available": False, "reason": reason}
    details = blocks.get("details")
    if details is None:
        return {"available": False, "reason": "details block not collected"}
    records = visualize_parsers.load_visualize_records(details.visualize_data)
    return _occupancy_digest(visualize_parsers.parse_occupancy(records))


def _roofline_result(blocks: dict, block_meta: dict, visualize_parsers) -> dict:
    reason = _block_reason(block_meta, "roofline", "roofline block not collected")
    if reason:
        return {"available": False, "reason": reason}
    roofline = blocks.get("roofline")
    if roofline is None:
        return {"available": False, "reason": "roofline block not collected"}
    records = visualize_parsers.load_visualize_records(roofline.visualize_data)
    return _roofline_digest(visualize_parsers.parse_roofline(records))


def _build_digest(args, collection: Path, scripts_dir: Path, normalized, visualize_parsers) -> dict:
    blocks = normalized.blocks
    block_meta = normalized.manifest.get("blocks") or {}
    return {
        "schema": SCHEMA,
        "collection": str(collection),
        "parsers_source": str(scripts_dir),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "blocks": {
            bid: {"metric": (meta or {}).get("metric"), "status": (meta or {}).get("status")}
            for bid, meta in block_meta.items()
            if bid in {"timeline", "details", "roofline"}
        },
        "timeline": _timeline_result(blocks, block_meta, visualize_parsers, args),
        "core_occupancy": _occupancy_result(blocks, block_meta, visualize_parsers),
        "roofline": _roofline_result(blocks, block_meta, visualize_parsers),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Timeline/Occupancy/Roofline digests for agent analysis.")
    parser.add_argument("collection", help="Collection directory containing collection_manifest.json")
    parser.add_argument("--output-dir", help="Digest output directory (default: collection dir)")
    parser.add_argument("--max-events", type=int, default=10000, help="Timeline event embed cap")
    parser.add_argument("--expect-iterations", type=int, default=None,
                        help="Loop trip count read from the DSL source; cross-checks detected burst count")
    parser.add_argument("--vis-skill-root", help="msopprof-visualization skill root (scripts/ must exist under it)")
    return parser


def _write_digest(digest: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "VISUAL_ANALYSIS.json"
    md_path = out_dir / "VISUAL_ANALYSIS.md"
    json_path.write_text(json.dumps(digest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(digest), encoding="utf-8")
    _LOGGER.info("VISUAL_ANALYSIS.json: %s", json_path)
    _LOGGER.info("VISUAL_ANALYSIS.md: %s", md_path)


def main() -> int:
    args = _build_parser().parse_args()
    collection = Path(args.collection).resolve()
    if not (collection / "collection_manifest.json").is_file():
        _LOGGER.error("ERROR: collection_manifest.json not found in %s", collection)
        return 1
    try:
        scripts_dir = _resolve_vis_scripts(args.vis_skill_root)
    except RuntimeError as exc:
        _LOGGER.error("%s", exc)
        return 1
    visualize_common, visualize_parsers = _import_parsers(scripts_dir)
    normalized = visualize_common.load_collection(collection)
    digest = _build_digest(args, collection, scripts_dir, normalized, visualize_parsers)
    out_dir = Path(args.output_dir).resolve() if args.output_dir else collection
    _write_digest(digest, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
