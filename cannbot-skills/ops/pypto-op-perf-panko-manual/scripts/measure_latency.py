#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""measure_latency.py — the frozen warm-latency measurement for PANKO E(x).

Used IDENTICALLY for the preopt baseline (INIT) and every candidate during the search, so the
`p` values are directly comparable. It fixes two bugs seen in ad-hoc measurement:

  1. Cold vs warm. The kernel's first call is cold (JIT compile / cache fill) and reads ~20-25%
     slower than the warm steady state. We run `warmup` calls and DISCARD them, then measure
     `runs` warm calls.
  2. Capture. The on-device profiler prints its "AICORE Prof Summary" at the native (fd) level,
     so Python's `redirect_stdout` (sys.stdout only) misses it. We redirect at the file-descriptor
     level (os.dup2) around the calls — keeping ONE warm process — then parse the captured text.

`p` = aggregate (median by default; min optional) of the warm per-call **"AICore End-to-End Time"**
the on-device profiler prints, which exists only under
`debug_options={"runtime_debug_mode": 1}`. It is the run's primary latency: `--p`, `--samples`
and `J` all mean this reading.

msprof's per-launch MIX_AIC `aicore_time(us)` is a SECOND reading of the same launch, carried
under its own name (`mix_aic`, `mix_aic_samples`, `mix_aic_sigma_log`); the CLI runs the
measurement under msprof for it (see `measure_under_msprof`). The two are different quantities,
their ratio is not a constant, and neither is derived from the other -- which is why both are
objectives. Only on-device AICore time is reported — never wall-clock (which includes
host/profiling overhead).

The measurement MECHANICS below are the frozen part. The op-specific `run_once()` (build inputs +
one wrapper call + synchronize) is supplied by the caller / CLI.
"""

import argparse
import csv
import glob
import importlib.util
import inspect
import json
import logging
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import types
from typing import Any, NamedTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jsonio  # noqa: E402
import workspace_probe  # noqa: E402

# One home for the two runtime questions, rather than a copy here: ask for an
# optional module, and treat a driver's refusal as the answer. `chip_profile`
# owns them because it is where the refusals are load-bearing.
from chip_profile import optional as _optional, quietly as _quietly  # noqa: E402

leaked_capture = []  # capture files a cleanup could not remove


def _read_text(path, errors="replace"):
    """The file's text, with the handle closed before the caller sees it."""
    with open(path, encoding="utf-8", errors=errors) as f:
        return f.read()


AICORE_RE = re.compile(r"AICore End-to-End Time:\s*([\d.]+)")
# Per-core scheduling record written by the profiler to
# output/output_<ts>/bubble_analysis.log. Bubble is NOT a string in the
# profiler summary: it is derived from these four fields, exactly as
# cannbot-skills/ops/pypto-op-perf-tune/perf-analyzer/scripts/analyze_perf.py
# derives it. Grepping stdout for "Bubble:" therefore always yielded None,
# which silently disabled the bubble half of the Stage-7 symptom index.
CORE_RE = re.compile(
    r"\[(AIC_\d+|AIV_\d+)\] Execute task num:(\d+)\s+"
    r"Core Total Work Time: ([\d.]+)\s+Total Wait Time: ([\d.]+)\s+"
    r"Wait Schedule Time: ([\d.]+)\s+Wait Predecessor Time: ([\d.]+)"
)
UTIL_RE = re.compile(r"AICore Utilization:\s*([\d.]+)")
BUBBLE_RE = re.compile(r"[Bb]ubble(?:\s*[Rr]ate)?:\s*([\d.]+)")


def _core_rows(path):
    """The per-core profiler records, or []. `CORE_RE` captures six fields."""
    try:
        return CORE_RE.findall(_read_text(path))
    except OSError:
        return []


def _bubble_from_log(path):
    """Mean bubble rate and mean core utilisation over all AIC/AIV cores.

    Definitions follow analyze_perf.py exactly:
        aicore_time = Core Total Work Time - Total Wait Time   (net busy)
        core_util   = aicore_time / (aicore_time + total_wait)
        bubble_rate = wait_schedule / (aicore_time + wait_schedule)
    Verified against the summary metric: both routes agree to a few points.
    """
    rows = _core_rows(path)
    if not rows:
        return None, None
    utils, bubbles = [], []
    for _name, _n, work, wait, sched, _pred in rows:
        work, wait, sched = float(work), float(wait), float(sched)
        busy = work - wait
        utils.append(busy / (busy + wait) * 100 if busy + wait else 0.0)
        bubbles.append(sched / (busy + sched) * 100 if busy + sched else 0.0)
    return (round(sum(bubbles) / len(bubbles), 4), round(sum(utils) / len(utils), 4))


def _pipes_from_log(path):
    """Per-pipe utilisation and dependency stall: {"AIC": {...}, "AIV": {...}}.

    Two of the six fields `CORE_RE` already captures were being discarded, and
    each of them is a bottleneck the aggregate cannot express.

    THE CORE NAME. `util` above is a mean over AIC and AIV together, so with 20
    cube cores against 40 vector cores the vector pipe carries two thirds of the
    weight. A kernel whose vector pipe is saturated while its cube pipe is
    nearly idle therefore reads comfortably above the 50% floor that would have
    raised a symptom, and a whole campaign can run with nothing pointing at the
    idle pipe. Splitting by the `AIC_`/`AIV_` prefix recovers it, and agrees
    with the swim-graph's own Cube and Vector Utilisation on the same kernel.

    WAIT PREDECESSOR. `bubble_rate` counts only `wait_schedule`. On a
    pipe-serialized core nearly all of the stall is `wait_predecessor` instead
    -- the whole cost of a serial cube->vector->cube recurrence -- so it is
    excluded from the metric by construction.

    Neither number costs a measurement: both are in the rows already parsed.
    """
    rows = _core_rows(path)
    if not rows:
        return {}
    acc = {}
    for name, n, work, wait, sched, pred in rows:
        work, wait, sched, pred = float(work), float(wait), float(sched), float(pred)
        busy = work - wait
        d = acc.setdefault(
            name.split("_")[0], {"u": [], "b": [], "s": [], "cores": 0, "tasks": 0}
        )
        d["u"].append(busy / (busy + wait) * 100 if busy + wait else 0.0)
        # `bubble_rate`'s own formula, per pipe rather than averaged over both.
        d["b"].append(sched / (busy + sched) * 100 if busy + sched else 0.0)
        # The same shape again, against the other wait.
        d["s"].append(pred / (busy + pred) * 100 if busy + pred else 0.0)
        d["cores"] += 1
        d["tasks"] += int(n)
    return {
        p: {
            "util": round(sum(d["u"]) / len(d["u"]), 2),
            "bubble": round(sum(d["b"]) / len(d["b"]), 2),
            "pred_stall": round(sum(d["s"]) / len(d["s"]), 2),
            "cores": d["cores"],
            "tasks": d["tasks"],
        }
        for p, d in sorted(acc.items())
    }


def _newest(roots, since, name):
    """Newest output/output_*/<name> created after `since`."""
    best, best_mt = None, since
    for root in roots:
        if not root:
            continue
        for d in glob.glob(os.path.join(root, "output", "output_*")):
            p = os.path.join(d, name)
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            if mt >= best_mt:
                best, best_mt = p, mt
    return best


def _find_bubble_log(roots, since):
    return _newest(roots, since, "bubble_analysis.log")


def _core_tasks(core):
    """(start, end) for every task on this core that carries both."""
    out = []
    for t in core.get("tasks") or []:
        s, e = t.get("execStart"), t.get("execEnd")
        if s is not None and e is not None:
            out.append((s, e))
    return out


def _core_totals(cores):
    """({coreType: {busy, cores, tasks}}, wall span) over the AIC/AIV cores."""
    acc, lo, hi = {}, None, None
    for c in cores:
        kind = c.get("coreType") if isinstance(c, dict) else None
        if kind not in ("AIC", "AIV"):
            continue
        d = acc.setdefault(kind, {"busy": 0, "cores": 0, "tasks": 0})
        d["cores"] += 1
        for s, e in _core_tasks(c):
            d["busy"] += e - s
            d["tasks"] += 1
            lo = s if lo is None or s < lo else lo
            hi = e if hi is None or e > hi else hi
    span = (hi - lo) if (lo is not None and hi is not None) else 0
    return acc, span


def _pipes_from_prof(path):
    """Per-pipe utilisation from the RAW profile, when the log is absent.

    `bubble_analysis.log` is written by `analysis_wait_cycles`, and the installed
    draw_swim_lane.py carries a "PANKO latency-only fast path" that exits
    before calling it -- so most evaluations of a run can produce no log at all
    and the per-pipe split is simply missing. That patch lives in
    site-packages, outside version control, alongside four earlier backups of the
    same file; adding a fifth would reproduce exactly the fragility that silently
    disabled the bubble half of the symptom index in the first place.

    `tilefwk_L1_prof_data.json` is the INPUT that fast path still parses, so it
    is upstream of the file people keep patching and is written on every run:
    one entry per core, `coreType` in {AIC, AIV}, and `tasks` carrying
    `execStart` / `execEnd`.

    What this recovers is utilisation only, and by print_aicore_summary's own
    definition -- busy task time over (core count x wall span) -- rather than the
    log's per-core busy/(busy+wait). The two nearly agree because a core's
    assigned window is nearly the whole span (a measured flash core: 23527 of
    23740). The WAIT BREAKDOWN is not recoverable: bubble and pred_stall come
    from analysis_wait_cycles, which is the thing the fast path skips. So
    `cube_starved`, which needs only utilisation, becomes available on every
    evaluation; `pipe_serialized`, which needs the predecessor wait, still needs
    the log. Absent is left absent rather than zeroed.
    """
    try:
        with open(path, encoding="utf-8") as f:
            cores = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(cores, list):
        return {}
    acc, span = _core_totals(cores)
    if not span:
        return {}
    return {
        p: {
            "util": round(d["busy"] / (d["cores"] * span) * 100, 2),
            "cores": d["cores"],
            "tasks": d["tasks"],
        }
        for p, d in sorted(acc.items())
        if d["cores"]
    }


def _sigma_log(vals):
    """Sample standard deviation of log(latency), or None below two readings."""
    xs = [math.log(v) for v in vals if v and v > 0]
    if len(xs) < 2:
        return None
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return round(math.sqrt(var), 6)


def _agg(vals, how):
    if not vals:
        return None
    return round(min(vals) if how == "min" else statistics.median(vals), 4)


FIRST_CALL_MARKER = "[panko-progress] first_call_done"


def _emit_first_call_marker(err_fd, compile_s):
    """Write the first-call marker to the saved stderr fd."""
    try:
        os.write(err_fd, f"{FIRST_CALL_MARKER} compile_s={compile_s}\n".encode("utf-8"))
    except OSError:
        # A marker we could not write costs the caller its compile/exec split
        # on a timeout. It must not cost the measurement, which is already
        # running.
        pass


def capture_readings(run_once, total):
    """Run `run_once` `total` times with fds 1 and 2 redirected to a temp file.

    Native (fd-level) capture, because the profiler writes from C rather than
    through `sys.stdout`. Returns (everything it printed, first-call seconds).

    The first call carries the JIT compile, and on this hardware that is the
    whole cost of a measurement: minutes of compile against a second or so for a
    warm call. A caller that kills us on a timeout therefore needs to know
    whether the compile finished, because a candidate whose COMPILE ran out of
    budget is unevaluated, while one that hung afterwards is a verdict about the
    candidate. So the marker goes to the SAVED stderr -- the real one, dup'd
    before the redirection -- which means it reaches the parent's pipe and
    survives in `TimeoutExpired.output`. Written into the redirected fd it would
    land in the temp file we are about to read and the parent would never see it.
    """
    sys.stdout.flush()
    sys.stderr.flush()  # don't let pre-existing buffered output leak into capture
    saved_out, saved_err = os.dup(1), os.dup(2)
    tf = tempfile.NamedTemporaryFile(mode="w+", suffix=".prof", delete=False)
    first_call_s = None
    try:
        os.dup2(tf.fileno(), 1)
        os.dup2(tf.fileno(), 2)
        for i in range(total):
            t_call = time.time()
            run_once()
            sys.stdout.flush()
            sys.stderr.flush()
            if i == 0:
                first_call_s = round(time.time() - t_call, 3)
                _emit_first_call_marker(saved_err, first_call_s)
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        tf.flush()
        tf.seek(0)
        text = tf.read()
        tf.close()
        try:
            os.unlink(tf.name)
        except OSError:
            # A capture file we could not remove is litter in a temp directory,
            # not a failed measurement. The numbers are already read.
            leaked_capture.append(tf.name)
    return text, first_call_s


def _pipe_readings(op_dir, t0):
    """Per-pipe counters, and which file they came from."""
    log = _find_bubble_log([op_dir, os.getcwd()], t0)
    log_bubble, log_util = _bubble_from_log(log) if log else (None, None)
    pipes = _pipes_from_log(log) if log else {}
    source = "bubble_analysis.log" if pipes else None
    if not pipes:  # the swimlane fast path skipped the log
        prof = _newest([op_dir, os.getcwd()], t0, "tilefwk_L1_prof_data.json")
        pipes = _pipes_from_prof(prof) if prof else {}
        source = "tilefwk_L1_prof_data.json" if pipes else None
    return {
        "log": log,
        "log_bubble": log_bubble,
        "log_util": log_util,
        "pipes": pipes,
        "source": source,
    }


def measure(run_once, warmup=1, runs=9, agg="median", op_dir=None):
    """Run `run_once` warmup+runs times capturing native (fd-level) profiler output; discard the
    first `warmup`, aggregate the warm `runs`. `run_once` must perform exactly one profiled kernel
    call (+ device sync) and let the profiler print its summary. Returns {p, util, bubble, samples}.
    """
    total = max(1, warmup) + max(1, runs)
    t0 = time.time() - 1.0          # tolerate coarse mtime granularity
    probe = getattr(run_once, "workspace_probe", None)
    text, compile_s = capture_readings(run_once, total)
    ws = (probe.reading(total) if probe is not None
          else {"workspace_b": None, "workspace_state": "unavailable",
                "workspace_error": "no probe (run_once was not built by this module)"})
    lat = [float(x) for x in AICORE_RE.findall(text)]
    util = [float(x) for x in UTIL_RE.findall(text)]
    bub = [float(x) for x in BUBBLE_RE.findall(text)]
    # No fallback on the primary instrument. These read `lat[-runs:]` when the
    # profiler printed `warmup` lines or fewer -- which at the default warmup=1
    # means exactly one line, and `lat[-runs:]` is then that one line: the COLD
    # call, which this module's own header says reads 20-25% slower than the warm
    # steady state. `J = 100 * P_ref / p` is inflated by that much, the
    # candidate is reverted, and nothing downstream can tell, because
    # `raw_count` is emitted and has no reader. The sibling instrument already
    # refuses a row count it cannot assign to calls (`mix_aic_reading`); `p` --
    # which J, the ratchet, SPRT and `verify` all run on -- guessed. It now
    # refuses the same way: `p` is None with `p_error`, which `evaluator` already
    # treats as an unmeasured candidate.
    lat_error = None
    if len(lat) <= warmup:
        lat_error = (f"{len(lat)} AICore End-to-End readings for "
                     f"{total} calls at warmup={warmup}: every reading is a "
                     f"warm-up, so there is no warm measurement to aggregate")
    warm_lat = lat[warmup:] if len(lat) > warmup else []
    warm_util = util[warmup:] if len(util) > warmup else []
    warm_bub = bub[warmup:] if len(bub) > warmup else []
    # Prefer the profiler's own per-core record; fall back to the (usually
    # empty) stdout regex so behaviour never regresses if the log is absent.
    read = _pipe_readings(op_dir, t0)
    log_bubble = read["log_bubble"]
    aic, aiv = read["pipes"].get("AIC") or {}, read["pipes"].get("AIV") or {}
    return {"p": _agg(warm_lat, agg),
            **({"p_error": lat_error} if lat_error else {}),
            "util": _agg(warm_util, "median"),
            "bubble": log_bubble if log_bubble is not None else _agg(warm_bub, "median"),
            "bubble_source": "bubble_analysis.log" if log_bubble is not None else "stdout",
            "bubble_log": read["log"],
            "util_from_log": read["log_util"],
            # Per-pipe. Absent (None) when no log was produced -- the installed
            # draw_swim_lane.py carries a latency-only fast path that exits
            # before writing one, so this is missing far more often than `util`
            # is, and "not measured" must stay distinguishable from "zero".
            "aic_util": aic.get("util"),
            "aiv_util": aiv.get("util"),
            # `bubble` above is a mean over both pipes. Per pipe it separates a
            # scheduler that cannot feed the cores from one that can: widening
            # a vec tile can drop AIV utilisation and raise AIV bubble while
            # latency ROSE, which is the signature of fewer, larger tasks
            # starving the vector cores rather than of more work.
            "aic_bubble": aic.get("bubble"),
            "aiv_bubble": aiv.get("bubble"),
            "pred_stall": max([d["pred_stall"] for d in read["pipes"].values()
                               if "pred_stall" in d], default=None),
            "pipes": read["pipes"],
            "pipe_source": read["source"],
            # The first call's wall time, which is the JIT compile plus one
            # execution. Reported because it is a free label: deterministic,
            # low-noise, and the quantity `eval_timeout_s` has to be set
            # against. It is a diagnostic and never enters `p` or `J`.
            "compile_s": compile_s,
            # The spread on the scale the search actually reasons on. Latency is
            # positive and multiplicative -- a 10% slowdown is 10% at any scale
            # -- so log is where its noise is nearly symmetric, and where cv and
            # sigma coincide for small cv. Reported so a run can calibrate its
            # own noise instead of carrying a transcribed constant: the cv
            # varies widely ACROSS OPERATORS, and a single number for all of
            # them makes a fixed stagnation count misbehave in both
            # directions.
            "sigma_log": _sigma_log(warm_lat),
            # The second objective (msprof "workspace"): bytes the runtime asked
            # for, read from the alloc callback. See workspace_probe for the
            # three states; `unavailable` is never reported as 0.
            **ws,
            "samples": warm_lat, "raw_count": len(lat)}


# ------------------------------------------------------------- mix_aic, from the profiler
# `p` is the on-device profiler's "AICore End-to-End Time", and it is what `p`
# carries everywhere `p` is used (J, the ratchet, the noise estimate, SPRT,
# verify). msprof's per-launch MIX_AIC time is parsed alongside it and reported
# as `mix_aic`. The two are separate readings of the same launch: they are not
# the same quantity, their ratio is not a constant, and neither is computed from
# the other, which is why the harness carries both as objectives.
#
# It cannot be read from inside the measuring process: msprof has to START the
# process, and `op_summary_*.csv` exists only after msprof has exited and parsed
# its raw data. So the CLI re-executes itself under msprof as a child, takes
# the child's reading, and replaces `p` with the MIX_AIC read afterwards.
#
# On a cube-bound kernel the two run close together, and there is one row per
# launch. On a vector-bound kernel the same ratio can reach most of an order of
# magnitude, so it is a property of the kernel, not a conversion factor.
MSPROF_CHILD_ENV = "PANKO_MSPROF_CHILD"
MIX_AIC_TASK_TYPE = "MIX_AIC"
MIX_AIC_COLUMN = "aicore_time(us)"

# Diagnostics from the SAME rows as `p`, so they cost nothing and -- unlike
# `util` / `aic_util` / `aiv_util`, which come from the stdout record and
# `tilefwk_L1_prof_data.json` -- they survive `runtime_debug_mode: 0`, which is
# the production configuration and now the default. They read the same under
# either mode.
#
# They are NOT substitutes for the fields they sit beside. The same run reports
# several different "utilisation" numbers -- pypto's stdout and msprof's
# cube_utilization among them -- and the symptom index's thresholds were set on
# the first. So these are carried under their own names and nothing reads them
# yet; a rule that wants them has to be calibrated against them.
#
# `_b` in a name means bytes, so time is kept out of that convention: `aiv_time`
# is summed per call like `p`, the ratios are medians of the warm rows.
MSPROF_TIME_COLUMNS = {"aiv_time": "aiv_time(us)"}
MSPROF_RATIO_COLUMNS = {
    "aic_mac_ratio": "aic_mac_ratio",
    "aiv_vec_ratio": "aiv_vec_ratio",
    "cube_utilization": "cube_utilization(%)",
    "vector_utilization": "vector_utilization(%)",
}
# The range each of those columns is DEFINED over, with slack for rounding. Not
# a judgement about whether msprof is a good instrument -- a fraction of a part
# cannot exceed the whole, whatever one thinks of the tool. Such a reading can
# arrive non-reproducibly on rows whose COUNT divides cleanly by the call
# count, so the row-count refusal passes it through. A percentage far outside
# the range means the columns of that CSV did not line up with their headers,
# and the MIX_AIC duration is read from the same rows by the same `_cell`. So
# the whole reading is refused rather than guessed, which is what the row-count
# check already does one function up.
MSPROF_RATIO_RANGES = {
    "aic_mac_ratio": (0.0, 1.05),
    "aiv_vec_ratio": (0.0, 1.05),
    "cube_utilization": (0.0, 100.5),
    "vector_utilization": (0.0, 100.5),
}


def _cell(row, column):
    """`row[column]` as a float, tolerating the header's unit suffix and tabs."""
    want = column.split("(")[0].strip().lower()
    for k, v in row.items():
        if str(k).split("(")[0].strip().lower() != want:
            continue
        try:
            return float(str(v).strip().strip("\t"))
        except (TypeError, ValueError):
            return None
    return None


def msprof_metrics(rows, total, warmup):
    """The diagnostic columns of the warm MIX_AIC rows.

    `rows` are the raw dicts, already filtered and ordered. Times are summed
    over a call's k rows and then aggregated; ratios are aggregated directly.
    """
    if not rows or total <= 0 or len(rows) % total:
        return {}
    k = len(rows) // total
    warm = rows[max(0, warmup) * k:]
    out_m = {}
    for field, col in MSPROF_TIME_COLUMNS.items():
        per_call = [
            [_cell(r, col) for r in warm[i * k:(i + 1) * k]]
            for i in range(len(warm) // k)
        ]
        sums = [
            round(sum(c), 4) for c in per_call if c and all(v is not None for v in c)
        ]
        if sums:
            out_m[field] = _agg(sums, "median")
    for field, col in MSPROF_RATIO_COLUMNS.items():
        vals = [v for v in (_cell(r, col) for r in warm) if v is not None]
        if not vals:
            continue
        lo, hi = MSPROF_RATIO_RANGES[field]
        bad = [v for v in vals if v < lo or v > hi]
        if bad:
            # See MSPROF_RATIO_RANGES. One column out of its own range condemns
            # the parse, not just that column.
            return {
                "p": None,
                "p_error": f"{col} out of range ({min(bad)}..{max(bad)}, expected "
                f"{lo}..{hi}): the op_summary columns do not line up, so "
                f"no reading from these rows is usable",
            }
        out_m[field] = _agg(vals, "median")
    return out_m


def _op_summary_csvs(root):
    return sorted(
        glob.glob(
            os.path.join(root, "**", "mindstudio_profiler_output", "op_summary_*.csv"),
            recursive=True,
        ),
        key=os.path.getmtime,
    )


def _row_start(row):
    for k in ("Task Start Time(us)", "Task Start Time"):
        try:
            return float(str(row.get(k, "")).strip().strip("\t"))
        except ValueError:
            continue
    return None


def mix_aic_rows(csv_path):
    """One op_summary's MIX_AIC rows in launch order.

    Returns ([(op name, us)], [every Task Type seen], [the raw rows]) -- the
    raw rows so the diagnostic columns are read from the same rows as `p`,
    never from a second, differently filtered pass.
    """
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as f:
        allrows = list(csv.DictReader(f))
    task_types = sorted({str(r.get("Task Type", "")).strip() for r in allrows})
    rows = [
        r for r in allrows if str(r.get("Task Type", "")).strip() == MIX_AIC_TASK_TYPE
    ]
    if rows and all(_row_start(r) is not None for r in rows):
        rows.sort(key=_row_start)
    out_rows, kept = [], []
    for r in rows:
        v = _cell(r, MIX_AIC_COLUMN)
        if v is None:
            continue
        out_rows.append((str(r.get("Op Name", "")).strip(), v))
        kept.append(r)
    return out_rows, task_types, kept


def mix_aic_reading(rows, total, warmup, agg="median", task_types=None):
    """`p` from the launch rows: drop the warmup, aggregate the rest.

    A wrapper call launches the kernel once (`workspace_launches` checks the same
    invariant), so there should be `total` rows. A wrapper that launches k MIX_AIC
    kernels per call gives k*total rows, and a call's value is then the sum of its
    k consecutive rows. Any other count means the rows cannot be assigned to calls,
    and the reading is refused rather than guessed: `p` is None with `p_error`.
    """
    base = {
        "p": None,
        "samples": [],
        "sigma_log": None,
        "mix_aic_rows": len(rows),
        "mix_aic_per_call": None,
    }
    if not rows:
        return {
            **base,
            "p_error": "no MIX_AIC rows in op_summary (task types seen: "
            f"{', '.join(task_types or []) or 'none'})",
        }
    if total <= 0 or len(rows) % total:
        return {
            **base,
            "p_error": f"{len(rows)} MIX_AIC rows for {total} calls: cannot assign rows to calls",
        }
    k = len(rows) // total
    per_call = [
        round(sum(v for _, v in rows[i * k:(i + 1) * k]), 4) for i in range(total)
    ]
    warm = per_call[warmup:] if len(per_call) > warmup else per_call
    return {
        **base,
        "p": _agg(warm, agg),
        "samples": warm,
        "sigma_log": _sigma_log(warm),
        "mix_aic_per_call": k,
        "mix_aic_ops": sorted({n for n, _ in rows}),
    }


def _last_json(text):
    for line in reversed((text or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


class ProfOptions(NamedTuple):
    """How the profiler child is launched and whether its output directory is kept."""

    keep_prof: bool = False
    prof_root: Any = None
    msprof: str = "msprof"


def measure_under_msprof(argv, warmup, runs, agg, prof_opts=ProfOptions()):
    """Run the measurement as a child of msprof; `p` becomes its MIX_AIC.

    stderr is inherited, so the child's FIRST_CALL_MARKER still reaches whoever
    runs us (the evaluator reads it on a timeout). The child's stdout -- msprof's
    log, any device error, the child's own JSON -- is passed through before our
    reading, so fault markers stay visible and our JSON is still the last line.
    """
    total = max(1, warmup) + max(1, runs)
    if shutil.which(prof_opts.msprof) is None:
        return {"p": None, "p_source": "msprof MIX_AIC",
                "p_error": f"{prof_opts.msprof} not found on PATH"}
    out_dir = tempfile.mkdtemp(prefix="panko_msprof_", dir=prof_opts.prof_root)
    env = dict(os.environ)
    env[MSPROF_CHILD_ENV] = "1"
    cmd = [prof_opts.msprof, f"--output={out_dir}", "--ai-core=on", "--task-time=on",
           "--ascendcl=on", sys.executable, os.path.abspath(__file__), *argv]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True, env=env,
                       check=False)
    if r.stdout:
        # The child's own lines, on the protocol channel: a fault marker it
        # printed stays visible and our JSON is still the last line.
        logging.getLogger(jsonio.CHANNEL).info(r.stdout.rstrip("\n"))
    child = _last_json(r.stdout) or {}
    # `p` IS the child's reading: the on-device profiler's "AICore End-to-End
    # Time". It is the run's primary latency, so it keeps the name every
    # downstream reader already uses for that (`--p`, `--samples`, `J`).
    reading = dict(child)
    csvs = _op_summary_csvs(out_dir)
    if child.get("p") is None and child.get("unevaluated"):
        mix = {}                             # the child's own verdict stands
    elif not csvs:
        mix = {"mix_aic_error":
               f"msprof exited {r.returncode} and wrote no op_summary_*.csv"}
    else:
        rows, task_types, raw = mix_aic_rows(csvs[-1])
        mix = mix_aic_reading(rows, total, warmup, agg, task_types)
        metrics = msprof_metrics(raw, total, warmup)
        if metrics.get("p_error"):
            # The diagnostic columns condemned the parse. Drop the MIX_AIC value
            # and its samples with it -- they came from the same rows.
            mix = {"p": None, "samples": [], "sigma_log": None,
                   "mix_aic_rows": len(rows), "mix_aic_per_call": None,
                   "p_error": metrics["p_error"]}
        else:
            mix.update(metrics)
    # MIX_AIC is a SECOND reading of the same launch, carried under its own
    # name. It is an objective in its own right (the two rank programs
    # differently), never a fallback for `p` and never mixed into it.
    for src, dst in (("p", "mix_aic"), ("samples", "mix_aic_samples"),
                     ("sigma_log", "mix_aic_sigma_log"),
                     ("p_error", "mix_aic_error")):
        if src in mix:
            mix[dst] = mix.pop(src)
    mix["mix_aic_source"] = "msprof MIX_AIC " + MIX_AIC_COLUMN
    mix["msprof_rc"] = r.returncode
    if prof_opts.keep_prof:
        mix["msprof_dir"] = out_dir
    else:
        shutil.rmtree(out_dir, ignore_errors=True)
    return {**reading, **mix}


# --------------------------------------------------------------------------- CLI, by convention
def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class InputsError(Exception):
    """`eval/test_inputs.py` could not produce the arguments.

    Its own type because it is NOT a verdict on the candidate and not a
    measurement that failed: nothing was run. The kernel was never called, so
    there is no latency and no `op_summary` -- and before this existed the
    builder's exception took the whole process down, msprof still exited 0, and
    the caller saw a measurement with no readings rather than a named fault.
    """


def call_builder(ti, device):
    """`build_inputs(device)`, `build_inputs()` or `INPUTS`, in that order.

    `device` is the `--device` value VERBATIM -- the id as a string, e.g. "7".
    It is not a torch device, and `torch.randn(device="7")` raises `Invalid
    device string`. By the time this runs, `torch.npu.set_device` has already
    been called and `TILE_FWK_DEVICE_ID` is set, so a builder can ignore the
    argument entirely and use `.npu()`; a builder that wants it explicitly
    writes `f"npu:{device}"` or `torch.device(f"npu:{device}")`.

    A zero-argument `build_inputs()` is accepted too, decided from the
    signature rather than by catching TypeError -- which would also swallow a
    TypeError raised INSIDE the builder and report it as the wrong thing.
    """
    builder = getattr(ti, "build_inputs", None)
    if not callable(builder):
        try:
            return getattr(ti, "INPUTS")
        except AttributeError:
            raise InputsError(
                f"{getattr(ti, '__file__', 'test_inputs.py')} exposes neither "
                f"`build_inputs` nor `INPUTS`. The objective cannot be measured "
                f"without the wrapper's arguments"
            ) from None
    try:
        takes_device = bool(inspect.signature(builder).parameters)
    except (TypeError, ValueError):
        takes_device = True
    try:
        return builder(device) if takes_device else builder()
    except Exception as e:
        raise InputsError(
            f"build_inputs({device!r}) raised {type(e).__name__}: {e}. The "
            f"argument is the device ID as given on the command line, a string; "
            f"`torch.npu.set_device` has already been called, so use `.npu()` "
            f'or `f"npu:{{device}}"` rather than passing it to `device=` '
            f"directly"
        ) from None


def build_run_once(op_dir, op, op_file, device, inputs_file=None):
    """Convention: `custom/<op>/eval/test_inputs.py` exposes `build_inputs()` (or `INPUTS`) giving
    the args for the wrapper `op`. `op_file` accepts a path or an already imported module.
    Adapt here if an op differs; the MECHANICS
    in `measure()` stay frozen regardless.
    """
    import torch  # noqa

    # Importing torch_npu is what registers the NPU backend; on a box without it
    # there is nothing to register and the caller finds out at the first device
    # call, with a message about the device rather than about an import.
    _optional("torch_npu")
    os.environ["TILE_FWK_DEVICE_ID"] = str(device)
    _quietly(torch.npu.set_device, int(device))
    impl = (
        op_file
        if isinstance(op_file, types.ModuleType)
        else _load_module(op_file, "op_impl")
    )
    # Before the first call, so the compile-and-launch of call 1 is seen too.
    # A probe that cannot install reports `unavailable`; it never stops a
    # measurement, whose latency is still the objective.
    probe = workspace_probe.WorkspaceProbe().install(impl)
    # `<op>_wrapper` is the more common convention in this repo, but `--op` names
    # the operator, not the callable. Without the fallback every trial on such a
    # kernel raises AttributeError inside the measurement subprocess, which the
    # evaluator cannot distinguish from a device fault -- so a whole campaign
    # scores as environment faults and no candidate is ever measured.
    wrapper = getattr(impl, op, None) or getattr(impl, f"{op}_wrapper")
    # `--inputs` names a DIFFERENT build_inputs, for measuring a kernel on a shape
    # other than the one the campaign optimised against. It defaults to the
    # campaign's own file, so E(x) is unchanged and every recorded number stays
    # comparable; this exists only for after-the-fact re-measurement.
    #
    # An operator's eval/test_inputs.py can measure at a smoke-test shape far
    # smaller than the one SPEC.md declares. Every loop-granularity result is
    # then drawn on a shape where the tile makes the loop run exactly once,
    # which it does not do at the declared size -- so the campaign optimises a
    # loop that is not there.
    ti = (
        inputs_file
        if isinstance(inputs_file, types.ModuleType)
        else _load_module(
            inputs_file or os.path.join(op_dir, "eval", "test_inputs.py"), "test_inputs"
        )
    )
    inputs = call_builder(ti, device)
    args = inputs if isinstance(inputs, (list, tuple)) else [inputs]

    def run_once():
        _ = wrapper(*args)
        # Synchronising is what makes the measured interval the kernel's; a
        # runtime that has no synchronize has already run it synchronously.
        _quietly(torch.npu.synchronize)

    run_once.workspace_probe = probe
    return run_once


def capture_output(run_once, total):
    """What `run_once` printed over `total` calls, without the compile time.

    `capture_readings` returns `(text, first_call_s)`; a caller that only wants
    the text -- the resident evaluation session reading a test run's output --
    takes it through here rather than indexing the pair at every call site.
    """
    return capture_readings(run_once, total)[0]


def main():
    ap = argparse.ArgumentParser(
        description="Frozen warm-latency measure for PANKO E(x)."
    )
    ap.add_argument("--op-dir", required=True)
    ap.add_argument("--op", required=True)
    ap.add_argument("--op-file", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--warmup", type=int, default=1)
    # 9, not 3. `min` reported the fast tail, so three hits were enough for it;
    # the median wants a sample. SE(median) ~ 1.2533*sigma/sqrt(n), so the six
    # extra calls cut the standard error by about a third, and the median then
    # tolerates four outliers instead of one -- which matters on a device whose
    # noise the repo itself calls bimodal. A profiled call is a small fraction
    # of an evaluation, so they are cheap.
    ap.add_argument("--runs", type=int, default=9)
    # `median`, not `min`. min reports a distribution's fast mode rather than
    # its centre, which flatters the noisier arm -- so the ratchet preferred
    # programs for their VARIANCE. The bias is systematic, not random: it grows
    # with the spread. The median is unbiased and halves the standard error.
    ap.add_argument("--agg", choices=["median", "min"], default="median")
    ap.add_argument(
        "--inputs",
        default=None,
        help="alternative build_inputs module; defaults to "
        "<op-dir>/eval/test_inputs.py. Re-measurement only -- "
        "passing it means the number is NOT comparable to the "
        "campaign's recorded latencies.",
    )
    ap.add_argument(
        "--keep-prof", action="store_true", help="keep the msprof output directory"
    )
    a = ap.parse_args()
    if not os.environ.get(MSPROF_CHILD_ENV):
        child = [x for x in sys.argv[1:] if x != "--keep-prof"]
        jsonio.emit(
            measure_under_msprof(
                child, a.warmup, a.runs, a.agg, ProfOptions(keep_prof=a.keep_prof)
            )
        )
        return
    try:
        run_once = build_run_once(a.op_dir, a.op, a.op_file, a.device, a.inputs)
    except InputsError as e:
        # A measurement that never happened, said so. `p: None` with `p_error`
        # is the shape the evaluator already reads: it raises an environment
        # fault of the unmeasured-objective kind, which ends the block naming
        # the cause instead of charging a candidate for it. Dying here
        # instead emitted nothing at all, and nothing at all is indistinguishable
        # from a device that went away.
        jsonio.emit({"p": None, "p_error": str(e)})
        return
    jsonio.emit(measure(run_once, a.warmup, a.runs, a.agg, op_dir=a.op_dir))


if __name__ == "__main__":
    main()
