#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
BSA Shared Test Utilities

Shared test infrastructure used by both test_bsa_fwd.py and test_bsa_bwd.py:
  - Performance tracking
  - Input generation
  - Tensor comparison helpers
"""

import os
import re
import sys
import math
import json
import logging
import time
import shutil
from collections import namedtuple
import torch
import numpy as np
from numpy.testing import assert_allclose
from bsa_common import DEFAULT_CONFIG, generate_block_sparse_mask

_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

from common_utils import compare

BSATestInputs = namedtuple('BSATestInputs',
    ['q', 'k', 'v', 'd_o', 'mask', 'asq', 'askv'])

# Namedtuple configs wrapping multi-argument function inputs
GenInputsConfig = namedtuple('GenInputsConfig',
    ['b', 'hq', 'hkv', 'sq', 'skv', 'sparsity', 'device', 'seed'])
GenInputsNonAlignedConfig = namedtuple('GenInputsNonAlignedConfig',
    ['b', 'hq', 'hkv', 'sq_max', 'skv_max',
     'actual_seq_lengths_q', 'actual_seq_lengths_kv',
     'sparsity', 'device', 'seed'])
PrecisionCaseConfig = namedtuple('PrecisionCaseConfig',
    ['name', 'b', 'hq', 'hkv', 'sq', 'skv', 'sparsity'])
NonAlignedCaseConfig = namedtuple('NonAlignedCaseConfig',
    ['name', 'b', 'hq', 'hkv', 'sq_max', 'skv_max', 'sparsity',
     'asq_list', 'askv_list'])
PerfBenchConfig = namedtuple('PerfBenchConfig',
    ['name', 'b', 'hq', 'hkv', 'sq', 'skv', 'sparsity'])
PerfWallConfig = namedtuple('PerfWallConfig',
    ['name', 'b', 'hq', 'hkv', 'sq', 'skv', 'sparsity'])
FwdCaseConfig = namedtuple('FwdCaseConfig',
    ['name', 'b', 'hq', 'hkv', 'sq', 'skv', 'sparsity'])
BwdCaseConfig = namedtuple('BwdCaseConfig',
    ['name', 'b', 'hq', 'hkv', 'sq', 'skv', 'sparsity'])
BSARunnerConfig = namedtuple('BSARunnerConfig',
    ['mode', 'precision_fn', 'perf_fn', 'perf_wall_fn', 'case_cfg_type'])
FwdNonAlignedCaseConfig = namedtuple('FwdNonAlignedCaseConfig',
    ['name', 'b', 'hq', 'hkv', 'sq_max', 'skv_max', 'sparsity',
     'asq_list', 'askv_list'])
BwdNonAlignedCaseConfig = namedtuple('BwdNonAlignedCaseConfig',
    ['name', 'b', 'hq', 'hkv', 'sq_max', 'skv_max', 'sparsity',
     'asq_list', 'askv_list'])
SwimlaneQueryConfig = namedtuple('SwimlaneQueryConfig',
    ['search_bases', 'tag', 'name', 'b', 'hq', 'sq', 'wall_ms'])
CompareNonAlignedConfig = namedtuple('CompareNonAlignedConfig',
    ['label', 'golden', 'actual', 'atol', 'rtol', 'actual_seq_lengths'])

cfg = DEFAULT_CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
_fmt = logging.Formatter(fmt='%(asctime)s [%(levelname)s] %(message)s', datefmt='[%H:%M:%S]')
_handler = logging.StreamHandler()
_handler.setFormatter(_fmt)
logger.handlers.clear()
logger.addHandler(_handler)

_BSA_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
_PERF_OUTPUT_BASE = os.path.abspath(os.path.join(_BSA_ROOT, "output"))
# PyPTO runtime writes output to CWD-relative "output/", not necessarily
# to the module-relative _PERF_OUTPUT_BASE above.  _collect_updated_dirs
# must search both to find swimlane data regardless of CWD.
_PERF_SEARCH_BASES = [_PERF_OUTPUT_BASE, os.path.abspath("output")]
_perf_records = []
_perf_timestamp = None


def get_device():
    """Get NPU device string."""
    did = os.environ.get("TILE_FWK_DEVICE_ID", "0")
    return f"npu:{did}"


def _resolve_bsa_root():
    """Locate BSA root directory (containing common/, FWD/, BWD/)."""
    _bsa_marker = ("BSA_README.md", os.path.join("common", "bsa_common.py"))

    env_root = os.environ.get("BSA_ROOT")
    if env_root and all(os.path.isfile(os.path.join(env_root, m)) for m in _bsa_marker):
        return os.path.abspath(env_root)

    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(20):
        if all(os.path.isfile(os.path.join(d, m)) for m in _bsa_marker):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent

    raise RuntimeError(
        "Cannot locate BSA root directory.\n"
        "  Set BSA_ROOT to the path containing common/, FWD/, BWD/.")


def _take_perf_timestamp():
    global _perf_timestamp
    _perf_timestamp = time.time()


def _measure_wall_clock_time(run_fn, n_runs=3):
    """Measure wall-clock time for a kernel call with warmup and median timing."""
    torch.npu.synchronize()
    run_fn()
    torch.npu.synchronize()
    times = []
    for _ in range(n_runs):
        torch.npu.synchronize()
        start = time.time()
        run_fn()
        torch.npu.synchronize()
        times.append((time.time() - start) * 1000)
    times.sort()
    median_ms = times[len(times) // 2]
    mean_ms = sum(times) / len(times)
    return median_ms, mean_ms


def _scan_base_for_updated_dirs(base, timestamp):
    """Scan one base directory for output dirs updated after timestamp."""
    updated = []
    if not os.path.isdir(base):
        return updated
    for name in os.listdir(base):
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        swim = os.path.join(d, "merged_swimlane.json")
        if os.path.isfile(swim):
            mtime = os.path.getmtime(swim)
            if mtime >= timestamp - 0.5:
                updated.append(d)
    return updated


def _collect_updated_dirs():
    """Collect output dirs whose merged_swimlane.json was modified after
    _take_perf_timestamp (minus a 0.5 s buffer).

    Searches across all _PERF_SEARCH_BASES (module-relative and CWD-relative)
    to find swimlane data regardless of where PyPTO runtime writes output.
    Callers that also pass extra_dirs to _record_perf_from_dirs
    can supplement this list with cached-kernel output dirs tracked by the
    implementation module.
    """
    if _perf_timestamp is None:
        return []
    updated = []
    for base in _PERF_SEARCH_BASES:
        updated.extend(_scan_base_for_updated_dirs(base, _perf_timestamp))
    updated.sort()
    return updated


def _identify_kernel(output_dir):
    """Identify kernel types from kernel_aicore directory.

    Returns a list of identified kernel type labels. For merged kernels that
    contain both dQ and dK/dV phases, the list will contain both labels.
    """
    kernel_aicore = os.path.join(output_dir, "kernel_aicore")
    if not os.path.isdir(kernel_aicore):
        return ["Unknown"]
    loop_prefixes = set()
    for f in os.listdir(kernel_aicore):
        m = re.match(r"TENSOR_LOOP_([a-z_]*?)(?:qblk|kblk|sub)", f)
        if m:
            prefix = m.group(1).rstrip("_")
            if prefix:
                loop_prefixes.add(prefix)
    prefix_map = {
        "fwd_s": "FWD", "fwd_d": "FWD", "fwd": "FWD",
        "dq": "dQ", "dqd": "dQ",
        "dkdv": "dK/dV", "dkdvd": "dK/dV",
    }
    identified = set()
    for prefix in loop_prefixes:
        for key, label in prefix_map.items():
            if prefix.startswith(key):
                identified.add(label)
    if not identified:
        return ["Unknown"]
    return sorted(identified)


def _parse_swimlane(output_dir):
    """Parse merged_swimlane.json, computing per-execution metrics.

    When the swimlane file accumulates events from multiple kernel
    executions (cached kernel reused across test cases), we isolate
    each execution by finding contiguous clusters of real-task events
    separated by large gaps (idle periods between test cases).

    Each cluster is treated as a single execution, and we return the
    metrics for the LAST cluster (the one most recently executed).
    """
    swim_path = os.path.join(output_dir, "merged_swimlane.json")
    if not os.path.exists(swim_path):
        return None
    with open(swim_path) as f:
        data = json.load(f)
    evts = data.get("traceEvents", [])
    if not evts:
        return None
    tid2core = {}
    for ev in evts:
        if ev.get("ph") == "M" and ev.get("name") == "thread_name":
            tid2core[ev["tid"]] = ev["args"]["name"]
    all_tasks = [ev for ev in evts if ev.get("cat") == "event" and ev.get("ph") == "X"]
    real_tasks = [t for t in all_tasks if "(fake)" not in t.get("name", "")]
    if not real_tasks:
        return None

    # Sort real tasks by start time
    real_tasks.sort(key=lambda t: t["ts"])

    # Split into clusters: a gap > 10ms between consecutive real tasks
    # indicates a new execution (idle period between test cases)
    cluster_gap_us = 10000  # 10ms in microseconds
    clusters = []
    current_cluster = [real_tasks[0]]
    for i in range(1, len(real_tasks)):
        prev_end = current_cluster[-1]["ts"] + current_cluster[-1]["dur"]
        curr_start = real_tasks[i]["ts"]
        if curr_start - prev_end > cluster_gap_us:
            clusters.append(current_cluster)
            current_cluster = [real_tasks[i]]
        else:
            current_cluster.append(real_tasks[i])
    clusters.append(current_cluster)

    latest_cluster = clusters[-1] if clusters else real_tasks

    global_first = min(t["ts"] for t in latest_cluster)
    global_last = max(t["ts"] + t["dur"] for t in latest_cluster)
    task_time = global_last - global_first
    aicore_time = sum(t["dur"] for t in latest_cluster)
    active_core_ids = set()
    for t in latest_cluster:
        core_name = tid2core.get(t["tid"], "")
        m = re.search(r"(\d+)$", core_name)
        if m:
            active_core_ids.add(int(m.group(1)))
    num_active_cores = len(active_core_ids) if active_core_ids else 1
    util = (aicore_time / (task_time * num_active_cores) * 100) if task_time > 0 else 0
    return task_time, aicore_time, util


def _record_perf_from_dirs(test_name, updated_dirs, kernel_filter=None):
    """Record perf data from output dirs, supporting merged kernels that
    produce multiple kernel types (e.g. both dQ and dK/dV) from a single dir.

    When _identify_kernel returns a list with multiple types, a separate perf
    record is created for each type, all using the same combined swimlane data.
    Dirs are deduplicated to avoid processing the same merged output twice.
    """
    seen_dirs = set()
    best_per_type = {}
    for d in updated_dirs:
        # Deduplicate dirs (important for merged kernels where both dQ and
        # dK/dV perf entries point to the same output directory)
        if d in seen_dirs:
            continue
        seen_dirs.add(d)

        ktypes = _identify_kernel(d)
        swim = os.path.join(d, "merged_swimlane.json")
        if not os.path.isfile(swim):
            continue
        mtime = os.path.getmtime(swim)

        for ktype in ktypes:
            if kernel_filter and ktype not in kernel_filter:
                continue
            if ktype == "Unknown":
                continue
            if ktype not in best_per_type or mtime > best_per_type[ktype][1]:
                best_per_type[ktype] = (d, mtime)

    for ktype, (d, _) in best_per_type.items():
        result = _parse_swimlane(d)
        if result is not None:
            task_time, aicore_time, util = result
            _perf_records.append({
                "test_name": test_name, "kernel": ktype,
                "task_time_us": task_time, "aicore_time_us": aicore_time,
                "utilization": util,
            })
        else:
            _perf_records.append({
                "test_name": test_name, "kernel": ktype,
                "task_time_us": None, "aicore_time_us": None, "utilization": None,
            })


def _print_perf_summary():
    if not _perf_records:
        logger.info("\n[PERF] No performance data collected.")
        return
    max_test_len = max(len(r["test_name"]) for r in _perf_records)
    w_test = max(max_test_len + 2, 20)
    w_kernel = 8
    w_task = 14
    w_aicore = 14
    w_util = 10

    def _fmt_us(val):
        if val is None:
            return "N/A".rjust(w_task)
        if val >= 1000:
            return f"{val / 1000:.2f} ms".rjust(w_task)
        return f"{val:.1f} us".rjust(w_task)

    def _fmt_pct(val):
        if val is None:
            return "N/A".rjust(w_util)
        return f"{val:.1f}%".rjust(w_util)

    sep = "+" + "-" * (w_test + 2) + "+" + "-" * (w_kernel + 2) + \
          "+" + "-" * (w_task + 2) + "+" + "-" * (w_aicore + 2) + \
          "+" + "-" * (w_util + 2) + "+"
    header = "|" + " Test Case".ljust(w_test + 1) + \
             "|" + " Kernel".ljust(w_kernel + 1) + \
             "|" + " Task Time".ljust(w_task + 1) + \
             "|" + " AICore Time".ljust(w_aicore + 1) + \
             "|" + " AICore Util".ljust(w_util + 1) + "|"

    logger.info("")
    logger.info("=" * len(sep))
    logger.info("Performance Summary (from merged_swimlane.json)")
    logger.info("=" * len(sep))
    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    current_test = None
    for rec in _perf_records:
        if rec["test_name"] != current_test:
            test_str = " " + rec["test_name"]
            current_test = rec["test_name"]
        else:
            test_str = ""
        row = "|" + test_str.ljust(w_test + 1) + \
              "|" + (" " + rec["kernel"]).ljust(w_kernel + 1) + \
              "|" + _fmt_us(rec["task_time_us"]) + " " + \
              "|" + _fmt_us(rec["aicore_time_us"]) + " " + \
              "|" + _fmt_pct(rec["utilization"]) + " |"
        logger.info(row)

    logger.info(sep)
    logger.info("")
    logger.info("Metric description (consistent with PyPTO toolkit definitions):")
    logger.info("  Task Time   = wall-clock elapsed time from the first task to the last task")
    logger.info("  AICore Time = sum of dur of all tasks (total computation)")
    logger.info("  AICore Util = AICoreTime / (TaskTime × active physical core count) × 100%")
    logger.info("=" * len(sep))


def _parse_swimlane_metrics(swim_path):
    """Parse merged_swimlane.json → (task_time_us, aicore_time_us, cores, util%, n_tasks).

    When the swimlane file accumulates events from multiple kernel executions,
    we isolate each execution by finding contiguous clusters of real-task events
    separated by large gaps (>10ms idle period between test cases).
    Returns metrics for the LAST cluster (the most recent kernel execution).
    """
    if not swim_path or not os.path.isfile(swim_path):
        return None
    with open(swim_path) as f:
        data = json.load(f)
    evts = [e for e in data.get('traceEvents', [])
            if e.get('ph') == 'X' and '(fake)' not in e.get('name', '')]
    if not evts:
        return None
    tid2core = {}
    for e in data.get('traceEvents', []):
        if e.get('ph') == 'M' and e.get('name') == 'thread_name':
            tid2core[e['tid']] = e['args']['name']
    active = set()
    for t in evts:
        c = tid2core.get(t['tid'], '')
        m = re.search(r'(\d+)$', c)
        if m:
            active.add(int(m.group(1)))
    cores = len(active) if active else 1

    # Sort events by start time and split into clusters
    evts.sort(key=lambda t: t['ts'])
    cluster_gap_us = 10000  # 10ms gap = new execution
    clusters = []
    current_cluster = [evts[0]]
    for i in range(1, len(evts)):
        prev_end = current_cluster[-1]['ts'] + current_cluster[-1]['dur']
        curr_start = evts[i]['ts']
        if curr_start - prev_end > cluster_gap_us:
            clusters.append(current_cluster)
            current_cluster = [evts[i]]
        else:
            current_cluster.append(evts[i])
    clusters.append(current_cluster)

    # Use the last cluster (most recent kernel execution)
    latest = clusters[-1] if clusters else evts
    g_first = min(t['ts'] for t in latest)
    g_last = max(t['ts'] + t['dur'] for t in latest)
    task_time = g_last - g_first
    aicore_time = sum(t['dur'] for t in latest)
    util = (aicore_time / (task_time * cores) * 100) if task_time > 0 else 0
    n_tasks = len(latest)
    return task_time, aicore_time, cores, util, n_tasks


def _strip_swimlane_copy(src_path, dst_path):
    """Copy merged_swimlane.json to dst, stripping Fake Core_0 events and
    C-phase counter events that inflate the timeline span with dispatch
    schedule bubbles.

    After stripping: timeline span == real TaskTime, no visual bubbles.
    """
    with open(src_path) as f:
        data = json.load(f)
    tid2core = {}
    for e in data.get('traceEvents', []):
        if e.get('ph') == 'M' and e.get('name') == 'thread_name':
            tid2core[e['tid']] = e['args']['name']
    fake_tids = {tid for tid, core in tid2core.items() if 'Fake' in core}
    stripped = [e for e in data['traceEvents']
                if e.get('tid') not in fake_tids and e.get('ph') != 'C']
    data['traceEvents'] = stripped
    with open(dst_path, 'w') as f:
        json.dump(data, f)


def _check_swimlane_in_dir(base_dir, newest_mtime, newest_swim):
    """Check one search base for the most recent merged_swimlane.json.

    Returns updated (newest_mtime, newest_swim) if a newer file is found.
    """
    if not os.path.isdir(base_dir):
        return newest_mtime, newest_swim
    for d in os.listdir(base_dir):
        swim = os.path.join(base_dir, d, 'merged_swimlane.json')
        if os.path.isfile(swim):
            mtime = os.path.getmtime(os.path.join(base_dir, d))
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest_swim = swim
    return newest_mtime, newest_swim


def _find_swimlane(search_bases):
    """Find the most recent merged_swimlane.json among search_bases.

    Args:
        search_bases: list of directory paths to search for output subdirs.
    """
    newest_swim = None
    newest_mtime = 0
    for base in search_bases:
        newest_mtime, newest_swim = _check_swimlane_in_dir(base, newest_mtime, newest_swim)
    return newest_swim


def _find_and_parse_swimlane(config):
    """Find and parse swimlane metrics for a single test case.

    First checks timestamp-tracked output dirs (from _collect_updated_dirs),
    then falls back to searching search_bases for the most recent file.
    Returns (row_dict, swim_path) for use in perf benchmark tables.

    Args:
        config: SwimlaneQueryConfig(search_bases, tag, name, b, hq, sq, wall_ms).
    """
    search_bases, tag, name, b, hq, sq, wall_ms = (
        config.search_bases, config.tag, config.name,
        config.b, config.hq, config.sq, config.wall_ms)
    updated_dirs = _collect_updated_dirs()
    swim_path = None
    newest_mtime = 0
    for out_dir in updated_dirs:
        swim = os.path.join(out_dir, 'merged_swimlane.json')
        if os.path.isfile(swim):
            mtime = os.path.getmtime(swim)
            if mtime > newest_mtime:
                newest_mtime = mtime
                swim_path = swim

    # Fallback to _find_swimlane if timestamp tracking didn't find anything
    if swim_path is None:
        swim_path = _find_swimlane(search_bases)

    result = _parse_swimlane_metrics(swim_path)

    row = {'name': name, 'BH': f"b={b},hq={hq}", 'Sq': sq, 'wall_ms': wall_ms}
    if result:
        task_time, aicore_time, cores, util, n_tasks = result
        row.update(task_time_us=task_time, aicore_time_us=aicore_time,
                   cores=cores, util=util, n_tasks=n_tasks)
        logger.info(f"  [{tag}] {name}: TaskTime={task_time}us "
                    f"AICoreTime={aicore_time}us Cores={cores} "
                    f"Util={util:.1f}% Tasks={n_tasks}")
    else:
        row.update(task_time_us=None, aicore_time_us=None,
                   cores=None, util=None, n_tasks=None)
        logger.info(f"  [{tag}] {name}: swimlane data not found")

    return row, swim_path


def _save_perf_case_data(swim_path, case_dir_name, *, extra_copy_dirs=None):
    """Save per-case swimlane data before the next case overwrites it.

    Strips merged_swimlane.json and copies supporting files (trace, analysis)
    to a named output directory. Also copies optional extra directories
    (e.g. kernel_aicore for BWD merged kernels).

    Args:
        swim_path: path to the source merged_swimlane.json.
        case_dir_name: name for the output subdirectory (spaces replaced with '_').
        extra_copy_dirs: optional list of (src_dir_name, dst_dir_name) pairs
            for directory-level copies (e.g. kernel_aicore).
    """
    if not swim_path or not os.path.isfile(swim_path):
        return
    case_dir = os.path.join(case_dir_name)
    os.makedirs(case_dir, exist_ok=True)
    _strip_swimlane_copy(swim_path, os.path.join(case_dir, 'merged_swimlane.json'))
    # Copy supporting files (perfetto trace, bubble analysis)
    for fname in ('machine_runtime_operator_trace.json', 'bubble_analysis.log'):
        src = os.path.join(os.path.dirname(swim_path), fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(case_dir, fname))
    # Copy extra directories (e.g. kernel_aicore for BWD)
    if extra_copy_dirs:
        for src_name, dst_name in extra_copy_dirs:
            kdir_src = os.path.join(os.path.dirname(swim_path), src_name)
            kdir_dst = os.path.join(case_dir, dst_name)
            if os.path.isdir(kdir_src) and not os.path.isdir(kdir_dst):
                shutil.copytree(kdir_src, kdir_dst)


def _clean_base_subdirs(base):
    """Remove all subdirectories inside base, keeping base itself intact."""
    if os.path.isdir(base):
        for name in os.listdir(base):
            d = os.path.join(base, name)
            if os.path.isdir(d):
                shutil.rmtree(d)
    os.makedirs(base, exist_ok=True)


def _clean_output_dirs(search_bases):
    """Remove timestamped output subdirs for clean swimlane data collection.

    Instead of removing the entire output directory (which prevents PyPTO from
    creating new subdirs), this function removes only the timestamped subdirs
    inside each base, keeping the parent directory intact. After cleaning,
    ensures each base directory exists (creating it if needed) so that PyPTO
    can write new output subdirs into it.

    Args:
        search_bases: list of directory paths whose subdirs to remove.
    """
    for base in search_bases:
        _clean_base_subdirs(base)


def _print_wall_table(rows, wall_label, wall_note):
    """Print wall-clock only performance table (extracted from _print_perf_table)."""
    w_case = 20
    w_bh = 14
    w_sq = 6
    w_wall = 14
    sep = "+" + "-" * w_case + "+" + "-" * w_bh + "+" + "-" * w_sq + "+" + "-" * w_wall + "+"
    hdr = "| " + "Case".ljust(w_case - 1) + \
          "| " + "BH".ljust(w_bh - 1) + \
          "| " + "Sq".ljust(w_sq - 1) + \
          "| " + "Wall(ms)".ljust(w_wall - 1) + "|"
    logger.info("")
    logger.info(sep)
    logger.info(hdr)
    logger.info(sep)
    for r in rows:
        wm = f"{r['wall_ms']:.2f}" if r.get('wall_ms') is not None else 'N/A'
        line = "| " + r['name'].ljust(w_case - 1) + \
               "| " + r['BH'].ljust(w_bh - 1) + \
               "| " + str(r['Sq']).ljust(w_sq - 1) + \
               "| " + wm.ljust(w_wall - 1) + "|"
        logger.info(line)
    logger.info(sep)
    logger.info("")
    logger.info(f"  {wall_label} = {wall_note}")
    logger.info("  NOTE: swimlane data unavailable — PyPTO/CANN SDK header incompatibility")


def _print_full_perf_table(rows):
    """Print full performance table with swimlane metrics (extracted from _print_perf_table)."""
    w_case = 20
    w_bh = 14
    w_sq = 6
    w_task = 14
    w_ai = 14
    w_cores = 7
    w_util = 8
    w_ntask = 7

    sep = "+" + "-" * w_case + "+" + "-" * w_bh + "+" + "-" * w_sq + \
          "+" + "-" * w_task + "+" + "-" * w_ai + "+" + "-" * w_cores + \
          "+" + "-" * w_util + "+" + "-" * w_ntask + "+"

    hdr = "| " + "Case".ljust(w_case - 1) + \
          "| " + "BH".ljust(w_bh - 1) + \
          "| " + "Sq".ljust(w_sq - 1) + \
          "| " + "TaskTime(us)".ljust(w_task - 1) + \
          "| " + "AICoreTime(us)".ljust(w_ai - 1) + \
          "| " + "Cores".ljust(w_cores - 1) + \
          "| " + "Util%".ljust(w_util - 1) + \
          "| " + "Tasks".ljust(w_ntask - 1) + "|"

    logger.info("")
    logger.info(sep)
    logger.info(hdr)
    logger.info(sep)

    for r in rows:
        tt = str(r['task_time_us']) if r['task_time_us'] is not None else 'N/A'
        ai = str(r['aicore_time_us']) if r['aicore_time_us'] is not None else 'N/A'
        co = str(r['cores']) if r['cores'] is not None else 'N/A'
        ut = f"{r['util']:.1f}" if r['util'] is not None else 'N/A'
        nt = str(r['n_tasks']) if r['n_tasks'] is not None else 'N/A'

        line = "| " + r['name'].ljust(w_case - 1) + \
               "| " + r['BH'].ljust(w_bh - 1) + \
               "| " + str(r['Sq']).ljust(w_sq - 1) + \
               "| " + tt.ljust(w_task - 1) + \
               "| " + ai.ljust(w_ai - 1) + \
               "| " + co.ljust(w_cores - 1) + \
               "| " + ut.ljust(w_util - 1) + \
               "| " + nt.ljust(w_ntask - 1) + "|"
        logger.info(line)

    logger.info(sep)
    logger.info("")
    logger.info("  TaskTime(us)   = wall-clock interval from first to last task")
    logger.info("  AICoreTime(us) = sum of all task durations (total compute)")
    logger.info("  Util%          = AICoreTime / (TaskTime x Cores) x 100")
    logger.info("  Cores          = active AICore physical core count")
    logger.info("  Tasks          = actual task count (excluding fake tasks)")


def _print_perf_table(rows, *, wall_only=False, wall_label="Wall(ms)",
                      wall_note="wall-clock time for kernel calls"):
    """Print performance table (dispatches to wall-only or full table)."""
    if not rows:
        logger.info("\n[PERF] No performance data collected.")
        return

    if wall_only:
        _print_wall_table(rows, wall_label, wall_note)
    else:
        _print_full_perf_table(rows)


def gen_inputs(config):
    """Generate random BSA inputs in FP16, BNSD layout.

    Args:
        config: GenInputsConfig namedtuple.
    """
    b, hq, hkv, sq, skv = config.b, config.hq, config.hkv, config.sq, config.skv
    sparsity, device, seed = config.sparsity, config.device, config.seed
    torch.manual_seed(seed)
    num_qb = math.ceil(sq / cfg.block_shape_x)
    num_kb = math.ceil(skv / cfg.block_shape_y)

    q = torch.empty(b, hq, sq, cfg.head_dim, dtype=cfg.torch_dtype, device=device).uniform_(-1, 1)
    k = torch.empty(b, hkv, skv, cfg.head_dim, dtype=cfg.torch_dtype, device=device).uniform_(-1, 1)
    v = torch.empty(b, hkv, skv, cfg.head_dim, dtype=cfg.torch_dtype, device=device).uniform_(-1, 1)
    d_o = torch.empty(b, hq, sq, cfg.head_dim, dtype=cfg.torch_dtype, device=device).uniform_(-1, 1)

    mask = generate_block_sparse_mask(
        b, hq, num_qb, num_kb, sparsity=sparsity, device=device, seed=seed)

    asq = torch.full([b], sq, dtype=torch.int64, device=device)
    askv = torch.full([b], skv, dtype=torch.int64, device=device)

    return BSATestInputs(q=q, k=k, v=v, d_o=d_o, mask=mask, asq=asq, askv=askv)


def gen_inputs_non_aligned(config):
    """Generate random BSA inputs with non-aligned/variable-length sequences.

    Args:
        config: GenInputsNonAlignedConfig namedtuple.

    Field descriptions:
        b: batch size
        hq, hkv: head counts
        sq_max: maximum Q sequence length (tensor shape dimension, block-aligned)
        skv_max: maximum KV sequence length (tensor shape dimension, block-aligned)
        actual_seq_lengths_q: list of per-batch actual Q lengths (e.g. [200, 300])
        actual_seq_lengths_kv: list of per-batch actual KV lengths (e.g. [256, 400])
        sparsity: mask sparsity ratio
        device: torch device
        seed: random seed
    """
    b, hq, hkv = config.b, config.hq, config.hkv
    sq_max, skv_max = config.sq_max, config.skv_max
    asq_list, askv_list = config.actual_seq_lengths_q, config.actual_seq_lengths_kv
    sparsity, device, seed = config.sparsity, config.device, config.seed
    torch.manual_seed(seed)
    num_qb = math.ceil(sq_max / cfg.block_shape_x)
    num_kb = math.ceil(skv_max / cfg.block_shape_y)

    q = torch.empty(b, hq, sq_max, cfg.head_dim, dtype=cfg.torch_dtype, device=device).uniform_(-1, 1)
    k = torch.empty(b, hkv, skv_max, cfg.head_dim, dtype=cfg.torch_dtype, device=device).uniform_(-1, 1)
    v = torch.empty(b, hkv, skv_max, cfg.head_dim, dtype=cfg.torch_dtype, device=device).uniform_(-1, 1)
    d_o = torch.empty(b, hq, sq_max, cfg.head_dim, dtype=cfg.torch_dtype, device=device).uniform_(-1, 1)

    mask = generate_block_sparse_mask(
        b, hq, num_qb, num_kb, sparsity=sparsity, device=device, seed=seed)

    asq = torch.tensor(asq_list, dtype=torch.int64, device=device)
    askv = torch.tensor(askv_list, dtype=torch.int64, device=device)

    return BSATestInputs(q=q, k=k, v=v, d_o=d_o, mask=mask, asq=asq, askv=askv)


def _compare_tensors(label, golden, actual, atol, rtol):
    """Compare two tensors, log result, and assert closeness."""
    g_np = golden.cpu().float().numpy().flatten()
    p_np = actual.cpu().float().numpy().flatten()
    diff = np.max(np.abs(p_np - g_np))
    assert_allclose(p_np, g_np, atol=atol, rtol=rtol)
    logger.info(f"    {label}: {list(actual.shape)} PASS (max_diff={diff:.6f})")
    return diff


def _compare_tensors_non_aligned(config):
    """Compare tensors for non-aligned scenarios, only checking valid positions.

    For output tensors of shape [B, Hq, Sq_max, D] or [B, Hq, Sq_max],
    only compares positions [:, :, :actual_seq_lengths[b], :] for each batch item.
    Positions beyond actual_seq_lengths[b] are masked out (should be zero in both
    golden and impl, but we skip them to avoid comparing meaningless padding data).

    Args:
        config: CompareNonAlignedConfig(label, golden, actual, atol, rtol, actual_seq_lengths).
    """
    label, golden, actual, atol, rtol, actual_seq_lengths = (
        config.label, config.golden, config.actual, config.atol, config.rtol, config.actual_seq_lengths)
    b_count = actual.shape[0]
    diffs = []
    for b in range(b_count):
        sq = actual_seq_lengths[b].item() if hasattr(actual_seq_lengths[b], 'item') else int(actual_seq_lengths[b])
        if actual.dim() == 4:
            g_slice = golden[b, :, :sq, :].cpu().float().numpy().flatten()
            p_slice = actual[b, :, :sq, :].cpu().float().numpy().flatten()
        elif actual.dim() == 3:
            g_slice = golden[b, :, :sq].cpu().float().numpy().flatten()
            p_slice = actual[b, :, :sq].cpu().float().numpy().flatten()
        elif actual.dim() == 2:
            g_slice = golden[b, :sq].cpu().float().numpy().flatten()
            p_slice = actual[b, :sq].cpu().float().numpy().flatten()
        else:
            g_slice = golden.cpu().float().numpy().flatten()
            p_slice = actual.cpu().float().numpy().flatten()
        if g_slice.size > 0 and p_slice.size > 0:
            diff = np.max(np.abs(p_slice - g_slice))
            diffs.append(diff)
            assert_allclose(p_slice, g_slice, atol=atol, rtol=rtol)
    max_diff = max(diffs) if diffs else 0.0
    logger.info(f"    {label}: {list(actual.shape)} PASS (max_diff={max_diff:.6f}, "
                f"checked {b_count} batches with variable Sq)")
    return max_diff


def _compare_grads(grad_pairs, atol, rtol):
    """Compare a list of (name, golden, actual) gradient pairs."""
    for grad_name, gg, gp in grad_pairs:
        g_np = gg.cpu().float().numpy().flatten()
        p_np = gp.cpu().float().numpy().flatten()
        diff = np.max(np.abs(p_np - g_np))
        assert_allclose(p_np, g_np, atol=atol, rtol=rtol)
        logger.info(f"    {grad_name}: PASS (max_diff={diff:.6f})")


def _extract_valid_slice(tensor, b, seq_len):
    """Extract valid slice from tensor for batch b up to seq_len."""
    if tensor.dim() == 4:
        return tensor[b, :, :seq_len, :].cpu().float().numpy().flatten()
    elif tensor.dim() == 3:
        return tensor[b, :, :seq_len].cpu().float().numpy().flatten()
    elif tensor.dim() == 2:
        return tensor[b, :seq_len].cpu().float().numpy().flatten()
    return tensor.cpu().float().numpy().flatten()


def _compare_grads_non_aligned(grad_pairs, atol, rtol,
                                actual_seq_lengths_q, actual_seq_lengths_kv):
    """Compare gradient pairs for non-aligned/variable-length scenarios.

    For each (name, golden, actual) pair:
      - dQ: shape [B, Hq, Sq_max, D] — compare [:, :, :asq[b], :]
      - dK: shape [B, Hkv, Skv_max, D] — compare [:, :, :askv[b], :]
      - dV: shape [B, Hkv, Skv_max, D] — compare [:, :, :askv[b], :]

    Positions beyond actual_seq_lengths are padding and skipped.
    """
    def _get_seq_len(g_name, b_idx):
        if g_name.startswith('dQ'):
            sl = actual_seq_lengths_q[b_idx]
            return sl.item() if hasattr(sl, 'item') else int(sl)
        sl = actual_seq_lengths_kv[b_idx]
        return sl.item() if hasattr(sl, 'item') else int(sl)

    b_count = actual_seq_lengths_q.shape[0]
    for grad_name, gg, gp in grad_pairs:
        diffs = []
        for b in range(b_count):
            seq_len = _get_seq_len(grad_name, b)
            g_slice = _extract_valid_slice(gg, b, seq_len)
            p_slice = _extract_valid_slice(gp, b, seq_len)
            if g_slice.size > 0 and p_slice.size > 0:
                diff = np.max(np.abs(p_slice - g_slice))
                diffs.append(diff)
                compare(torch.tensor(p_slice), torch.tensor(g_slice),
                        "bsa_out", atol=atol, rtol=rtol, max_error_ratio=0.005)

        max_diff = max(diffs) if diffs else 0.0
        logger.info(f"    {grad_name}: {list(gp.shape)} PASS (max_diff={max_diff:.6f}, "
                     f"checked {b_count} batches with variable seq lengths)")


def _run_test_cases(cases, runner_cfg):
    """Generic test case runner for both FWD and BWD tests.

    Args:
        cases: list of (name, b, hq, hkv, sq, skv, sparsity) tuples.
        runner_cfg: TestRunnerConfig(mode, precision_fn, perf_fn, perf_wall_fn, case_cfg_type).
    """
    mode = runner_cfg.mode
    precision_fn = runner_cfg.precision_fn
    perf_fn = runner_cfg.perf_fn
    perf_wall_fn = runner_cfg.perf_wall_fn
    case_cfg_type = runner_cfg.case_cfg_type
    passed = 0
    failed = 0
    perf_rows = []
    for case in cases:
        case_cfg = case_cfg_type(*case)
        logger.info(f"--- {case_cfg.name} ---")
        try:
            if mode == "precision":
                precision_fn(case_cfg)
                passed += 1
            elif mode in ("perf", "perf-table") and perf_fn:
                row = perf_fn(case_cfg)
                perf_rows.append(row)
            elif mode == "perf-wall" and perf_wall_fn:
                row = perf_wall_fn(case_cfg)
                perf_rows.append(row)
            logger.info(f"  >> DONE")
        except Exception as e:
            logger.error(f"  >> FAILED: {e}")
            failed += 1
    return passed, failed, perf_rows


def _run_non_aligned_test_cases(cases, non_aligned_fn,
                                 case_cfg_type=FwdNonAlignedCaseConfig):
    """Generic non-aligned test case runner.

    Args:
        cases: list of (name, b, hq, hkv, sq_max, skv_max, sp, asq_list, askv_list) tuples.
        non_aligned_fn: function(case_cfg) accepting FwdNonAlignedCaseConfig or BwdNonAlignedCaseConfig.
        case_cfg_type: FwdNonAlignedCaseConfig or BwdNonAlignedCaseConfig namedtuple type.
    """
    passed = 0
    failed = 0
    for case in cases:
        case_cfg = case_cfg_type(*case)
        logger.info(f"--- {case_cfg.name} (non-aligned) ---")
        try:
            non_aligned_fn(case_cfg)
            passed += 1
            logger.info(f"  >> PASSED")
        except Exception as e:
            logger.error(f"  >> FAILED: {e}")
            failed += 1
    return passed, failed


def _print_mode_results(passed, failed, perf_rows, mode):
    """Print results based on mode and return exit code."""
    if mode == "precision":
        _print_test_results(passed, failed, mode)
    elif mode in ("perf", "perf-table"):
        _print_perf_table(perf_rows)
    elif mode == "perf-wall":
        _print_perf_table(perf_rows, wall_only=True)
    return 0 if failed == 0 else 1


def _print_test_results(passed, failed, mode):
    """Print test results summary for both FWD and BWD tests.

    Args:
        passed: number of passed cases.
        failed: number of failed cases.
        mode: test mode string ("precision", "perf", etc.).
    """
    if mode in ("precision", "non-aligned"):
        logger.info("=" * 70)
        logger.info(f"Results: {passed}/{passed + failed} PASSED")
        logger.info("=" * 70)
        _print_perf_summary()
