#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""msprof Task Wait 深度分析脚本

用法:
  python3 analyze_msprof_taskwait.py /path/to/PROF_xxx/[/mindstudio_profiler_output/op_summary_*.csv]

支持:
  - 单目录分析: 直接传 PROF_ 目录或 op_summary CSV 路径
  - 对比模式: 传多个目录，自动标注 [0]/[1] 对比
"""

import csv
import glob
import logging
import os
import statistics
import sys
from collections import defaultdict

LOGGER = logging.getLogger(__name__)


def _output(message=""):
    LOGGER.info("%s", message)


def find_op_summary(prof_dir: str) -> str:
    """在 PROF_* 目录中查找 op_summary CSV"""
    # 直接路径
    if os.path.isfile(prof_dir) and prof_dir.endswith(".csv"):
        return prof_dir
    # PROF_xxx/mindstudio_profiler_output/op_summary_*.csv
    pattern = os.path.join(prof_dir, "mindstudio_profiler_output", "op_summary_*.csv")
    candidates = sorted(glob.glob(pattern))
    if candidates:
        return candidates[-1]  # 取最新的
    # 递归搜索
    for root, _, files in os.walk(prof_dir):
        for f in files:
            if f.startswith("op_summary_") and f.endswith(".csv"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"找不到 op_summary CSV: {prof_dir}")


def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _compute_op_stats(rows):
    """Iterate rows and compute total_wait, total_dur, per-op-type stats."""
    total_wait = 0.0
    total_dur = 0.0
    op_stats = defaultdict(lambda: {"wait": 0.0, "dur": 0.0, "cnt": 0})
    for row in rows:
        wait = float(row.get("Task Wait Time(us)", 0) or 0)
        dur = float(row.get("Task Duration(us)", 0) or 0)
        op_type = row.get("OP Type", "Unknown")
        total_wait += wait
        total_dur += dur
        op_stats[op_type]["wait"] += wait
        op_stats[op_type]["dur"] += dur
        op_stats[op_type]["cnt"] += 1
    return total_wait, total_dur, op_stats


def _print_global_stats(prefix, total_wait, total_dur, op_stats, n_rows):
    """Print global statistics and top-15 op-type ranking."""
    total = total_wait + total_dur
    wait_pct = total_wait / total * 100 if total > 0 else 0
    _output(f"\n{'=' * 70}")
    _output(f"{prefix}Overall Statistics")
    _output(
        f"  Total Tasks: {n_rows:,}  Total Wait: {total_wait / 1e6:.2f}s  "
        f"Total Duration: {total_dur / 1e6:.3f}s  Wait Ratio: {wait_pct:.1f}%"
    )
    sorted_ops = sorted(op_stats.items(), key=lambda x: x[1]["wait"], reverse=True)
    _output(f"\n{'=' * 70}")
    _output(f"{prefix}Task Wait Ranking by OP Type")
    _output(f"{'OP Type':28s} {'Count':>7s} {'Wait(s)':>8s} {'Dur(s)':>8s} {'AvgWait':>8s}")
    _output("-" * 62)
    for op_type, s in sorted_ops[:15]:
        avg = s["wait"] / s["cnt"] if s["cnt"] > 0 else 0
        _output(f"{op_type:28s} {s['cnt']:>7d} {s['wait'] / 1e6:>8.2f} {s['dur'] / 1e6:>8.3f} {avg:>8.1f}us")


def _analyse_single_layer_delay(rows, prefix):
    """Segment analysis around FlashAttentionScore ops, return seg_stats."""
    fa_indices = [i for i, r in enumerate(rows) if "FlashAttentionScore" in r.get("OP Type", "")]
    if len(fa_indices) >= 5:
        quarter = len(fa_indices) // 4
        start = min(quarter * 2, len(fa_indices) - 5)
        anchor = fa_indices[start:start + 5]  # 中间区域取 5 个
        _output(f"\n{'=' * 70}")
        _output(f"{prefix}Per-Layer Latency Analysis (FlashAttentionScore segments, Total FA Count={len(fa_indices)})")
        _output(f"{'Seg':>4s} {'Tasks':>6s} {'TotalDur':>8s} {'wait avg':>8s} {'median':>8s} {'max':>8s} {'min':>8s}")
        seg_stats = []
        for seg_id in range(len(anchor) - 1):
            seg = rows[anchor[seg_id] + 1:anchor[seg_id + 1]]
            waits = [float(r.get("Task Wait Time(us)", 0) or 0) for r in seg]
            durs = [float(r.get("Task Duration(us)", 0) or 0) for r in seg]
            if not waits:
                continue
            total_dur_seg = sum(durs)
            avg = statistics.mean(waits)
            med = statistics.median(waits)
            mx = max(waits)
            mn = min(waits)
            seg_stats.append({
                "tasks": len(seg),
                "dur": total_dur_seg,
                "wait_avg": avg, "wait_med": med,
                "wait_max": mx, "wait_min": mn,
            })
            _output(
                f"  {seg_id + 1:>2d}  {len(seg):>6d}  {total_dur_seg:>7.1f}us "
                f"{avg:>8.1f}us {med:>8.1f}us {mx:>8.1f}us {mn:>8.1f}us"
            )
        if seg_stats:
            first_seg = rows[anchor[0] + 1:anchor[1]]
            type_dist = defaultdict(int)
            for r in first_seg:
                type_dist[r.get("OP Type", "?")] += 1
            _output(f"\n  Per-Layer OP Distribution (first segment): {dict(type_dist)}")
        return seg_stats
    elif fa_indices:
        _output(f"\n  FlashAttentionScore count < 5 ({len(fa_indices)}), skipping per-layer analysis")
        return []
    else:
        _output("\n  FlashAttentionScore not found, skipping per-layer analysis")
        return []


def analyse_single(rows, label=""):
    """Analyze a single op_summary"""
    prefix = f"[{label}] " if label else ""
    total_wait, total_dur, op_stats = _compute_op_stats(rows)
    _print_global_stats(prefix, total_wait, total_dur, op_stats, len(rows))
    return _analyse_single_layer_delay(rows, prefix)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if len(sys.argv) < 2:
        _output(__doc__)
        sys.exit(1)

    csv_paths = []
    for arg in sys.argv[1:]:
        try:
            csv_paths.append(find_op_summary(arg))
        except FileNotFoundError as e:
            _output(f"WARNING: {e}")
            continue

    if not csv_paths:
        _output("No usable op_summary CSV")
        sys.exit(1)

    all_seg_stats = []
    for csv_path in csv_paths:
        label = os.path.basename(os.path.dirname(os.path.dirname(csv_path))) if len(csv_paths) > 1 else ""
        label = label[-20:] if label else ""
        _output(f"Source: {csv_path}")
        rows = load_csv(csv_path)
        segs = analyse_single(rows, label)
        if segs:
            all_seg_stats.append((label, segs))

    # 对比模式
    if len(all_seg_stats) >= 2:
        _output(f"\n{'=' * 70}")
        _output("Comparison Summary")
        _output(f"{'Label':>20s} {'PerLayer':>8s} {'Compute(us)':>9s} {'avgWait':>8s} {'medWait':>8s}")
        _output("-" * 62)
        for label, segs in all_seg_stats:
            avg_tasks = statistics.mean([s["tasks"] for s in segs])
            avg_dur = statistics.mean([s["dur"] for s in segs])
            avg_w = statistics.mean([s["wait_avg"] for s in segs])
            med_w = statistics.mean([s["wait_med"] for s in segs])
            _output(f"{label:>20s} {avg_tasks:>8.0f} {avg_dur:>9.1f} {avg_w:>8.1f}us {med_w:>8.1f}us")


if __name__ == "__main__":
    main()
