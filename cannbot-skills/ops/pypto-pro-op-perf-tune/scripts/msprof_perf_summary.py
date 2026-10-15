#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
# msprof 解析 & 归档 & 对比测试脚本（统一入口）
#
# 支持五种模式：
#   1. 标准模式 (默认): 解析 PROF_GROUP，生成 summary.txt 并归档 CSV
#      python3 msprof_perf_summary.py <PROF_GROUP_dir> <ops_dir>
#
#   2. 对比模式 (--compare): 按 manifest 采集 PyPTO target kernel；
#      Golden 完整覆盖时计算默认目标比值（Golden E2E / PyPTO target kernel）。
#      python3 msprof_perf_summary.py --compare --output-dir <op_dir> --case-manifest <path>
#
#   3. 批量模式 (--batch): 扫描多个算子目录，汇总批量报告
#      python3 msprof_perf_summary.py --batch <base_dir> [--output-md <path>] [--output-json <path>]
#
#   4. lowering 名称发现 (--list-op-names): 从 discovery profile 列出精确 Op Name
#
#   5. 流水时间线 (--timeline): 对 final compare 的一个 case 补采指令级时间线
#      python3 msprof_perf_summary.py --timeline --output-dir <op_dir> \
#        --case-manifest <path> --case-id <id> --op-name <exact-name>
#
# 归档位置 (与 perf_summary.py 保持一致):
#     <ops_dir>/docs/perf/round_NNN/
#         op_summary_<Metric>.csv    (7 份)
#         task_time.csv              (若存在)
#         op_statistic.csv           (若存在)
#         summary.txt                (合并后的统计摘要)
# ----------------------------------------------------------------------------------------------------------

import argparse
import csv
import glob
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from golden_contract import (  # noqa: E402  (standalone sibling module)
    performance_case_source_error,
)

LOGGER = logging.getLogger(__name__)

METRICS = [
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "L2Cache",
    "ResourceConflictRatio",
]

DEFAULT_GOLDEN_TARGET_THRESHOLD = 1.0
BOUND_ROUTE_HIGH_THRESHOLD = 0.80
BOUND_ROUTE_MAX_THRESHOLD = 0.70


# ============================================================================
# 通用工具函数
# ============================================================================

def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    s = str(val).strip().rstrip("\t ")
    if s in ("", "N/A", "NA", "-"):
        return default
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    return int(safe_float(val, default))


def _is_positive_finite(value: Any) -> bool:
    """Return whether value is a finite, strictly positive real number."""
    if isinstance(value, bool):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and parsed > 0


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def reserve_next_round(perf_dir: str, round_name: str = None) -> str:
    """Create the next local round directory for one operator collection."""
    os.makedirs(perf_dir, exist_ok=True)
    if round_name:
        if not re.fullmatch(r"round_\d+", round_name):
            raise ValueError("--round-name must match round_NNN")
        candidate = os.path.join(perf_dir, round_name)
        try:
            os.mkdir(candidate)
        except FileExistsError as exc:
            raise ValueError(f"round directory already exists: {candidate}") from exc
        return candidate
    index = 1
    while True:
        candidate = os.path.join(perf_dir, f"round_{index:03d}")
        try:
            os.mkdir(candidate)
            return candidate
        except FileExistsError:
            index += 1


# ============================================================================
# 标准模式：解析 PROF_GROUP
# ============================================================================


def find_next_round(perf_dir: str) -> str:
    if not os.path.exists(perf_dir):
        return os.path.join(perf_dir, "round_001")
    existing = [d for d in os.listdir(perf_dir) if re.match(r"round_\d+", d)]
    if not existing:
        return os.path.join(perf_dir, "round_001")
    nums = [int(re.search(r"\d+", d).group()) for d in existing]
    return os.path.join(perf_dir, f"round_{max(nums) + 1:03d}")


# ============================================================================
# 标准模式：解析 PROF_GROUP
# ============================================================================

def find_op_summary(prof_metric_dir: str, strict: bool = False) -> Optional[str]:
    pattern = os.path.join(prof_metric_dir, "**", "mindstudio_profiler_output", "op_summary_*.csv")
    hits = sorted(glob.glob(pattern, recursive=True))
    if strict:
        # manifest 协议：一次指标采集只应产出一份证据样本；多份时拒绝按路径排序猜测。
        return hits[0] if len(hits) == 1 else None
    # 既有行为：取最后一个命中。
    return hits[-1] if hits else None


def pick_target_row(
    rows: List[Dict[str, str]],
    target_name: Optional[str],
    strict: bool = False,
) -> Optional[Dict[str, str]]:
    if not rows:
        return None
    if target_name:
        matched = [r for r in rows if r.get("Op Name", "").strip() == target_name]
        if strict:
            # manifest 协议的正式样本必须唯一归属一次 measured launch；既有非严格路径保留取最大行。
            if len(matched) == 1:
                return matched[0]
            # 显式目标是证据合同而不是提示；取最长行会把其它 kernel 的计时归到拼错或缺失的目标上。
            return None
        if matched:
            return max(matched, key=lambda r: safe_float(r.get("Task Duration(us)")))
    ai_core_rows = []
    for r in rows:
        if "AI_CORE" in r.get("Task Type", "") \
                and "Op Name" in r:
            ai_core_rows.append(r)
    if ai_core_rows:
        return max(ai_core_rows, key=lambda r: safe_float(r.get("Task Duration(us)")))
    return None


def _merge_row_values(merged: Dict[str, Any], row: Dict[str, str]) -> None:
    for k, v in row.items():
        if k in (None, ""):
            continue
        if k not in merged:
            merged[k] = v
        else:
            old = merged[k]
            if (old in (None, "", "N/A", "NA")) and v not in (None, "", "N/A", "NA"):
                merged[k] = v
            elif safe_float(old) == 0 and safe_float(v) != 0:
                merged[k] = v


def merge_metric_rows(group_dir: str, target_name: Optional[str], strict: bool = False) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    merged["_metric_sources"] = {}
    merged["_missing_metrics"] = []

    for metric in METRICS:
        prof_metric_dir = os.path.join(group_dir, f"PROF_{metric}")
        if not os.path.isdir(prof_metric_dir):
            merged["_missing_metrics"].append(metric)
            continue
        csv_path = find_op_summary(prof_metric_dir)
        if not csv_path:
            merged["_missing_metrics"].append(metric)
            continue
        rows = read_csv_rows(csv_path)
        row = pick_target_row(rows, target_name)
        if not row:
            merged["_missing_metrics"].append(metric)
            continue
        merged["_metric_sources"][metric] = csv_path
        _merge_row_values(merged, row)
    return merged


def load_per_core_cycles(group_dir: str) -> List[Tuple[int, int]]:
    candidates = glob.glob(os.path.join(group_dir, "PROF_Sample", "PROF_*", "device_0", "sqlite", "aicore.db"))
    if not candidates:
        return []
    db = sorted(candidates)[-1]
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        rows = list(cur.execute(
            "SELECT coreid, SUM(task_cyc) FROM AICoreOriginalData WHERE task_cyc>0 GROUP BY coreid ORDER BY coreid"
        ))
        conn.close()
        return [(int(cid), int(cyc)) for cid, cyc in rows if cid is not None]
    except sqlite3.Error:
        return []


def load_per_core_cycles_scoped(group_dir: str) -> Tuple[List[Tuple[int, int]], str]:
    """manifest 协议：进程级逐核样本（device_*，唯一性要求 + 作用域标注）。"""
    candidates = glob.glob(
        os.path.join(group_dir, "PROF_Sample", "PROF_*", "device_*", "sqlite", "aicore.db")
    )
    if not candidates:
        return [], "unavailable:no_aicore_db"
    if len(candidates) != 1:
        LOGGER.warning(
            "Expected one sample-based aicore.db under %s, found %s; "
            "per-core evidence is ambiguous and will be omitted",
            group_dir, len(candidates),
        )
        return [], f"unavailable:ambiguous_aicore_db:{len(candidates)}"
    db = sorted(candidates)[-1]
    try:
        conn = sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro", uri=True)
        cur = conn.cursor()
        rows = list(cur.execute(
            "SELECT coreid, SUM(task_cyc) FROM AICoreOriginalData WHERE task_cyc>0 GROUP BY coreid ORDER BY coreid"
        ))
        conn.close()
        parsed = [(int(cid), int(cyc)) for cid, cyc in rows if cid is not None]
        if not parsed:
            return [], "unavailable:no_positive_core_cycles"
        return parsed, "process_scope_unattributed"
    except sqlite3.Error as error:
        return [], f"unavailable:sqlite_error:{error}"


def per_core_balance_section(merged: Dict[str, Any], group_dir: str) -> List[str]:
    core_rows = load_per_core_cycles(group_dir)
    if not core_rows:
        return []
    aicore_time_us = safe_float(merged.get("aicore_time(us)"))
    if aicore_time_us <= 0:
        return []
    max_cyc = max(c for _, c in core_rows)
    if max_cyc <= 0:
        return []
    ns_per_cyc = aicore_time_us * 1000.0 / max_cyc
    freq_ghz = 1.0 / ns_per_cyc
    times = [(cid, cyc * ns_per_cyc / 1000.0) for cid, cyc in core_rows]
    t_values = [t for _, t in times]
    t_min = min(t_values)
    t_max = max(t_values)
    t_avg = statistics.mean(t_values)
    spread_pct = (t_max - t_min) / t_max * 100.0 if t_max > 0 else 0.0

    if spread_pct < 10:
        verdict = "PASS (<10%)"
    elif spread_pct < 30:
        verdict = "Warning (10~30%)"
    else:
        verdict = "Critical issue (>30%)"

    lines = ["", "--- Per-core load balancing (sample-based aicore.db) ---"]
    lines.append(
        f"  Effective cores: {len(times)}  | Estimated frequency: "
        f"{freq_ghz:.3f} GHz ({ns_per_cyc:.4f} ns/cycle)"
    )
    lines.append(f"  min={t_min:.3f}us  avg={t_avg:.3f}us  max={t_max:.3f}us")
    lines.append(f"  (max-min)/max = {spread_pct:.2f}%  ->  {verdict}")

    sorted_desc = sorted(times, key=lambda x: -x[1])
    slow_top = sorted_desc[:3]
    fast_top = sorted_desc[-3:][::-1]
    lines.append("  Top-3 slowest cores: " + ", ".join(f"Core{cid}={t:.2f}us" for cid, t in slow_top))
    lines.append("  Top-3 fastest cores: " + ", ".join(f"Core{cid}={t:.2f}us" for cid, t in fast_top))

    sorted_by_id = sorted(times, key=lambda x: x[0])
    if len(sorted_by_id) >= 4:
        mid = len(sorted_by_id) // 2
        g1 = [t for _, t in sorted_by_id[:mid]]
        g2 = [t for _, t in sorted_by_id[mid:]]
        g1_avg = statistics.mean(g1)
        g2_avg = statistics.mean(g2)
        gap = abs(g1_avg - g2_avg) / max(g1_avg, g2_avg) * 100.0
        if gap >= 2.0:
            lines.append(
                f"  [Hint] first-half core avg {g1_avg:.2f}us vs second-half {g2_avg:.2f}us, "
                f"gap {gap:.2f}%"
            )
            lines.append(
                "         疑似两簇 (NUMA / L2 slice) 负载偏斜，"
                "建议尝试 block swat / 尾轮均衡策略。"
            )
    return lines


def archive_per_core_csv(group_dir: str, round_dir: str) -> Optional[str]:
    core_rows = load_per_core_cycles(group_dir)
    if not core_rows:
        return None
    out = os.path.join(round_dir, "per_core_cycles.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coreid", "task_cycles"])
        for cid, cyc in core_rows:
            w.writerow([cid, cyc])
    return out


def archive_per_core_csv_scoped(group_dir: str, round_dir: str) -> Tuple[Optional[str], str]:
    core_rows, scope = load_per_core_cycles_scoped(group_dir)
    if not core_rows:
        return None, scope
    filename = "per_core_cycles.csv" if scope == "target_op" else "process_scope_core_cycles.csv"
    out = os.path.join(round_dir, filename)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coreid", "task_cycles"])
        for cid, cyc in core_rows:
            w.writerow([cid, cyc])
    return out, scope


def archive_csvs(group_dir: str, round_dir: str) -> List[str]:
    os.makedirs(round_dir, exist_ok=True)
    copied = []
    for metric in METRICS:
        prof_metric_dir = os.path.join(group_dir, f"PROF_{metric}")
        if not os.path.isdir(prof_metric_dir):
            continue
        op_csv = find_op_summary(prof_metric_dir)
        if op_csv:
            dst = os.path.join(round_dir, f"op_summary_{metric}.csv")
            shutil.copy2(op_csv, dst)
            copied.append(os.path.basename(dst))
        mso_dir = os.path.dirname(op_csv) if op_csv else None
        if mso_dir:
            _copy_extra_csvs(mso_dir, metric, round_dir, copied)
    return copied


def archive_deep_profile(group_dir: str, case_dir: str,
                         target_name: Optional[str],
                         collection_id: Optional[str] = None,
                         case_name: Optional[str] = None) -> Optional[str]:
    """Validate and archive one seven-metric repeat for a case."""
    merged = merge_metric_rows(group_dir, target_name, strict=True)
    missing = merged.get("_missing_metrics", [])
    if missing or not merged.get("Op Name"):
        detail = ", ".join(missing) if missing else "target op row"
        return f"deep profile is incomplete: missing {detail}"

    copied = archive_csvs(group_dir, case_dir)
    if len([name for name in copied if name.startswith("op_summary_")]) != len(METRICS):
        return "deep profile archive did not contain all seven op_summary metrics"
    per_core, per_core_status = archive_per_core_csv_scoped(group_dir, case_dir)
    if per_core:
        copied.append(os.path.basename(per_core))
    collection_log = Path(group_dir).parent / "collection.log"
    if collection_log.is_file():
        shutil.copy2(collection_log, Path(case_dir) / "collection.log")
        copied.append("collection.log")
    from evidence_cli import diagnose_bound_route
    diagnosis = diagnose_bound_route(merged)
    summary = generate_summary(merged, case_dir, group_dir)
    with open(os.path.join(case_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(summary)
    with open(os.path.join(case_dir, "evidence_status.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "collection_id": collection_id,
            "case": case_name,
            "target_op_name": target_name,
            "seven_metric_status": "complete",
            "per_core_status": per_core_status,
            "per_core_claim_allowed": per_core_status == "target_op",
            "bound_diagnosis": diagnosis,
        }, handle, indent=2, ensure_ascii=False)
    return None


def _copy_extra_csvs(mso_dir: str, metric: str, round_dir: str, copied: List[str]) -> None:
    for extra in ("op_statistic_", "task_time_", "api_statistic_"):
        for f in sorted(glob.glob(os.path.join(mso_dir, f"{extra}*.csv"))):
            name = f"{extra.rstrip('_')}_{metric}.csv"
            dst = os.path.join(round_dir, name)
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
                copied.append(os.path.basename(dst))
            break


def fmt_ratio(val: Any, width: int = 6) -> str:
    # Non-pipe ratios such as resource-conflict ratios may exceed 1.  The
    # strict pipe-unit check belongs in ``diagnose_bound_route`` only.
    v = safe_float(val) * 100.0
    return f"{v:>{width}.2f}%"


def _add_memory_section(lines: List[str], merged: Dict[str, Any]) -> None:
    _mem_keys = [
        "aic_main_mem_read_bw(GB/s)", "aic_main_mem_write_bw(GB/s)",
        "aiv_main_mem_read_bw(GB/s)", "aiv_main_mem_write_bw(GB/s)",
        "aic_l1_read_bw(GB/s)", "aic_l1_write_bw(GB/s)",
        "aiv_ub_read_bw(GB/s)", "aiv_ub_write_bw(GB/s)",
    ]
    has_mem = any(safe_float(merged.get(k)) > 0 for k in _mem_keys)
    if not has_mem:
        return
    lines.append("")
    lines.append("--- Memory bandwidth (aic-metrics=Memory) ---")
    mem_rows = [
        ("aic main_mem read", "aic_main_mem_read_bw(GB/s)"),
        ("aic main_mem write", "aic_main_mem_write_bw(GB/s)"),
        ("aiv main_mem read", "aiv_main_mem_read_bw(GB/s)"),
        ("aiv main_mem write", "aiv_main_mem_write_bw(GB/s)"),
        ("aic L1 read", "aic_l1_read_bw(GB/s)"),
        ("aic L1 write", "aic_l1_write_bw(GB/s)"),
        ("aiv UB read", "aiv_ub_read_bw(GB/s)"),
        ("aiv UB write", "aiv_ub_write_bw(GB/s)"),
    ]
    for label, key in mem_rows:
        v = safe_float(merged.get(key))
        if v > 0:
            lines.append(f"  {label}: {v:.2f} GB/s")


def _add_memory_l0_section(lines: List[str], merged: Dict[str, Any]) -> None:
    _l0_keys = [
        "aic_l0a_read_bw(GB/s)", "aic_l0a_write_bw(GB/s)",
        "aic_l0b_read_bw(GB/s)", "aic_l0b_write_bw(GB/s)",
        "aic_l0c_read_bw_cube(GB/s)", "aic_l0c_write_bw_cube(GB/s)",
    ]
    has_l0 = any(safe_float(merged.get(k)) > 0 for k in _l0_keys)
    if not has_l0:
        return
    lines.append("")
    lines.append("--- MemoryL0 ---")
    for label, key in [
        ("L0A read", "aic_l0a_read_bw(GB/s)"),
        ("L0A write", "aic_l0a_write_bw(GB/s)"),
        ("L0B read", "aic_l0b_read_bw(GB/s)"),
        ("L0B write", "aic_l0b_write_bw(GB/s)"),
        ("L0C read (cube)", "aic_l0c_read_bw_cube(GB/s)"),
        ("L0C write (cube)", "aic_l0c_write_bw_cube(GB/s)"),
    ]:
        v = safe_float(merged.get(key))
        if v > 0:
            lines.append(f"  {label}: {v:.2f} GB/s")


def _add_memory_ub_section(lines: List[str], merged: Dict[str, Any]) -> None:
    _ub_keys = [
        "aiv_ub_read_bw_vector(GB/s)", "aiv_ub_write_bw_vector(GB/s)",
        "aiv_ub_read_bw_scalar(GB/s)", "aiv_ub_write_bw_scalar(GB/s)",
        "aic_ub_read_bw_scalar(GB/s)", "aic_ub_write_bw_scalar(GB/s)",
        "aiv_fixp2ub_write_bw(GB/s)", "aic_fixp2ub_write_bw(GB/s)",
    ]
    has_ub = any(safe_float(merged.get(k)) > 0 for k in _ub_keys)
    if not has_ub:
        return
    lines.append("")
    lines.append("--- MemoryUB ---")
    for label, key in [
        ("UB read (vector)", "aiv_ub_read_bw_vector(GB/s)"),
        ("UB write (vector)", "aiv_ub_write_bw_vector(GB/s)"),
        ("UB read (scalar)", "aiv_ub_read_bw_scalar(GB/s)"),
        ("UB write (scalar)", "aiv_ub_write_bw_scalar(GB/s)"),
        ("aic UB read (scalar)", "aic_ub_read_bw_scalar(GB/s)"),
        ("aic UB write (scalar)", "aic_ub_write_bw_scalar(GB/s)"),
        ("aiv fixp2ub write", "aiv_fixp2ub_write_bw(GB/s)"),
        ("aic fixp2ub write", "aic_fixp2ub_write_bw(GB/s)"),
    ]:
        v = safe_float(merged.get(key))
        if v > 0:
            lines.append(f"  {label}: {v:.2f} GB/s")


def _add_l2cache_section(lines: List[str], merged: Dict[str, Any]) -> None:
    l2_fields_aic = [
        ("aic read hit", "aic_read_local_l2_hit"),
        ("aic read miss", "aic_read_local_l2_miss"),
        ("aic read victim", "aic_read_local_l2_victim"),
        ("aic write hit", "aic_write_local_l2_hit"),
        ("aic write miss", "aic_write_local_l2_miss"),
        ("aic write victim", "aic_write_local_l2_victim"),
    ]
    l2_fields_aiv = [
        ("aiv read hit", "aiv_read_local_l2_hit"),
        ("aiv read miss", "aiv_read_local_l2_miss"),
        ("aiv read victim", "aiv_read_local_l2_victim"),
        ("aiv write hit", "aiv_write_local_l2_hit"),
        ("aiv write miss", "aiv_write_local_l2_miss"),
        ("aiv write victim", "aiv_write_local_l2_victim"),
    ]
    l2_has = any(safe_float(merged.get(k)) > 0 for _, k in l2_fields_aic + l2_fields_aiv)
    if not l2_has:
        return
    lines.append("")
    lines.append("--- L2Cache ---")

    def _emit(group_label, fields):
        hit = safe_float(merged.get(fields[0][1]))
        miss = safe_float(merged.get(fields[1][1]))
        total = hit + miss
        if total > 0:
            rate = hit / total * 100.0
            lines.append(f"  {group_label} read: hit={int(hit)} miss={int(miss)} hit_rate={rate:.2f}%")
        whit = safe_float(merged.get(fields[3][1]))
        wmiss = safe_float(merged.get(fields[4][1]))
        wtotal = whit + wmiss
        if wtotal > 0:
            rate = whit / wtotal * 100.0
            lines.append(f"  {group_label} write: hit={int(whit)} miss={int(wmiss)} hit_rate={rate:.2f}%")

    _emit("aic", l2_fields_aic)
    _emit("aiv", l2_fields_aiv)


def _add_rc_section(lines: List[str], merged: Dict[str, Any]) -> None:
    rc_fields = [
        ("vec_bank_cflt", "aiv_vec_bank_cflt_ratio"),
        ("vec_resc_cflt", "aiv_vec_resc_cflt_ratio"),
    ]
    if not any(safe_float(merged.get(k)) > 0 for _, k in rc_fields):
        return
    lines.append("")
    lines.append("--- ResourceConflict ---")
    parts = [f"{label}={fmt_ratio(merged.get(key))}" for label, key in rc_fields]
    lines.append("  " + " | ".join(parts))


def _add_arith_section(lines: List[str], merged: Dict[str, Any]) -> None:
    arith_fields = [
        ("mac_fp16", "aic_mac_fp16_ratio"),
        ("mac_int8", "aic_mac_int8_ratio"),
    ]
    has_arith_fields = any(
        safe_float(merged.get(k)) > 0 for _, k in arith_fields
    )
    arith_has = has_arith_fields or safe_float(merged.get("aic_cube_fops")) > 0
    if not arith_has:
        return
    lines.append("")
    lines.append("--- ArithmeticUtilization ---")
    parts = []
    for label, key in arith_fields:
        v = safe_float(merged.get(key))
        if v > 0:
            parts.append(f"{label}={fmt_ratio(merged.get(key))}")
    fops = safe_float(merged.get("aic_cube_fops"))
    if fops > 0:
        parts.append(f"cube_fops={fops:.0f}")
    if parts:
        lines.append("  " + " | ".join(parts))


def _add_basic_info(lines: List[str], merged: Dict[str, Any]) -> Tuple[float, float, float]:
    op_name = merged.get("Op Name", "unknown")
    op_type = merged.get("OP Type", "unknown")
    task_type = merged.get("Task Type", "")
    duration = safe_float(merged.get("Task Duration(us)"))
    block_dim = safe_int(merged.get("Block Num", 0))
    mix_block = safe_int(merged.get("Mix Block Num", 0))
    aicore_time = safe_float(merged.get("aicore_time(us)"))
    aiv_time = safe_float(merged.get("aiv_time(us)"))

    lines.append("=== On-device performance statistics summary (msprof) ===")
    lines.append(f"Op: {op_name}")
    lines.append(
        f"Type: {op_type} | TaskType: {task_type} | Duration: {duration}us"
        f" | BlockDim: {block_dim} (mix={mix_block})"
    )
    if merged.get("_missing_metrics"):
        lines.append(f"[WARN] Missing metrics: {', '.join(merged['_missing_metrics'])}")
    lines.append("")
    lines.append("[Note] msprof op_summary is a per-op aggregate value (not per-core min/avg/max);")
    lines.append("     For per-core data, switch to msprof op (requires the msopprof binary).")
    return duration, aicore_time, aiv_time


def _add_pipe_ratios(lines: List[str], merged: Dict[str, Any], aicore_time: float, aiv_time: float) -> None:
    lines.append("")
    aic_cube_like = max(safe_float(merged.get("aic_mac_ratio")), safe_float(merged.get("aic_mte2_ratio")))
    aiv_vec_like = safe_float(merged.get("aiv_vec_ratio"))
    prefix = "aic" if aic_cube_like >= aiv_vec_like else "aiv"
    lines.append(f"--- Pipe ratios (dominant core = {prefix}) ---")
    lines.append(f"  aicore_time: {aicore_time:.3f}us | aiv_time: {aiv_time:.3f}us")

    aic_fields = [
        ("aic_mac_ratio", "mac"),
        ("aic_cube_ratio", "cube"),
        ("aic_mte1_ratio", "mte1"),
        ("aic_mte2_ratio", "mte2"),
        ("aic_mte3_ratio", "mte3"),
        ("aic_fixpipe_ratio", "fixpipe"),
        ("aic_scalar_ratio", "scalar"),
        ("aic_icache_miss_rate", "icache_miss"),
    ]
    parts = [f"{label}={fmt_ratio(merged.get(key))}" for key, label in aic_fields if safe_float(merged.get(key)) > 0]
    if parts:
        lines.append("  aic: " + " | ".join(parts))

    aiv_fields = [
        ("aiv_vec_ratio", "vec"),
        ("aiv_scalar_ratio", "scalar"),
        ("aiv_mte2_ratio", "mte2"),
        ("aiv_mte3_ratio", "mte3"),
        ("aiv_icache_miss_rate", "icache_miss"),
    ]
    parts = [f"{label}={fmt_ratio(merged.get(key))}" for key, label in aiv_fields if safe_float(merged.get(key)) > 0]
    if parts:
        lines.append("  aiv: " + " | ".join(parts))

    util = safe_float(merged.get("cube_utilization(%)"))
    if util > 0:
        lines.append(f"  cube_utilization: {util:.2f}%")


def _add_overhead(lines: List[str], duration: float, aicore_time: float, aiv_time: float) -> None:
    lines.append("")
    core_time_max = max(aicore_time, aiv_time)
    overhead = max(0.0, duration - core_time_max)
    overhead_pct = (overhead / duration * 100.0) if duration > 0 else 0.0
    lines.append("--- Overhead ---")
    lines.append(
        f"  Task Duration: {duration}us | max core time: {core_time_max:.3f}us"
        f" | overhead: {overhead:.3f}us ({overhead_pct:.1f}%)"
    )


def _add_footer(lines: List[str], merged: Dict[str, Any], round_dir: str, group_dir: str) -> None:
    lines.append("")
    lines.append("--- Raw data location ---")
    lines.append(f"  Archived CSV : {round_dir}/")
    lines.append(f"  source PROF (compare default cleanup): {group_dir}/")
    lines.append("  op_summary_<Metric>.csv split by aic-metrics have all been copied to the archive directory, ")
    lines.append("  for per-column inspection, just Read.")
    if merged.get("_metric_sources"):
        lines.append("")
        lines.append("--- Metric source ---")
        for m in METRICS:
            src = merged["_metric_sources"].get(m)
            if src:
                lines.append(f"  {m:<22s} <- {src}")
            else:
                lines.append(f"  {m:<22s} <MISSING>")


def generate_summary(merged: Dict[str, Any], round_dir: str, group_dir: str) -> str:
    lines: List[str] = []
    duration, aicore_time, aiv_time = _add_basic_info(lines, merged)
    _add_pipe_ratios(lines, merged, aicore_time, aiv_time)
    _add_pipe_ratios(lines, merged, aicore_time, aiv_time)
    from evidence_cli import _add_bound_route
    _add_bound_route(lines, merged)
    _add_overhead(lines, duration, aicore_time, aiv_time)
    _add_memory_section(lines, merged)
    _add_memory_l0_section(lines, merged)
    _add_memory_ub_section(lines, merged)
    _add_l2cache_section(lines, merged)
    _add_rc_section(lines, merged)
    _add_arith_section(lines, merged)
    lines.extend(per_core_balance_section(merged, group_dir))
    _add_footer(lines, merged, round_dir, group_dir)
    return "\n".join(lines)


# ============================================================================


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _manifest_case_functions(args) -> Dict[str, str]:
    source = getattr(args, "performance_cases", None)
    if not isinstance(source, dict) or source.get("type") != "case_manifest":
        return {}
    value = source.get("case_functions")
    return value if isinstance(value, dict) else {}


def uses_function_adapter(args, case_ids=None) -> bool:
    if getattr(args, "case_arg", None) or getattr(args, "case_env", None):
        return False
    mapping = _manifest_case_functions(args)
    expected = {str(case_id) for case_id in (case_ids or mapping)}
    return bool(expected) and set(mapping) == expected


def profile_case_contract_error(cases, args):
    if len(cases) <= 1:
        return None
    if uses_function_adapter(args, [case[0] for case in cases]):
        return None
    return (
        f"the selected performance case source contains {len(cases)} cases, but test_<op>.py "
        "needs an explicit per-case selector. Refusing to reuse one aggregate "
        "msprof duration for every case; rerun with --case-arg/--case-env, or add a "
        "validated test_function to every manifest case for the function adapter."
    )


def _find_test_script(out_dir: Path, strict: bool = False):
    test_scripts = sorted(out_dir.glob("test_*.py"))
    if not test_scripts:
        return None, "no test_*.py found in operator directory"
    if strict:
        # manifest 协议：优先 test_{dirname}.py；多个候选时拒绝猜测。
        preferred = out_dir / f"test_{out_dir.name}.py"
        if preferred in test_scripts:
            return str(preferred), None
        if len(test_scripts) == 1:
            return str(test_scripts[0]), None
        names = ", ".join(path.name for path in test_scripts)
        return None, (
            f"multiple test_*.py files found ({names}); expected the runner "
            f"test_{out_dir.name}.py and refusing to guess"
        )
    # 既有行为：取排序后第一个。
    return str(test_scripts[0]), None


def _selected_executable_sha256(out_dir: Path):
    """Resolve the runner under the manifest protocol and hash its current bytes."""
    test_script, error = _find_test_script(out_dir, strict=True)
    if not test_script:
        return None, None, error
    try:
        digest = hashlib.sha256(Path(test_script).read_bytes()).hexdigest()
    except OSError as source_error:
        return None, None, f"cannot hash runner executable: {source_error}"
    return test_script, digest, None


def _resolved_evidence_round(op_dir: Path, raw_round: Any) -> Path:
    if raw_round is None or not str(raw_round).strip():
        raise ValueError("deep_profile_round is missing")
    perf_root = (op_dir / "docs" / "perf").resolve(strict=True)
    candidate = Path(str(raw_round))
    if not candidate.is_absolute():
        candidate = op_dir / candidate
    probe = candidate.absolute()
    while probe != perf_root and probe != probe.parent:
        if probe.is_symlink():
            raise ValueError(f"evidence path must not contain symlinks: {probe}")
        probe = probe.parent
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(perf_root)
    except ValueError as error:
        raise ValueError(f"evidence round resolves outside {perf_root}: {resolved}") from error
    return resolved


def _find_msprof_script():
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir / "msprof_profile_run.sh"
    if candidate.exists():
        return str(candidate)
    return "msprof_profile_run.sh"


def _profile_env(device_id, seed=None, case_env=None, case_name=None, tile_fwk=False):
    env = os.environ.copy()
    if tile_fwk:
        # manifest 协议：PyPTO-Pro runner 用 TILE_FWK_DEVICE_ID 作物理设备 id。
        # 同时设置 visibility mask 会把物理卡重编号为逻辑 0，使 TILE_FWK_DEVICE_ID>0 失效。
        env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
        env["TILE_FWK_DEVICE_ID"] = str(device_id)
    else:
        # 既有行为：visible mask + 种子环境变量透传给 runner。
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
        env["PYPTO_PERF_SEED"] = str(seed)
        env["PYTHONHASHSEED"] = str(seed)
    if case_env:
        env[case_env] = str(case_name)
    return env


def _l2_cache_clean(env):
    """在 profiled run 之前刷新目标 NPU 的 L2 cache（cannbench 式）。

    单独进程执行，落在 msprof 采集窗口之外；失败只告警，不中断采集。
    """
    helper = Path(__file__).resolve().parent / "l2_cache_clean.py"
    result = subprocess.run(
        [sys.executable, str(helper)], capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        LOGGER.warning("[WARN] L2 cache clean failed: %s", result.stderr[-300:])


@dataclass(frozen=True)
class RunSelector:
    """逐 case 选择器参数组（case-arg/case-env/manifest 与适配器开关）。"""
    case_arg: Optional[str] = None
    case_env: Optional[str] = None
    case_name: Optional[str] = None
    case_manifest: Optional[str] = None
    function_adapter: bool = False


@dataclass(frozen=True)
class RunRequest:
    """单次采集请求（runner、输出目录与协议参数）。"""
    test_script: str
    output_dir: str
    warmup: int = 3
    device_id: int = 0
    seed: int = 0
    selector: RunSelector = RunSelector()
    l2_clean: bool = True


def _test_command(test_script, selector: RunSelector):
    if selector.function_adapter:
        return [
            sys.executable, str(Path(__file__).resolve()),
            "--run-case-function", "--test-script", str(test_script),
            "--case-manifest", str(selector.case_manifest),
            "--case-id", str(selector.case_name),
        ]
    command = [sys.executable, test_script]
    if selector.case_arg:
        command.extend([selector.case_arg, str(selector.case_name)])
    return command


def _run_msprof_standard(request: RunRequest):
    """调用 msprof_profile_run.sh 进行完整采集（7 组 aic-metrics + sample-based）。

    直接采集 test_{op}.py 脚本，不生成 wrapper。
    """
    os.makedirs(request.output_dir, exist_ok=True)
    selector = request.selector
    cmd = [
        "bash", _find_msprof_script(),
        f"--warm-up={request.warmup}",
        f"--output={request.output_dir}",
        "--l2-clean" if request.l2_clean else "--no-l2-clean",
        "--",
        *_test_command(request.test_script, selector)
    ]
    env = _profile_env(
        request.device_id, request.seed, selector.case_env,
        selector.case_name, bool(selector.case_manifest),
    )
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if selector.case_manifest:
        _write_collection_log(request.output_dir, cmd, result)

    if result.returncode != 0:
        return None, f"msprof failed: {result.stderr[-500:]}"

    prof_dirs = sorted(Path(request.output_dir).glob("PROF_GROUP_*"))
    if not prof_dirs:
        return None, "no PROF_GROUP directory found"
    return str(prof_dirs[-1]), None


def _run_warmups(request: RunRequest, env):
    for _ in range(max(0, request.warmup)):
        result = subprocess.run(
            _test_command(request.test_script, request.selector),
            capture_output=True, text=True, env=env
        )
        if result.returncode != 0:
            return f"warmup failed: {result.stderr[-500:]}"
    return None


def _run_msprof_quick(request: RunRequest):
    """快速模式：单次采集只获取 kernel 时间，不采集 7 个 aic-metrics。

    直接调用 msprof 命令（不通过 msprof_profile_run.sh，避免循环调用）。
    直接采集 test_{op}.py 脚本，不生成 wrapper。
    """
    os.makedirs(request.output_dir, exist_ok=True)
    selector = request.selector
    env = _profile_env(
        request.device_id, request.seed, selector.case_env,
        selector.case_name, bool(selector.case_manifest),
    )
    warmup_error = _run_warmups(request, env)
    if warmup_error:
        return None, warmup_error
    if request.l2_clean:
        _l2_cache_clean(env)
    cmd = [
        "msprof",
        f"--output={request.output_dir}",
        "--task-time=on",
        "--ascendcl=on",
        *_test_command(request.test_script, selector)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if selector.case_manifest:
        _write_collection_log(request.output_dir, cmd, result)

    if result.returncode != 0:
        return None, f"msprof failed: {result.stderr[-500:]}"

    prof_dirs = sorted(Path(request.output_dir).glob("PROF_*"))
    if not prof_dirs:
        return None, "no PROF directory found"
    return str(prof_dirs[-1]), None


def _canonical_timing_csv(prof_group_dir: str):
    """Return the sole PipeUtilization op_summary used as compare timing truth."""
    pattern = os.path.join(
        prof_group_dir,
        "PROF_PipeUtilization", "PROF_*", "mindstudio_profiler_output", "op_summary_*.csv",
    )
    csv_files = sorted(glob.glob(pattern))
    if len(csv_files) != 1:
        return None, (
            "expected exactly one canonical PipeUtilization op_summary csv; "
            f"found {len(csv_files)}"
        )
    return csv_files[0], None


def _parse_msprof_duration(prof_group_dir: str, op_name: str = None, strict: bool = False):
    if strict:
        csv_path, source_error = _canonical_timing_csv(prof_group_dir)
        if not csv_path:
            return None, None, source_error
    else:
        # 既有行为：PROF_* 下任一 op_summary CSV，取排序后第一个。
        csv_pattern = os.path.join(prof_group_dir, "PROF_*/PROF_*/mindstudio_profiler_output/op_summary_*.csv")
        csv_files = sorted(glob.glob(csv_pattern))
        if not csv_files:
            return None, None, "no op_summary csv found"
        csv_path = csv_files[0]

    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        return None, None, f"cannot read op_summary csv: {e}"

    target = pick_target_row(rows, op_name, strict)
    if target is None:
        if strict:
            exact_count = sum(
                1 for row in rows if row.get("Op Name", "").strip() == op_name
            )
            return None, None, (
                f"expected exactly one row for op_name={op_name}; found {exact_count}. "
                "Use a runner that emits one measured target launch per profiling process, "
                "or add a verified occurrence/correlation selector."
            )
        return None, None, f"no matching row for op_name={op_name}"

    duration = float(target.get("Task Duration(us)", 0) or 0)
    if strict and (not math.isfinite(duration) or duration <= 0):
        return None, None, f"invalid Task Duration for op_name={op_name}: {duration}"
    name = target.get("Op Name", "unknown")
    return duration, name, None


def _write_collection_log(output_dir, command, result):
    """Persist the collection command and result without serializing the environment."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    payload = [
        f"command_argv={json.dumps(command, ensure_ascii=False)}",
        f"returncode={result.returncode}",
        "--- stdout ---",
        result.stdout or "",
        "--- stderr ---",
        result.stderr or "",
    ]
    (Path(output_dir) / "collection.log").write_text(
        "\n".join(payload), encoding="utf-8"
    )


def _read_profiler_csv(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as csv_file:
            return list(csv.DictReader(csv_file)), None
    except (OSError, csv.Error) as exc:
        return None, str(exc)


def _collect_kernel_times(rows):
    kernel_times: Dict[str, List[float]] = {}
    ignored_types = {"PROFILING_ENABLE", "TASK_TIMEOUT_SET", ""}
    for row in rows:
        task_time = row.get("task_time(us)", "")
        if row.get("kernel_type", "") in ignored_types or not task_time:
            continue
        try:
            duration = float(task_time)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            kernel_times.setdefault(row.get("kernel_name", "unknown"), []).append(duration)
    return kernel_times


def _select_kernel_times(kernel_times, op_name, strict: bool = False):
    if not kernel_times:
        return None, None, None
    if op_name and op_name not in kernel_times:
        return None, None, f"no task_time entry for op_name={op_name}"
    kernel_name = op_name or max(
        kernel_times.items(), key=lambda item: (sum(item[1]), len(item[1]))
    )[0]
    times = kernel_times[kernel_name]
    if strict and op_name and len(times) != 1:
        return None, None, (
            f"expected exactly one task_time entry for op_name={op_name}; "
            f"found {len(times)}. A quick profiling process must contain one "
            "measured target launch unless a verified occurrence/correlation "
            "selector is implemented."
        )
    if len(times) >= 3:
        duration = statistics.mean(sorted(times)[1:-1])
    else:
        duration = statistics.median(times)
    return duration, kernel_name, None


def _parse_task_time(path, op_name):
    rows, error = _read_profiler_csv(path)
    if error:
        return None, None, "no task_time or api_statistic csv found"
    return _select_kernel_times(_collect_kernel_times(rows), op_name)


def _parse_api_statistic(path):
    rows, error = _read_profiler_csv(path)
    if error:
        return None, None, "no task_time or api_statistic csv found"
    for row in rows:
        if row.get("Level", "") != "node" or row.get("API Name", "") != "launch":
            continue
        try:
            duration = float(row.get("Time(us)", ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            return duration, "launch", None
    return None, None, None


def _parse_msprof_duration_quick(prof_group_dir: str, op_name: str = None, strict: bool = False):
    """快速解析 task_time，缺失时回退到 api_statistic 的 launch 行。"""
    output_dir = os.path.join(prof_group_dir, "mindstudio_profiler_output")
    task_time_files = sorted(glob.glob(os.path.join(output_dir, "task_time_*.csv")))
    if task_time_files:
        result = _parse_task_time(task_time_files[0], op_name)
        if result[2] is not None or result[0] is not None:
            return result
    if strict and op_name:
        return None, None, (
            f"no exact task_time entry for op_name={op_name}; "
            "refusing to substitute host API launch time"
        )
    api_files = sorted(glob.glob(os.path.join(output_dir, "api_statistic_*.csv")))
    if api_files:
        result = _parse_api_statistic(api_files[0])
        if result[2] is not None or result[0] is not None:
            return result
    return None, None, "no task_time or api_statistic csv found"


def pick_idle_npu(default=0):
    try:
        p = subprocess.run(["/usr/local/bin/npu-smi", "info"], capture_output=True, text=True, timeout=10)
        if p.returncode != 0:
            return default
    except Exception:
        return default

    devices = {}
    cur = None
    head_re = re.compile(r"^\|\s+(\d+)\s+\S+\s+\|\s+\w+\s+\|")
    bus_re = re.compile(
        r"^\|\s+\d+\s+\|\s+[0-9A-Fa-f:.]+\s+\|\s+(\d+)\s+(\d+)\s*/\s*(\d+)(?:\s+(\d+)\s*/\s*(\d+))?"
    )
    for line in p.stdout.splitlines():
        m2 = bus_re.match(line)
        if m2 and cur is not None:
            aicore = int(m2.group(1))
            mem_used = int(m2.group(2))
            mem_total = max(int(m2.group(3)), 1)
            hbm_used = int(m2.group(4)) if m2.group(4) else 0
            hbm_total = max(int(m2.group(5)), 1) if m2.group(5) else 1
            mem_ratio = max(mem_used / mem_total, hbm_used / hbm_total)
            devices[cur] = (aicore, mem_ratio)
            cur = None
            continue
        m1 = head_re.match(line)
        if m1:
            cur = int(m1.group(1))

    if not devices:
        return default
    best_id, _ = min(devices.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    return best_id


def _select_device_id(args, tile_fwk: bool = False):
    """Select NPU device from CLI arg, env var, or auto-detect."""
    if args.device is not None:
        return args.device, "cli"
    if tile_fwk:
        # manifest 协议：TILE_FWK_DEVICE_ID 是物理 id；只有非零 visibility mask 且未设置
        # TILE_FWK_DEVICE_ID 时拒绝猜测（mask 会把物理卡重编号为逻辑 0）。
        if os.environ.get("TILE_FWK_DEVICE_ID"):
            return int(os.environ["TILE_FWK_DEVICE_ID"].split(",")[0]), "env.TILE_FWK_DEVICE_ID"
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
            visible = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")[0]
            if visible.strip() not in ("", "0"):
                raise ValueError(
                    "ASCEND_RT_VISIBLE_DEVICES masks/renumbers devices but TILE_FWK_DEVICE_ID "
                    "is unset; pass --device=<physical-id> or unset the visibility mask"
                )
            return 0, "env.ASCEND_RT_VISIBLE_DEVICES.logical0"
        return pick_idle_npu(default=0), "auto"
    # 既有行为：visible mask 首项或自动探测空闲卡。
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        return int(os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")[0]), "env"
    return pick_idle_npu(default=0), "auto"


def _aggregate_durations(durations):
    if len(durations) >= 3:
        return statistics.mean(sorted(durations)[1:-1])
    return statistics.median(durations)


def safe_case_dir_name(case_name):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_name)).strip("._")
    digest = hashlib.sha256(str(case_name).encode("utf-8")).hexdigest()[:8]
    return f"{value or 'case'}_{digest}"


class MeasureMeta(NamedTuple):
    """逐 case 测量的证据元数据输入（字段顺序与历史字典保持一致）。"""
    durations: list
    selected_op_names: list
    timing_source_files: list
    quick: bool
    archived_repeat_dirs: list
    repeats: int
    raw_prof_dir: Optional[str]
    repeat_bound_diagnoses: list


def _measure_meta(meta: MeasureMeta):
    """构造逐 case 测量的证据元数据字典。"""
    return {
        "duration_samples_us": meta.durations,
        "selected_op_names": meta.selected_op_names,
        "timing_source_metric": "task_time" if meta.quick else "PipeUtilization",
        "timing_source_files": (
            meta.timing_source_files if meta.quick or not meta.archived_repeat_dirs else [
                os.path.join(path, "op_summary_PipeUtilization.csv")
                for path in meta.archived_repeat_dirs
            ]
        ),
        "aggregation_method": "trimmed_mean" if meta.repeats >= 3 else "median",
        "raw_prof_dir": meta.raw_prof_dir,
        "repeat_bound_diagnoses": meta.repeat_bound_diagnoses,
    }


class CaseRun(NamedTuple):
    """单个 case 的 repeat 采集设置。"""
    case_name: Optional[str]
    quick: bool
    manifest: bool
    function_adapter: bool


def _run_repeat_loop(
    test_script, session, args, device_id, case_run: CaseRun,
):
    """执行逐 repeat 采集与解析；返回 (durations, names, sources, dirs, err)。"""
    runner = _run_msprof_quick if case_run.quick else _run_msprof_standard
    parser = _parse_msprof_duration_quick if case_run.quick else _parse_msprof_duration
    durations = []
    selected_op_names = []
    timing_source_files = []
    repeat_prof_dirs = []
    for repeat_index in range(max(1, getattr(args, "repeats", 1))):
        output_dir = session / f"repeat_{repeat_index}"
        selector = RunSelector(
            getattr(args, "case_arg", None),
            getattr(args, "case_env", None),
            case_run.case_name,
            getattr(args, "case_manifest", None),
            case_run.function_adapter,
        )
        request = RunRequest(
            test_script, str(output_dir), args.warmup, device_id, args.seed,
            selector, getattr(args, "l2_clean", True),
        )
        prof_dir, err = runner(request)
        if not prof_dir:
            break
        duration, selected_op_name, parse_err = parser(
            prof_dir, getattr(args, "op_name", None), strict=case_run.manifest
        )
        if duration is None:
            err = parse_err
            break
        durations.append(duration)
        selected_op_names.append(selected_op_name)
        if case_run.quick:
            timing_source_files.append("task_time_*.csv")
        else:
            timing_csv, timing_csv_error = _canonical_timing_csv(prof_dir)
            if not timing_csv:
                err = timing_csv_error
                break
            timing_source_files.append(timing_csv)
        repeat_prof_dirs.append(prof_dir)
    return durations, selected_op_names, timing_source_files, repeat_prof_dirs, err


class MeasuredRuns(NamedTuple):
    """逐 case 测量样本集（时长、目标名与 repeat 数）。"""
    durations: list
    selected_op_names: list
    repeats: int


def _archive_deep_round(
    args, case_name, repeat_prof_dirs, samples: MeasuredRuns,
    last_session,
):
    """归档七指标证据到 deep round；返回 (evidence_dir, err)。"""
    deep_round = getattr(args, "deep_round_dir", None)
    case_dir = os.path.join(deep_round, f"case_{safe_case_dir_name(case_name)}")
    archived_repeat_dirs = []
    repeat_bound_diagnoses = []
    archive_error = None
    for repeat_index, repeat_prof_dir in enumerate(repeat_prof_dirs, 1):
        repeat_dir = os.path.join(case_dir, f"repeat_{repeat_index:03d}")
        archive_error = archive_deep_profile(
            repeat_prof_dir, repeat_dir, getattr(args, "op_name", None),
            getattr(args, "collection_id", None),
            case_name,
        )
        if archive_error:
            break
        archived_repeat_dirs.append(repeat_dir)
        evidence_status = json.loads(
            (Path(repeat_dir) / "evidence_status.json").read_text(
                encoding="utf-8"
            )
        )
        repeat_bound_diagnoses.append(
            evidence_status.get("bound_diagnosis")
        )
    if archive_error:
        _cleanup_prof_dirs(case_dir)
        if not getattr(args, "keep_prof", False):
            _cleanup_prof_dirs(last_session)
        return None, [], [], archive_error
    with open(os.path.join(case_dir, "measurement.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "collection_id": args.collection_id,
            "target_op_name": getattr(args, "op_name", None),
            "case": case_name,
            "duration_samples_us": samples.durations,
            "selected_op_names": samples.selected_op_names,
            "timing_source_metric": "PipeUtilization",
            "timing_source_files": [
                os.path.join(path, "op_summary_PipeUtilization.csv")
                for path in archived_repeat_dirs
            ],
            "aggregate_us": _aggregate_durations(samples.durations),
            "aggregation_method": "trimmed_mean" if samples.repeats >= 3 else "median",
            "repeat_evidence_dirs": archived_repeat_dirs,
            "repeat_bound_diagnoses": repeat_bound_diagnoses,
        }, handle, indent=2, ensure_ascii=False)
    return case_dir, archived_repeat_dirs, repeat_bound_diagnoses, None


class MeasureAttempt(NamedTuple):
    """单次采集尝试的结果（成功时 aggregate/meta 非空，失败时 error 非空）。"""
    durations: list
    selected_op_names: list
    timing_source_files: list
    aggregate: Optional[float]
    evidence_dir: Optional[str]
    meta: Optional[dict]
    error: Optional[str]


def _measure_pypto_attempt(
    test_script, session, args, device_id, case_run: CaseRun,
) -> MeasureAttempt:
    """执行一次采集尝试；失败时按既有语义完成清理与退避。"""
    repeats = max(1, getattr(args, "repeats", 1))
    durations, selected_op_names, timing_source_files, repeat_prof_dirs, err = (
        _run_repeat_loop(test_script, session, args, device_id, case_run)
    )
    if len(durations) != repeats:
        if not getattr(args, "keep_prof", False):
            _cleanup_prof_dirs(str(session))
        time.sleep(0.5)
        return MeasureAttempt(
            durations, selected_op_names, timing_source_files,
            None, None, None, err,
        )
    keep_prof = getattr(args, "keep_prof", False)
    evidence_dir = str(session) if keep_prof else None
    archived_repeat_dirs = []
    repeat_bound_diagnoses = []
    if not case_run.quick and getattr(args, "deep_round_dir", None):
        evidence_dir, archived_repeat_dirs, repeat_bound_diagnoses, err = (
            _archive_deep_round(
                args, case_run.case_name, repeat_prof_dirs,
                MeasuredRuns(durations, selected_op_names, repeats),
                str(session),
            )
        )
        if err:
            return MeasureAttempt(
                durations, selected_op_names, timing_source_files,
                None, None, None, err,
            )
    if not keep_prof:
        _cleanup_prof_dirs(str(session))
    return MeasureAttempt(
        durations, selected_op_names, timing_source_files,
        _aggregate_durations(durations), evidence_dir,
        _measure_meta(
            MeasureMeta(
                durations, selected_op_names, timing_source_files, case_run.quick,
                archived_repeat_dirs, repeats,
                str(session) if keep_prof else None, repeat_bound_diagnoses,
            ),
        ),
        None,
    )


def _measure_pypto_runs(out_dir, args, device_id, quick, case_name=None):
    manifest = bool(getattr(args, "case_manifest", None))
    test_script, err = _find_test_script(out_dir, strict=manifest)
    if not test_script:
        return None, err, None, _measure_meta(
            MeasureMeta([], [], [], quick, [], 1, None, []),
        )
    repeats = max(1, getattr(args, "repeats", 1))
    mode_name = "quick" if quick else "standard"
    case_ids = list((_manifest_case_functions(args) or {}).keys())
    function_adapter = uses_function_adapter(args, case_ids)
    case_run = CaseRun(case_name, quick, manifest, function_adapter)
    last_session = None
    result = None
    for _attempt in range(1 + args.retry):
        session = (
            out_dir / ".msprof" /
            f"{mode_name}_{os.getpid()}_{time.time_ns()}"
        )
        last_session = str(session)
        result = _measure_pypto_attempt(
            test_script, session, args, device_id, case_run,
        )
        if result.aggregate is not None:
            return result.aggregate, None, result.evidence_dir, result.meta
        err = result.error
    retained_session = last_session if getattr(args, "keep_prof", False) else None
    return None, err, retained_session, _measure_meta(
        MeasureMeta(
            result.durations, result.selected_op_names, result.timing_source_files,
            quick, [], repeats, retained_session, [],
        ),
    )


def _measure_pypto(out_dir: Path, args, device_id: int, case_name=None):
    """Measure PyPTO with standard metrics, honoring repeats and seed."""
    return _measure_pypto_runs(
        out_dir, args, device_id, quick=False, case_name=case_name
    )


def _measure_pypto_quick(out_dir: Path, args, device_id: int, case_name=None):
    """Measure PyPTO with quick profiling, warmups, retries and repeats."""
    return _measure_pypto_runs(
        out_dir, args, device_id, quick=True, case_name=case_name
    )


@dataclass
class CompareSummaryInput:
    """封装 compute_compare_summary 的汇总计算参数。

    Args:
        out_dir: 算子输出目录（Path）
        rows: 逐 case 的测量结果列表
        speedups: 有效 speedup 值列表
        ref_times: 参考实现耗时列表（us）
        asc_times: PyPTO-Pro 实现耗时列表（us）
        n_cases: case 总数
        args: argparse.Namespace（需含 warmup, repeats 属性）
        device_id: NPU 设备 ID
        device_src: 设备来源描述（cli/env/auto）
    """
    out_dir: Path
    rows: list
    speedups: list
    ref_times: list
    asc_times: list
    n_cases: int
    args: argparse.Namespace
    device_id: int
    device_src: str


class RatioRows(NamedTuple):
    """逐 case 比值重算结果。"""
    rows: list
    ratios: list
    ref_times: list
    asc_times: list
    valid_pypto_cases: int
    case_ids: list


def _ratio_rows(csi: CompareSummaryInput) -> RatioRows:
    """逐 case 重算比值；返回 RatioRows（rows/ratios/ref_times/asc_times/valid_pypto_cases/case_ids）。"""
    rows = []
    ratios = []
    ref_times = []
    asc_times = []
    valid_pypto_cases = 0
    case_ids = []
    for raw_row in csi.rows:
        row = dict(raw_row)
        case_ids.append(str(row.get("case", "")).strip())
        golden_us = row.get("ref_us")
        pypto_us = row.get("asc_us")
        pypto_valid = _is_positive_finite(pypto_us)
        golden_valid = _is_positive_finite(golden_us)
        if pypto_valid:
            valid_pypto_cases += 1
            asc_times.append(float(pypto_us))
        ratio = None
        if golden_valid and pypto_valid:
            golden_value = float(golden_us)
            ratio = golden_value / float(pypto_us)
            if _is_positive_finite(ratio):
                ratios.append(ratio)
                ref_times.append(golden_value)
            else:
                ratio = None
        # `speedup` is retained as a compatibility alias.  This ratio is the
        # default Golden target metric, not baseline-to-final optimization speedup.
        row["speedup"] = ratio
        row["golden_reference_ratio"] = ratio
        row["default_target_ratio"] = ratio
        row["default_target_met"] = (
            ratio >= DEFAULT_GOLDEN_TARGET_THRESHOLD if ratio is not None else None
        )
        rows.append(row)
    return RatioRows(rows, ratios, ref_times, asc_times, valid_pypto_cases, case_ids)


def _target_validity(csi: CompareSummaryInput, case_set_complete: bool,
                     valid_pypto_cases: int, ratios: list) -> bool:
    """判断默认 Golden 目标证据是否有效（逐 case 全量且协议一致）。"""
    performance_cases = getattr(csi.args, "performance_cases", None) or {}
    golden_source = performance_cases.get("golden_diagnostic") or {}
    golden_contract = performance_cases.get("golden_contract") or {}
    golden_protocol = golden_contract.get("protocol") or {}
    golden_exact_id_joined = (
        golden_source.get("status") == "joined"
        and isinstance(performance_cases.get("golden_contract"), dict)
    )
    return (
        golden_exact_id_joined
        and golden_protocol.get("device_id") == csi.device_id
        and golden_protocol.get("seed") == 42
        and case_set_complete
        and valid_pypto_cases == csi.n_cases
        and len(ratios) == csi.n_cases
        and not getattr(csi.args, "quick", False)
    )


def _selector_entry(args) -> dict:
    """构造 case_selector 条目（cli/env/function_adapter/single_case）。"""
    return {
        "kind": "cli" if getattr(args, "case_arg", None) else (
            "env" if getattr(args, "case_env", None) else (
                "function_adapter" if _manifest_case_functions(args)
                else "single_case"
            )
        ),
        "name": (
            getattr(args, "case_arg", None)
            or getattr(args, "case_env", None)
            or ("PERFORMANCE_CASES.test_function" if _manifest_case_functions(args) else None)
        ),
    }


def _target_fields(csi: CompareSummaryInput, has_golden_ratio: bool,
                   valid_for_target_met: bool, default_target_met: bool) -> dict:
    """构造与默认 Golden 目标判定相关的字段段。"""
    return {
        "comparison_scope": (
            "golden_e2e_to_pypto_target_kernel"
            if has_golden_ratio
            else "pypto_target_kernel_measurement"
        ),
        "ratio_semantics": (
            "golden_per_iteration_e2e_us / pypto_target_kernel_us"
            if has_golden_ratio
            else None
        ),
        "default_target_metric": (
            "golden_per_iteration_e2e_us / pypto_target_kernel_us"
        ),
        "default_target_threshold": DEFAULT_GOLDEN_TARGET_THRESHOLD,
        "default_target_met": bool(default_target_met),
        "default_target_status": (
            "met" if default_target_met else (
                "not_met" if valid_for_target_met else "unavailable"
            )
        ),
        "valid_for_optimization_speedup": False,
        "valid_for_target_met": bool(valid_for_target_met),
        "golden_timing_method": (
            "torch_npu.kernel_details.all_golden_npu_kernel_sum"
            if has_golden_ratio
            else None
        ),
    }


class SummaryStats(NamedTuple):
    """compare 汇总的统计字典组。"""
    speedup_stats: dict
    timing_stats: dict


class SummaryDecisions(NamedTuple):
    """compare 汇总的目标判定输入。"""
    valid_pypto_cases: int
    has_golden_ratio: bool
    valid_for_target_met: bool
    default_target_met: bool


def _summary_payload(csi: CompareSummaryInput, rows: list, ratios: list,
                    stats: SummaryStats, decisions: SummaryDecisions) -> dict:
    """组装 compare 汇总的返回字典（结构与字段名保持稳定）。"""
    return {
        "task": csi.out_dir.name,
        "task_dir": str(csi.out_dir),
        "n_cases_total": csi.n_cases,
        **stats.speedup_stats,
        "golden_reference_ratio_stats": {
            "n_cases_valid": stats.speedup_stats["n_cases_valid"],
            "geomean": stats.speedup_stats["geomean_speedup"],
            "mean": stats.speedup_stats["mean_speedup"],
            "median": stats.speedup_stats["median_speedup"],
            "min": stats.speedup_stats["min_speedup"],
            "max": stats.speedup_stats["max_speedup"],
        },
        "n_cases_valid": decisions.valid_pypto_cases,
        "n_default_target_cases": len(ratios),
        # Compatibility alias retained for existing report consumers.
        "n_cross_scope_cases": stats.speedup_stats["n_cases_valid"],
        **stats.timing_stats,
        "warmup": csi.args.warmup,
        "repeats": csi.args.repeats,
        "seed": csi.args.seed if csi.args.seed is not None else 42,
        "seed_source": "cli_override" if csi.args.seed is not None else "runner_default",
        "l2_clean": getattr(csi.args, "l2_clean", True),
        "device_id": csi.device_id,
        "device_select_source": csi.device_src,
        **_target_fields(
            csi, decisions.has_golden_ratio,
            decisions.valid_for_target_met, decisions.default_target_met,
        ),
        "golden_iterations": getattr(csi.args, "golden_iterations", None),
        "performance_cases": getattr(csi.args, "performance_cases", None),
        "target_op_name": csi.args.op_name,
        "case_selector": _selector_entry(csi.args),
        "timing_method": "msprof.op_summary.Task_Duration",
        "timing_source_metric": "task_time" if getattr(csi.args, "quick", False) else "PipeUtilization",
        "profiling_mode": "quick" if getattr(csi.args, "quick", False) else "compare",
        "collection_id": getattr(csi.args, "collection_id", None),
        "deep_profile_round": getattr(csi.args, "deep_round_dir", None),
        "per_case": rows,
    }


def compute_compare_summary(csi: CompareSummaryInput):
    """Compute the summary statistics dict for compare mode."""
    # Recompute every machine-decision value from per-case evidence instead of
    # trusting caller-maintained aggregate lists.  This keeps NaN/Inf and
    # partial-case results from accidentally satisfying the default target.
    ratio_rows = _ratio_rows(csi)
    rows = ratio_rows.rows
    ratios = ratio_rows.ratios
    ref_times = ratio_rows.ref_times
    asc_times = ratio_rows.asc_times
    valid_pypto_cases = ratio_rows.valid_pypto_cases
    case_ids = ratio_rows.case_ids

    speedup_stats = _compute_speedup_stats(ratios)
    timing_stats = _compute_timing_stats(ref_times, asc_times)
    has_golden_ratio = bool(ratios)
    case_set_complete = (
        csi.n_cases > 0
        and len(rows) == csi.n_cases
        and len(case_ids) == len(set(case_ids))
        and all(case_ids)
    )
    valid_for_target_met = _target_validity(
        csi, case_set_complete, valid_pypto_cases, ratios
    )
    default_target_met = valid_for_target_met and all(
        ratio >= DEFAULT_GOLDEN_TARGET_THRESHOLD for ratio in ratios
    )
    return _summary_payload(
        csi, rows, ratios,
        SummaryStats(speedup_stats, timing_stats),
        SummaryDecisions(
            valid_pypto_cases, has_golden_ratio,
            valid_for_target_met, default_target_met,
        ),
    )


def _compute_speedup_stats(speedups: list) -> dict:
    """Compute speedup statistics from a list of speedup values.

    Returns a dict with n_cases_valid and geomean/mean/median/min/max speedup.
    """
    if not speedups:
        return {
            "n_cases_valid": 0,
            "geomean_speedup": None,
            "mean_speedup": None,
            "median_speedup": None,
            "min_speedup": None,
            "max_speedup": None,
        }
    return {
        "n_cases_valid": len(speedups),
        "geomean_speedup": statistics.geometric_mean(speedups),
        "mean_speedup": statistics.mean(speedups),
        "median_speedup": statistics.median(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
    }


def _timing_stat(prefix: str, values: list):
    """按前缀生成 mean/median/total 统计字典（values 为空时全部置 None）。"""
    if values:
        return {
            f"mean_{prefix}_us": statistics.mean(values),
            f"median_{prefix}_us": statistics.median(values),
            f"total_{prefix}_us": sum(values),
        }
    return {
        f"mean_{prefix}_us": None,
        f"median_{prefix}_us": None,
        f"total_{prefix}_us": None,
    }


def _compute_timing_stats(ref_times: list, asc_times: list) -> dict:
    """Compute timing statistics for reference and PyPTO-Pro implementations.

    Returns a dict with mean/median/total for ref and asc, plus total_speedup.
    """
    ref_stats = _timing_stat("ref", ref_times)
    asc_stats = _timing_stat("asc", asc_times)

    total_speedup = None
    if ref_times and asc_times and len(ref_times) == len(asc_times):
        asc_sum = sum(asc_times)
        if asc_sum > 0:
            total_speedup = sum(ref_times) / asc_sum

    return {
        **ref_stats,
        **asc_stats,
        "aggregate_golden_reference_ratio": total_speedup,
        # Compatibility alias for existing JSON consumers.
        "total_speedup": total_speedup,
    }


def _atomic_write_text(path: Path, content: str):
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _cleanup_prof_dirs(*dirs):
    """Remove profiling directories if they exist."""
    for d in dirs:
        if d and os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


def _run_measurement_loop(out_dir, performance_cases, args, device_id, measure):
    rows, speedups, ref_times, asc_times = [], [], [], []
    for case_name, case_shape, case_dtype, golden_us in performance_cases:
        pypto_us, pypto_err, pypto_prof_dir, measurement_meta = measure(
            out_dir, args, device_id, case_name
        )
        if _is_positive_finite(pypto_us):
            pypto_us = float(pypto_us)
            asc_times.append(pypto_us)
            if _is_positive_finite(golden_us):
                golden_us = float(golden_us)
                speedup = golden_us / pypto_us
                speedups.append(speedup)
                ref_times.append(golden_us)
                LOGGER.info(f"{case_name:<20} {case_shape:<35} {case_dtype:<10} "
                      f"{golden_us:>12.2f} {pypto_us:>12.2f} {speedup:>9.3f}x")
            else:
                LOGGER.info(f"{case_name:<20} {case_shape:<35} {case_dtype:<10} "
                      f"{'N/A':>12} {pypto_us:>12.2f} {'N/A':>10}")
        else:
            LOGGER.warning(f"{case_name:<20} {case_shape:<35} {case_dtype:<10} "
                  f"{'N/A' if golden_us is None else f'{golden_us:.2f}':>12} "
                  f"{'N/A' if pypto_us is None else f'{pypto_us:.2f}':>12} "
                  f"{'N/A':>10}  (pypto_err={pypto_err})")

        rows.append({
            "case": case_name, "shape": case_shape, "dtype": case_dtype,
            "ref_us": golden_us, "asc_us": pypto_us,
            "speedup": (
                golden_us / pypto_us
                if _is_positive_finite(golden_us) and _is_positive_finite(pypto_us)
                else None
            ),
            "default_target_ratio": (
                golden_us / pypto_us
                if _is_positive_finite(golden_us) and _is_positive_finite(pypto_us)
                else None
            ),
            "golden_reference_ratio": (
                golden_us / pypto_us
                if _is_positive_finite(golden_us) and _is_positive_finite(pypto_us)
                else None
            ),
            "ref_error": None,
            "asc_error": pypto_err,
            "ref_prof_dir": None,
            "asc_prof_dir": pypto_prof_dir,
            "deep_profile_dir": pypto_prof_dir if not getattr(args, "quick", False) else None,
            **measurement_meta,
        })
    return rows, speedups, ref_times, asc_times


def _run_compare_loop(out_dir, performance_cases, args, device_id):
    """Run standard msprof measurement for all selected performance cases."""
    return _run_measurement_loop(
        out_dir, performance_cases, args, device_id, _measure_pypto
    )


def _run_quick_loop(out_dir, performance_cases, args, device_id):
    """Run lightweight msprof measurement for selected performance cases."""
    return _run_measurement_loop(
        out_dir, performance_cases, args, device_id, _measure_pypto_quick
    )


def _build_collection_record(args, out_dir, device_id, performance_cases):
    """构造 compare 轮次的 collection.json 初始记录。"""
    return {
        "collection_id": args.collection_id,
        "mode": "compare",
        "status": "in_progress",
        "operator_dir": str(out_dir),
        "target_op_name": args.op_name,
        "device_id": device_id,
        "seed": args.seed if args.seed is not None else 42,
        "seed_source": "cli_override" if args.seed is not None else "runner_default",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "expected_cases": [str(case[0]) for case in performance_cases],
        "performance_cases": args.performance_cases,
        "executable_sha256": args.executable_sha256,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _write_collection_json(collection_path: Path, collection_record: dict) -> None:
    """原子写入 collection.json 的当前状态。"""
    _atomic_write_text(
        collection_path,
        json.dumps(collection_record, indent=2, ensure_ascii=False) + "\n",
    )


class DeviceSelection(NamedTuple):
    """NPU 设备选择结果（设备 id 与来源描述）。"""
    device_id: int
    device_src: str


class CollectionState(NamedTuple):
    """compare 轮次的 collection 状态（记录与文件路径）。"""
    record: dict
    path: Path


def _execute_compare_loop(
    args, out_dir, device: DeviceSelection, performance_cases,
    collection: CollectionState,
):
    """执行 compare 采集、汇总与报告；成功返回 0，失败写 failed 状态返回 1。"""
    from evidence_cli import _log_and_save_compare_reports, _log_compare_header
    try:
        _log_compare_header(out_dir, args)
        rows, speedups, ref_times, asc_times = _run_compare_loop(
            out_dir, performance_cases, args, device.device_id)

        csi = CompareSummaryInput(
            out_dir, rows, speedups, ref_times, asc_times, len(performance_cases),
            args, device.device_id, device.device_src)
        summary = compute_compare_summary(csi)
        if summary["n_cases_valid"] != summary["n_cases_total"]:
            raise RuntimeError(
                f"only {summary['n_cases_valid']}/{summary['n_cases_total']} cases "
                "produced valid performance evidence"
            )
        source_error = performance_case_source_error(args.performance_cases)
        if source_error:
            raise RuntimeError(source_error)
        _, executable_sha256, executable_error = _selected_executable_sha256(out_dir)
        if executable_error:
            raise RuntimeError(executable_error)
        if executable_sha256 != collection.record.get("executable_sha256"):
            raise RuntimeError("runner executable changed during formal compare")
        summary["executable_sha256"] = executable_sha256
        _log_and_save_compare_reports(summary, out_dir, speedups, len(performance_cases))
        collection.record["status"] = "complete"
        collection.record["completed_cases"] = summary["n_cases_valid"]
        _write_collection_json(collection.path, collection.record)
        return 0
    except Exception as error:
        collection.record["status"] = "failed"
        collection.record["failure"] = str(error)
        _write_collection_json(collection.path, collection.record)
        LOGGER.error("[ERROR] %s", error)
        return 1


def _validated_case_suite(out_dir, args, device_id):
    """校验逐 case 证据套件（manifest/selector/runtime）；失败返回 None。"""
    from evidence_cli import (
        _validate_case_selector,
        _validate_selector_runtime,
        _validated_performance_cases,
    )
    performance_cases = _validated_performance_cases(out_dir, args)
    if performance_cases is None:
        return None
    if not _validate_case_selector(performance_cases, args):
        return None
    if not _validate_selector_runtime(out_dir, performance_cases, args, device_id):
        return None
    return performance_cases


def _default_collection_id(args, prefix):
    """生成缺省 collection id（显式参数 > 环境变量 > 时间戳 + pid）。"""
    return (
        getattr(args, "collection_id", None)
        or os.environ.get("PYPTO_PERF_COLLECTION_ID")
        or f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    )


def _run_compare_mode(args, out_dir, device_id, device_src):
    """Collect PyPTO cases and evaluate the Golden-backed default target when available."""
    args.quick = False
    LOGGER.info(f"[INFO] Using NPU device {device_id} (source={device_src})")

    performance_cases = _validated_case_suite(out_dir, args, device_id)
    if performance_cases is None:
        return 1
    args.collection_id = _default_collection_id(args, "compare")
    args.deep_round_dir = reserve_next_round(str(out_dir / "docs" / "perf"))
    _, executable_sha256, script_error = _selected_executable_sha256(out_dir)
    if script_error:
        raise ValueError(script_error)
    args.executable_sha256 = executable_sha256
    collection_record = _build_collection_record(
        args, out_dir, device_id, performance_cases
    )
    collection_path = Path(args.deep_round_dir) / "collection.json"
    _write_collection_json(collection_path, collection_record)
    return _execute_compare_loop(
        args, out_dir, DeviceSelection(device_id, device_src),
        performance_cases, CollectionState(collection_record, collection_path),
    )


def _resolve_mode_entry(args):
    """compare/quick 共用入口解析；校验失败抛 ValueError（上层统一转为 1）。"""
    out_dir = Path(args.output_dir).resolve()
    manifest = bool(getattr(args, "case_manifest", None))
    from evidence_cli import _validate_measurement_args
    if not _validate_measurement_args(args, manifest=manifest):
        raise ValueError("invalid measurement arguments")
    device_id, device_src = _select_device_id(args, tile_fwk=manifest)
    return manifest, out_dir, device_id, device_src


def run_compare_mode(args):
    """执行对比模式。

    未传 --case-manifest：既有 GOLDEN_PERF_REPORT.md 流程；
    传入 --case-manifest：manifest 证据协议。
    """
    try:
        manifest, out_dir, device_id, device_src = _resolve_mode_entry(args)
        if manifest:
            return _run_compare_mode(args, out_dir, device_id, device_src)
        from legacy_compare import _run_compare_mode_legacy
        return _run_compare_mode_legacy(args, out_dir, device_id, device_src)
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.error("[ERROR] %s", error)
        return 1


def _run_quick_mode(args, out_dir, device_id, device_src):
    """执行快速模式：每次 repeat 只获取 kernel 时间。"""
    from evidence_cli import (
        _log_and_save_compare_reports,
        _log_compare_header,
    )
    args.quick = True
    LOGGER.info(f"[INFO] Using NPU device {device_id} (source={device_src})")
    LOGGER.info("[INFO] Quick mode: kernel timing only (no aic-metrics)")

    performance_cases = _validated_case_suite(out_dir, args, device_id)
    if performance_cases is None:
        return 1
    args.collection_id = _default_collection_id(args, "quick")
    args.deep_round_dir = None

    _log_compare_header(out_dir, args)
    rows, speedups, ref_times, asc_times = _run_quick_loop(
        out_dir, performance_cases, args, device_id)

    source_error = performance_case_source_error(args.performance_cases)
    if source_error:
        LOGGER.error("[ERROR] %s", source_error)
        return 1

    csi = CompareSummaryInput(
        out_dir, rows, speedups, ref_times, asc_times, len(performance_cases), args, device_id, device_src)
    summary = compute_compare_summary(csi)
    summary["timing_method"] = "msprof.quick.Task_Duration"
    summary["profiling_mode"] = "quick"
    if summary["n_cases_valid"] != summary["n_cases_total"]:
        LOGGER.error(
            "[ERROR] only %s/%s cases produced valid performance evidence",
            summary["n_cases_valid"], summary["n_cases_total"],
        )
        return 1
    _log_and_save_compare_reports(summary, out_dir, speedups, len(performance_cases))
    return 0


def run_quick_mode(args):
    """执行快速模式。

    未传 --case-manifest：既有流程；传入：manifest 证据协议。
    """
    try:
        manifest, out_dir, device_id, device_src = _resolve_mode_entry(args)
        if manifest:
            return _run_quick_mode(args, out_dir, device_id, device_src)
        from legacy_compare import _run_quick_mode_legacy
        return _run_quick_mode_legacy(args, out_dir, device_id, device_src)
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.error("[ERROR] %s", error)
        return 1


def _is_valid_table_row(line: str) -> bool:
    return line.startswith('|') and 'Level' not in line and '---' not in line and len(line) > 5


def _extract_trace_table_rows(trace_file_path: str) -> List[str]:
    if not os.path.exists(trace_file_path):
        return []
    try:
        with open(trace_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return []
    start_idx = content.find('## 汇总表报告')
    if start_idx == -1:
        return []
    section_content = content[start_idx:]
    lines = section_content.split('\n')
    valid_rows = []
    for line in lines:
        line = line.strip()
        if _is_valid_table_row(line):
            valid_rows.append(line)
    return valid_rows


def _load_performance_json(op_dir: Path) -> Optional[Dict[str, Any]]:
    perf_json = op_dir / "performance.json"
    if perf_json.exists():
        try:
            with open(perf_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            LOGGER.warning("Failed to load performance.json from %s: %s", op_dir, e)
    return None


def _collect_batch_results(base_dir: Path, collection_id, evidence_check) -> Tuple[list, list]:
    """扫描 base_dir 收集 performance.json 与 trace 表格行。"""
    op_results = []
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        perf_data = _load_performance_json(subdir)
        if perf_data and (not collection_id or perf_data.get("collection_id") == collection_id):
            op_results.append({
                "name": subdir.name,
                "data": perf_data,
                "dir": subdir,
                "evidence_error": evidence_check(subdir, perf_data),
            })
    trace_rows = []
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        trace_file = subdir / "trace.md"
        if trace_file.exists():
            rows = _extract_trace_table_rows(str(trace_file))
            trace_rows.extend(rows)
    return op_results, trace_rows


def _incomplete_operators(op_results, expected_names):
    """返回缺失与证据不完整的算子目录名列表。"""
    found_names = {item["name"] for item in op_results}
    missing_names = sorted(expected_names - found_names)
    incomplete_names = []
    for item in op_results:
        has_no_total = not item["data"].get("n_cases_total")
        valid_mismatch = item["data"].get("n_cases_valid") != item["data"].get("n_cases_total")
        if has_no_total or valid_mismatch or item.get("evidence_error"):
            incomplete_names.append(item["name"])
    return missing_names, sorted(incomplete_names)


def run_batch_mode(args):
    """执行批量模式：扫描 base_dir 下所有子目录，汇总性能报告。"""
    base_dir = Path(args.batch).resolve()

    if not base_dir.is_dir():
        raise ValueError("'%s' is not a directory." % base_dir)

    if not getattr(args, "expect_operator", None):
        # 既有批量流程：扫描 performance.json 并汇总（保持既有行为）。
        from legacy_compare import run_batch_mode_legacy
        return run_batch_mode_legacy(args)

    expected_names = set(getattr(args, "expect_operator", []) or [])
    from batch_evidence import (
        batch_evidence_error,
        _generate_batch_md_report,
        generate_batch_json_report,
    )
    collection_id = getattr(args, "collection_id", None)
    if not expected_names:
        raise ValueError("batch mode requires at least one --expect-operator")

    # 收集所有子目录的 performance.json 与 trace.md 表格行
    op_results, trace_rows = _collect_batch_results(
        base_dir, collection_id, batch_evidence_error
    )

    LOGGER.info("Found %d operators with performance.json in %s", len(op_results), base_dir)

    missing_names, incomplete_names = _incomplete_operators(op_results, expected_names)
    if missing_names or incomplete_names:
        raise ValueError(
            "batch evidence incomplete; missing=%s incomplete=%s collection_id=%s"
            % (missing_names, incomplete_names, collection_id)
        )

    if args.output_md:
        _generate_batch_md_report(args, op_results, trace_rows, base_dir)

    if args.output_json:
        generate_batch_json_report(args, op_results, base_dir)
    return 0


def _run_standard_mode(args):
    """Execute standard mode: parse PROF_GROUP and generate summary."""
    group_dir = os.path.abspath(args.prof_group_dir)
    ops_dir = os.path.abspath(args.ops_dir)

    if not os.path.isdir(group_dir):
        raise ValueError("'%s' is not a directory." % group_dir)
    if not os.path.isdir(ops_dir):
        raise ValueError("'%s' is not a directory." % ops_dir)

    merged = merge_metric_rows(group_dir, args.op_name)
    if not merged.get("Op Name"):
        raise ValueError("no op_summary_*.csv rows discovered under %s" % group_dir)

    perf_dir = os.path.join(ops_dir, "docs", "perf")
    round_dir = os.path.join(perf_dir, args.round_name) if args.round_name else find_next_round(perf_dir)

    copied = archive_csvs(group_dir, round_dir)
    pc_csv = archive_per_core_csv(group_dir, round_dir)
    if pc_csv:
        copied.append(os.path.basename(pc_csv))
    LOGGER.info("Archived %d CSV files to: %s", len(copied), round_dir)

    summary = generate_summary(merged, round_dir, group_dir)
    summary_path = os.path.join(round_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    LOGGER.info("Summary written to: %s", summary_path)
    LOGGER.info("\n%s", summary)


def _add_standard_args(parser) -> None:
    """注册标准模式参数。"""
    parser.add_argument("prof_group_dir", nargs="?", help="PROF_GROUP_<timestamp> directory")
    parser.add_argument("ops_dir", nargs="?", help="Operator directory")
    parser.add_argument("--op-name", default=None, help="Exact Op Name to pick in op_summary.csv")
    parser.add_argument(
        "--list-op-names", action="store_true",
        help="list exact lowering Op Name values from a discovery PROF_GROUP, then exit",
    )
    parser.add_argument(
        "--timeline", action="store_true",
        help=(
            "collect one supplemental instruction timeline for a completed final "
            "compare case; requires --output-dir/--case-manifest/--case-id/--op-name"
        ),
    )
    parser.add_argument(
        "--timeline-timeout", type=int, default=600,
        help="timeout in seconds for each instruction-profile collect/export command",
    )
    parser.add_argument("--round-name", default=None, help="Override round directory name")


def _add_compare_args(parser) -> None:
    """注册对比/快速模式参数。"""
    parser.add_argument("--compare", action="store_true", help="启用对比模式（8 轮采集：7 metrics + sample）")
    parser.add_argument(
        "--quick", action="store_true",
        help="启用快速模式：每次 repeat 只获取 kernel 时间，不采集 7 个 aic-metrics",
    )
    parser.add_argument("--output-dir", dest="output_dir", help="算子输出目录（对比模式/快速模式）")
    parser.add_argument("--warmup", type=int, default=3, help="msprof warmup 次数")
    parser.add_argument("--repeats", type=int, default=1, help="重复采集次数（既有流程默认 1；manifest 协议建议 3）")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="既有流程：随机种子，经 PYPTO_PERF_SEED/PYTHONHASHSEED 传入 runner；"
             "manifest（--case-manifest）协议仅接受 42（0 视为未指定）",
    )
    parser.add_argument("--retry", type=int, default=2, help="单 case 解析失败重试次数")
    parser.add_argument("--device", type=int, default=None, help="NPU 设备 id")
    parser.add_argument("--keep-prof", action="store_true", help="保留 msprof 原始 PROF 目录")
    parser.add_argument(
        "--l2-clean", dest="l2_clean", action="store_true", default=True,
        help="每次测量前刷新目标 NPU 的 L2 cache（cannbench 式），默认开启",
    )
    parser.add_argument(
        "--no-l2-clean", dest="l2_clean", action="store_false",
        help="关闭测量前的 L2 cache 刷新（与未含该机制的历史协议对齐）",
    )
    parser.add_argument(
        "--case-arg",
        help="逐 case 选择器参数名；每次 runner 调用追加 '<case-arg> <case-id>'",
    )
    parser.add_argument(
        "--case-env",
        help="逐 case 选择器环境变量名；每次 runner 调用设置 '<case-env>=<case-id>'",
    )
    parser.add_argument(
        "--case-manifest",
        help=(
            "UTF-8 JSON performance-case manifest (schema_version=1, non-empty cases); "
            "GOLDEN_PERF_REPORT.json is joined only when its "
            "manifest identity and canonical case metadata match exactly"
        ),
    )
    parser.add_argument(
        "--validate-case-source",
        action="store_true",
        help="validate the required manifest and optional Golden exact-id join, then exit",
    )
    parser.add_argument("--run-case-function", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-script", help=argparse.SUPPRESS)
    parser.add_argument("--case-id", help=argparse.SUPPRESS)


def _add_batch_args(parser) -> None:
    """注册批量模式参数。"""
    parser.add_argument("--batch", metavar="BASE_DIR", help="启用批量模式，指定根目录")
    parser.add_argument("--output-md", help="批量模式 Markdown 输出路径")
    parser.add_argument("--output-json", help="批量模式 JSON 输出路径")
    parser.add_argument(
        "--collection-id",
        help="本轮采集标识；batch 汇总只接受相同标识的 performance.json",
    )
    parser.add_argument(
        "--expect-operator", action="append", default=[],
        help="batch 预期算子目录名，可重复；缺失时汇总非零退出",
    )


def _build_arg_parser():
    """构建并返回统一的命令行解析器。"""
    parser = argparse.ArgumentParser(description="msprof 解析 & 归档 & 对比测试脚本（统一入口）")
    _add_standard_args(parser)
    _add_compare_args(parser)
    _add_batch_args(parser)
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _build_arg_parser()

    args = parser.parse_args()

    # 模式路由
    if args.run_case_function:
        if not args.test_script or not args.case_manifest or not args.case_id:
            parser.error("--run-case-function requires --test-script, --case-manifest and --case-id")
        try:
            from evidence_cli import run_case_function
            sys.exit(run_case_function(args.test_script, args.case_manifest, args.case_id))
        except (OSError, RuntimeError, ValueError) as error:
            LOGGER.error("[ERROR] %s", error)
            sys.exit(1)
    elif args.list_op_names:
        if not args.prof_group_dir:
            parser.error("--list-op-names requires prof_group_dir")
        try:
            from evidence_cli import list_op_names_mode
            sys.exit(list_op_names_mode(args))
        except ValueError as error:
            LOGGER.error("[ERROR] %s", error)
            sys.exit(1)
    elif args.timeline:
        try:
            from evidence_cli import run_timeline_mode
            sys.exit(run_timeline_mode(args))
        except (OSError, RuntimeError, ValueError) as error:
            LOGGER.error("[ERROR] %s", error)
            sys.exit(1)
    elif args.validate_case_source:
        if not args.output_dir or not args.case_manifest:
            parser.error(
                "--validate-case-source requires --output-dir and --case-manifest"
            )
        from evidence_cli import _validate_case_source_mode
        sys.exit(_validate_case_source_mode(args))
    elif args.compare:
        if not args.output_dir:
            parser.error("--compare mode requires --output-dir")
        sys.exit(run_compare_mode(args))
    elif args.quick:
        if not args.output_dir:
            parser.error("--quick mode requires --output-dir")
        sys.exit(run_quick_mode(args))
    elif args.batch:
        try:
            run_batch_mode(args)
        except ValueError as e:
            LOGGER.error("%s", e)
            sys.exit(1)
    else:
        if not args.prof_group_dir or not args.ops_dir:
            parser.error("Standard mode requires prof_group_dir and ops_dir arguments")
        try:
            _run_standard_mode(args)
        except ValueError as e:
            LOGGER.error("%s", e)
            sys.exit(1)


if __name__ == "__main__":
    main()

