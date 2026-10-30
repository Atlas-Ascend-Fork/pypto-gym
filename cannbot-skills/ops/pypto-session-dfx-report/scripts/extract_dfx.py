#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Extract DFX data from PyPTO session exports (root_full.json) for one operator.

Deterministic extraction — renders time/token/exec-record sections automatically.
The LLM only authors report_body.md (sections 0/1/4) from the compact summary.

Usage:
    python3 extract_dfx.py <op_dir> [--out <out_dir>]

op_dir  : report/pto_case/<op_name> (must contain pypto_sessions/attempt_XX/root_full.json)
out_dir : default <op_dir>/dfx

Outputs:
  dfx_data.json                      structured full data (visualization)
  summary.txt                        compact per-node stats (for LLM, low tokens)
  sections/2_time.md                 section 2 tables (deterministic)
  sections/3_token.md                section 3 tables (deterministic)
   sections/5_exec.md                 section 5 exec records (attempt-separated, ms precision)
   sections/6_data.md                 section 6 data-prep (static)
  exec_records/attempt_XX/{order}_{id8}.md   per-session tool tables
"""
import argparse
import collections
import datetime
import json
import logging
import os
import re
import sys
from pathlib import Path

TZ = datetime.timezone(datetime.timedelta(hours=8))
LOGGER = logging.getLogger(__name__)

CATEGORY = {
    "bash": "执行命令",
    "read": "读取文件",
    "edit": "修改文件",
    "write": "写入文件",
    "skill": "加载技能",
    "task": "派发子任务",
    "grep": "内容搜索",
    "glob": "文件搜索",
    "state_transition": "状态流转",
    "invalid": "无效调用",
    "webfetch": "网页抓取",
    # 非工具类事件（同样具备具体耗时/时间戳，必须保留以填满执行轨迹时间轴的空档）
    "reasoning": "推理/思考",
    "text": "文本输出",
}

# 说明：dfx 报告（5.2 节逐事件表）与 dfx_data.json 均保留**完整说明、不做截断**——
# 推理/思考等长文本原样写入报告（表格内由 esc() 转义管道符、换行折叠为空格），
# 提示窗中的截断（capLines + 省略号）仅发生在 dashboard 悬停 tooltip 展示层。


def esc(s):
    return (s or "").replace("|", "\\|").replace("\n", " ")


def fmt_ts(ms):
    if ms is None:
        return "-"
    return datetime.datetime.fromtimestamp(ms / 1000, TZ).strftime("%Y-%m-%d %H:%M:%S")


def fmt_hms_ms(ms):
    """Trigger time with millisecond precision: HH:MM:SS.mmm"""
    if ms is None:
        return "-"
    return datetime.datetime.fromtimestamp(ms / 1000, TZ).strftime("%H:%M:%S.%f")[:-3]


def dur_s(start_ms, end_ms):
    if start_ms is None or end_ms is None:
        return None
    return (end_ms - start_ms) / 1000.0


def _clean(s):
    """Normalize whitespace but do NOT truncate content.

    说明项需要输出完整信息（dashboard 选中提示框 / tooltip / dfx 报告表均展示完整 desc），
    只折叠多余空白、保留换行，不做长度裁剪、不加省略号。
    """
    s = (s or "").strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def describe(tool, inp):
    if not isinstance(inp, dict):
        return ""
    if tool == "read":
        return "读取 " + (inp.get("filePath") or "")
    if tool == "write":
        return "写入 " + (inp.get("filePath") or "")
    if tool == "edit":
        return "修改 " + (inp.get("filePath") or "")
    if tool == "bash":
        # 说明包含命令描述与命令本身（完整内容，不裁剪）
        desc = (inp.get("description") or "").strip()
        cmd = (inp.get("command") or "").strip()
        if desc and cmd:
            return _clean(f"{desc}：{cmd}")
        return _clean(desc or cmd)
    if tool == "skill":
        return "加载技能 " + str(inp.get("name") or "")
    if tool == "task":
        return f"派发[{inp.get('subagent_type') or '?'}] {_clean(inp.get('description') or inp.get('prompt') or '')}"
    if tool == "grep":
        return f"搜索 {_clean(inp.get('pattern') or '')}"
    if tool == "glob":
        return f"匹配 {_clean(inp.get('pattern') or '')}"
    if tool == "state_transition":
        return f"状态流转 {inp.get('stage')} -> {inp.get('action')}"
    if tool == "webfetch":
        return "抓取 " + _clean(inp.get("url") or "")
    if tool == "todowrite":
        items = inp.get("todos") or []
        return f"更新任务清单（{len(items)} 项）"
    return _clean(json.dumps(inp, ensure_ascii=False))


STAGE_MAP = {
    1: "Stage 1 规格规划",
    2: "Stage 2 数学建模",
    3: "Stage 3 架构设计",
    4: "Stage 4 模块接口/脚手架",
    5: "Stage 5 实现与验证",
    6: "Stage 6 E2E 门禁",
    7: "Stage 7 性能调优",
}
# 兼容 "Stage 5" / "Stage5" / "stage 5" 等大小写与空格差异
_STAGE_RE = re.compile(r"[Ss]tage\s*([1-7])\b")
# optimizer S1_SETUP..S5_REPORT 均属 Stage 7 性能调优
_STAGE7_PHASE_RE = re.compile(r"\bS[1-5]_(SETUP|COLLECT|ANALYZE|FRONTEND|SWIMLANE|INCORE|REPORT)\b")


def categorize(title):
    m = _STAGE_RE.search(title)
    if m:
        return STAGE_MAP[int(m.group(1))]
    tl = title.lower()
    if _STAGE7_PHASE_RE.search(title):
        return STAGE_MAP[7]
    if "debug phase" in tl or "verify phase" in tl:
        return STAGE_MAP[5]
    return "编排/其他"


def build_message_entry(msg_info: dict) -> dict:
    return {
        "id": msg_info.get("id"),
        "role": msg_info.get("role"),
        "agent": msg_info.get("agent"),
        "mode": msg_info.get("mode"),
        "model": msg_info.get("modelID"),
        "time": msg_info.get("time") or {},
        "tokens": msg_info.get("tokens") or {},
        "cost": msg_info.get("cost"),
        "tools": [],
    }


def event_context(node_id, title: str, msg_info: dict) -> dict:
    return {
        "session_id": node_id,
        "session_title": title,
        "agent": msg_info.get("agent"),
        "role": msg_info.get("role"),
    }


def build_tool_record(ctx: dict, part: dict, start, end) -> dict:
    tool = part.get("tool")
    state = part.get("state", {})
    return {
        **ctx,
        "tool": tool,
        "kind": "tool",
        "status": state.get("status"),
        "start_ms": start,
        "end_ms": end,
        "duration_sec": dur_s(start, end),
        "category": CATEGORY.get(tool, tool),
        "desc": describe(tool, state.get("input")),
    }


def build_timed_record(ctx: dict, part_type: str, start, end, text) -> dict:
    return {
        **ctx,
        "tool": part_type,
        "kind": part_type,
        "status": "completed",
        "start_ms": start,
        "end_ms": end,
        "duration_sec": dur_s(start, end),
        "category": CATEGORY.get(part_type, part_type),
        "desc": _clean(text) if isinstance(text, str) else "",
    }


def tool_entry(part: dict, state: dict, start, end) -> dict:
    return {
        "tool": part.get("tool"),
        "callID": part.get("callID"),
        "status": state.get("status"),
        "title": state.get("title"),
        "time": {"start": start, "end": end},
        "input": state.get("input"),
    }


def collect_thinking_events(ctx: dict, parts: list) -> list:
    """reasoning / text events that carry real timestamps (text must have duration)."""
    records = []
    for part in parts:
        part_type = part.get("type")
        if part_type not in ("reasoning", "text"):
            continue
        times = part.get("time") or {}
        start, end = times.get("start"), times.get("end")
        if start is None:
            continue
        if part_type == "text" and (end is None or end <= start):
            continue  # 瞬时文本输出（0 耗时），跳过
        records.append(build_timed_record(ctx, part_type, start, end, part.get("text")))
    return records


def collect_message_events(node: dict, node_id, title: str) -> tuple:
    """Walk the node messages once: message entries + flattened event records."""
    msgs = []
    exec_records = []
    for msg in node.get("messages", []):
        msg_info = msg.get("info", {})
        entry = build_message_entry(msg_info)
        ctx = event_context(node_id, title, msg_info)
        parts = msg.get("parts", [])
        for part in parts:
            if part.get("type") != "tool":
                continue
            state = part.get("state", {})
            times = state.get("time") or {}
            start = times.get("start")
            if start is None:
                start = (msg_info.get("time") or {}).get("created")
            end = times.get("end")
            entry["tools"].append(tool_entry(part, state, start, end))
            exec_records.append(build_tool_record(ctx, part, start, end))
        exec_records.extend(collect_thinking_events(ctx, parts))
        msgs.append(entry)
    exec_records.sort(key=lambda r: r["start_ms"] or 0)
    return msgs, exec_records


def node_stats(msgs: list) -> dict:
    """Wall-clock bounds, active seconds and token totals for one node."""
    lo, hi, active = None, None, 0.0
    tokens = {"total": 0, "input": 0, "output": 0, "reasoning": 0,
              "cache_read": 0, "cache_write": 0}
    for msg in msgs:
        times = msg["time"]
        created, completed = times.get("created"), times.get("completed")
        if created is not None:
            lo = created if lo is None else min(lo, created)
        if completed is not None:
            hi = completed if hi is None else max(hi, completed)
        duration = dur_s(created, completed)
        if duration is not None:
            active += duration
        token_info = msg["tokens"] or {}
        tokens["total"] += token_info.get("total") or 0
        tokens["input"] += token_info.get("input") or 0
        tokens["output"] += token_info.get("output") or 0
        tokens["reasoning"] += token_info.get("reasoning") or 0
        cache = token_info.get("cache") or {}
        tokens["cache_read"] += cache.get("read") or 0
        tokens["cache_write"] += cache.get("write") or 0
    return {"lo": lo, "hi": hi, "active": active, "tokens": tokens}


def walk_tree(node, order, out) -> None:
    info = node.get("info", {})
    node_id = info.get("id")
    title = info.get("title", "")
    idx = order[0]
    order[0] += 1
    msgs, exec_records = collect_message_events(node, node_id, title)
    stats = node_stats(msgs)
    out.append({
        "order": idx,
        "id": node_id,
        "parent_id": info.get("parent_id"),
        "title": title,
        "directory": info.get("directory", ""),
        "stage": categorize(title),
        "msg_count": len(msgs),
        "duration_sec": dur_s(stats["lo"], stats["hi"]),
        "active_sec": stats["active"],
        "tokens": stats["tokens"],
        "messages": msgs,
        "exec_records": exec_records,
    })
    for child in node.get("children", []):
        walk_tree(child, order, out)


def _check_message(msg, where, problems) -> None:
    if not isinstance(msg, dict):
        problems.append(f"{where} is not an object")
        return
    if msg.get("info") is not None and not isinstance(msg["info"], dict):
        problems.append(f"{where}.info is not an object")
    parts = msg.get("parts")
    if parts is not None and not isinstance(parts, list):
        problems.append(f"{where}.parts is not an array")


def _check_node(node, where, problems) -> None:
    if not isinstance(node, dict):
        problems.append(f"{where} is not an object")
        return
    if node.get("info") is not None and not isinstance(node["info"], dict):
        problems.append(f"{where}.info is not an object")
    msgs = node.get("messages")
    if isinstance(msgs, list):
        for i, msg in enumerate(msgs):
            _check_message(msg, f"{where}.messages[{i}]", problems)
    elif msgs is not None:
        problems.append(f"{where}.messages is not an array")
    children = node.get("children")
    if isinstance(children, list):
        for i, child in enumerate(children):
            _check_node(child, f"{where}.children[{i}]", problems)
    elif children is not None:
        problems.append(f"{where}.children is not an array")


def validate_root(root, path) -> bool:
    """Stage-0 structural check: info/messages/children must be usable types."""
    problems = []
    _check_node(root, "root", problems)
    if problems:
        LOGGER.error("invalid session export %s: %s", path, "; ".join(problems[:5]))
        return False
    return True


def load_attempt(attempt_dir):
    path = os.path.join(attempt_dir, "root_full.json")
    with open(path) as f:
        root = json.load(f)
    if not validate_root(root, path):
        return None, None
    nodes = []
    walk_tree(root, [0], nodes)
    return root, nodes


def find_attempts(op_dir):
    base = os.path.join(op_dir, "pypto_sessions")
    if not os.path.isdir(base):
        return []
    attempts = sorted(
        (d for d in os.listdir(base) if re.fullmatch(r"attempt_\d+", d)),
        key=lambda d: (int(d.rsplit("_", 1)[1]), d),
    )
    return [d for d in attempts if os.path.isfile(os.path.join(base, d, "root_full.json"))]


def render_time_section(data) -> str:
    lines = ["## 2. 端到端耗时与各阶段/agent 耗时", ""]
    for i, attempt in enumerate(data, start=1):
        nodes = data[attempt]["nodes"]
        lines.append(f"### 2.{i} {attempt} 端到端时间")
        lines.append("")
        lines.append("| Session | 开始 | 结束 | 墙钟(s) | 活跃(s) | 阶段 | 说明 |")
        lines.append("|---------|------|------|--------|--------|------|------|")
        for node in nodes:
            lines.append(render_time_row(node))
        lines.append("")
    return "\n".join(lines)


def render_time_row(node: dict) -> str:
    """Node wall-clock row: prefer exec-record ms bounds, fall back to messages."""
    rec_starts = [r["start_ms"] for r in node["exec_records"] if r.get("start_ms") is not None]
    rec_ends = [r["end_ms"] for r in node["exec_records"] if r.get("end_ms") is not None]
    first_msg = node["messages"][0]["time"].get("created") if node["messages"] else None
    lo = min(rec_starts) if rec_starts else first_msg
    if rec_ends:
        hi = max(rec_ends)
    elif lo and node["duration_sec"]:
        hi = lo + node["duration_sec"] * 1000
    else:
        hi = None
    duration = f"{node['duration_sec']:.1f}" if node["duration_sec"] is not None else "-"
    active = f"{node['active_sec']:.1f}"
    start_text = fmt_ts(lo) if lo else "-"
    end_text = fmt_ts(hi) if hi else "-"
    return (f"| `{node['id'][:12]}` | {start_text} | {end_text} | {duration} | {active} | "
            f"{node['stage']} | {esc(node['title'][:60])} |")


def render_token_section(data) -> str:
    lines = ["## 3. 各阶段/agent/subagent 的 Token 消耗", ""]
    for i, attempt in enumerate(data, start=1):
        nodes = data[attempt]["nodes"]
        lines.append(f"### 3.{2*i-1} {attempt} 各 subagent Token")
        lines.append("")
        lines.append("| Session | total | input | output | reasoning | cache-read | 阶段 |")
        lines.append("|---------|------:|------:|-------:|----------:|----------:|------|")
        for node in nodes:
            lines.append(render_token_row(node))
        lines.append("")
        lines.extend(render_stage_agg(attempt, i, nodes))
        lines.append("")
    return "\n".join(lines)


def render_token_row(node: dict) -> str:
    token_info = node["tokens"]
    return (f"| `{node['id'][:12]}` | {token_info['total']} | {token_info['input']} | "
            f"{token_info['output']} | {token_info['reasoning']} | "
            f"{token_info['cache_read']} | {node['stage']} |")


def aggregate_stage_tokens(nodes: list) -> dict:
    agg = collections.OrderedDict()
    for node in nodes:
        if not node["parent_id"]:
            continue
        entry = agg.setdefault(node["stage"], {
            "count": 0, "total": 0, "input": 0, "output": 0, "reasoning": 0, "cache_read": 0,
        })
        entry["count"] += 1
        for key in ("total", "input", "output", "reasoning", "cache_read"):
            entry[key] += node["tokens"][key]
    return agg


def render_stage_agg(attempt: str, i: int, nodes: list) -> list:
    lines = [f"### 3.{2*i} {attempt} 各阶段聚合", ""]
    lines.append("| 阶段 | subagent 数 | total | input | output | reasoning | cache-read |")
    lines.append("|------|:---:|---:|---:|---:|---:|---:|")
    for stage, value in aggregate_stage_tokens(nodes).items():
        lines.append(f"| {stage} | {value['count']} | {value['total']} | {value['input']} | "
                     f"{value['output']} | {value['reasoning']} | {value['cache_read']} |")
    return lines


def summarize_events(nodes: list) -> dict:
    """Event totals by category/status plus kind counters for one attempt."""
    cat = collections.Counter()
    status = collections.Counter()
    errors = collections.Counter()
    counts = {"total": 0, "tool": 0, "reasoning": 0, "text": 0}
    for node in nodes:
        for record in node["exec_records"]:
            cat[record["category"]] += 1
            status[record["status"]] += 1
            counts["total"] += 1
            if record["status"] == "error":
                errors[record["category"]] += 1
            kind = record.get("kind") or "tool"
            if kind in counts:
                counts[kind] += 1
            else:
                counts["tool"] += 1
    return {"cat": cat, "status": status, "errors": errors, "counts": counts}


def render_event_stats(attempt: str, i: int, summary: dict) -> list:
    counts = summary["counts"]
    lines = [f"### 5.{2*i-1} 事件/工具调用统计（{attempt}）", ""]
    parts = [f"总事件 **{counts['total']}** 条（工具调用 {counts['tool']} 次，"
             f"推理/思考 {counts['reasoning']} 次，文本输出 {counts['text']} 次）"]
    if summary["status"].get("error"):
        parts.append(f"错误 **{summary['status'].get('error')}** 次")
    lines.append("，".join(parts) + "；按操作类别：")
    lines.append("")
    lines.append("| 操作类别 | 次数 | 错误 |")
    lines.append("|---------|-----:|-----:|")
    for category, count in summary["cat"].most_common():
        lines.append(f"| {esc(category)} | {count} | {summary['errors'].get(category, 0)} |")
    return lines


EXEC_TABLE_HEADER = [
    "| 角色 | 触发时间 | 原始时间 | 耗时 | 操作类别 | 状态 | 说明 |",
    "|------|---------|---------|------|---------|------|------|",
]


def render_exec_row(record: dict) -> str:
    duration = f"{record['duration_sec']:.3f}s" if record["duration_sec"] is not None else "-"
    start_text = fmt_hms_ms(record["start_ms"]) if record["start_ms"] is not None else "-"
    raw = str(record["start_ms"]) if record["start_ms"] is not None else "-"
    return (f"| {esc(record['agent'] or record['role'])} | {start_text} | {raw} | {duration} | "
            f"{esc(record['category'])} | {esc(record['status'])} | {esc(record['desc'])} |")


def render_node_events(attempt: str, node: dict) -> list:
    lines = [f"#### {node['title']}", ""]
    lines.append(f"- **attempt**: {attempt} | **session**: `{node['id']}` | "
                 f"**事件次数**: {len(node['exec_records'])}")
    lines.append("")
    lines.extend(EXEC_TABLE_HEADER)
    for record in node["exec_records"]:
        lines.append(render_exec_row(record))
    lines.append("")
    return lines


def render_exec_section(data) -> str:
    lines = ["## 5. 执行轨迹", ""]
    for i, attempt in enumerate(data, start=1):
        nodes = data[attempt]["nodes"]
        summary = summarize_events(nodes)
        lines.extend(render_event_stats(attempt, i, summary))
        lines.append("")
        lines.append(f"### 5.{2*i} 各 subagent 逐次执行轨迹（{attempt}）")
        lines.append("")
        lines.append("> 字段说明：**角色**=执行 agent；**触发时间**=事件开始时刻（HH:MM:SS.mmm，毫秒级）；"
                     "**原始时间**=事件触发的 epoch 毫秒时间戳（如 1786269062333，便于全文检索定位）；"
                     "**耗时**=事件 start→end（秒，毫秒级）；**操作类别**=执行命令/读取文件/修改文件/写入文件/"
                     "加载技能/派发子任务/内容搜索/文件搜索/状态流转/推理/思考/文本输出等；"
                     "**状态**=completed/error。")
        lines.append("")
        active_nodes = sum(1 for node in nodes if node["exec_records"])
        lines.append(f"以下按调度顺序列出 {attempt} 全部 {active_nodes} 个有事件的节点，"
                     f"合计 {summary['counts']['total']} 条"
                     f"（推理/思考、文本输出等具备时间戳的事件已全量保留，不再留空档）。")
        lines.append("")
        for node in sorted(nodes, key=lambda item: item["order"]):
            if not node["exec_records"]:
                continue
            lines.extend(render_node_events(attempt, node))
    return "\n".join(lines)


DATA_DOC = """## 6. 可视化网页数据准备

以下数据文件已生成，可直接供可视化网页（时间线/泳道图/柱状图）消费：

| 文件 | 说明 |
|------|------|
| `dfx/dfx_data.json` | **结构化全量数据**：按 attempt 组织，每个节点含 id/title/stage/time/messages（消息级 time 与 tokens）/exec_records（事件级 start_ms/end_ms/类别/说明，含工具调用 + 推理/思考 + 有耗时的文本输出） |
| `dfx/exec_records/attempt_XX/*.md` | 各 attempt 各节点逐事件 Markdown 表格（毫秒级，含原始 epoch 时间戳） |
| `dfx/summary.txt` | 各节点墙钟/活跃/token 摘要 |
| `dfx/sections/*.md` | 脚本预渲染的确定性章节（2/3/5/6） |

### dfx_data.json 结构

```jsonc
{
  "attempt_01": {                          // 键 = attempt 目录名（与 dfx_data.json 一致）
    "root_info": {"id": "...", "title": "...", "time": {"created": ms, "updated": ms}},
    "nodes": [
      {
        "id": "ses_...", "parent_id": null, "title": "...",
        "stage": "Stage 1 规格规划",          // 阶段分类
        "msg_count": 30,
        "duration_sec": 283.9, "active_sec": 282.5,
        "tokens": {"total": ..., "input": ..., "output": ...,
                   "reasoning": ..., "cache_read": ..., "cache_write": ...},
        "messages": [                       // 消息级（可视化时间线主数据）
          { "id": "...", "role": "assistant", "agent": "pypto-op-planner",
            "time": {"created": ms, "completed": ms},
            "tokens": {...},
            "tools": [ {"tool": "bash", "status": "completed",
                        "time": {"start": ms, "end": ms}, "input": {...}} ] }
        ],
        "exec_records": [                   // 事件级扁平记录（泳道图/事件流）
          // 工具调用：kind="tool"
          { "agent": "...", "tool": "bash", "kind": "tool", "start_ms": ms, "end_ms": ms,
            "duration_sec": 1.234, "category": "执行命令", "desc": "...", "status": "completed" },
          // 模型推理/思考（parts[].time.start/end）：kind="reasoning"
          { "agent": "...", "tool": "reasoning", "kind": "reasoning", "start_ms": ms, "end_ms": ms,
            "duration_sec": 0.932, "category": "推理/思考", "desc": "模型思考内容", "status": "completed" }
        ]
      }
    ]
  },
  "attempt_02": { ... }
}
```

### 可视化建议

- **端到端时间线/泳道图**: 以 `exec_records[].start_ms/end_ms` 或 `messages[].time` 为时间轴，按 attempt / stage 分色、按 agent 分行。
- **执行轨迹不留空档**: 具备具体耗时的事件全量保留（工具调用 + 推理/思考 + 有耗时的文本输出），
  泳道图中模型推理时段也以矩形呈现，避免工具调用之间的长空档。
- **耗时分布**: 用 `duration_sec` / `active_sec` 绘制各 subagent 横向条形图。
- **Token 分布**: 用 `tokens.total/input/output/reasoning` 绘制堆叠条形图（按 stage / subagent 聚合）。
- **性能调优趋势**: 报告 4.2 节趋势数据用折线图。
- **工具调用统计**: 5.x 节类别计数渲染为饼图/柱状图。
"""


def write_exec_tables(attempts: list, data: dict, exec_root: str) -> None:
    """Per-attempt, per-node Markdown tables with ms-precision events."""
    for attempt in attempts:
        attempt_dir = os.path.join(exec_root, attempt)
        os.makedirs(attempt_dir, exist_ok=True)
        for node in data[attempt]["nodes"]:
            if not node["exec_records"]:
                continue
            lines = [f"### {node['title']}", "",
                     f"- **session**: `{node['id']}` (order {node['order']})",
                     f"- **事件次数**: {len(node['exec_records'])}", ""]
            lines.extend(EXEC_TABLE_HEADER)
            for record in node["exec_records"]:
                lines.append(render_exec_row(record))
            fname = f"{node['order']:02d}_{node['id'][:8]}.md"
            with open(os.path.join(attempt_dir, fname), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")


def build_summary_lines(attempts: list, data: dict) -> list:
    """Compact per-node stats consumed by the LLM (low token count)."""
    summary_lines = []
    for attempt in attempts:
        summary_lines.append(f"===== ATTEMPT {attempt} =====")
        root_info = data[attempt]["root_info"]
        root_time = root_info.get("time") or {}
        wall = dur_s(root_time.get("created"), root_time.get("updated"))
        wall_text = f"{wall:.1f}s" if wall is not None else "-"
        summary_lines.append(f"root={root_info.get('id')} created={fmt_ts(root_time.get('created'))} "
                             f"updated={fmt_ts(root_time.get('updated'))} wall={wall_text}")
        for node in data[attempt]["nodes"]:
            summary_lines.append(render_summary_row(node))
    return summary_lines


def render_summary_row(node: dict) -> str:
    tokens = node["tokens"]
    duration = f"{node['duration_sec']:.1f}" if node["duration_sec"] is not None else "-"
    active = f"{node['active_sec']:.1f}" if node["active_sec"] is not None else "-"
    return (f"[{node['order']:02d}][{node['stage']}] {node['id'][:12]} msgs={node['msg_count']} "
            f"wall={duration}s active={active}s "
            f"tokens(t/i/o/r/cR)={tokens['total']}/{tokens['input']}/{tokens['output']}/"
            f"{tokens['reasoning']}/{tokens['cache_read']} {node['title'][:60]}")


def write_sections(sections_dir: str, data: dict) -> None:
    rendered = {
        "2_time.md": render_time_section(data),
        "3_token.md": render_token_section(data),
        "5_exec.md": render_exec_section(data),
        "6_data.md": DATA_DOC,
    }
    for name, content in rendered.items():
        with open(os.path.join(sections_dir, name), "w", encoding="utf-8") as f:
            f.write(content + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("op_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    op_dir = os.path.abspath(args.op_dir)
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(op_dir, "dfx")
    exec_root = os.path.join(out_dir, "exec_records")
    sections_dir = os.path.join(out_dir, "sections")

    attempts = find_attempts(op_dir)
    if not attempts:
        LOGGER.error("no attempt_XX/root_full.json under %s/pypto_sessions", op_dir)
        return 1

    data = {}
    for attempt in attempts:
        session_dir = os.path.join(op_dir, "pypto_sessions", attempt)
        root, nodes = load_attempt(session_dir)
        if root is None:
            return 1
        data[attempt] = {"root_info": root.get("info") or {}, "nodes": nodes}

    os.makedirs(exec_root, exist_ok=True)
    os.makedirs(sections_dir, exist_ok=True)
    write_exec_tables(attempts, data, exec_root)
    with open(os.path.join(out_dir, "dfx_data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    summary_lines = build_summary_lines(attempts, data)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    write_sections(sections_dir, data)

    LOGGER.info("OK: attempts=%s nodes=%d", list(data.keys()),
                sum(len(item["nodes"]) for item in data.values()))
    LOGGER.info("out_dir=%s", out_dir)
    LOGGER.info("%s", summary_lines[0:3])
    LOGGER.info("... (full summary in dfx/summary.txt)")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
