# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ast
import json
import os
import re
import subprocess

from ..ast_helpers import _get_jit_functions
from ..core import (
    API_REPORT_FILE,
    DESIGN_FILE,
    GOLDEN_PERF_REPORT_FILE,
    PYTHON_BIN,
    SPEC_FILE,
    CheckContext,
    Finding,
    register,
)
from ..utils import (
    _extract_markdown_headings,
    _extract_section_text,
    _has_heading_like,
    _impl_files_to_scan,
    _parse_front_matter,
    _phase_to_module_suffix,
    _validate_doc_schema,
)

_REQUIRED_SPEC_HEADINGS = ("数学公式", "输入输出规格", "精度要求")


def _first_section(content: str, *keywords: str) -> str:
    for keyword in keywords:
        section = _extract_section_text(content, keyword)
        if section:
            return section
    return ""


def _spec_content_issues(content: str) -> list[str]:
    headings = _extract_markdown_headings(content)
    missing = [
        heading
        for heading in _REQUIRED_SPEC_HEADINGS
        if not _has_heading_like(headings, heading)
    ]
    issues: list[str] = []
    if missing:
        issues.append(
            f"Missing required sections: {', '.join(missing)}"
            f"(a keyword may fail to match if the heading uses an English equivalent)"
        )
    math_text = _first_section(content, "数学", "算法", "基础信息")
    formula_tokens = ("$$", "=", "\\begin{equation}", "round", "clamp")
    if not math_text or not any(token in math_text for token in formula_tokens):
        issues.append(
            "Math definition section content is insufficient (missing parseable formula traits)"
        )
    io_text = _first_section(content, "输入输出规格", "数据规格")
    if not io_text or "dtype" not in io_text.lower() or "shape" not in io_text.lower():
        issues.append(
            "Input/output spec section content is insufficient (must include shape/dtype)"
        )
    precision_text = _extract_section_text(content, "精度")
    tolerance_tokens = ("atol", "rtol", "mare")
    if not precision_text or not any(
        token in precision_text.lower() for token in tolerance_tokens
    ):
        issues.append(
            "Accuracy requirements section content is insufficient "
            "(must include atol/rtol or metric thresholds)"
        )
    return issues


def _spec_schema_issues(content: str) -> list[str]:
    spec_meta, _ = _parse_front_matter(content)
    if not spec_meta:
        return [
            "Missing front matter"
            "(must start with ---, include schema_version/op_name/supported_dtypes/p0_shapes/tolerance)"
        ]
    schema_errors = _validate_doc_schema("SPEC", spec_meta)
    if schema_errors:
        return [f"front matter schema invalid: {'; '.join(schema_errors)}"]
    return []


@register("OL09")
def check_ol09(ctx: CheckContext) -> Finding:
    if not ctx.file_exists(SPEC_FILE):
        return ctx.make_finding("OL09", "FAIL", f"{SPEC_FILE} does not exist")
    content = ctx.read_file(SPEC_FILE)
    if ctx.op_name not in content:
        return ctx.make_finding(
            "OL09",
            "FAIL",
            f"{SPEC_FILE} does not contain operator name '{ctx.op_name}'",
            file=SPEC_FILE,
        )

    issues = _spec_content_issues(content)
    issues.extend(_spec_schema_issues(content))

    if issues:
        return ctx.make_finding(
            "OL09",
            "FAIL",
            f"{SPEC_FILE} has {len(issues)} issues:\n"
            + "\n".join(f"  - {i}" for i in issues),
            file=SPEC_FILE,
        )

    return ctx.make_finding(
        "OL09",
        "PASS",
        f"{SPEC_FILE} contains operator name, formulas, input/output spec "
        "and accuracy requirements; front matter schema is valid",
        file=SPEC_FILE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# OL54 — Phase M_k 自我评审证据（MEMORY.md）
# ─────────────────────────────────────────────────────────────────────────────


_SELF_REVIEW_HEADING_TMPL = re.compile(
    r"^\s*##\s+Phase\s+(M\d+)\s+self-review.*$",
    re.IGNORECASE | re.MULTILINE,
)
_CHECKLIST_ITEM_RE = re.compile(
    r"^\s*[-*]\s+\[([ xX✓✗❌])\]\s+(.+?)(?:\s*$)",
    re.MULTILINE,
)


_REQUIRED_SELF_REVIEW_KEYWORDS = [
    "signature",  # host_wrapper signature == module_interfaces.yaml
    "output",  # outputs written via assemble / slice
    "view",  # pypto.view shape/offsets rank
    "inventory",  # SPEC golden inventory cross-check
    "for ... in range",  # Layer K Python for-range absent
    "exactly once",  # Layer K JIT call exactly once
]


def _self_review_section(text: str, phase_scope: str) -> str | None:
    target = next(
        (
            match
            for match in _SELF_REVIEW_HEADING_TMPL.finditer(text)
            if match.group(1).upper() == phase_scope.upper()
        ),
        None,
    )
    if target is None:
        return None
    start = target.end()
    next_heading = re.search(r"^\s*##\s+", text[start:], re.MULTILINE)
    if next_heading is None:
        return text[start:]
    end = start + next_heading.start()
    return text[start:end]


def _self_review_problems(body: str) -> list[str]:
    items = _CHECKLIST_ITEM_RE.findall(body)
    matched = {
        keyword: next(
            ((mark, desc) for mark, desc in items if keyword.lower() in desc.lower()),
            None,
        )
        for keyword in _REQUIRED_SELF_REVIEW_KEYWORDS
    }
    missing = [keyword for keyword, item in matched.items() if item is None]
    unchecked = [
        keyword
        for keyword, item in matched.items()
        if item is not None and item[0].strip().lower() not in ("x", "✓")
    ]
    problems: list[str] = []
    if missing:
        problems.append(
            f"Missing required items ({len(missing)}): {', '.join(missing)}"
        )
    if unchecked:
        problems.append(f"Unchecked items ({len(unchecked)}): {', '.join(unchecked)}")
    return problems


@register("OL54")
def check_ol54(ctx: CheckContext) -> Finding:
    """complete_phase 时, `MEMORY.md` 必须存在 `## Phase M_k self-review` 章节,
    且 6 个必填检查项均已 ✅ 标记。

    必填项 (按子串匹配):
      1. host_wrapper signature 与 module_interfaces.yaml 一致
      2. 所有 output 通过 pypto.assemble / `[:] =` 写回
      3. 所有 pypto.view 的 shape/offsets/valid_shape rank 一致
      4. SPEC golden inventory 每行均含 impl 侧 line ref
      5. Layer K 内不存在 `for ... in range(...)`
      6. Layer K 的 JIT call 恰好一次

    `phase_scope` 未设置 (即 complete_stage / general check) 时 SKIP。
    """
    phase_scope = getattr(ctx, "phase_scope", None)
    if not phase_scope:
        return ctx.make_finding(
            "OL54",
            "SKIP",
            "phase_scope not set — this rule only takes effect during complete_phase",
        )
    memory_file = "MEMORY.md"
    if not ctx.file_exists(memory_file):
        return ctx.make_finding(
            "OL54",
            "FAIL",
            f"{memory_file} does not exist — Phase {phase_scope} self-review is required",
            file=memory_file,
        )
    text = ctx.read_file(memory_file)
    expected_heading = f"## Phase {phase_scope} self-review"
    body = _self_review_section(text, phase_scope)
    if body is None:
        return ctx.make_finding(
            "OL54",
            "FAIL",
            f"{memory_file} is missing the `{expected_heading}` section."
            f"All 6 required checklist items must be completed before complete_phase."
            f"See skill `pypto-memory-template` SKILL.md for the template.",
            file=memory_file,
        )
    problems = _self_review_problems(body)
    if problems:
        return ctx.make_finding(
            "OL54",
            "FAIL",
            f"{memory_file} `{expected_heading}` section self-review is not complete.\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\nFix policy: fill each item as `- [x] <description>`, attach evidence when necessary "
            "(impl line number or corresponding code snippet). Once any `- [ ]` / missing item exists, "
            "complete_phase will not pass.",
            file=memory_file,
        )
    return ctx.make_finding(
        "OL54",
        "PASS",
        f"Phase {phase_scope} self-review: all 6 items ✅",
        file=memory_file,
    )


@register("OL10")
def check_ol10(ctx: CheckContext) -> Finding:
    if not ctx.file_exists(API_REPORT_FILE):
        return ctx.make_finding("OL10", "FAIL", f"{API_REPORT_FILE} does not exist")
    content = ctx.read_file(API_REPORT_FILE)
    headings = _extract_markdown_headings(content)
    missing: list[str] = []
    if not _has_heading_like(headings, "API mapping"):
        missing.append("API mapping")
    if not _has_heading_like(headings, "constraints"):
        missing.append("constraints")
    if not _has_heading_like(headings, "Tiling"):
        missing.append("Tiling")
    if missing:
        return ctx.make_finding(
            "OL10",
            "FAIL",
            f"{API_REPORT_FILE} is missing required content: {', '.join(missing)}\n"
            f"Note: the above keywords may not match if the heading uses an English equivalent (e.g. API Mapping). "
            f"Please check whether the corresponding section heading contains the above keywords.",
            file=API_REPORT_FILE,
        )
    return ctx.make_finding(
        "OL10",
        "PASS",
        f"{API_REPORT_FILE} contains API mapping, constraints, and Tiling description",
        file=API_REPORT_FILE,
    )


@register("OL11")
def check_ol11(ctx: CheckContext) -> Finding:
    """进入 Stage 4 需 {op}_golden.py 可导入"""
    golden_file = f"{ctx.op_name}_golden.py"
    if not ctx.file_exists(golden_file):
        return ctx.make_finding("OL11", "FAIL", f"{golden_file} does not exist")
    probe_code = (
        "import importlib, sys\n"
        f"sys.path.insert(0, {json.dumps(ctx.op_dir)})\n"
        f"importlib.import_module({json.dumps(f'{ctx.op_name}_golden')})\n"
    )
    try:
        result = subprocess.run(
            [PYTHON_BIN, "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return ctx.make_finding(
            "OL11", "FAIL", f"{golden_file} import timed out (>10s)", file=golden_file
        )
    except OSError as e:
        return ctx.make_finding(
            "OL11", "FAIL", f"{golden_file} import probe failed: {e}", file=golden_file
        )
    if result.returncode != 0:
        return ctx.make_finding(
            "OL11",
            "FAIL",
            f"{golden_file} import failed: {result.stderr[:200]}",
            file=golden_file,
        )
    return ctx.make_finding(
        "OL11", "PASS", f"{golden_file} can be imported", file=golden_file
    )


@register("OL12")
def check_ol12(ctx: CheckContext) -> Finding:
    if not ctx.file_exists(DESIGN_FILE):
        return ctx.make_finding("OL12", "FAIL", f"{DESIGN_FILE} does not exist")
    content = ctx.read_file(DESIGN_FILE)
    headings = _extract_markdown_headings(content)
    missing: list[str] = []
    if not _has_heading_like(headings, "计算图") and not _has_heading_like(
        headings, "API 映射"
    ):
        missing.append("compute graph")
    if not _has_heading_like(headings, "Tiling") and not _has_heading_like(
        headings, "数据切分"
    ):
        missing.append("Tiling")
    if not _has_heading_like(headings, "verification plan"):
        missing.append("verification plan")
    if missing:
        return ctx.make_finding(
            "OL12",
            "FAIL",
            f"{DESIGN_FILE} is missing required content: {', '.join(missing)}\n"
            "Note: the above keywords may not match if the heading uses an "
            "English equivalent (e.g. Compute Graph, Verification Plan). "
            f"Please check whether the corresponding section heading contains the above keywords.",
            file=DESIGN_FILE,
        )
    return ctx.make_finding(
        "OL12",
        "PASS",
        f"{DESIGN_FILE} contains compute graph, Tiling and verification plan",
        file=DESIGN_FILE,
    )


@register("OL13")
def check_ol13(ctx: CheckContext) -> Finding:
    """Stage 5 cleanup 三件套：{op}_impl.py + test_{op}.py + README.md。"""
    files = [
        f"{ctx.op_name}_impl.py",
        f"test_{ctx.op_name}.py",
        "README.md",
    ]
    missing = [f for f in files if not ctx.file_exists(f)]
    if missing:
        return ctx.make_finding(
            "OL13",
            "FAIL",
            f"Stage 5 cleanup artifact trio is incomplete, missing: {', '.join(missing)}",
        )
    return ctx.make_finding("OL13", "PASS", "Stage 5 cleanup artifact trio is complete")


# module_count 来源: MEMORY.md (`module_count: 1`, construct skill 写) 为主,
# DESIGN.md §0.3 Decision (`module_count = 1`) 为后备。两源都用 1 表示 L0 单模块。
_MEMORY_MODULE_COUNT_RE = re.compile(r"module_count\s*:\s*(\d+)")
_DESIGN_MODULE_COUNT_RE = re.compile(r"Decision\D+module_count\s*=\s*(\d+)")


def _detect_module_count(ctx: CheckContext) -> int | None:
    """检测 module_count（L0=1 / L1≥2）。无法判定时返回 None，调用方维持 L1 行为。"""
    mem = _MEMORY_MODULE_COUNT_RE.search(ctx.read_file("MEMORY.md"))
    if mem:
        return int(mem.group(1))
    design = _DESIGN_MODULE_COUNT_RE.search(ctx.read_file(DESIGN_FILE))
    if design:
        return int(design.group(1))
    return None


def _load_state_file(state_path: str) -> tuple[dict | None, Exception | None]:
    try:
        with open(state_path, "r", encoding="utf-8") as state_file:
            return json.load(state_file), None
    except (ValueError, OSError) as exc:
        return None, exc


def _state_active_phase(state: dict) -> object | None:
    stage5_phases = state.get("stage5_phases")
    if not isinstance(stage5_phases, dict):
        return None
    return stage5_phases.get("active_phase")


def _phase_artifacts(op_name: str, suffix: str) -> list[str]:
    return [
        f"modules/{op_name}_module{suffix}_impl.py",
        f"modules/{op_name}_module{suffix}_golden.py",
        f"modules/test_{op_name}_module{suffix}.py",
    ]


def _ol44_active_phase(ctx: CheckContext) -> tuple:
    """当前活跃 Phase；无法判定、状态不可用或 L0 单模块时返回对应 Finding。

    返回 (active_phase, finding)：finding 非 None 时调用方应直接返回它。
    """
    state_path = ctx.file_path(".orchestrator_state.json")
    if not os.path.isfile(state_path):
        return None, ctx.make_finding(
            "OL44",
            "SKIP",
            ".orchestrator_state.json does not exist (stateless run), cannot determine Stage 5 modules/ status",
        )
    state, state_error = _load_state_file(state_path)
    if state_error is not None:
        return None, ctx.make_finding(
            "OL44", "FAIL", f"Failed to parse .orchestrator_state.json: {state_error}"
        )
    active_phase = _state_active_phase(state or {})
    if not active_phase:
        return None, ctx.make_finding(
            "OL44", "SKIP", "No active Phase M_k recorded in stage5_phases"
        )

    # L0 单模块: module_count == 1 时 Stage 5 直接产出 <op>_impl.py, 无 modules/
    # 目录, 强制三件套会误伤 L0 算子。仅当确定为 L0 时跳过, 否则维持 L1 行为。
    if _detect_module_count(ctx) == 1:
        return None, ctx.make_finding(
            "OL44",
            "SKIP",
            "L0 single-module: modules/ not expected (module_count == 1)",
        )
    return active_phase, None


@register("OL44")
def check_ol44(ctx: CheckContext) -> Finding:
    """Stage 5 当前 Phase 三件套：modules/<op>_module<k>_impl.py +
    modules/<op>_module<k>_golden.py + modules/test_<op>_module<k>.py。

    从 .orchestrator_state.json 读取当前活跃 Phase
    （stage5_phases.active_phase），解析后缀后验证三件套是否存在。

    L0 单模块（module_count == 1）：Stage 5 直接产出 <op>_impl.py，无
    modules/ 目录，故跳过本规则。module_count 无法判定时维持 L1 行为。
    """
    active_phase, finding = _ol44_active_phase(ctx)
    if finding is not None:
        return finding

    if not isinstance(active_phase, str):
        return ctx.make_finding(
            "OL44", "FAIL", f"Malformed active_phase: {active_phase!r}"
        )
    try:
        suffix = _phase_to_module_suffix(active_phase)
    except ValueError as e:
        return ctx.make_finding(
            "OL44", "FAIL", f"Malformed active_phase: {active_phase!r} ({e})"
        )

    modules_dir = ctx.file_path("modules")
    if not os.path.isdir(modules_dir):
        return ctx.make_finding(
            "OL44",
            "FAIL",
            "Stage 5 is active but the custom/<op>/modules/ directory does not exist",
        )

    expected = _phase_artifacts(ctx.op_name, suffix)
    missing = [p for p in expected if not ctx.file_exists(p)]
    if missing:
        return ctx.make_finding(
            "OL44",
            "FAIL",
            f"Active Phase {active_phase} artifact trio is incomplete, missing: {', '.join(missing)}",
        )
    return ctx.make_finding(
        "OL44",
        "PASS",
        f"Active Phase {active_phase} artifact trio is complete (impl + golden + test)",
    )


@register("OL14")
def check_ol14(ctx: CheckContext) -> Finding:
    """Stage 6（结构验证）进入前需要 Stage 5（含 cleanup）已完成。"""
    state_path = ctx.file_path(".orchestrator_state.json")
    if not os.path.isfile(state_path):
        return ctx.make_finding("OL14", "FAIL", "State file does not exist")
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        status = data.get("stage_status", {})
        # 新 Stage 1-7 模型：Stage 5 = Construction（per-Phase + cleanup）必须在
        # 进入 Stage 6 = 结构验证前完成。
        if status.get("5") == "completed":
            return ctx.make_finding(
                "OL14",
                "PASS",
                "Stage 5 complete, ready to enter Stage 6 (structure verification)",
            )
    except ValueError:
        pass
    return ctx.make_finding(
        "OL14",
        "FAIL",
        "Stage 6 entry blocked: Stage 5 not yet complete",
    )


@register("OL24")
def check_ol24(ctx: CheckContext) -> Finding:
    """.orchestrator_state.json 结构合法（schema v2.0）。"""
    state_path = ctx.file_path(".orchestrator_state.json")
    if not os.path.isfile(state_path):
        return ctx.make_finding(
            "OL24", "FAIL", ".orchestrator_state.json does not exist"
        )
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return ctx.make_finding("OL24", "FAIL", f"JSON parse failed: {e}")
    # schema v2.0 必需字段（同时兼容 v1 旧格式 — max_stage 缺失时视为旧版放行）。
    required = ["operator_name", "current_stage", "stage_status"]
    missing = [k for k in required if k not in data]
    if missing:
        return ctx.make_finding(
            "OL24",
            "FAIL",
            f"Missing required fields: {', '.join(missing)}",
        )
    # 软检查：max_stage 缺失时警告（旧版 schema）。
    if "max_stage" not in data:
        return ctx.make_finding(
            "OL24",
            "WARN",
            "State file does not declare max_stage (old schema v1 format); "
            "reinitialize via state_transition to upgrade to v2.0",
        )
    # v2.0 字段为可选，但若存在则必须格式正确。
    if "stage5_phases" in data:
        s5 = data["stage5_phases"]
        if not isinstance(s5, dict) or "phase_status" not in s5:
            return ctx.make_finding(
                "OL24",
                "FAIL",
                "stage5_phases must be a dict containing phase_status",
            )
    if "rollback_history" in data and not isinstance(data["rollback_history"], list):
        return ctx.make_finding(
            "OL24",
            "FAIL",
            "rollback_history must be a list",
        )
    if "artifact_hashes" in data and not isinstance(data["artifact_hashes"], dict):
        return ctx.make_finding(
            "OL24",
            "FAIL",
            "artifact_hashes must be a dict",
        )
    return ctx.make_finding(
        "OL24", "PASS", "State file structure is valid (schema v2.0)"
    )


@register("OL59")
def check_ol59(ctx: CheckContext) -> Finding:
    """Stage 2 完成时 GOLDEN_PERF_REPORT.md 必须存在且包含 Op Performance 表头"""
    if not ctx.file_exists(GOLDEN_PERF_REPORT_FILE):
        return ctx.make_finding(
            "OL59",
            "FAIL",
            f"{GOLDEN_PERF_REPORT_FILE} does not exist — follow the guidance in pypto-golden-generate SKILL.md §15, "
            f"run profile_golden.py to collect real NPU performance data and generate the report. "
            f"For operators with semantic constraints, use the --factory _make_inputs mode.",
            file=GOLDEN_PERF_REPORT_FILE,
        )
    content = ctx.read_file(GOLDEN_PERF_REPORT_FILE)
    if "| op | count | mean_duration | total |" not in content:
        return ctx.make_finding(
            "OL59",
            "FAIL",
            f"{GOLDEN_PERF_REPORT_FILE} is missing the Op Performance header "
            "— report format does not meet requirements, "
            f"please regenerate following pypto-golden-generate SKILL.md §15.",
            file=GOLDEN_PERF_REPORT_FILE,
        )
    return ctx.make_finding(
        "OL59",
        "PASS",
        f"{GOLDEN_PERF_REPORT_FILE} exists and contains the Op Performance header",
        file=GOLDEN_PERF_REPORT_FILE,
    )


_PREFLIGHT_TABLE_RE = re.compile(
    r"^\s*\|.{2,}\|",
    re.MULTILINE,
)
_PREFLIGHT_CHECKLIST_ITEM_RE = re.compile(
    r"^\s*-\s+\[([ xX✓✗❌\-])\]",
    re.MULTILINE,
)
_PREFLIGHT_PENDING_RE = re.compile(
    r"^\s*-\s+\[-\]",
    re.MULTILINE,
)
_PREFLIGHT_WARNING_ANNOTATION_RE = re.compile(
    r"^\s*>\s*⚠️\s*待验证",
    re.MULTILINE,
)
_PREFLIGHT_ACCEPTED_ANNOTATION_RE = re.compile(
    r"^\s*>\s*✅\s*已知风险",
    re.MULTILINE,
)


_PREFLIGHT_PLACEHOLDERS = (
    "This section is created by the preflight process",
    "no placeholder content needed here",
    "no specific preflight items found",
)


def _preflight_format_failures(section: str) -> list[str]:
    if not section or any(
        placeholder in section for placeholder in _PREFLIGHT_PLACEHOLDERS
    ):
        return [
            "[R5 Experience Preflight not executed]: MEMORY.md → "
            "'## Experience Preflight' is still a placeholder or missing.\n"
            "Fix policy: Coder must run the preflight scan before writing impl in Stage 5 "
            "(pypto-op-knowledge → references/experience_preflight.md), "
            "and write the checklist into MEMORY.md."
        ]
    failures: list[str] = []
    if _PREFLIGHT_TABLE_RE.search(section):
        failures.append(
            "[Preflight format violation]: MEMORY.md → "
            "'## Experience Preflight' uses table format (| ... |).\n"
            "Fix policy: must use the standard markdown checklist format "
            "(`- [x]`/`- [-]`/`- [ ]`), tables cannot carry "
            "the `> ⚠️ pending verification` sub-annotation and cannot be parsed by the gate."
        )
    checklist_items = _PREFLIGHT_CHECKLIST_ITEM_RE.findall(section)
    if not checklist_items:
        failures.append(
            "[Preflight format violation]: MEMORY.md → "
            "'## Experience Preflight': no checklist items detected.\n"
            "Fix policy: each item must have its own line "
            "`- [x]/[-] [S0/S1/S2] {description}`."
        )
    elif len(checklist_items) > 20:
        failures.append(
            f"[Preflight item count exceeded]: MEMORY.md → "
            f"'## Experience Preflight' contains {len(checklist_items)} items, exceeding the limit of 20.\n"
            f"Fix policy: trim the checklist, remove N/A items (API rules unused by the operator), "
            f"items duplicated with the fixed checklist, S2/S3 items, and standalone DEBUG_GUIDEBOOK entries."
        )
    pending_count = len(_PREFLIGHT_PENDING_RE.findall(section))
    warning_count = len(_PREFLIGHT_WARNING_ANNOTATION_RE.findall(section))
    if pending_count > 0 and warning_count < pending_count:
        failures.append(
            f"[Preflight [-] items missing pending-verification annotation]: "
            f"Detected {pending_count} `[-]` items, but only "
            f"{warning_count} `> ⚠️ pending verification` sub-annotations.\n"
            "Fix policy: each `[-]` item must be immediately followed by "
            "a `> ⚠️ pending verification: {specific item to confirm}` sub-annotation line."
        )
    return failures


def _next_nonempty_line(lines: list[str], start: int) -> str:
    for line in lines[start:]:
        if line.strip():
            return line
    return ""


def _unresolved_preflight_count(section: str) -> int:
    lines = section.splitlines()
    unresolved = 0
    for index, line in enumerate(lines):
        if not re.match(r"^\s*-\s+\[-\]", line):
            continue
        annotation = _next_nonempty_line(lines, index + 1)
        if not _PREFLIGHT_ACCEPTED_ANNOTATION_RE.match(annotation):
            unresolved += 1
    return unresolved


def _stage5_preflight_failures(section: str) -> list[str]:
    if not section:
        return [
            "[OL61 Impl stage]: MEMORY.md → "
            "'## Experience Preflight' does not exist.\n"
            "Fix policy: the Preflight checklist must be generated before writing impl in Stage 5."
        ]
    unresolved = _unresolved_preflight_count(section)
    if unresolved == 0:
        return []
    return [
        f"[OL61 Stage 5 Preflight [-] not resolved]: "
        f"MEMORY.md → '## Experience Preflight' still has "
        f"{unresolved} unresolved `[-]` items.\n"
        "Fix policy: every `[-]` must be resolved before coder dispatch:\n"
        "  (a) change to `- [x]` (verified compliant), or\n"
        "  (b) change `> ⚠️ pending verification` to `> ✅ known risk, accepted` (keep after confirming the risk)."
    ]


def _memory_preflight_section(ctx: CheckContext) -> str:
    if not ctx.file_exists("MEMORY.md"):
        return ""
    return _extract_section_text(ctx.read_file("MEMORY.md"), "Experience Preflight")


@register("OL61")
def check_ol61(ctx: CheckContext) -> Finding:
    """Experience Preflight 门禁检查。

    仅 Stage 5/6 运行（Stage 1-4 不触碰 MEMORY.md）：
      1. MEMORY.md preflight section 存在性（非占位符，Coder 创建）
      2. Preflight checklist 格式合规（markdown checklist，非表格；[-] 项有 ⚠️ 待验证）
      3. 所有 [-] 项已消除（改为 [x] 或标注 > ✅ 已知风险，接受）
      4. AST code scan (F1/F2/F4/F8)
    """
    failures = []

    if not ctx.file_exists(DESIGN_FILE):
        if ctx.file_scope and ctx.stage in (5, 6):
            _ol61_code_scan(ctx, failures)
            if failures:
                return ctx.make_finding(
                    "OL61",
                    "FAIL",
                    f"Stage 5 OL61 AST code scan failed ({len(failures)} items):\n"
                    + "\n".join(failures),
                    file=DESIGN_FILE,
                )
            return ctx.make_finding(
                "OL61", "PASS", "AST code scan passed", file=DESIGN_FILE
            )
        return ctx.make_finding(
            "OL61",
            "SKIP",
            f"{DESIGN_FILE} does not exist",
            file=DESIGN_FILE,
        )

    section = _memory_preflight_section(ctx)
    if ctx.file_exists("MEMORY.md") and ctx.stage in (5, 6):
        failures.extend(_preflight_format_failures(section))
    if ctx.stage == 5 and not ctx.file_scope:
        failures.extend(_stage5_preflight_failures(section))
    if ctx.stage in (5, 6):
        _ol61_code_scan(ctx, failures)

    if failures:
        return ctx.make_finding(
            "OL61",
            "FAIL",
            f"Stage {ctx.stage} OL61 check failed ({len(failures)} items):\n"
            + "\n".join(failures),
            file=DESIGN_FILE,
        )

    # OL61 仅在 stage 5/6 调度（rules.json stages=[5,6]）；Stage 1-4 不触碰 MEMORY.md
    return ctx.make_finding(
        "OL61",
        "PASS",
        f"Stage {ctx.stage} Experience Preflight validation passed (existence + format + [-] resolution + AST scan)",
        file=DESIGN_FILE,
    )


_LEGAL_CAST_PATHS: dict[str, set[str]] = {
    "DT_FP16": {"DT_FP32", "DT_INT32", "DT_INT16", "DT_INT8", "DT_UINT8", "DT_INT4"},
    "DT_BF16": {"DT_FP32", "DT_INT32"},
    "DT_INT32": {"DT_FP32", "DT_INT16", "DT_INT64", "DT_FP16"},
    "DT_FP32": {"DT_BF16", "DT_FP16", "DT_INT16", "DT_INT32", "DT_INT64"},
    "DT_UINT8": {"DT_FP16"},
    "DT_INT8": {"DT_FP16"},
    "DT_INT4": {"DT_FP16"},
    "DT_INT16": {"DT_FP32", "DT_FP16"},
    "DT_INT64": {"DT_FP32", "DT_INT32"},
}

_ARITH_OPS = {"div", "mul", "add", "sub"}

_ALLOC_OPS = {"zeros", "ones"}


def _is_pypto_attr(node: ast.AST, aliases: set[str], attr: str | None = None) -> bool:
    if not isinstance(node, ast.Attribute):
        return False
    if not isinstance(node.value, ast.Name):
        return False
    if node.value.id not in aliases:
        return False
    if attr is not None:
        return node.attr == attr
    return True


def _get_pypto_call_name(node: ast.Call, aliases: set[str]) -> str | None:
    if isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name) and node.func.value.id in aliases:
            return node.func.attr
    return None


def _is_pypto_cast_result(node: ast.AST, aliases: set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    return _get_pypto_call_name(node, aliases) == "cast"


def _tensor_annotation_dtype(
    annotation: ast.AST | None, aliases: set[str]
) -> str | None:
    if not isinstance(annotation, ast.Call):
        return None
    if not _is_pypto_attr(annotation.func, aliases, "Tensor"):
        return None
    return next(
        (
            arg.attr
            for arg in annotation.args
            if isinstance(arg, ast.Attribute) and arg.attr.startswith("DT_")
        ),
        None,
    )


def _get_jit_param_dtypes(tree: ast.Module, aliases: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for func in _get_jit_functions(tree, aliases):
        for arg in func.args.args:
            dtype = _tensor_annotation_dtype(arg.annotation, aliases)
            if dtype is not None:
                result[arg.arg] = dtype
    return result


def _ol61_code_scan(ctx: CheckContext, failures: list[str]) -> None:
    impl_files = _impl_files_to_scan(ctx)
    for impl_file in impl_files:
        tree = ctx.parse_file(impl_file)
        if tree is None:
            continue
        aliases = ctx.pypto_aliases(impl_file)
        param_dtypes = _get_jit_param_dtypes(tree, aliases)
        _ol61_5a_cast_path(tree, aliases, param_dtypes, impl_file, failures)
        _ol61_5b_element_wrap(tree, aliases, impl_file, failures)
        _ol61_5c_scalar_first_arg(tree, aliases, impl_file, failures)
        _ol61_5d_alloc_dtype(tree, aliases, impl_file, failures)


def _ol61_5a_cast_path(
    tree: ast.Module,
    aliases: set[str],
    param_dtypes: dict[str, str],
    impl_file: str,
    failures: list[str],
) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _get_pypto_call_name(node, aliases) != "cast":
            continue
        if len(node.args) < 2:
            continue
        target_dt = None
        for arg in node.args[1:]:
            if isinstance(arg, ast.Attribute) and arg.attr.startswith("DT_"):
                target_dt = arg.attr
                break
        if target_dt is None:
            continue
        src = node.args[0]
        src_is_cast = _is_pypto_cast_result(src, aliases)
        if src_is_cast:
            continue
        src_dt = None
        if isinstance(src, ast.Name) and src.id in param_dtypes:
            src_dt = param_dtypes[src.id]
        if src_dt is None:
            continue
        legal = _LEGAL_CAST_PATHS.get(src_dt, set())
        if target_dt not in legal:
            failures.append(
                f"[OL61 Preflight F4 illegal cast path] {impl_file}: "
                f"pypto.cast({src_dt} → {target_dt}) is not in the legal direct-cast path table.\n"
                f"Fix policy: use a stepping-stone cast, e.g. INT8→FP32 must go via FP16: "
                f"pypto.cast(pypto.cast(x, pypto.DT_FP16), pypto.DT_FP32).\n"
                f"Legal direct casts: {src_dt} → {{{', '.join(sorted(legal))}}}"
            )


def _ol61_5b_element_wrap(
    tree: ast.Module,
    aliases: set[str],
    impl_file: str,
    failures: list[str],
) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _get_pypto_call_name(node, aliases)
        if call_name is None:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Call) and _is_pypto_attr(
                arg.func, aliases, "Element"
            ):
                failures.append(
                    f"[OL61 Preflight F2 Element double wrapping] {impl_file}: "
                    f"pypto.{call_name}(..., pypto.Element(...), ...) — "
                    "Passing pypto.Element() as an argument to other pypto "
                    "operations causes double-wrapping crashes.\n"
                    f"Fix policy: use a plain Python scalar directly, e.g. pypto.mul(tensor, 127.0), "
                    f"do not construct pypto.Element(DT_FP32, 127.0)."
                )
                break


def _ol61_5c_scalar_first_arg(
    tree: ast.Module,
    aliases: set[str],
    impl_file: str,
    failures: list[str],
) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _get_pypto_call_name(node, aliases)
        if call_name not in _ARITH_OPS:
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, (int, float)):
            failures.append(
                f"[OL61 Preflight F1 scalar first argument] {impl_file}: "
                f"pypto.{call_name}({first.value!r}, ...) — "
                f"The first argument is a Python scalar and must be a Tensor.\n"
                f"Fix policy: swap the argument order, e.g. pypto.{call_name}(tensor, {first.value!r}), "
                f"or construct a scalar Tensor with pypto.full()."
            )


def _ol61_5d_alloc_dtype(
    tree: ast.Module,
    aliases: set[str],
    impl_file: str,
    failures: list[str],
) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _get_pypto_call_name(node, aliases)
        if call_name not in _ALLOC_OPS:
            continue
        if any(kw.arg == "dtype" for kw in node.keywords):
            continue
        for _, arg in enumerate(node.args):
            if isinstance(arg, ast.Attribute) and arg.attr.startswith("DT_"):
                failures.append(
                    f"[OL61 Preflight F8 {call_name} dtype position] {impl_file}: "
                    f"pypto.{call_name}(..., {arg.attr}) — "
                    f"dtype is swallowed by the positional *size argument; use a keyword argument instead.\n"
                    f"Fix policy: pypto.{call_name}(shape, dtype={arg.attr})"
                )
                break


def _stage7_tuning_mode(ctx: CheckContext) -> str:
    """Which Stage 7 path this operator is on, per the authoritative state file.

    Returns "" when the state file is absent or unreadable, which callers treat
    the same as "stepwise": a run with no orchestrator state has not declared a
    PANKO run, and a gate must not assume one.
    """
    path = ctx.file_path(".orchestrator_state.json")
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            mode = json.load(f).get("stage7_tuning_mode")
    except (json.JSONDecodeError, OSError):
        return ""
    return mode if isinstance(mode, str) else ""


def _ol63_off_path(ctx: CheckContext, mode, have_state) -> Finding:
    """The verdict for a run that is not on the PANKO path.

    A state file with no declared mode is the inconsistency, not a skip: PANKO
    demonstrably ran -- only panko_harness.py writes that file -- but the run was
    never declared, and skipping there would let the gate be switched off simply
    by not recording the mode.
    """
    if have_state:
        return ctx.make_finding(
            "OL63",
            "FAIL",
            "optimization/search_state.json exists (PANKO ran) but "
            f"stage7_tuning_mode is {mode or 'unset'} — record the path with "
            'state_transition(action="start_stage", stage=7, tuning_mode="panko") '
            "so the Stage 7 stop gate applies to this run",
            file=".orchestrator_state.json",
        )
    return ctx.make_finding(
        "OL63",
        "SKIP",
        f"stage7_tuning_mode={mode or 'unset'} (stepwise path) — OL63 governs "
        "the PANKO auto-tuning path only",
    )


def _ol63_progress(ctx: CheckContext, state_path):
    """(progress, failure) for the harness state file; exactly one is set."""
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return None, ctx.make_finding(
            "OL63", "FAIL", f"search_state.json failed to parse as JSON: {e}"
        )
    progress = data.get("progress")
    if not isinstance(progress, dict):
        return None, ctx.make_finding(
            "OL63", "FAIL", "search_state.json has no progress field"
        )
    return progress, None


@register("OL63")
def check_ol63(ctx: CheckContext) -> Finding:
    """Stage 7 completion gate: block complete_stage(7) until the harness recorded a real STOP.

    Applies to the PANKO tuning path ONLY. The path is read from
    .orchestrator_state.json's stage7_tuning_mode, which start_stage(7) records
    and which nothing else may write. It is not inferred from the artifacts: the
    stepwise path leaves no search_state.json, and so does a PANKO run the
    orchestrator abandoned before the first init -- exactly the failure this gate
    exists to catch. Absent or "stepwise" means the stepwise path, which this
    rule does not govern, so the gate returns SKIP and the default Stage 7 flow
    completes as it always has.

    On the PANKO path the orchestrator is a thin relay with no authority to end
    the search on "good enough" (background: an orchestrator once called
    complete_stage(7) long before the harness stop() would have fired,
    discarding most of the unexplored action space). The stop decision belongs to
    panko_harness.py's stop(), which persists it to progress.stop_reason in
    optimization/search_state.json. This gate only checks that the marker is
    non-empty; it does not duplicate the stop logic, so the two cannot drift.
    """
    mode = _stage7_tuning_mode(ctx)
    state_path = ctx.file_path(os.path.join("optimization", "search_state.json"))
    have_state = os.path.isfile(state_path)
    if mode != "panko":
        return _ol63_off_path(ctx, mode, have_state)
    if not have_state:
        return ctx.make_finding(
            "OL63",
            "FAIL",
            "stage7_tuning_mode=panko but optimization/search_state.json does not "
            "exist — the harness never ran, so no stop was decided; cannot "
            "complete_stage(7)",
        )
    progress, failure = _ol63_progress(ctx, state_path)
    if failure is not None:
        return failure
    reason = progress.get("stop_reason")
    if not isinstance(reason, str) or not reason.strip():
        return ctx.make_finding(
            "OL63",
            "FAIL",
            "the harness recorded no STOP (progress.stop_reason is empty) — the "
            "orchestrator must not end Stage 7 on its own judgement; relay to "
            "pypto-op-auto-tuner-panko until the harness stop() emits STOP",
            file="optimization/search_state.json",
        )
    return ctx.make_finding(
        "OL63",
        "PASS",
        f"harness STOP recorded (stop_reason={reason})",
        file="optimization/search_state.json",
    )
