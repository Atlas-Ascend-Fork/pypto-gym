#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import os

from . import checks  # noqa: F401
from .core import (
    CheckContext,
    Finding,
    MODE_ENV,
    POST_EDIT_BLOCK_ENV,
    _run_checks,
)
from .infer import (
    _build_context,
    _infer_op_dir,
    _infer_op_name_from_filename,
    _load_hook_input,
    _rule_ids_for_filename,
)
from .observability import (
    _output_hook_json,
)


def _post_edit_target(data: dict) -> tuple[CheckContext, str] | None:
    """Resolve (CheckContext, basename) for the edited file.

    Returns None if the file is not an operator artifact.
    """
    file_path = data.get("tool_input", {}).get("file_path", "")
    if not file_path:
        return None
    file_path = os.path.abspath(file_path)
    basename = os.path.basename(file_path)
    op_name = _infer_op_name_from_filename(
        basename, staged=os.path.basename(os.path.dirname(file_path)) == "modules"
    )
    op_dir = _infer_op_dir(file_path, op_name)
    if not op_dir:
        return None
    ctx = _build_context(op_dir, op_name=op_name)
    ctx.file_scope = file_path
    return ctx, basename


def _split_findings(
    findings: list[Finding],
) -> tuple[list[Finding], list[Finding], list[Finding], list[Finding]]:
    fails = [f for f in findings if f.status == "FAIL"]
    error_fails = [f for f in fails if f.severity in ("S0", "S1")]
    warns = [f for f in findings if f.status == "WARN"]
    infos = [f for f in findings if f.status == "INFO"]
    return fails, error_fails, warns, infos


def _format_finding(finding: Finding) -> str:
    loc = ""
    message_already_locates_file = bool(
        finding.file
        and any(
            line.startswith(f"{finding.file}:")
            for line in finding.message.splitlines()
        )
    )
    if finding.file and not message_already_locates_file:
        loc = f" {finding.file}"
        if finding.line:
            loc += f":{finding.line}"
    return f"  [{finding.rule_id}][{finding.severity}]{loc} {finding.message}"


def _finding_context(
    fails: list[Finding],
    warns: list[Finding],
    infos: list[Finding],
) -> str:
    sections: list[str] = []
    if fails:
        lines = [_format_finding(f) for f in fails]
        sections.append(
            "[pypto-pro-op-lint] The following rules were violated, fix them and rewrite the file:\n"
            + "\n".join(lines)
        )
    if warns:
        lines = [_format_finding(f) for f in warns]
        sections.append(
            "[pypto-pro-op-lint] The following reminders are recommended for confirmation:\n" + "\n".join(lines)
        )
    if infos:
        lines = [_format_finding(f) for f in infos]
        sections.append(
            "[pypto-pro-op-lint] The following messages are for information "
            "(does not affect the result):\n" + "\n".join(lines)
        )
    return "\n\n".join(sections)


def _rule_fix_hint(rule_id: str, ctx: CheckContext) -> str:
    rule = ctx.get_rule(rule_id)
    return rule.get("fix_hint", "Refer to the rule description in rules.json to fix")


def _blocking_reason(error_fails: list[Finding], ctx: object) -> str:
    lines = [_format_finding(f) for f in error_fails]
    hint_lines = [
        f"  - {f.rule_id}: {_rule_fix_hint(f.rule_id, ctx)}"
        for f in error_fails
    ]
    blocking_rules = sorted({f.rule_id for f in error_fails})
    footer = (
        "\n\nFix the reported violations using the hints above, then rerun this check."
    )
    return (
        "[pypto-pro-op-lint] Post-edit checks failed (S0/S1):\n"
        + "\n".join(lines)
        + "\n\nblocking_rules: "
        + ", ".join(blocking_rules)
        + "\nfix_hints:\n"
        + "\n".join(hint_lines)
        + footer
    )


def hook_post_edit() -> int:
    """按文件类型检查，输出 decision/context JSON，由调用方处理。"""
    os.environ[MODE_ENV] = "post-edit"
    target = _post_edit_target(_load_hook_input())
    if target is None:
        _output_hook_json(
            "PostToolUse",
            decision="block",
            reason="[pypto-pro-op-lint] cannot locate the Pro operator directory of the edited file",
            additionalContext="",
        )
        return 0
    ctx, basename = target

    rule_ids = _rule_ids_for_filename(basename)
    if not rule_ids:
        _output_hook_json(
            "PostToolUse",
            decision="block",
            reason=f"[pypto-pro-op-lint] cannot select post-edit rules for {basename}",
            additionalContext="",
        )
        return 0

    findings, _ = _run_checks(ctx, rule_ids)
    fails, error_fails, warns, infos = _split_findings(findings)
    if not fails and not warns and not infos:
        _output_hook_json(
            "PostToolUse", decision="allow", reason="", additionalContext=""
        )
        return 0

    context_msg = _finding_context(fails, warns, infos)
    strict_block = os.environ.get(POST_EDIT_BLOCK_ENV, "1") == "1"
    if strict_block and error_fails:
        _output_hook_json(
            "PostToolUse",
            decision="block",
            reason=_blocking_reason(error_fails, ctx),
            additionalContext=context_msg,
        )
        return 0

    _output_hook_json(
        "PostToolUse", decision="allow", reason="", additionalContext=context_msg
    )
    return 0
