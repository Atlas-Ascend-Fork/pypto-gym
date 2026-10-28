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
"""D3 Golden 纯度: PL11 (禁止 import pypto/pypto_pro) + PL13 (golden_stage 独立性)."""

from __future__ import annotations

import os

from ..ast_helpers import (
    has_module_component,
    is_framework_import,
    iter_imported_names,
    load_python_ast,
)
from ..core import CheckContext, Finding, register


def _scan_golden_files(ctx: CheckContext) -> list[tuple[str, str]]:
    """List (relative_filename, abs_path) of golden files to scan.

    In post-edit mode (file_scope set), only the edited file.
    Otherwise, scan top-level Golden files, plus modules/ in static context 4.
    """
    if ctx.file_scope:
        basename = os.path.basename(ctx.file_scope)
        return [(basename, ctx.file_scope)]

    results: list[tuple[str, str]] = []
    op_name = ctx.op_name

    for fname in [f"{op_name}_golden.py", f"{op_name}_golden_cpu.py"]:
        if ctx.file_exists(fname):
            results.append((fname, ctx.file_path(fname)))

    # Only static context 4 includes module Golden files.
    if ctx.stage != 4:
        return results

    modules_dir = os.path.join(ctx.op_dir, "modules")
    if os.path.isdir(modules_dir):
        for entry in sorted(os.listdir(modules_dir)):
            if "_golden_stage" in entry and entry.endswith(".py"):
                rel = os.path.join("modules", entry)
                results.append((rel, os.path.join(modules_dir, entry)))

    return results


@register("PL11")
def check_pl11(ctx: CheckContext) -> Finding:
    """golden 文件禁止 import pypto_pro 或 import pypto."""
    targets = _scan_golden_files(ctx)
    if not targets:
        return ctx.make_finding("PL11", "SKIP", "No golden files found")

    failures: list[str] = []
    for rel, abs_path in targets:
        tree, parse_error = load_python_ast(abs_path, rel)
        if tree is None:
            failures.append(f"{rel}: {parse_error}")
            continue

        forbidden = []
        for name in iter_imported_names(tree):
            if is_framework_import(name):
                forbidden.append(name)
        forbidden = sorted(set(forbidden))
        if forbidden:
            failures.append(
                f"{rel}: found forbidden import ({', '.join(forbidden)})"
            )

    if failures:
        return ctx.make_finding(
            "PL11", "FAIL", "\n".join(failures), file=targets[0][0]
        )
    return ctx.make_finding(
        "PL11", "PASS",
        f"All golden files ({len(targets)} total) passed the purity check"
    )


@register("PL13")
def check_pl13(ctx: CheckContext) -> Finding:
    """{op}_golden_stage*.py 禁止 import 任何 test_{op}_module*.py（golden 独立性）."""
    if ctx.file_scope:
        basename = os.path.basename(ctx.file_scope)
        if "_golden_stage" not in basename or not basename.endswith(".py"):
            return ctx.make_finding("PL13", "SKIP", "The current file is not a golden_stage file")
        stage_files = [(basename, ctx.file_scope)]
    else:
        stage_files = []

    modules_dir = os.path.join(ctx.op_dir, "modules")
    if not stage_files and not os.path.isdir(modules_dir):
        return ctx.make_finding(
            "PL13", "SKIP", "modules/ directory does not exist"
        )

    op_name = ctx.op_name
    if not stage_files:
        for entry in sorted(os.listdir(modules_dir)):
            if "_golden_stage" in entry and entry.endswith(".py"):
                rel = os.path.join("modules", entry)
                stage_files.append((rel, os.path.join(modules_dir, entry)))

    if not stage_files:
        return ctx.make_finding(
            "PL13", "SKIP", "No golden_stage files found"
        )

    failures: list[str] = []
    staged_module_prefix = f"test_{op_name}_module"
    for rel, abs_path in stage_files:
        tree, parse_error = load_python_ast(abs_path, rel)
        if tree is None:
            failures.append(f"{rel}: {parse_error}")
            continue

        staged_imports = []
        for name in iter_imported_names(tree):
            if has_module_component(name, staged_module_prefix):
                staged_imports.append(name)
        staged_imports = sorted(set(staged_imports))
        if staged_imports:
            failures.append(
                f"{rel}: found import of staged impl ({', '.join(staged_imports)})"
            )

    if failures:
        return ctx.make_finding(
            "PL13", "FAIL", "\n".join(failures), file=stage_files[0][0]
        )
    return ctx.make_finding(
        "PL13", "PASS",
        f"All golden_stage files ({len(stage_files)} total) passed the independence check"
    )
