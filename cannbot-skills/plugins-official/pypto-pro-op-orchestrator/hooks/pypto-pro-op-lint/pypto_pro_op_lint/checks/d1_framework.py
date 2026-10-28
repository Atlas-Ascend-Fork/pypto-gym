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
"""D1 框架合规: PL01 (import 门禁) + PL02 (单 kernel 铁律)."""

from __future__ import annotations

import os

from ..ast_helpers import (
    count_decorators,
    has_import_alias,
    imports_module,
    load_python_ast,
)
from ..core import CheckContext, Finding, register


def _scan_test_files(ctx: CheckContext) -> list[tuple[str, str]]:
    """List (relative_filename, abs_path) of test files to scan.

    In post-edit mode (file_scope set), only the edited file is returned.
    With module_scope set, only that module's staged file.
    Otherwise, top-level test_{op}.py + all staged files.
    """
    results: list[tuple[str, str]] = []
    op_name = ctx.op_name

    if ctx.file_scope:
        basename = os.path.basename(ctx.file_scope)
        results.append((basename, ctx.file_scope))
        return results

    if ctx.module_scope:
        staged = os.path.join("modules", f"test_{op_name}_module{ctx.module_scope}.py")
        abs_path = ctx.file_path(staged)
        if os.path.isfile(abs_path):
            results.append((staged, abs_path))
        return results

    top_test = f"test_{op_name}.py"
    if ctx.file_exists(top_test):
        results.append((top_test, ctx.file_path(top_test)))

    modules_dir = os.path.join(ctx.op_dir, "modules")
    if os.path.isdir(modules_dir):
        for entry in sorted(os.listdir(modules_dir)):
            if entry.startswith(f"test_{op_name}_module") and entry.endswith(".py"):
                rel = os.path.join("modules", entry)
                results.append((rel, os.path.join(modules_dir, entry)))

    return results


@register("PL01")
def check_pl01(ctx: CheckContext) -> Finding:
    """PL01: import 门禁 — 必须 import pypto_pro.language as pl，禁止 classic API."""
    targets = _scan_test_files(ctx)
    if not targets:
        return ctx.make_finding("PL01", "SKIP", "No test/staged files found")

    failures: list[str] = []
    for rel, abs_path in targets:
        tree, parse_error = load_python_ast(abs_path, rel)
        if tree is None:
            failures.append(f"{rel}: {parse_error}")
            continue

        has_pro_import = has_import_alias(tree, "pypto_pro.language", "pl")
        if not has_pro_import:
            failures.append(
                f"{rel}: `import pypto_pro.language as pl` not found"
            )

        has_classic_jit = count_decorators(tree, "pypto.frontend.jit") > 0
        if has_classic_jit:
            failures.append(
                f"{rel}: found `@pypto.frontend.jit` (classic API forbidden)"
            )

        if imports_module(tree, "pypto"):
            failures.append(
                f"{rel}: found non-Pro pypto import (classic API forbidden)"
            )

    if failures:
        return ctx.make_finding(
            "PL01", "FAIL", "\n".join(failures), file=targets[0][0]
        )

    return ctx.make_finding(
        "PL01", "PASS",
        f"All test/staged files ({len(targets)} total) passed the import gate"
    )


@register("PL02")
def check_pl02(ctx: CheckContext) -> Finding:
    """test/staged 文件中 @pl.jit 必须且仅能出现 1 次（单 kernel 铁律）."""
    targets = _scan_test_files(ctx)
    if not targets:
        return ctx.make_finding("PL02", "SKIP", "No test/staged files found")

    failures: list[str] = []
    for rel, abs_path in targets:
        tree, parse_error = load_python_ast(abs_path, rel)
        if tree is None:
            failures.append(f"{rel}: {parse_error}")
            continue

        jit_count = count_decorators(tree, "pl.jit")
        if jit_count == 0:
            failures.append(f"{rel}: no kernel function decorated with @pl.jit found")
        elif jit_count > 1:
            failures.append(
                f"{rel}: found {jit_count} @pl.jit(s) (single-kernel rule: only 1 allowed)"
            )

    if failures:
        return ctx.make_finding(
            "PL02", "FAIL", "\n".join(failures), file=targets[0][0]
        )

    return ctx.make_finding(
        "PL02", "PASS",
        f"All test/staged files ({len(targets)} total) passed the single-kernel rule"
    )
