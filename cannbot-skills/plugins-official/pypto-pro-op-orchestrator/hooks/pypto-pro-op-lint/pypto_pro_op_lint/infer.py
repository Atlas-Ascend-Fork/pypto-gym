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

import json
import os
import re
import select
import sys
from typing import Any, Optional

from .core import (
    CheckContext,
    HOOK_INPUT_ENV,
    LintConfigurationError,
    _load_rules,
)


def _infer_op_dir(file_path: str, op_name: Optional[str] = None) -> Optional[str]:
    """Resolve operator artifacts without consulting a workflow ledger."""
    if not file_path:
        return None
    directory = os.path.dirname(os.path.abspath(file_path))
    if op_name is None:
        op_name = _infer_op_name_from_filename(
            os.path.basename(file_path), staged=os.path.basename(directory) == "modules"
        )
    if os.path.basename(directory) == "modules":
        parent = os.path.dirname(directory)
        if op_name and _looks_like_stateless_op_dir(parent, op_name):
            return parent
    if op_name and _looks_like_stateless_op_dir(directory, op_name):
        return directory
    return None


def _infer_op_name_from_filename(filename: str, *, staged: bool = False) -> str:
    if staged:
        match = re.fullmatch(r"(?:test_(.+)_module|(.+)_golden_stage).*\.py", filename)
        if match:
            return match.group(1) or match.group(2)
    if filename.startswith("test_") and filename.endswith(".py"):
        return filename[len("test_"):-len(".py")]
    if filename.endswith("_golden.py"):
        return filename[:-len("_golden.py")]
    if filename.endswith("_golden_cpu.py"):
        return filename[:-len("_golden_cpu.py")]
    return ""


def _looks_like_stateless_op_dir(op_dir: str, op_name: str) -> bool:
    try:
        files = set(os.listdir(op_dir))
    except OSError:
        return False
    expected = {
        f"{op_name}_golden.py",
        f"{op_name}_golden_cpu.py",
        f"test_{op_name}.py",
        "SPEC.md",
        "DESIGN.md",
        "module_interfaces.yaml",
        "GOLDEN_PERF_REPORT.md",
    }
    return len(files & expected) >= 2


def _get_op_name(op_dir: str) -> str:
    filenames = os.listdir(op_dir)
    names = {_infer_op_name_from_filename(filename) for filename in filenames} - {""}
    directory_name = os.path.basename(op_dir)
    directory_goldens = {f"{directory_name}_golden.py", f"{directory_name}_golden_cpu.py"}
    if directory_name in names or not names or directory_goldens.intersection(filenames):
        return directory_name
    modules_dir = os.path.join(op_dir, "modules")
    if os.path.isdir(modules_dir) and any(
        filename.startswith((f"test_{directory_name}_module", f"{directory_name}_golden_stage"))
        and filename.endswith(".py") for filename in os.listdir(modules_dir)
    ):
        return directory_name
    paired_names = {
        name for name in names if f"test_{name}.py" in filenames
        and any(f"{name}{suffix}" in filenames for suffix in ("_golden.py", "_golden_cpu.py"))
    }
    if paired_names:
        names = paired_names
    else:
        golden_names = {
            _infer_op_name_from_filename(filename) for filename in filenames
            if filename.endswith(("_golden.py", "_golden_cpu.py"))
        }
        if len(golden_names) == 1:
            return golden_names.pop()
    if len(names) == 1:
        return names.pop()
    raise LintConfigurationError(f"Ambiguous operator names in {op_dir}: {', '.join(sorted(names))}")


def _build_context(op_dir: str, stage: Optional[int] = None, *, op_name: Optional[str] = None) -> CheckContext:
    rules = _load_rules()
    op_dir = os.path.abspath(op_dir)
    return CheckContext(op_dir=op_dir, op_name=_get_op_name(op_dir) if op_name is None else op_name,
                        stage=4 if stage is None else stage, rules=rules)


def _load_hook_input() -> dict[str, Any]:
    raw = ""
    if not sys.stdin.closed:
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                raw = sys.stdin.read()
        except (ValueError, OSError):
            pass
    if not raw.strip():
        raw = os.environ.get(HOOK_INPUT_ENV, "")
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _is_test_file(filename: str) -> bool:
    """Match test_{op}.py and modules/test_{op}_module*.py."""
    basename = os.path.basename(filename)
    if basename.startswith("test_") and basename.endswith(".py"):
        return True
    return False


def _is_golden_file(filename: str) -> bool:
    """Match {op}_golden.py, {op}_golden_cpu.py, {op}_golden_stage*.py."""
    basename = os.path.basename(filename)
    if basename.endswith("_golden.py"):
        return True
    if basename.endswith("_golden_cpu.py"):
        return True
    if "_golden_stage" in basename and basename.endswith(".py"):
        return True
    return False


def _rule_ids_for_filename(filename: str) -> list[str]:
    """Return rule IDs for post-edit checks on a single file.

    Only includes rules that validate the edited file itself. Cross-file
    rules remain available through explicit standalone checks.
    """
    from .core import POST_EDIT_GOLDEN_RULES, POST_EDIT_TEST_RULES

    if _is_test_file(filename):
        return POST_EDIT_TEST_RULES
    if _is_golden_file(filename):
        return POST_EDIT_GOLDEN_RULES
    return []
