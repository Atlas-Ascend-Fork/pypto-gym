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
"""D2 工件完整性: PL03-PL09."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from ..core import (
    CheckContext,
    DESIGN_FILE,
    Finding,
    MODULE_INTERFACES_FILE,
    SPEC_FILE,
    register,
)


_SPEC_VALIDATOR_RELATIVE_PATHS = (
    Path("skills/pypto-pro-intent-understand/scripts/validate_spec.py"),
    Path("ops/pypto-pro-intent-understand/scripts/validate_spec.py"),
)


def _load_spec_contract_text(content: str):
    """Load the canonical SPEC parser from an installed or source layout."""
    validator_path = None
    for ancestor in Path(__file__).resolve().parents:
        for relative in _SPEC_VALIDATOR_RELATIVE_PATHS:
            candidate = ancestor / relative
            if candidate.is_file():
                validator_path = candidate
                break
        if validator_path is not None:
            break
    if validator_path is None:
        raise RuntimeError("找不到 pypto-pro-intent-understand 的 canonical SPEC validator")

    spec = importlib.util.spec_from_file_location(
        "_pypto_pro_canonical_spec_validator", validator_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 canonical SPEC validator: {validator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = getattr(module, "load_spec_contract_text", None)
    if not callable(loader):
        raise RuntimeError("canonical SPEC validator 缺少 load_spec_contract_text()")
    return loader(content)


def _check_file_exists(ctx: CheckContext, rule_id: str, filename: str) -> Finding:
    if ctx.file_exists(filename):
        return ctx.make_finding(
            rule_id, "PASS", f"{filename} exists", file=filename
        )
    return ctx.make_finding(
        rule_id, "FAIL", f"{filename} does not exist", file=filename
    )


@register("PL03")
def check_pl03(ctx: CheckContext) -> Finding:
    """SPEC.md satisfies the canonical JSON contract."""
    if not ctx.file_exists(SPEC_FILE):
        return ctx.make_finding("PL03", "FAIL", f"{SPEC_FILE} does not exist", file=SPEC_FILE)
    content = ctx.read_file(SPEC_FILE)
    if not content.strip():
        return ctx.make_finding("PL03", "FAIL", f"{SPEC_FILE} is empty", file=SPEC_FILE)
    try:
        contract = _load_spec_contract_text(content)
    except Exception as exc:
        return ctx.make_finding(
            "PL03", "FAIL", f"SPEC.md JSON machine-contract invalid: {exc}", file=SPEC_FILE
        )
    if contract.get("op_name") != ctx.op_name:
        return ctx.make_finding(
            "PL03", "FAIL",
            f"SPEC.md op_name={contract.get('op_name')!r}, expected {ctx.op_name!r}",
            file=SPEC_FILE,
        )
    return ctx.make_finding(
        "PL03", "PASS", "SPEC.md JSON machine-contract passed the canonical validator", file=SPEC_FILE
    )


@register("PL04")
def check_pl04(ctx: CheckContext) -> Finding:
    """{op}_golden.py 存在."""
    filename = f"{ctx.op_name}_golden.py"
    return _check_file_exists(ctx, "PL04", filename)


@register("PL05")
def check_pl05(ctx: CheckContext) -> Finding:
    """{op}_golden_cpu.py 存在."""
    filename = f"{ctx.op_name}_golden_cpu.py"
    return _check_file_exists(ctx, "PL05", filename)


@register("PL07")
def check_pl07(ctx: CheckContext) -> Finding:
    """DESIGN.md 存在."""
    return _check_file_exists(ctx, "PL07", DESIGN_FILE)


@register("PL08")
def check_pl08(ctx: CheckContext) -> Finding:
    """module_interfaces.yaml 存在."""
    return _check_file_exists(ctx, "PL08", MODULE_INTERFACES_FILE)


@register("PL09")
def check_pl09(ctx: CheckContext) -> Finding:
    """test_{op}.py 存在."""
    filename = f"test_{ctx.op_name}.py"
    return _check_file_exists(ctx, "PL09", filename)
