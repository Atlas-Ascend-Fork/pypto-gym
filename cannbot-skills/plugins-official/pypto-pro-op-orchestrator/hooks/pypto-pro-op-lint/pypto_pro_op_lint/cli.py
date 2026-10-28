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

import argparse

from . import checks  # noqa: F401
from .core import (
    Finding,
    _has_error_fail,
    _run_checks,
)
from .hooks import hook_post_edit
from .infer import _build_context
from .observability import _print_findings


def _cmd_run(findings: list[Finding]) -> int:
    _print_findings(findings)
    return 2 if _has_error_fail(findings) else 0


def cmd_lint_test(op_dir: str, stage: int) -> int:
    from .core import TEST_RULE_IDS
    ctx = _build_context(op_dir, stage)
    findings, _ = _run_checks(ctx, TEST_RULE_IDS)
    return _cmd_run(findings)


def cmd_lint_golden(op_dir: str, stage: int) -> int:
    from .core import GOLDEN_RULE_IDS
    ctx = _build_context(op_dir, stage)
    findings, _ = _run_checks(ctx, GOLDEN_RULE_IDS)
    return _cmd_run(findings)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyPTO-Pro standalone static checks")
    parser.add_argument("--hook", choices=["post-edit"], help="Explicit post-edit check")
    parser.add_argument("--lint-test", action="store_true", help="Check test artifacts")
    parser.add_argument("--lint-golden", action="store_true", help="Check CPU Golden artifacts")
    parser.add_argument("--op-dir", help="Operator artifact directory")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3, 4), default=4,
                        help="Optional static rule context; never reads or advances workflow state")
    return parser


def _require_op_dir(parser: argparse.ArgumentParser, args: argparse.Namespace, option: str) -> str:
    if not args.op_dir:
        parser.error(f"{option} requires --op-dir")
    return args.op_dir


def _dispatch_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.hook:
        return hook_post_edit()
    if args.lint_test:
        return cmd_lint_test(_require_op_dir(parser, args, "--lint-test"), args.stage)
    if args.lint_golden:
        return cmd_lint_golden(_require_op_dir(parser, args, "--lint-golden"), args.stage)
    parser.print_help()
    return 0


def main() -> int:
    parser = _build_parser()
    return _dispatch_command(parser, parser.parse_args())
