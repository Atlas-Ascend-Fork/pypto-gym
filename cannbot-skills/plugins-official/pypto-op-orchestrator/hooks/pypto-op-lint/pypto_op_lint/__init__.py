#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

from . import checks  # noqa: F401
from .cli import (
    cmd_check_gate,
    cmd_lint_consistency,
    cmd_lint_golden,
    cmd_lint_impl,
    cmd_lint_test,
    main,
)
from .core import (
    API_REPORT_FILE,
    CONSISTENCY_RULE_IDS,
    DESIGN_FILE,
    GOLDEN_RULE_IDS,
    IMPL_RULE_IDS,
    MODE_ENV,
    POST_EDIT_BLOCK_ENV,
    POST_EDIT_CONSISTENCY_RULE_IDS,
    SPEC_FILE,
    STRICT_ENV,
    TEST_RULE_IDS,
    CheckContext,
    Finding,
    _has_error_fail,
    _run_checks,
    register,
)
from .hooks import hook_post_bash, hook_post_edit, hook_pre_edit_backup, hook_stop
from .infer import _build_context
from .observability import LOGS_DIR, LOGS_EVENTS_FILE
from .utils import (
    _parse_front_matter,
    _validate_doc_schema,
    parse_front_matter,
    validate_doc_schema,
)

__all__ = [
    "API_REPORT_FILE",
    "CONSISTENCY_RULE_IDS",
    "CheckContext",
    "DESIGN_FILE",
    "Finding",
    "GOLDEN_RULE_IDS",
    "IMPL_RULE_IDS",
    "LOGS_DIR",
    "LOGS_EVENTS_FILE",
    "MODE_ENV",
    "POST_EDIT_BLOCK_ENV",
    "POST_EDIT_CONSISTENCY_RULE_IDS",
    "SPEC_FILE",
    "STRICT_ENV",
    "TEST_RULE_IDS",
    "cmd_check_gate",
    "cmd_lint_consistency",
    "cmd_lint_golden",
    "cmd_lint_impl",
    "cmd_lint_test",
    "hook_post_bash",
    "hook_post_edit",
    "hook_pre_edit_backup",
    "hook_stop",
    "main",
    "parse_front_matter",
    "register",
    "validate_doc_schema",
    "_build_context",
    "_has_error_fail",
    "_parse_front_matter",
    "_run_checks",
    "_validate_doc_schema",
]
