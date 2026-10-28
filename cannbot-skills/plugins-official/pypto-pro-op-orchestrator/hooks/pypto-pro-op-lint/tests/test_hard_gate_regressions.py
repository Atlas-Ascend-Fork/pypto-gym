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

"""Regression tests for standalone AST checks and invalid configurations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

LINT_ROOT = Path(__file__).resolve().parents[1]
if str(LINT_ROOT) not in sys.path:
    sys.path.insert(0, str(LINT_ROOT))

from pypto_pro_op_lint import core  # noqa: E402
from pypto_pro_op_lint import observability  # noqa: E402
from pypto_pro_op_lint.checks import d2_artifact  # noqa: E402
from pypto_pro_op_lint.cli import cmd_lint_test, cmd_lint_golden, _build_parser, main  # noqa: E402
from pypto_pro_op_lint.core import (  # noqa: E402
    HOOK_INPUT_ENV,
    CheckContext,
    _load_rules,
    _run_checks,
)
from pypto_pro_op_lint.hooks import hook_post_edit  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_observability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_dir = tmp_path / "lint-state"
    monkeypatch.setattr(observability, "LOGS_DIR", str(state_dir))
    monkeypatch.setattr(
        observability, "LOGS_EVENTS_FILE", str(state_dir / "lint_events.jsonl")
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _context(op_dir: Path, stage: int = 4) -> CheckContext:
    return CheckContext(
        op_dir=str(op_dir),
        op_name="demo",
        stage=stage,
        rules=_load_rules(),
    )


def _run(op_dir: Path, rule_id: str, stage: int = 4):
    findings, _ = _run_checks(_context(op_dir, stage), [rule_id])
    return findings[0]


def _python_comment(content: str) -> str:
    return f"# {content}"


def _valid_spec(op_name: str = "demo") -> str:
    return f'''```json machine-contract
{{
  "schema_version": 1,
  "op_name": "{op_name}",
  "formula": "y = x",
  "supported_dtypes": ["float32"],
  "inputs": [{{"name": "x", "shape": [8, 16], "dtype": "float32", "value_range": [-4, 4]}}],
  "outputs": [{{"name": "y", "shape": [8, 16], "dtype": "float32", "value_range": [-4, 4]}}],
  "default_params": {{}},
  "tolerance": {{"atol": 0.001, "rtol": 0.001}},
  "dynamic_axes_ranges": {{}},
  "shape_constraints": [],
  "p0_cases": [{{"name": "p0", "params": {{}}, "input_shapes": {{"x": [8, 16]}}, "output_shapes": {{"y": [8, 16]}}}}]
}}
```

## 语义说明
y = x
'''


def _build_valid_operator(op_dir: Path) -> None:
    _write(op_dir / "SPEC.md", _valid_spec())
    _write(op_dir / "DESIGN.md", "# DESIGN\n")
    _write(op_dir / "module_interfaces.yaml", "is_fusion: false\n")
    _write(op_dir / "demo_golden.py", "import torch\n\ndef demo_golden(x): return x\n")
    _write(
        op_dir / "demo_golden_cpu.py",
        "import torch\n\ndef demo_golden_cpu(x): return x.float()\n",
    )
    _write(
        op_dir / "test_demo.py",
        """import pypto_pro.language as pl
from demo_golden_cpu import demo_golden_cpu

@pl.jit(auto_mutex=True)
def demo_kernel():
    pass

def _assert_precision(actual, expected):
    return actual == expected

def test_one(): pass
def test_two(): pass
def test_three(): pass
def test_four(): pass
""",
    )


def test_pl03_accepts_canonical_contract_from_source_layout(tmp_path: Path) -> None:
    _write(tmp_path / "SPEC.md", _valid_spec())
    assert _run(tmp_path, "PL03", stage=1).status == "PASS"


def test_pl03_rejects_invalid_contract_and_operator_mismatch(tmp_path: Path) -> None:
    _write(tmp_path / "SPEC.md", _valid_spec().replace("demo", "{{OP_NAME}}"))
    assert _run(tmp_path, "PL03", stage=1).status == "FAIL"

    _write(tmp_path / "SPEC.md", _valid_spec("another_op"))
    finding = _run(tmp_path, "PL03", stage=1)
    assert finding.status == "FAIL"
    assert "op_name" in finding.message


def test_pl03_loads_canonical_contract_from_installed_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_root = tmp_path / "config"
    installed_validator = (
        config_root / "skills/pypto-pro-intent-understand/scripts/validate_spec.py"
    )
    source_validator = (
        Path(__file__).resolve().parents[5]
        / "ops/pypto-pro-intent-understand/scripts/validate_spec.py"
    )
    _write(installed_validator, source_validator.read_text(encoding="utf-8"))
    fake_check = (
        config_root
        / "hooks/pypto-pro-op-lint/pypto_pro_op_lint/checks/d2_artifact.py"
    )
    monkeypatch.setattr(d2_artifact, "__file__", str(fake_check))
    _write(tmp_path / "SPEC.md", _valid_spec())
    assert _run(tmp_path, "PL03", stage=1).status == "PASS"


def test_pl03_fails_closed_when_canonical_validator_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        d2_artifact,
        "__file__",
        str(tmp_path / "isolated/hooks/lint/package/checks/d2_artifact.py"),
    )
    _write(tmp_path / "SPEC.md", _valid_spec())
    finding = _run(tmp_path, "PL03", stage=1)
    assert finding.status == "FAIL"
    assert "canonical SPEC validator" in finding.message


def test_pl01_rejects_from_import_of_classic_pypto(tmp_path: Path) -> None:
    _write(
        tmp_path / "test_demo.py",
        """import pypto_pro.language as pl
from pypto import frontend

@pl.jit
def kernel():
    pass
""",
    )
    assert _run(tmp_path, "PL01").status == "FAIL"


@pytest.mark.parametrize(
    "source",
    [
        "import importlib\nimportlib.import_module('pypto.frontend')\n",
        "import importlib as il\nil.import_module('pypto')\n",
        "from importlib import import_module as load\nload('pypto')\n",
        "__import__('pypto.frontend')\n",
    ],
)
def test_pl01_rejects_literal_dynamic_classic_imports(
    tmp_path: Path, source: str,
) -> None:
    _write(
        tmp_path / "test_demo.py",
        "import pypto_pro.language as pl\n@pl.jit\ndef kernel(): pass\n" + source,
    )
    assert _run(tmp_path, "PL01").status == "FAIL"


def test_pl02_ignores_decorator_text_in_comments_and_strings(tmp_path: Path) -> None:
    _write(
        tmp_path / "test_demo.py",
        '''import pypto_pro.language as pl
TEXT = "@pl.jit"
# @pl.jit

@pl.jit(auto_mutex=True)
def kernel():
    pass
''',
    )
    assert _run(tmp_path, "PL02").status == "PASS"


def test_pl02_fails_closed_on_invalid_python(tmp_path: Path) -> None:
    _write(tmp_path / "test_demo.py", "@pl.jit\ndef kernel(:\n    pass\n")
    finding = _run(tmp_path, "PL02")
    assert finding.status == "FAIL"
    assert "syntax error" in finding.message


@pytest.mark.parametrize(
    "statement",
    ["from pypto_pro import language", "from pypto import frontend"],
)
def test_pl11_rejects_from_imports(tmp_path: Path, statement: str) -> None:
    _write(tmp_path / "demo_golden.py", f"{statement}\n")
    assert _run(tmp_path, "PL11").status == "FAIL"


@pytest.mark.parametrize(
    "statement",
    [
        "import importlib\nimportlib.import_module('pypto_pro.language')",
        "__import__('pypto')",
    ],
)
def test_pl11_rejects_literal_dynamic_imports(
    tmp_path: Path, statement: str,
) -> None:
    _write(tmp_path / "demo_golden.py", f"{statement}\n")
    assert _run(tmp_path, "PL11").status == "FAIL"


@pytest.mark.parametrize(
    "statement",
    [
        "from modules.test_demo_module1 import kernel",
        "from modules import test_demo_module1",
        "import modules.test_demo_module1",
    ],
)
def test_pl13_rejects_qualified_staged_imports(tmp_path: Path, statement: str) -> None:
    _write(tmp_path / "modules" / "demo_golden_stage1.py", f"{statement}\n")
    assert _run(tmp_path, "PL13").status == "FAIL"


def test_pl13_rejects_literal_dynamic_staged_import(tmp_path: Path) -> None:
    _write(
        tmp_path / "modules" / "demo_golden_stage1.py",
        "import importlib\nimportlib.import_module('modules.test_demo_module1')\n",
    )
    assert _run(tmp_path, "PL13").status == "FAIL"


def test_pl13_post_edit_only_checks_current_golden_stage(tmp_path: Path) -> None:
    clean = tmp_path / "modules" / "demo_golden_stage1.py"
    polluted = tmp_path / "modules" / "demo_golden_stage12.py"
    _write(clean, "import torch\n")
    _write(polluted, "from modules.test_demo_module12 import kernel\n")
    ctx = _context(tmp_path)
    ctx.file_scope = str(clean)
    findings, _ = _run_checks(ctx, ["PL13"])
    assert findings[0].status == "PASS"


def test_static_golden_scope_excludes_modules_outside_context4(tmp_path: Path) -> None:
    _write(tmp_path / "demo_golden.py", "import torch\n")
    _write(tmp_path / "demo_golden_cpu.py", "import torch\n")
    _write(
        tmp_path / "modules" / "demo_golden_stage1.py",
        "import pypto_pro\n",
    )
    assert _run(tmp_path, "PL11", stage=2).status == "PASS"
    assert _run(tmp_path, "PL11", stage=4).status == "FAIL"


def test_pl15_ignores_assert_close_in_comments_and_strings(tmp_path: Path) -> None:
    source = "\n".join([
        'TEXT = "assert_close"',
        _python_comment("assert_close(x, y)"),
        "def _assert_precision(actual, expected):",
        "    return actual == expected",
    ])
    _write(tmp_path / "test_demo.py", source)
    assert _run(tmp_path, "PL15").status == "PASS"


def test_pl15_does_not_accept_assert_precision_string(tmp_path: Path) -> None:
    _write(tmp_path / "test_demo.py", 'TEXT = "_assert_precision"\n')
    assert _run(tmp_path, "PL15").status == "FAIL"


def test_pl15_does_not_accept_assert_precision_variable(tmp_path: Path) -> None:
    _write(tmp_path / "test_demo.py", "_assert_precision = object()\n")
    assert _run(tmp_path, "PL15").status == "FAIL"


def test_missing_checker_is_s0_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _context(tmp_path)
    monkeypatch.delitem(core.CHECKERS, "PL01")
    findings, _ = _run_checks(ctx, ["PL01"])
    finding = findings[0]
    assert finding.status == "FAIL"
    assert finding.severity == "S0"


def test_checker_exception_is_s0_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _context(tmp_path)

    def broken_checker(_ctx):
        raise RuntimeError("simulated checker failure")

    monkeypatch.setitem(core.CHECKERS, "PL01", broken_checker)
    findings, _ = _run_checks(ctx, ["PL01"])
    assert findings[0].status == "FAIL"
    assert findings[0].severity == "S0"
    assert "simulated checker failure" in findings[0].message


def test_rules_checker_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = _load_rules()
    _write(tmp_path / "rules.json", json.dumps({"rules": rules}))
    monkeypatch.setattr(core, "SCRIPT_DIR", str(tmp_path))
    monkeypatch.delitem(core.CHECKERS, "PL01")
    with pytest.raises(core.LintConfigurationError, match="Missing checkers for rules"):
        _load_rules()


def test_static_group_with_undefined_rule_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "TEST_RULE_IDS", ["MISSING"])
    with pytest.raises(core.LintConfigurationError, match="静态检查组引用了未定义规则"):
        _load_rules()


@pytest.mark.parametrize("stages", [[], [0], [True], [5]])
def test_invalid_static_context_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stages: list,
) -> None:
    rules = _load_rules()
    rules[0]["stages"] = stages
    _write(tmp_path / "rules.json", json.dumps({"rules": rules}))
    monkeypatch.setattr(core, "SCRIPT_DIR", str(tmp_path))
    with pytest.raises(core.LintConfigurationError, match="非法 stages"):
        _load_rules()


@pytest.mark.parametrize("rule_id", ["PL01", "PL11"])
def test_static_group_cannot_skip_default_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rule_id: str,
) -> None:
    rules = _load_rules()
    for rule in rules:
        if rule["id"] == rule_id:
            rule["stages"] = [1]
    _write(tmp_path / "rules.json", json.dumps({"rules": rules}))
    monkeypatch.setattr(core, "SCRIPT_DIR", str(tmp_path))
    with pytest.raises(core.LintConfigurationError, match="不适用于默认静态范围"):
        _load_rules()


def test_empty_triggers_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = _load_rules()
    for rule in rules:
        if rule["id"] == "PL03":
            rule["triggers"] = []
    _write(tmp_path / "rules.json", json.dumps({"rules": rules}))
    monkeypatch.setattr(core, "SCRIPT_DIR", str(tmp_path))
    with pytest.raises(core.LintConfigurationError, match="缺少合法 triggers"):
        _load_rules()


@pytest.mark.parametrize(
    ("rule_id", "trigger"),
    [("PL01", "lint-test"), ("PL11", "lint-golden"), ("PL02", "post-edit")],
)
def test_missing_static_group_trigger_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    rule_id: str, trigger: str,
) -> None:
    rules = _load_rules()
    for rule in rules:
        if rule["id"] == rule_id:
            rule["triggers"].remove(trigger)
    _write(tmp_path / "rules.json", json.dumps({"rules": rules}))
    monkeypatch.setattr(core, "SCRIPT_DIR", str(tmp_path))
    with pytest.raises(core.LintConfigurationError, match=f"缺少 {trigger} trigger"):
        _load_rules()


def test_unmapped_declared_static_trigger_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = _load_rules()
    for rule in rules:
        if rule["id"] == "PL03":
            rule["triggers"].append("lint-test")
    _write(tmp_path / "rules.json", json.dumps({"rules": rules}))
    monkeypatch.setattr(core, "SCRIPT_DIR", str(tmp_path))
    with pytest.raises(core.LintConfigurationError, match="未进入对应静态检查组"):
        _load_rules()


@pytest.mark.parametrize("trigger", ["gate:S2", "gate:M", "unknown"])
def test_unknown_trigger_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trigger: str,
) -> None:
    rules = _load_rules()
    rules[0]["triggers"].append(trigger)
    _write(tmp_path / "rules.json", json.dumps({"rules": rules}))
    monkeypatch.setattr(core, "SCRIPT_DIR", str(tmp_path))
    with pytest.raises(core.LintConfigurationError, match="未知 trigger"):
        _load_rules()


def test_missing_rule_definition_is_s0_failure(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    ctx.rules = [rule for rule in ctx.rules if rule["id"] != "PL01"]
    findings, _ = _run_checks(ctx, ["PL01"])
    assert findings[0].status == "FAIL"
    assert findings[0].severity == "S0"


def test_rules_load_failure_raises_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "SCRIPT_DIR", str(tmp_path))
    with pytest.raises(core.LintConfigurationError):
        _load_rules()


def test_empty_rules_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path / "rules.json", json.dumps({"rules": []}))
    monkeypatch.setattr(core, "SCRIPT_DIR", str(tmp_path))
    with pytest.raises(core.LintConfigurationError):
        _load_rules()


def test_empty_check_selection_is_s0_failure(tmp_path: Path) -> None:
    findings, _ = _run_checks(_context(tmp_path), [])
    assert len(findings) == 1
    assert findings[0].status == "FAIL"
    assert findings[0].severity == "S0"


def test_removed_rules_are_not_registered_or_referenced() -> None:
    removed = {"PL06", "PL10", "PL12", "PL14", "PL17"}
    rule_ids = {rule["id"] for rule in _load_rules()}
    referenced = set(
        core.TEST_RULE_IDS
        + core.GOLDEN_RULE_IDS
        + core.POST_EDIT_TEST_RULES
        + core.POST_EDIT_GOLDEN_RULES
    )
    assert removed.isdisjoint(rule_ids)
    assert removed.isdisjoint(core.CHECKERS)
    assert removed.isdisjoint(referenced)


def test_all_rules_have_registered_checkers_and_static_group_triggers() -> None:
    rules = _load_rules()
    rule_by_id = {rule["id"]: rule for rule in rules}
    assert set(rule_by_id) == set(core.CHECKERS)
    groups = {
        "lint-test": core.TEST_RULE_IDS,
        "lint-golden": core.GOLDEN_RULE_IDS,
        "post-edit": core.POST_EDIT_TEST_RULES + core.POST_EDIT_GOLDEN_RULES,
    }
    for trigger, rule_ids in groups.items():
        for rule_id in rule_ids:
            assert trigger in rule_by_id[rule_id]["triggers"]


def _post_edit_output(
    tmp_path: Path,
    source: str,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> dict:
    _write(tmp_path / "SPEC.md", _valid_spec())
    test_file = tmp_path / "test_demo.py"
    _write(test_file, source)
    payload = {"tool_input": {"file_path": str(test_file)}}
    monkeypatch.setenv(HOOK_INPUT_ENV, json.dumps(payload))
    assert hook_post_edit() == 0
    return json.loads(capfd.readouterr().out)


@pytest.mark.parametrize(
    ("source", "expected_decision"),
    [
        pytest.param(
            """import pypto_pro.language as pl
from pypto import frontend
@pl.jit
def kernel(): pass
""",
            "block",
            id="ast-import-violation",
        ),
        pytest.param(
            """import pypto_pro.language as pl
@pl.jit
def kernel(): pass
""",
            "allow",
            id="valid-file",
        ),
    ],
)
def test_post_edit_hook_returns_expected_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    source: str,
    expected_decision: str,
) -> None:
    output = _post_edit_output(tmp_path, source, monkeypatch, capfd)
    assert output["hookSpecificOutput"]["decision"] == expected_decision
    assert "test_demo.py test_demo.py:" not in output["hookSpecificOutput"]["reason"]


def test_standalone_checks_do_not_need_or_consult_workflow_state(tmp_path: Path) -> None:
    from pypto_pro_op_lint.infer import _build_context
    _build_valid_operator(tmp_path)
    assert cmd_lint_test(str(tmp_path), 4) == 0
    assert cmd_lint_golden(str(tmp_path), 4) == 0
    state = tmp_path / ".orchestrator_state.json"
    original = '{"operator_name":"wrong","current_stage":99}'
    state.write_text(original, encoding="utf-8")
    context = _build_context(str(tmp_path))
    assert context.op_name == "demo"
    assert context.stage == 4
    assert cmd_lint_test(str(tmp_path), 4) == 0
    assert cmd_lint_golden(str(tmp_path), 4) == 0
    assert state.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("command", "filename", "source"),
    [
        (cmd_lint_test, "test_demo.py", "import pypto\n"),
        (cmd_lint_golden, "demo_golden.py", "import pypto_pro\n"),
    ],
)
def test_standalone_cli_reports_actual_static_failure(
    tmp_path: Path, command, filename: str, source: str,
    capfd: pytest.CaptureFixture[str],
) -> None:
    _build_valid_operator(tmp_path)
    _write(tmp_path / filename, source)
    assert command(str(tmp_path), 4) == 2
    result = json.loads(capfd.readouterr().out)
    assert result["summary"]["has_error_fail"] is True
    assert any(finding["status"] == "FAIL" for finding in result["findings"])


def test_cli_has_no_workflow_transition_or_stop_gate_entry() -> None:
    parser = _build_parser()
    options = parser._option_string_actions
    assert "--check-gate" not in options
    assert "--check-module-gate" not in options
    assert options["--hook"].choices == ["post-edit"]


@pytest.mark.parametrize("directory", ["where", "output"])
@pytest.mark.parametrize("command", [cmd_lint_test, cmd_lint_golden])
def test_ancillary_test_does_not_redirect_operator_checks(tmp_path: Path, directory, command, capfd) -> None:
    op_dir = tmp_path / directory
    _write(op_dir / "test_where.py", "import pypto\n")
    _write(op_dir / "where_golden_cpu.py", "import pypto_pro\n")
    _write(op_dir / "test_accuracy.py", "def test_accuracy(): pass\n")
    assert command(str(op_dir), 4) == 2
    result = json.loads(capfd.readouterr().out)
    assert any(item["status"] == "FAIL" and "where" in (item["file"] or "")
               for item in result["findings"])


def test_ambiguous_operator_checks_do_not_silently_pass(tmp_path: Path) -> None:
    _write(tmp_path / "test_where.py", "import pypto\n")
    _write(tmp_path / "test_accuracy.py", "def test_accuracy(): pass\n")
    with pytest.raises(core.LintConfigurationError, match="Ambiguous operator names.*accuracy, where"):
        cmd_lint_test(str(tmp_path), 4)


@pytest.mark.parametrize("op_name", ["test_shift", "foo_golden_cpu"])
@pytest.mark.parametrize("suffix", ["_golden.py", "_golden_cpu.py"])
def test_canonical_golden_check_does_not_require_implementation_test(tmp_path: Path, op_name, suffix, capfd) -> None:
    op_dir = tmp_path / op_name
    filename = f"{op_name}{suffix}"
    _write(op_dir / filename, "import pypto_pro\n")
    assert cmd_lint_golden(str(op_dir), 4) == 2
    result = json.loads(capfd.readouterr().out)
    assert any(item["rule_id"] == "PL11" and item["status"] == "FAIL" and item["file"] == filename
               for item in result["findings"])


def test_arbitrary_output_keeps_golden_like_operator_test_identity(tmp_path: Path, capfd) -> None:
    _write(tmp_path / "test_foo_golden_cpu.py", "import pypto\n")
    assert cmd_lint_test(str(tmp_path), 4) == 2
    result = json.loads(capfd.readouterr().out)
    assert any(item["rule_id"] == "PL01" and item["status"] == "FAIL"
               for item in result["findings"])


@pytest.mark.parametrize("suffix", ["12", "X", ""])
@pytest.mark.parametrize("golden", [False, True])
def test_staged_only_operator_is_not_redirected_by_ancillary_test(tmp_path: Path, suffix, golden, capfd) -> None:
    op_dir = tmp_path / "where"
    filename = f"where_golden_stage{suffix}.py" if golden else f"test_where_module{suffix}.py"
    source = f"from modules.test_where_module{suffix} import kernel\n" if golden else "import pypto\n"
    _write(op_dir / "modules" / filename, source)
    _write(op_dir / "test_accuracy.py", "def test_accuracy(): pass\n")
    command, rule = (cmd_lint_golden, "PL13") if golden else (cmd_lint_test, "PL01")
    assert command(str(op_dir), 4) == 2
    result = json.loads(capfd.readouterr().out)
    assert any(item["rule_id"] == rule and item["status"] == "FAIL" and filename in item["file"]
               for item in result["findings"])


@pytest.mark.parametrize("op_name", ["test_shift", "foo_golden_cpu"])
@pytest.mark.parametrize("suffixes", [("_golden.py",), ("_golden_cpu.py",), ("_golden.py", "_golden_cpu.py")])
@pytest.mark.parametrize("golden", [False, True])
def test_paired_artifacts_keep_identity_in_arbitrary_output(tmp_path: Path, op_name, suffixes, golden, capfd) -> None:
    _write(tmp_path / f"test_{op_name}.py", "import pypto\n")
    for suffix in suffixes:
        _write(tmp_path / f"{op_name}{suffix}", "import pypto_pro\n")
    command, rule = (cmd_lint_golden, "PL11") if golden else (cmd_lint_test, "PL01")
    assert command(str(tmp_path), 4) == 2
    result = json.loads(capfd.readouterr().out)
    expected = f"{op_name}{suffixes[0]}" if golden else f"test_{op_name}.py"
    assert any(item["rule_id"] == rule and item["status"] == "FAIL" and item["file"] == expected
               for item in result["findings"])


def test_multiple_pairs_remain_ambiguous_when_generic_golden_names_overlap(tmp_path: Path) -> None:
    for filename in ("test_test_foo.py", "test_foo_golden_cpu.py", "foo_golden_cpu_golden_cpu.py"):
        _write(tmp_path / filename, "# Ambiguous artifact roles\n")
    with pytest.raises(core.LintConfigurationError, match="Ambiguous operator names"):
        cmd_lint_golden(str(tmp_path), 4)


@pytest.mark.parametrize(
    ("relative", "source", "blocking_rule"),
    [
        ("test_where.py", "import pypto_pro.language as pl\n@pl.jit\ndef kernel(): pass\n", None),
        ("test_where.py", "import pypto\n", "PL01"),
        ("where_golden_cpu.py", "import torch\n", None),
        ("where_golden_cpu.py", "import pypto_pro\n", "PL11"),
        ("modules/test_where_module12.py", "import pypto_pro.language as pl\n@pl.jit\ndef kernel(): pass\n", None),
        ("modules/test_where_module12.py", "import pypto\n", "PL01"),
        ("modules/where_golden_stage12.py", "import torch\n", None),
        ("modules/where_golden_stage12.py", "from modules.test_where_module12 import kernel\n", "PL13"),
        ("modules/test_where_moduleX.py", "import pypto_pro.language as pl\n@pl.jit\ndef kernel(): pass\n", None),
        ("modules/test_where_moduleX.py", "import pypto\n", "PL01"),
        ("modules/where_golden_stageX.py", "import torch\n", None),
        ("modules/where_golden_stageX.py", "from modules.test_where_moduleX import kernel\n", "PL13"),
        ("modules/test_where_module.py", "import pypto_pro.language as pl\n@pl.jit\ndef kernel(): pass\n", None),
        ("modules/test_where_module.py", "import pypto\n", "PL01"),
        ("modules/where_golden_stage.py", "import torch\n", None),
        ("modules/where_golden_stage.py", "from modules.test_where_module import kernel\n", "PL13"),
    ],
)
def test_post_edit_uses_selected_artifact_amid_ambiguous_siblings(
    tmp_path: Path, relative: str, source: str, blocking_rule: str | None,
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str],
) -> None:
    op_dir = tmp_path / "chosen-output"
    _write(op_dir / "SPEC.md", _valid_spec("where"))
    _write(op_dir / "DESIGN.md", "# DESIGN\n")
    _write(op_dir / "test_where.py", "import pypto\n")
    _write(op_dir / "test_accuracy.py", "def test_accuracy(): pass\n")
    if relative == "where_golden_cpu.py":
        _write(op_dir / "other_golden_cpu.py", "import torch\n")
    target = op_dir / relative
    _write(target, source)
    monkeypatch.setenv(HOOK_INPUT_ENV, json.dumps({"tool_input": {"file_path": str(target)}}))
    monkeypatch.setenv(core.POST_EDIT_BLOCK_ENV, "1")
    monkeypatch.setattr(sys, "argv", ["pypto-pro-op-lint", "--hook", "post-edit"])
    assert main() == 0
    output = json.loads(capfd.readouterr().out)["hookSpecificOutput"]
    assert output["decision"] == ("block" if blocking_rule else "allow")
    if blocking_rule:
        assert f"[{blocking_rule}]" in output["reason"]
