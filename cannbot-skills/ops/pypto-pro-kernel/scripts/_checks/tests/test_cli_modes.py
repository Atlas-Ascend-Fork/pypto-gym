# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CLI operations preserve structured reports and source-gated execution."""

import json
import subprocess
import sys

import pytest

from _checks import feedback
from _checks.paths import SKILL_ROOT

ENTRY = SKILL_ROOT / "scripts/check.py"


def call_cli(tmp_path, *args):
    return subprocess.run(
        [sys.executable, str(ENTRY), *map(str, args)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "mode",
    [
        "query",
        "symbol",
        "discovery",
        "document",
        "docstring",
        "example",
        "admission",
        "rules",
        "native",
        "run",
        "suggest",
        "explain",
        "experiment",
    ],
)
def test_cli_operations(tmp_path, mode):
    source = tmp_path / "candidate.py"
    source.write_text("def run(): return 1\n")
    spec = tmp_path / "native.json"
    spec.write_text('{"entry":"candidate.py:run","args":[]}')
    identity = tmp_path / "toolchain.json"
    identity.write_text('{"scope":"no-device CLI test"}')
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "pypto-experiment-plan/1",
                "toolchain": feedback.file_identity(identity),
                "max_trials": 1,
                "baseline": {
                    "root": str(tmp_path),
                    "files": [source.name],
                    "entry": "candidate.py:run",
                    "configuration": {},
                },
                "scope": {
                    "cases": {"one": {"rows": 1}},
                    "timing_cases": ["one"],
                    "metric": "native_task_sum_us",
                    "numerical_contract": {
                        "atol": 0,
                        "rtol": 0,
                        "oracle_sha256": "f" * 64,
                    },
                },
            }
        )
    )
    commands = {
        "query": ["query", "vf.full", "--format", "json"],
        "symbol": ["query", "pl.DT_FP16", "--kind", "constant", "--format", "json"],
        "discovery": [
            "query",
            "--filters",
            '{"source_memory":"UB","destination_memory":"L1"}',
            "--format",
            "json",
        ],
        "document": [
            "query",
            "document:pro-api/Utils-API/debugging/pto_assert.md#功能说明",
            "--format",
            "json",
        ],
        "docstring": ["query", "docstring:pl.pto_assert", "--format", "json"],
        "example": [
            "query",
            "example:algorithms/batched_transposed_gemm",
            "--format",
            "json",
        ],
        "admission": ["query", "--validate", "--format", "json"],
        "rules": ["rules", source, "--format", "json"],
        "native": ["native", "--spec", spec, "--format", "json"],
        "run": [
            "run",
            "--source",
            source,
            "--format",
            "json",
            "--",
            sys.executable,
            "-c",
            "print('CLI_ONLY_OK')",
        ],
        "suggest": ["suggest", "--source", source],
        "explain": ["explain", "--source", source],
        "experiment": [
            "experiment",
            "--state",
            tmp_path / "current.json",
            "--init",
            plan,
        ],
    }
    result = call_cli(tmp_path, *commands[mode])
    assert result.returncode == (2 if mode == "native" else 0), result.stderr
    report = json.loads(result.stdout)
    assert report and isinstance(report, dict)
    if mode == "native":
        assert report["status"] == "unverified"
        assert report["validation"]["numerical"] == "not_established"
    if mode == "run":
        assert report["command"]["started"] and report["command"]["exit_code"] == 0
        assert "CLI_ONLY_OK" in result.stderr
    if mode == "experiment":
        shown = call_cli(
            tmp_path, "experiment", "--state", tmp_path / "current.json"
        )
        assert shown.returncode == 0
        assert json.loads(shown.stdout) == report


def test_cli_run_retains_source_gate(tmp_path):
    source = tmp_path / "candidate.py"
    source.write_text("missing_name()\n")
    marker = tmp_path / "started"
    result = call_cli(
        tmp_path,
        "run",
        "--source",
        source,
        "--format",
        "json",
        "--",
        sys.executable,
        "-c",
        f"open({str(marker)!r}, 'w').write('ran')",
    )
    assert result.returncode == 1
    assert not json.loads(result.stdout)["command"]["started"]
    assert not marker.exists()
