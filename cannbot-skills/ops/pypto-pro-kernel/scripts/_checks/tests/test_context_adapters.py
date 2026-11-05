# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared task/batch semantics survive real MCP and literal host adaptation."""

import asyncio
import hashlib
import json
import shlex
import subprocess
import sys

import pytest

from _checks import host_hooks
from _checks.paths import SKILL_ROOT


def policy(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "schema": "pypto-context-policy/1",
                "roots": [str(tmp_path)],
                "limits": {"max_output_bytes": 4096},
            }
        )
    )
    return path


def test_mixed_literal_reads_keep_ranges_and_one_total_budget(tmp_path):
    source = tmp_path / "a.txt"
    source.write_text("one\ntwo\nthree\n")
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "cwd": str(tmp_path),
        "tool_input": {
            "command": "cat a.txt && sed -n '2,2p' a.txt && head -n 1 a.txt"
        },
    }
    result = host_hooks.pre_tool(payload, policy_path=policy(tmp_path))
    command = result["hookSpecificOutput"]["updatedInput"]["command"]
    run = subprocess.run(
        shlex.split(command), cwd=tmp_path, capture_output=True, text=True, check=True
    )
    data = json.loads(run.stdout)
    assert [item["text"] for item in data["items"]] == [
        source.read_text(),
        "two\n",
        "one\n",
    ]
    assert len(run.stdout.encode()) == data["budget"]["result_bytes"] <= 4096


@pytest.mark.parametrize("suffix", ["--task task.json", "--task=task.json"])
def test_task_query_receives_fixed_host_policy(tmp_path, suffix):
    fixed = policy(tmp_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "cwd": str(tmp_path),
        "tool_input": {
            "command": f"{sys.executable} {SKILL_ROOT / 'scripts/check.py'} query {suffix} --policy ignored.json"
        },
    }
    command = host_hooks.pre_tool(payload, policy_path=fixed)["hookSpecificOutput"][
        "updatedInput"
    ]["command"]
    words = shlex.split(command)
    assert words.count("--policy") == 1 and words[-1] == str(fixed.resolve())
    assert "ignored.json" not in words


def test_mcp_batch_and_task_match_cli(tmp_path):
    pytest.importorskip("mcp")
    from _checks.mcp_server import Bridge

    source = tmp_path / "source.txt"
    source.write_text("alpha\nbeta\ngamma\n")
    manifest = tmp_path / "task.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "pypto-task-context/1",
                "goal": "select_experiment",
                "artifacts": [
                    {
                        "role": "source",
                        "path": source.name,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    bridge = Bridge(workspace=tmp_path, context_policy=policy(tmp_path))
    for name, args in (
        ("kernel_query", {"task": manifest.name}),
        (
            "kernel_read",
            {
                "requests": [
                    {"path": source.name, "start_line": 2, "max_lines": 1},
                    {"path": source.name, "start_line": 3, "max_lines": 1},
                ]
            },
        ),
    ):
        command, cwd = bridge.command(name, args)
        cli = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, check=True
        )
        response = asyncio.run(bridge.call(name, args))
        body = json.loads(response.content[0].text)
        assert body["delivery"] == "complete" and body["report"] == json.loads(
            cli.stdout
        )
        assert (
            len(response.model_dump_json(by_alias=True, exclude_none=True).encode())
            <= body["budget"]["limit_bytes"]
        )


def test_mcp_claim_preserves_unknown_without_replaying(tmp_path):
    pytest.importorskip("mcp")
    from _checks.mcp_server import Bridge

    spec = tmp_path / "claim.json"
    spec.write_text(
        json.dumps(
            {
                "schema": "pypto-performance-claim/1",
                "claim": {
                    "speedup_established": True,
                    "workload": {"rows": 1, "cols": 256},
                    "metric": "synchronized_wall_us",
                    "changes_wrapper": False,
                },
            }
        )
    )
    bridge = Bridge(workspace=tmp_path)
    command, cwd = bridge.command("kernel_claim", {"spec": spec.name})
    cli = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    result = asyncio.run(bridge.call("kernel_claim", {"spec": spec.name}))
    data = json.loads(result.content[0].text)
    assert data["delivery"] == "complete" and data["exit_code"] == cli.returncode == 2
    assert (
        data["report"] == json.loads(cli.stdout)
        and data["report"]["supported_speedup"] is None
    )
