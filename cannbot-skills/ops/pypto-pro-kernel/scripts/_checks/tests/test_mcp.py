# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Real stdio/CLI parity, fresh inputs, bounded replies and process ownership."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import version

import pytest

pytest.importorskip(
    "mcp", reason="Install scripts/requirements-mcp.txt for MCP acceptance"
)
if version("mcp") != "2.0.0":
    pytest.skip("MCP acceptance uses the pinned SDK", allow_module_level=True)

from mcp import Client, types
from mcp.client.stdio import StdioServerParameters, stdio_client

from _checks import feedback, mcp_server
from _checks.paths import PYPTO_ROOT, SKILL_ROOT
from _checks.tests.knowledge_fixtures import make_corpus, write


def payload(result):
    data = json.loads(result.content[0].text)
    size = len(result.model_dump_json(by_alias=True, exclude_none=True).encode())
    assert data["budget"]["result_bytes"] == size
    assert size <= data["budget"]["limit_bytes"]
    return data


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text("value = 1\n")
    return path


@pytest.fixture
def bridge(tmp_path):
    return mcp_server.Bridge(workspace=tmp_path)


@pytest.fixture
def knowledge_bundle(tmp_path):
    skill = tmp_path / "kit"
    shutil.copytree(
        SKILL_ROOT, skill, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache")
    )
    shutil.rmtree(skill / "references")
    shutil.rmtree(skill / "examples")
    corpus = make_corpus(tmp_path, references=skill / "references")
    bridge = mcp_server.Bridge(
        workspace=tmp_path,
        entry=skill / "scripts/check.py",
        pypto_root=corpus.framework,
        examples_root=corpus.examples,
    )
    return corpus, bridge


@pytest.mark.parametrize(
    "name,args",
    [
        ("kernel_search", {"paths": ["candidate.py"], "pattern": "value"}),
        ("kernel_read", {"paths": ["candidate.py"]}),
        ("kernel_query", {"text": "UB to L1"}),
        ("kernel_query", {"text": "acc-final-output"}),
        ("kernel_query", {"text": "acc-final-output", "goal": "select_experiment"}),
        ("kernel_query", {"text": "pl.make_tile_group"}),
        ("kernel_query", {"text": "vf.full"}),
        (
            "kernel_query",
            {
                "text": "VF add with carry UINT32 addc output carry target declaration",
                "limit": 3,
            },
        ),
        (
            "kernel_query",
            {
                "text": "row softmax runtime prefix length empty rows physical tail padding",
                "limit": 3,
            },
        ),
        (
            "kernel_query",
            {
                "text": "L0B reuse across two matmul operations, explicit operand movement, and matmul chain",
                "limit": 3,
            },
        ),
        (
            "kernel_query",
            {
                "text": "gathered dot product matrix multiplication repeated row selector",
                "limit": 1,
            },
        ),
        (
            "kernel_query",
            {
                "text": "分段前缀和需要重置标记跨越寄存器和UB分块，保留局部状态与块间传递",
                "limit": 1,
            },
        ),
        ("kernel_query", {"text": "请比较vf.add和vf.sub的输出声明和调用", "limit": 4}),
        (
            "kernel_query",
            {
                "text": "pl.jit native entry classification after binding a wrapper",
                "limit": 3,
            },
        ),
        (
            "kernel_query",
            {
                "text": "pl.jit scalar parameter annotation, not postponed Tensor annotations",
                "limit": 3,
            },
        ),
        (
            "kernel_query",
            {
                "text": "segmented softmax weighted sum ragged segments scan reduce VF SIMT",
                "limit": 1,
            },
        ),
        (
            "kernel_query",
            {
                "text": "VF Vector per-column reduction across rows for contiguous 2D FP32 input; inspect suitable complete example and explicit buffers/stages",
                "kinds": ["example"],
                "limit": 1,
            },
        ),
        (
            "kernel_query",
            {
                "text": "stable ordered compaction using VF vector registers, prefix ranks and tail initialization",
                "kinds": ["example"],
                "limit": 1,
            },
        ),
        (
            "kernel_query",
            {
                "text": "UB Vec TileType capacity streaming tile",
                "kinds": ["example"],
                "limit": 3,
            },
        ),
        (
            "kernel_query",
            {"text": "UB Vec TileType capacity streaming tile", "limit": 3},
        ),
        ("kernel_query", {"text": "pl.system.bar_all", "kinds": ["api"]}),
        ("kernel_query", {"text": "pl.DT_FP16"}),
        ("kernel_query", {"text": "pl.DT_FP16", "limit": 5}),
        (
            "kernel_query",
            {"text": "pl.DT_FP16", "limit": mcp_server.knowledge.MAX_LIMIT},
        ),
        ("kernel_query", {"kinds": ["contract"], "limit": 2}),
        (
            "kernel_query",
            {
                "filters": {"source_memory": "UB", "destination_memory": "L1"},
                "limit": 3,
            },
        ),
        ("kernel_query", {"text": "UB to L1", "kinds": ["document"]}),
        ("kernel_query", {"text": "vf.addc", "kinds": ["document"]}),
        (
            "kernel_query",
            {
                "text": "document:pro-api/SIMD-API/Reg-API/basic_arithmetic/addc.md#调用示例"
            },
        ),
        (
            "kernel_query",
            {
                "text": "document:pro-guide/development/cube_vector_fusion.md#跨核中间数据传递"
            },
        ),
        ("kernel_query", {"text": "docstring:pl.pto_assert"}),
        (
            "kernel_query",
            {"text": "docstring:pl.pto_assert", "reference_sha256": "f" * 64},
        ),
        (
            "kernel_query",
            {"text": "document:pro-api/Utils-API/debugging/pto_assert.md#missing"},
        ),
        (
            "kernel_query",
            {"text": "batched transposed GEMM with M tails", "kinds": ["example"]},
        ),
        (
            "kernel_query",
            {"text": "pl.simt.atomic_add", "kinds": ["example"], "limit": 2},
        ),
        ("kernel_query", {"text": "example:algorithms/batched_transposed_gemm"}),
        (
            "kernel_query",
            {
                "text": "example:algorithms/batched_transposed_gemm/kernel.py",
                "reference_sha256": "f" * 64,
            },
        ),
        (
            "kernel_query",
            {"text": "batched transposed GEMM with N tails", "kinds": ["example"]},
        ),
        ("kernel_query", {"validate": True}),
        ("kernel_query", {"kinds": ["contract"], "cursor": "invalid-cursor"}),
        (
            "kernel_read",
            {
                "paths": [
                    str(
                        PYPTO_ROOT
                        / "docs/zh/api/pro_api/Utils-API/debugging/pto_assert.md"
                    )
                ],
                "fragment": "功能说明",
            },
        ),
        (
            "kernel_read",
            {
                "requests": [
                    {
                        "path": str(
                            PYPTO_ROOT
                            / "docs/zh/api/pro_api/Utils-API/debugging/pto_assert.md"
                        ),
                        "fragment": "功能说明",
                    }
                ]
            },
        ),
        ("kernel_query", {"text": "--source=not-an-option"}),
        ("kernel_check", {"sources": ["candidate.py"]}),
        ("kernel_suggest", {"sources": ["candidate.py"]}),
        ("kernel_suggest", {"sources": ["candidate.py"], "goal": "select_experiment"}),
        ("kernel_explain", {"source": "candidate.py"}),
        (
            "kernel_run",
            {
                "sources": ["candidate.py"],
                "argv": [sys.executable, "-c", "print('ok')"],
            },
        ),
        (
            "kernel_run",
            {
                "sources": ["candidate.py"],
                "argv": [
                    sys.executable,
                    "-c",
                    "print('opaque failure'); raise SystemExit(7)",
                ],
            },
        ),
        (
            "kernel_run",
            {
                "sources": ["candidate.py"],
                "argv": [
                    sys.executable,
                    "-c",
                    "print('RuntimeError: make_tile_group() depth is required when mutex_ids is \\\"auto\\\", None, or empty'); raise SystemExit(9)",
                ],
            },
        ),
    ],
)
def test_cli_parity_for_identical_inputs(bridge, source, name, args):
    # Corpus-wide discovery can exceed the default transport page; parity compares a complete, explicitly budgeted reply.
    if name == "kernel_query":
        args = {"max_output_bytes": 65536, **args}
    command, cwd = bridge.command(name, args)
    cli = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=30)
    result = asyncio.run(bridge.call(name, args))
    data = payload(result)
    assert data["delivery"] == "complete", data
    assert data["exit_code"] == cli.returncode
    assert data["report"] == json.loads(cli.stdout)
    assert result.is_error == (cli.returncode != 0)


def test_status_cli_parity_keeps_failed_process_outcome(bridge, source, tmp_path):
    import hashlib

    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "request_id": "existing",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "state": "completed",
                "exit_code": 7,
            }
        )
    )
    args = {
        "receipt": receipt.name,
        "request_id": "existing",
        "source": source.name,
        "wait_seconds": 0,
    }
    command, cwd = bridge.command("kernel_status", args)
    cli = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=10)
    data = payload(asyncio.run(bridge.call("kernel_status", args)))
    report = json.loads(cli.stdout)
    # Elapsed wall time may differ; all receipt identity/outcome fields must agree.
    data["report"].pop("elapsed_seconds")
    report.pop("elapsed_seconds")
    assert data["report"] == report and report["outcome"] == "failed"
    assert report["exit_code"] == 7 and set(report["validation"].values()) == {
        "not_established"
    }


@pytest.mark.parametrize(
    "name,args",
    [
        ("kernel_run", {"sources": [], "argv": ["python"]}),
        ("kernel_run", {"sources": ["candidate.py"], "argv": "echo no"}),
        (
            "kernel_run",
            {"sources": ["candidate.py"], "argv": ["echo"], "timeout_seconds": 0},
        ),
        ("kernel_query", {"text": "pl.move", "max_output_bytes": 999999}),
        ("kernel_query", {"text": "pl.move", "unexpected": True}),
        ("kernel_query", {"query": "vf.reduce_sum"}),
        ("kernel_query", {"symbol": "vf.reduce_sum"}),
        ("kernel_query", {}),
        ("kernel_query", {"text": 12}),
        ("kernel_query", {"text": "acc-final-output", "goal": "skip_validation"}),
        ("kernel_query", {"kinds": ["function"]}),
        ("kernel_query", {"text": "x", "filters": {"source": "UB"}}),
        ("kernel_query", {"text": "x", "filters": {"dtype": True}}),
        ("kernel_query", {"text": "x", "limit": 0}),
        ("kernel_query", {"cursor": "previous"}),
        ("kernel_query", {"task": "task.json", "kinds": ["api"]}),
        ("kernel_query", {"task": "task.json", "reference_sha256": "f" * 64}),
        ("kernel_query", {"text": "docstring:pl.insert", "reference_sha256": "bad"}),
        ("kernel_query", {"validate": False}),
        ("kernel_query", {"validate": True, "text": "pl.move"}),
        ("kernel_query", {"validate": True, "task": "task.json"}),
        ("kernel_query", {"validate": True, "goal": "select_experiment"}),
        ("kernel_read", {"paths": ["a.md"], "fragment": ""}),
        ("kernel_suggest", {"sources": ["candidate.py"], "limit": 6}),
        ("kernel_experiment", {"state": "state.json", "action": "init"}),
        (
            "kernel_experiment",
            {"state": "state.json", "action": "show", "input": "ignored.json"},
        ),
        ("kernel_unknown", {}),
    ],
)
def test_bad_arguments_do_not_start_a_process(bridge, name, args, monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail("invalid input must not execute")

    monkeypatch.setattr(bridge, "execute", unexpected)
    result = asyncio.run(bridge.call(name, args))
    assert result.is_error and payload(result)["exit_code"] == 2
    if payload(result)["delivery"] == "invalid_arguments":
        assert payload(result)["input_schema"] == mcp_server.BY_NAME[name].input_schema
        assert "No CLI process was started" in payload(result)["next_action"]


@pytest.mark.parametrize(
    "name,args,path,constraint,expected",
    [
        ("kernel_query", {"text": "vf.add", "limit": 0}, ["limit"], "minimum", 1),
        (
            "kernel_query",
            {"text": "vf.add", "limit": mcp_server.knowledge.MAX_LIMIT + 1},
            ["limit"],
            "maximum",
            mcp_server.knowledge.MAX_LIMIT,
        ),
        (
            "kernel_query",
            {"text": "vf.add", "limit": "5"},
            ["limit"],
            "type",
            "integer",
        ),
        (
            "kernel_read",
            {"paths": ["a.md"], "max_lines": 0},
            ["max_lines"],
            "minimum",
            1,
        ),
        (
            "kernel_read",
            {"requests": [{"path": "a.md", "max_lines": 0}]},
            ["requests", 0, "max_lines"],
            "minimum",
            1,
        ),
        (
            "kernel_run",
            {"sources": ["a.py"], "argv": ["python"], "timeout_seconds": 0},
            ["timeout_seconds"],
            "minimum",
            0.1,
        ),
    ],
)
def test_invalid_argument_feedback_locates_the_constraint_before_execution(
    bridge, monkeypatch, name, args, path, constraint, expected
):
    async def unexpected(*args, **kwargs):
        pytest.fail("invalid input must not execute")

    monkeypatch.setattr(bridge, "execute", unexpected)
    original = json.loads(json.dumps(args))
    result = asyncio.run(bridge.call(name, {**args, "max_output_bytes": 2048}))
    body = payload(result)
    assert result.is_error and body["delivery"] == "invalid_arguments"
    assert body["argument_error"] == {
        "path": path,
        "constraint": constraint,
        "expected": expected,
    }
    assert args == original


def test_nested_unknown_field_feedback_lists_local_fields(bridge, monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail("invalid input must not execute")

    monkeypatch.setattr(bridge, "execute", unexpected)
    result = asyncio.run(
        bridge.call(
            "kernel_query",
            {"text": "vf.add", "filters": {"source": "UB"}, "max_output_bytes": 2048},
        )
    )
    detail = payload(result)["argument_error"]
    assert (
        detail["path"] == ["filters"] and detail["constraint"] == "additionalProperties"
    )
    assert (
        "source_memory" in detail["allowed_fields"]
        and "source" not in detail["allowed_fields"]
    )


@pytest.mark.parametrize(
    "name,args,option",
    [
        (
            "kernel_query",
            {"text": "vf.add", "limit": 5.0, "max_output_bytes": 2048.0},
            "--limit",
        ),
        ("kernel_read", {"paths": ["a.md"], "max_lines": 2000.0}, "--max-lines"),
        (
            "kernel_read",
            {"requests": [{"path": "a.md", "max_lines": 2000.0}]},
            "--request",
        ),
        ("kernel_run", {"sources": ["a.py"], "argv": ["echo", "5.0"]}, "--"),
    ],
)
def test_json_integer_syntax_is_canonicalized_without_changing_the_request(
    bridge, monkeypatch, name, args, option
):
    calls = []

    async def execute(command, cwd, timeout, env):
        calls.append(command)
        return {"exit_code": 0, "delivery": "complete", "report": {}}

    monkeypatch.setattr(bridge, "execute", execute)
    original = json.dumps(args)
    result = asyncio.run(bridge.call(name, args))
    body = payload(result)
    assert not result.is_error and len(calls) == 1 and json.dumps(args) == original
    command = calls[0]
    if option == "--request":
        value = json.loads(command[command.index(option) + 1])["max_lines"]
        assert type(value) is int and value == 2000
    elif option == "--":
        assert command[command.index(option) + 1 :] == ["echo", "5.0"]
    elif option == "--max-lines":
        assert "--max-lines=2000" in command
    else:
        assert "--limit=5" in command
    if "max_output_bytes" in args:
        assert body["budget"]["limit_bytes"] == 2048


def test_fractional_integer_argument_is_rejected_without_rounding(bridge, monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail("fractional input must not execute")

    monkeypatch.setattr(bridge, "execute", unexpected)
    result = asyncio.run(bridge.call("kernel_query", {"text": "vf.add", "limit": 5.5}))
    assert result.is_error and payload(result)["argument_error"] == {
        "path": ["limit"],
        "constraint": "type",
        "expected": "integer",
    }


@pytest.mark.parametrize(
    "invalid", [{"query": "vf.reduce_sum"}, {"错误😀" * 400: "value"}]
)
def test_invalid_argument_guidance_respects_the_requested_budget_without_starting_cli(
    bridge, invalid, monkeypatch
):
    async def unexpected(*args, **kwargs):
        pytest.fail("schema rejection must not execute")

    monkeypatch.setattr(bridge, "execute", unexpected)
    result = asyncio.run(
        bridge.call("kernel_query", {**invalid, "max_output_bytes": 2048})
    )
    body = payload(result)
    assert (
        result.is_error
        and body["delivery"] == "invalid_arguments"
        and body["report"] is None
    )
    assert (
        body["budget"]["limit_bytes"] == 2048 and body["budget"]["result_bytes"] <= 2048
    )
    assert (
        body["input_schema_omitted"] == "output_budget" and "input_schema" not in body
    )
    assert body["argument_error"]["path"] == []
    assert body["argument_error"]["constraint"] == "additionalProperties"
    assert "text" in body["argument_names"] and "query" not in body["argument_names"]
    assert "No CLI process was started" in body["next_action"]


def test_source_gate_and_native_gate_prevent_child_execution(bridge, source, tmp_path):
    marker = tmp_path / "started"
    source.write_text("missing_name()\n")
    args = {
        "sources": [source.name],
        "argv": [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('ran')"],
    }
    data = payload(asyncio.run(bridge.call("kernel_run", args)))
    assert (
        data["exit_code"] == 1
        and not data["report"]["command"]["started"]
        and not marker.exists()
    )
    source.write_text("value=1\n")
    # A real spec error is sufficient to close the gate without an NPU/runtime.
    spec = tmp_path / "spec.json"
    spec.write_text('{"entry":"missing.py:kernel","args":[]}')
    args["native_spec"] = spec.name
    data = payload(asyncio.run(bridge.call("kernel_run", args)))
    assert data["exit_code"] == 2 and not marker.exists()


def test_run_retains_original_failure_context_and_signal(bridge, source):
    async def check():
        result = await bridge.call(
            "kernel_run",
            {
                "sources": [source.name],
                "argv": [
                    sys.executable,
                    "-c",
                    "print('move: NZ destination uses align16 rows'); raise SystemExit(9)",
                ],
            },
        )
        data = payload(result)
        assert data["exit_code"] == data["report"]["command"]["exit_code"] == 9
        assert data["report"]["diagnostics"][0]["rule"] == "native-transfer-contract"
        assert (
            data["report"]["diagnostics"][0]["evidence"]["condition_status"]
            == "unverified"
        )
        assert not data["report"]["context"]
        result = await bridge.call(
            "kernel_run",
            {
                "sources": [source.name],
                "argv": [
                    sys.executable,
                    "-c",
                    "import os,signal; os.kill(os.getpid(),signal.SIGTERM)",
                ],
            },
        )
        assert payload(result)["exit_code"] == 143

    asyncio.run(check())


def test_conditional_failure_reviews_match_cli_without_numerical_acceptance(
    bridge, source
):
    source.write_text("""import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
@pl.vector_function
def compute(src, dst, pred):
    values = vf.load_align(src, [0, 0])
    total = vf.reduce_sum(values, pred)
    vf.store_align(dst, total, pred, [0, 0])
""")
    args = {
        "sources": [source.name],
        "argv": [sys.executable, "-c", "raise SystemExit(9)"],
    }
    command, cwd = bridge.command("kernel_run", args)
    cli = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=30)
    result = asyncio.run(bridge.call("kernel_run", args))
    data = payload(result)
    assert result.is_error and data["delivery"] == "complete"
    assert data["exit_code"] == cli.returncode == 9
    assert data["report"] == json.loads(cli.stdout)
    assert [r["id"] for r in data["report"]["related_contracts"]] == [
        "vf-reduction-result"
    ]
    assert "not a diagnosis" in data["report"]["contract_review_scope"]
    assert data["report"]["validation"]["numerical"] == "not_established"


def test_run_preserves_cwd_argv_and_does_not_interpret_shell_text(
    bridge, source, tmp_path
):
    folder = tmp_path / "space folder"
    folder.mkdir()
    shutil.copyfile(source, folder / source.name)
    dangerous_literal = "$(touch unintended); `touch unintended` $HOME"
    program = "import sys,pathlib; pathlib.Path('observed.txt').write_text(sys.argv[1])"
    data = payload(
        asyncio.run(
            bridge.call(
                "kernel_run",
                {
                    "sources": [source.name],
                    "cwd": str(folder),
                    "argv": [sys.executable, "-c", program, dangerous_literal],
                },
            )
        )
    )
    assert (
        data["exit_code"] == 0
        and (folder / "observed.txt").read_text() == dangerous_literal
    )
    assert not (folder / "unintended").exists()


def test_budget_excess_is_explicit_and_never_reexecutes_a_mutation(
    bridge, source, tmp_path
):
    count = tmp_path / "counter"
    program = f"from pathlib import Path; p=Path({str(count)!r}); p.write_text(str(int(p.read_text())+1) if p.exists() else '1')"
    result = asyncio.run(
        bridge.call(
            "kernel_run",
            {
                "sources": [source.name],
                "argv": [sys.executable, "-c", program],
                "details": True,
                "max_output_bytes": 2048,
            },
        )
    )
    data = payload(result)
    assert data["delivery"] == "budget_exceeded" and data["report"] is None
    assert data["exit_code"] == 0 and count.read_text() == "1"
    assert result.is_error and data["report_summary"]["status"] == "passed"
    assert "read_only_retry" not in data


@pytest.mark.parametrize(
    "returned,cursor", [(1, None), (3, None), (3, "existing-cursor")]
)
def test_query_budget_feedback_uses_observed_page_size_without_executing_it(
    bridge, monkeypatch, returned, cursor
):
    calls = []

    async def execute(*args):
        calls.append(args)
        return {
            "exit_code": 0,
            "delivery": "complete",
            "report": {
                "stage": "query",
                "status": "passed",
                "knowledge": {"page": {"returned": returned}},
                "large": "x" * 10000,
            },
        }

    monkeypatch.setattr(bridge, "execute", execute)
    arguments = {"text": "vf.add", "kinds": ["api"], "max_output_bytes": 2048}
    arguments["limit"] = 7
    if cursor is not None:
        arguments["cursor"] = cursor
    original = dict(arguments)
    result = asyncio.run(bridge.call("kernel_query", arguments))
    data = payload(result)
    assert (
        result.is_error
        and data["delivery"] == "budget_exceeded"
        and data["report"] is None
    )
    assert len(calls) == 1 and arguments == original
    assert data["budget"]["result_bytes"] <= 2048
    retry = data["read_only_retry"]
    assert retry["tool"] == "kernel_query" and retry["preserve_other_arguments"] is True
    if returned == 1 or cursor:
        assert (
            10000
            < retry["arguments_patch"]["max_output_bytes"]
            <= mcp_server.MAX_BUDGET
        )
    else:
        assert retry["arguments_patch"] == {"limit": 1}


def test_exact_program_budget_retry_and_original_role_reads_use_cli_bytes(bridge):
    arguments = {"text": "example:algorithms/streaming_segmented_scan/kernel.py"}
    first = payload(asyncio.run(bridge.call("kernel_query", arguments)))
    assert first["delivery"] == "budget_exceeded" and first["exit_code"] == 0
    patch = first["read_only_retry"]["arguments_patch"]
    assert (
        patch["max_output_bytes"] > mcp_server.DEFAULT_BUDGET and "limit" not in patch
    )
    data = payload(asyncio.run(bridge.call("kernel_query", {**arguments, **patch})))
    assert data["delivery"] == "complete"
    program = data["report"]["knowledge"]["items"][0]["program"]
    assert set(program["read_requests"]) == {"kernel", "reference", "test"}
    for request in program["read_requests"].values():
        command, cwd = bridge.command("kernel_read", {"requests": [request]})
        cli = json.loads(
            subprocess.run(
                command, cwd=cwd, capture_output=True, text=True, check=True
            ).stdout
        )
        read = payload(asyncio.run(bridge.call("kernel_read", {"requests": [request]})))
        assert read["report"] == cli and read["delivery"] == "complete"
    stale = {**program["read_requests"]["kernel"], "source_sha256": "f" * 64}
    read = payload(asyncio.run(bridge.call("kernel_read", {"requests": [stale]})))
    assert read["exit_code"] != 0 and read["report"]["items"] == []
    assert "Source changed" in read["report"]["error"]


def test_real_stdio_exact_program_budget_recovery(tmp_path):
    async def check():
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(mcp_server.ENTRY), "mcp", "--workspace", str(tmp_path)],
        )
        async with Client(stdio_client(params)) as client:
            arguments = {
                "text": "example:algorithms/streaming_segmented_scan/kernel.py",
                "limit": 1,
            }
            first = payload(await client.call_tool("kernel_query", arguments))
            retry = first["read_only_retry"]
            assert (
                first["delivery"] == "budget_exceeded"
                and retry["tool"] == "kernel_query"
            )
            data = payload(
                await client.call_tool(
                    retry["tool"], {**arguments, **retry["arguments_patch"]}
                )
            )
            assert data["delivery"] == "complete"
            request = data["report"]["knowledge"]["items"][0]["program"][
                "read_requests"
            ]["reference"]
            read = payload(
                await client.call_tool("kernel_read", {"requests": [request]})
            )
            assert read["report"]["items"][0]["sha256"] == request["source_sha256"]

    asyncio.run(check())


def test_real_stdio_source_lines_and_log_conditions_share_cli_semantics(tmp_path):
    import hashlib

    path = tmp_path / "entry.py"
    data = b"\r\n\r\ndef wrapper(value):\r\n    return value\r\n"
    path.write_bytes(data)
    log = tmp_path / "error.txt"
    log.write_text("RuntimeError: Expected a native @pl.jit Pro kernel\n")

    async def check():
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(mcp_server.ENTRY), "mcp", "--workspace", str(tmp_path)],
        )
        async with Client(stdio_client(params)) as client:
            read = payload(
                await client.call_tool(
                    "kernel_read",
                    {"paths": [str(path)], "start_line": 2, "max_lines": 3},
                )
            )
            item = read["report"]["items"][0]
            assert item["sha256"] == hashlib.sha256(data).hexdigest()
            assert (
                item["numbered_text"]
                == "2|\r\n3|def wrapper(value):\r\n4|    return value\r\n"
            )
            arguments = {"source": str(path), "error_log": str(log)}
            explained = payload(await client.call_tool("kernel_explain", arguments))
            cli = subprocess.run(
                [
                    sys.executable,
                    str(mcp_server.ENTRY),
                    "explain",
                    "--source",
                    str(path),
                    "--error-log",
                    str(log),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            assert explained["report"] == json.loads(cli.stdout)
            assert explained["report"]["diagnostics"][0]["rule"] == "native-entry-kind"
            assert (
                explained["report"]["diagnostics"][0]["evidence"]["stage"] == "unknown"
            )

    asyncio.run(check())


def test_real_stdio_large_line_request_keeps_budgeted_continuation(tmp_path):
    source = tmp_path / "lines.txt"
    data = "".join(f"line {index}\n" for index in range(600)).encode()
    source.write_bytes(data)

    async def check():
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(mcp_server.ENTRY), "mcp", "--workspace", str(tmp_path)],
        )
        async with Client(stdio_client(params)) as client:
            first = payload(
                await client.call_tool(
                    "kernel_read", {"paths": [source.name], "max_lines": 2000}
                )
            )
            assert first["delivery"] == "complete"
            page = first["report"]
            item = page["items"][0]
            assert (
                item["text"]
                and not item["request_complete"]
                and not item["file_complete"]
            )
            assert item["text"].encode() == data[: item["end_offset_bytes"]]
            integral = payload(
                await client.call_tool(
                    "kernel_read",
                    {"requests": [{"path": source.name, "max_lines": 2000.0}]},
                )
            )
            assert integral["delivery"] == "complete"
            integral_item = integral["report"]["items"][0]
            assert integral_item["request_index"] == 0
            assert {
                key: value
                for key, value in integral_item.items()
                if key != "request_index"
            } == item
            following = payload(
                await client.call_tool("kernel_read", {"requests": [page["next"]]})
            )
            assert following["delivery"] == "complete"
            next_item = following["report"]["items"][0]
            assert next_item["offset_bytes"] == item["end_offset_bytes"]
            assert next_item["sha256"] == item["sha256"]
            assert (
                next_item["text"].encode()
                == data[item["end_offset_bytes"] : next_item["end_offset_bytes"]]
            )

    asyncio.run(check())


@pytest.mark.parametrize(
    "tool,args",
    [
        (
            "kernel_query",
            {"text": "cube-01", "kinds": ["document"], "max_output_bytes": 65536},
        ),
        ("kernel_query", {"validate": True, "max_output_bytes": 65536}),
        ("kernel_suggest", {"sources": ["candidate.py"]}),
    ],
)
def test_optional_card_source_cli_parity(knowledge_bundle, tmp_path, tool, args):
    pytest.importorskip("yaml")
    from _checks.tests.test_performance_cards import library, relation

    corpus, original = knowledge_bundle
    root = tmp_path / "card source with spaces"
    library(root, links=[relation()])
    write(tmp_path / "candidate.py", "value = 1\n")
    bridge = mcp_server.Bridge(
        workspace=tmp_path,
        entry=original.entry,
        pypto_root=corpus.framework,
        examples_root=corpus.examples,
        cards_root=root,
    )
    command, cwd = bridge.command(tool, args)
    expected = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, timeout=30
    )
    actual = payload(asyncio.run(bridge.call(tool, args)))
    assert (
        actual["delivery"] == "complete"
        and actual["exit_code"] == expected.returncode == 0
    )
    assert actual["report"] == json.loads(expected.stdout)


def test_external_card_continuations_preserve_bytes_and_sha(knowledge_bundle, tmp_path):
    pytest.importorskip("yaml")
    from _checks.tests.test_performance_cards import library, relation

    corpus, original = knowledge_bundle
    root = tmp_path / "external card library"
    path = library(root, links=[relation()])
    path.write_text(path.read_text() + "\n" + "卡片原文逐字节续读。\n" * 1800)
    expected = path.read_bytes()
    workspace = tmp_path / "execution workspace"
    workspace.mkdir()
    bridge = mcp_server.Bridge(
        workspace=workspace,
        entry=original.entry,
        pypto_root=corpus.framework,
        examples_root=corpus.examples,
        cards_root=root,
    )

    async def check():
        value = payload(
            await bridge.call(
                "kernel_query",
                {"text": "document:perf-cards/cube/cube-01-fixture.md"},
            )
        )
        assert value["delivery"] == "complete" and value["exit_code"] == 0
        content = value["report"]["knowledge"]["content"]
        received = bytearray()
        pages = 0
        while True:
            pages += 1
            previous = len(received)
            for row in content["items"]:
                assert row["offset_bytes"] == len(received)
                received.extend(row["text"].encode("utf-8"))
            if content["status"] == "complete":
                break
            assert content["status"] == "partial" and len(received) > previous
            request = {
                key: item
                for key, item in content["next"].items()
                if key != "request_index"
            }
            continued = payload(
                await bridge.call("kernel_read", {"requests": [request]})
            )
            assert continued["delivery"] == "complete" and continued["exit_code"] == 0
            content = continued["report"]
        assert pages > 1 and bytes(received) == expected

    asyncio.run(check())


def test_real_stdio_card_reads_refresh_and_reject_stale_sha(knowledge_bundle, tmp_path):
    pytest.importorskip("yaml")
    from _checks.tests.test_performance_cards import library, relation

    corpus, bridge = knowledge_bundle
    root = tmp_path / "card source with spaces"
    path = library(root, links=[relation()])

    async def check():
        params = StdioServerParameters(
            command=sys.executable,
            args=[
                str(bridge.entry),
                "--pypto-root",
                str(corpus.framework),
                "--examples-root",
                str(corpus.examples),
                "--cards-root",
                str(root),
                "mcp",
                "--workspace",
                str(tmp_path),
            ],
        )
        async with Client(stdio_client(params)) as client:
            first = payload(
                await client.call_tool(
                    "kernel_query",
                    {"text": "cube-01", "kinds": ["document"], "limit": 1},
                )
            )
            assert first["delivery"] == "complete" and first["exit_code"] == 0
            item = first["report"]["knowledge"]["items"][0]
            assert item["document_role"] == "performance_card"
            read = payload(await client.call_tool("kernel_query", item["resolve"]))
            assert (
                read["exit_code"] == 0
                and read["report"]["knowledge"]["content"]["items"]
            )
            path.write_text(path.read_text() + "\nChanged original bytes.\n")
            stale = payload(await client.call_tool("kernel_query", item["resolve"]))
            assert (
                stale["exit_code"] == 2
                and stale["report"]["knowledge"]["lookup"]["status"] == "source_changed"
            )
            fresh = payload(
                await client.call_tool("kernel_query", {"text": item["id"]})
            )
            assert fresh["exit_code"] == 0
            assert (
                fresh["report"]["knowledge"]["items"][0]["reference"]["sha256"]
                != item["reference"]["sha256"]
            )

    asyncio.run(check())


def test_real_stdio_program_identity_and_resource_negatives(tmp_path):
    async def check():
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(mcp_server.ENTRY), "mcp", "--workspace", str(tmp_path)],
        )
        async with Client(stdio_client(params)) as client:
            for text, expected in [
                (
                    "VF Vector per-column reduction across rows for contiguous 2D FP32 input; inspect suitable complete example and explicit buffers/stages",
                    "column_statistics",
                ),
                (
                    "stable ordered compaction using VF vector registers, prefix ranks and tail initialization",
                    "stable_filter",
                ),
                ("UB Vec TileType capacity streaming tile", None),
            ]:
                data = payload(
                    await client.call_tool(
                        "kernel_query",
                        {
                            "text": text,
                            "kinds": ["example"],
                            "limit": 1,
                            "max_output_bytes": 65536,
                        },
                    )
                )
                assert data["delivery"] == "complete"
                view = data["report"]["knowledge"]
                assert (
                    view["lookup"]["status"] == "retrieval_miss" and view["items"] == []
                )
                rows = view.get("recovery", {}).get("items", [])
                assert [row["id"] for row in rows] == (
                    [f"example:algorithms/{expected}/kernel.py"] if expected else []
                )
                if rows:
                    request = rows[0]["program"]["read_requests"]["reference"]
                    read = payload(
                        await client.call_tool("kernel_read", {"requests": [request]})
                    )
                    assert (
                        read["report"]["items"][0]["sha256"] == request["source_sha256"]
                    )
            data = payload(
                await client.call_tool(
                    "kernel_query",
                    {
                        "text": "UB Vec TileType capacity streaming tile",
                        "limit": 3,
                        "max_output_bytes": 65536,
                    },
                )
            )
            assert data["delivery"] == "complete"
            view = data["report"]["knowledge"]
            assert view["items"][0]["id"] == "contract:tile-capacity"
            assert not any(row["kind"] == "example" for row in view["items"])
            assert "recovery" not in view

    asyncio.run(check())


def test_unicode_and_large_stderr_do_not_break_budget_or_protocol(bridge, source):
    data = payload(
        asyncio.run(
            bridge.call(
                "kernel_run",
                {
                    "sources": [source.name],
                    "max_output_bytes": 4096,
                    "argv": [
                        sys.executable,
                        "-c",
                        "import sys; sys.stderr.write('错误' * 100000)",
                    ],
                },
            )
        )
    )
    assert data["exit_code"] == 0 and data["stderr_truncated"]
    assert data["cli_stderr_bytes"] >= 600000
    assert data["budget"]["context_tokens"] is None


def test_native_missing_spec_matches_cli(bridge):
    args = {"spec": "missing-spec.json"}
    command, cwd = bridge.command("kernel_native", args)
    cli = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    data = payload(asyncio.run(bridge.call("kernel_native", args)))
    assert data["exit_code"] == cli.returncode == 2 and data["report"] is None


def test_state_read_and_mutation_use_existing_cli(tmp_path, bridge, source):
    toolchain = tmp_path / "toolchain.json"
    toolchain.write_text('{"compiler":"test fixture"}')
    plan = {
        "schema": "pypto-experiment-plan/1",
        "toolchain": feedback.file_identity(toolchain),
        "max_trials": 1,
        "scope": {
            "cases": {"small": {"rows": 8}},
            "timing_cases": ["small"],
            "metric": "native_task_sum_us",
            "numerical_contract": {"rtol": 0, "atol": 0, "oracle_sha256": "a" * 64},
        },
        "baseline": {
            "root": str(tmp_path),
            "files": [source.name],
            "entry": "candidate.py:kernel",
            "configuration": {},
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))

    async def check():
        args = {"state": "state.json", "action": "init", "input": "plan.json"}
        data = payload(await bridge.call("kernel_experiment", args))
        assert data["report"]["phase"] == "baseline_unverified"
        read = {"state": "state.json"}
        command, cwd = bridge.command("kernel_experiment", read)
        cli = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, check=True
        )
        assert payload(await bridge.call("kernel_experiment", read))[
            "report"
        ] == json.loads(cli.stdout)
        before = (tmp_path / "state.json").read_bytes()
        assert payload(await bridge.call("kernel_experiment", args))["exit_code"] == 2
        assert (tmp_path / "state.json").read_bytes() == before

    asyncio.run(check())


def test_no_cache_survives_source_document_example_or_framework_changes(tmp_path):
    skill = tmp_path / "skill"
    shutil.copytree(
        SKILL_ROOT, skill, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache")
    )
    examples = tmp_path / "examples"
    example = examples / "basic_api/datamove/ub_l1_handover/kernel.py"
    example.parent.mkdir(parents=True)
    example.write_text("# example A\n")
    framework = tmp_path / "framework"
    for name in feedback.FRAMEWORK_FILES:
        target = framework / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PYPTO_ROOT / name, target)
    bridge = mcp_server.Bridge(
        workspace=tmp_path,
        entry=skill / "scripts/check.py",
        examples_root=examples,
        pypto_root=framework,
    )
    source = tmp_path / "source.py"
    source.write_text("value=1\n")

    def rewrite(path, old, new):
        stat = path.stat()
        text = path.read_text()
        assert old in text
        path.write_text(text.replace(old, new, 1))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    async def check():
        args = {
            "text": "UB to L1",
            "source": source.name,
            "max_output_bytes": 65536,
        }
        command, cwd = bridge.command("kernel_query", args)
        subprocess.run(command, cwd=cwd, capture_output=True, check=True)
        a = payload(await bridge.call("kernel_query", args))["report"]
        rewrite(source, "value=1", "value=2")
        rewrite(example, "example A", "example B")
        doc = skill / "references/api/datamove.md"
        rewrite(doc, "physical copy", "physical COPY")
        stub = framework / "python/pypto_pro/language/_api.py"
        stub.write_text(stub.read_text() + "\n# source variant\n")
        rewrite(
            skill / "scripts/_checks/query.py",
            'feedback.envelope("query", pypto_root, [source] if source else [])',
            'feedback.envelope("probe", pypto_root, [source] if source else [])',
        )
        b = payload(await bridge.call("kernel_query", args))["report"]
        assert a["stage"] == "query" and b["stage"] == "probe"
        assert a["identity"]["sources"] != b["identity"]["sources"]
        assert a["identity"]["selected_source"] != b["identity"]["selected_source"]
        assert (
            a["context"][0]["reference"]["sha256"]
            != b["context"][0]["reference"]["sha256"]
        )
        assert b["context"][0]["examples"][0]["text"] == "# example B\n"
        old = next(row for row in a["knowledge"]["items"] if row["kind"] == "example")
        current = next(
            row for row in b["knowledge"]["items"] if row["kind"] == "example"
        )
        assert (
            old["id"] == current["id"]
            and old["reference"]["sha256"] != current["reference"]["sha256"]
        )
        assert old["program"]["program_sha256"] != current["program"]["program_sha256"]

    asyncio.run(check())


def test_official_document_and_docstring_refresh_keep_pinned_reads_honest(tmp_path):
    root = tmp_path / "framework"
    initializer = root / "python/pypto_pro/language/__init__.py"
    initializer.parent.mkdir(parents=True)
    initializer.write_text(
        'raise RuntimeError("do not import")\n__all__=["sample"]\nfrom ._api import sample\n'
    )
    source = initializer.with_name("_api.py")
    source.write_text(
        'def sample():\n    """Source alpha."""\n    raise RuntimeError("do not execute")\n'
    )
    owner = root / "docs/zh/api/pro_api/sample.md"
    owner.parent.mkdir(parents=True)
    owner.write_text("# pypto_pro.language.sample\n\n## Example\nDocument alpha.\n")
    bridge = mcp_server.Bridge(workspace=tmp_path, pypto_root=root)

    async def check():
        args = {"text": "pl.sample", "kinds": ["document"]}
        first = payload(await bridge.call("kernel_query", args))["report"]["knowledge"]
        by_role = {row["document_role"]: row for row in first["items"]}
        for path in (source, owner):
            stamp = path.stat()
            path.write_bytes(path.read_bytes().replace(b"alpha", b"omega"))
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        second = payload(await bridge.call("kernel_query", args))["report"]["knowledge"]
        assert first["corpus_sha256"] != second["corpus_sha256"]
        for role in ("official_pro_api", "source_docstring"):
            rejected = payload(
                await bridge.call("kernel_query", by_role[role]["resolve"])
            )
            assert (
                rejected["exit_code"] == 2
                and rejected["report"]["knowledge"]["lookup"]["status"]
                == "source_changed"
            )
            current = next(
                row for row in second["items"] if row["document_role"] == role
            )
            resolved = payload(await bridge.call("kernel_query", current["resolve"]))[
                "report"
            ]["knowledge"]
            assert "omega" in resolved["content"]["items"][0]["text"]

    asyncio.run(check())


def test_metadata_admission_and_partial_queries_share_cli_results(knowledge_bundle):
    corpus, bridge = knowledge_bundle
    second = write(
        corpus.references / "second.md",
        '<!-- kernel-query: {"id":"second"} -->\n## Second\nBody\n',
    )

    async def matched(args):
        command, cwd = bridge.command("kernel_query", args)
        expected = subprocess.run(
            command, cwd=cwd, text=True, capture_output=True, timeout=30
        )
        data = payload(await bridge.call("kernel_query", args))
        assert (
            data["delivery"] == "complete" and data["exit_code"] == expected.returncode
        )
        assert data["report"] == json.loads(expected.stdout)
        return data["report"]

    async def check():
        assert (await matched({"validate": True}))["status"] == "passed"
        page = (await matched({"kinds": ["contract"], "limit": 1}))["knowledge"]
        stamp = second.stat()
        second.write_text(
            second.read_text().replace('"id":"second"', '"id":"second","facets":42')
        )
        os.utime(second, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        failed = await matched({"validate": True})
        assert failed["status"] == "failed"
        assert any(
            row["rule"] == "knowledge-metadata-invalid"
            and row["location"]["file"] == str(second)
            for row in failed["diagnostics"]
        )
        good = (await matched({"text": "contract:copy", "kinds": ["contract"]}))[
            "knowledge"
        ]
        assert (
            good["lookup"]["status"] == "matched"
            and good["lookup"]["coverage"] == "partial"
        )
        missing = (await matched({"text": "contract:second", "kinds": ["contract"]}))[
            "knowledge"
        ]
        assert missing["lookup"]["status"] == "source_unavailable"
        stale = (
            await matched(
                {
                    "kinds": ["contract"],
                    "limit": 1,
                    "cursor": page["page"]["next_cursor"],
                }
            )
        )["knowledge"]
        assert stale["lookup"]["status"] == "cursor_invalidated"
        second.unlink()
        assert (await matched({"validate": True}))["status"] == "passed"

    asyncio.run(check())


def test_document_and_program_lifecycle_stays_fresh_through_mcp(knowledge_bundle):
    corpus, bridge = knowledge_bundle

    async def call(args):
        return payload(await bridge.call("kernel_query", args))["report"]

    async def check():
        original = (await call({"text": "contract:copy", "kinds": ["contract"]}))[
            "knowledge"
        ]["items"][0]
        corpus.owner.rename(corpus.references / "renamed.md")
        changed = (await call({"text": "contract:copy", "kinds": ["contract"]}))[
            "knowledge"
        ]["items"][0]
        assert (
            original["id"] == changed["id"]
            and original["reference"]["path"] != changed["reference"]["path"]
        )
        assert (await call({"text": "document:skill/owner.md"}))["knowledge"]["lookup"][
            "status"
        ] == "source_unavailable"
        folder = corpus.examples / "algorithms/composition"
        folder.rename(folder.with_name("renamed"))
        assert (await call({"validate": True}))["status"] == "failed"
        corpus.catalog.write_text(
            corpus.catalog.read_text().replace(
                "algorithms/composition/", "algorithms/renamed/"
            )
        )
        assert (await call({"validate": True}))["status"] == "passed"
        resolved = (await call({"text": "example:algorithms/renamed"}))["knowledge"]
        assert set(resolved["items"][0]["program"]["entries"]) == {
            "kernel",
            "reference",
            "test",
        }
        assert (await call({"text": "example:algorithms/composition"}))["knowledge"][
            "lookup"
        ]["status"] == "source_unavailable"

    asyncio.run(check())


def test_real_stdio_discovery_and_budget(tmp_path):
    async def check():
        params = StdioServerParameters(
            command=sys.executable,
            args=[
                str(mcp_server.ENTRY),
                "--pypto-root",
                str(PYPTO_ROOT),
                "mcp",
                "--workspace",
                str(tmp_path),
            ],
        )
        async with Client(stdio_client(params)) as client:
            tools = (await client.list_tools()).tools
            assert {t.name for t in tools} == set(mcp_server.BY_NAME)
            assert (
                next(
                    t for t in tools if t.name == "kernel_run"
                ).annotations.read_only_hint
                is False
            )
            result = await client.call_tool(
                "kernel_query", {"text": "从UB向L1搬运", "max_output_bytes": 65536}
            )
            assert payload(result)["report"]["context"][0]["id"] == "ub-l1"
            schema = next(t for t in tools if t.name == "kernel_query").input_schema
            assert schema["properties"]["filters"] == mcp_server.knowledge.FILTER_SCHEMA
            assert schema["properties"]["validate"]["const"] is True
            result = await client.call_tool("kernel_query", {"query": "vf.reduce_sum"})
            invalid = payload(result)
            assert result.is_error and invalid["delivery"] == "invalid_arguments"
            assert invalid["report"] is None and invalid["input_schema"] == schema
            valid = payload(
                await client.call_tool(
                    "kernel_query", {"text": "pl.DT_FP16", "limit": 5.0}
                )
            )
            assert valid["delivery"] == "complete" and valid["exit_code"] == 0
            assert invalid["budget"]["result_bytes"] <= mcp_server.DEFAULT_BUDGET
            admitted = payload(
                await client.call_tool("kernel_query", {"validate": True})
            )["report"]
            assert (
                admitted["stage"] == "knowledge_admission"
                and admitted["status"] == "passed"
            )
            result = await client.call_tool(
                "kernel_query", {"text": "pl.DT_FP16", "kinds": ["constant"]}
            )
            view = payload(result)["report"]["knowledge"]
            assert (
                not result.is_error
                and view["items"][0]["kind"] == "constant"
                and view["items"][0]["callable"] is False
            )
            args = {
                "kinds": ["contract"],
                "limit": 1,
                "filters": {"source_memory": "UB", "destination_memory": "L1"},
            }
            result = await client.call_tool("kernel_query", args)
            assert (
                payload(result)["report"]["knowledge"]["items"][0]["id"]
                == "contract:ub-l1"
            )
            args = {"kinds": ["contract"], "limit": 1}
            page = payload(await client.call_tool("kernel_query", args))["report"][
                "knowledge"
            ]
            following = payload(
                await client.call_tool(
                    "kernel_query", {**args, "cursor": page["page"]["next_cursor"]}
                )
            )["report"]["knowledge"]
            assert page["items"][0]["id"] != following["items"][0]["id"]
            command, cwd = mcp_server.Bridge(workspace=tmp_path).command(
                "kernel_query", {**args, "cursor": page["page"]["next_cursor"]}
            )
            cli = subprocess.run(
                command, cwd=cwd, capture_output=True, text=True, timeout=30
            )
            assert json.loads(cli.stdout)["knowledge"] == following
            discovery = payload(
                await client.call_tool(
                    "kernel_query", {"text": "vf.addc", "kinds": ["document"]}
                )
            )["report"]["knowledge"]
            original = next(
                item
                for item in discovery["items"]
                if item["document_role"] == "official_pro_api"
            )
            resolved = payload(
                await client.call_tool(
                    "kernel_query",
                    {**original["resolve"], "text": original["id"] + "#调用示例"},
                )
            )["report"]["knowledge"]
            assert "carry = vf.create_mask(" in resolved["content"]["items"][0]["text"]
            assert resolved["items"][0]["evidence"]["kind"] == "official_documentation"
            read = payload(
                await client.call_tool(
                    "kernel_read",
                    {
                        "paths": [original["reference"]["path"]],
                        "fragment": "调用示例",
                        "source_sha256": original["reference"]["sha256"],
                    },
                )
            )["report"]
            assert read["items"][0]["text"] == resolved["content"]["items"][0]["text"]
            programs = payload(
                await client.call_tool(
                    "kernel_query",
                    {"text": "带M尾块的批量转置GEMM", "kinds": ["example"]},
                )
            )["report"]["knowledge"]
            (program,) = programs["items"]
            arguments = dict(program["resolve"])
            response = await client.call_tool("kernel_query", arguments)
            resolved = payload(response)
            if resolved["delivery"] == "budget_exceeded":
                assert response.is_error and resolved["report"] is None
                retry = resolved["read_only_retry"]
                assert retry["tool"] == "kernel_query"
                assert retry["preserve_other_arguments"] is True
                assert (
                    retry["arguments_patch"]["max_output_bytes"]
                    > resolved["budget"]["limit_bytes"]
                )
                arguments.update(retry["arguments_patch"])
                response = await client.call_tool("kernel_query", arguments)
                resolved = payload(response)
            assert not response.is_error and resolved["delivery"] == "complete"
            resolved_program = resolved["report"]["knowledge"]
            assert set(resolved_program["items"][0]["program"]["entries"]) == {
                "kernel",
                "reference",
                "test",
            }
            reference = resolved_program["items"][0]["program"]["read_requests"][
                "reference"
            ]
            source = payload(
                await client.call_tool("kernel_read", {"requests": [reference]})
            )["report"]
            assert "torch.matmul" in source["items"][0]["text"]
            result = await client.call_tool(
                "kernel_query", {"text": "UB to L1", "max_output_bytes": 2048}
            )
            assert result.is_error and payload(result)["delivery"] == "budget_exceeded"
            source = tmp_path / "source.py"
            source.write_text("value = 7\n")
            result = await client.call_tool("kernel_read", {"paths": [source.name]})
            assert payload(result)["report"]["items"][0]["text"] == source.read_text()
            result = await client.call_tool(
                "kernel_search", {"paths": [source.name], "pattern": "value"}
            )
            assert payload(result)["report"]["items"][0]["line"] == 1
            import hashlib

            manifest = tmp_path / "task-context.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "pypto-task-context/1",
                        "artifacts": [
                            {
                                "role": "source",
                                "path": source.name,
                                "sha256": hashlib.sha256(
                                    source.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                    }
                )
            )
            result = await client.call_tool("kernel_query", {"task": manifest.name})
            assert payload(result)["report"]["items"][0]["status"] == "verified_bytes"
            manifest.write_text("Find the applicable complete program.\n")
            result = await client.call_tool("kernel_query", {"task": manifest.name})
            assert payload(result)["exit_code"] == 2
            assert (
                "pypto-task-context/1 JSON inventory"
                in payload(result)["report"]["error"]
            )
            literal = tmp_path / "literal.py"
            literal.write_text("one = vf.full(1.0, preg)\n")
            result = await client.call_tool(
                "kernel_search", {"paths": [literal.name], "pattern": "vf.full("}
            )
            assert payload(result)["exit_code"] == 2
            assert "MCP fixed=true" in payload(result)["report"]["error"]
            result = await client.call_tool(
                "kernel_search",
                {"paths": [literal.name], "pattern": "vf.full(", "fixed": True},
            )
            assert payload(result)["report"]["items"][0]["line"] == 1
            result = await client.call_tool(
                "kernel_read", {"requests": [{"path": source.name, "max_lines": 1}]}
            )
            assert payload(result)["report"]["items"][0]["request_index"] == 0
            receipt = tmp_path / "receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "request_id": "original",
                        "source_sha256": hashlib.sha256(
                            source.read_bytes()
                        ).hexdigest(),
                        "state": "completed",
                        "exit_code": 7,
                    }
                )
            )
            result = await client.call_tool(
                "kernel_status",
                {
                    "receipt": receipt.name,
                    "source": source.name,
                    "request_id": "original",
                    "wait_seconds": 0,
                },
            )
            assert payload(result)["report"]["outcome"] == "failed"
            claim = tmp_path / "claim.json"
            claim.write_text(
                json.dumps(
                    {
                        "schema": "pypto-performance-claim/1",
                        "claim": {
                            "speedup_established": True,
                            "workload": {"rows": 1},
                            "metric": "synchronized_wall_us",
                            "changes_wrapper": False,
                        },
                    }
                )
            )
            result = await client.call_tool("kernel_claim", {"spec": claim.name})
            assert payload(result)["report"]["supported_speedup"] is None
            assert (
                payload(result)["delivery"] == "complete"
                and payload(result)["exit_code"] == 2
            )

    asyncio.run(check())


async def raw_start(tmp_path):
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(mcp_server.ENTRY),
        "mcp",
        "--workspace",
        str(tmp_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await raw_send(
        proc,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "legacy-test", "version": "1"},
            },
        },
    )
    response = await raw_read(proc, 1)
    assert response["result"]["protocolVersion"] == "2025-11-25"
    await raw_send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    return proc


async def raw_send(proc, message):
    proc.stdin.write((json.dumps(message) + "\n").encode())
    await proc.stdin.drain()


async def raw_read(proc, request_id):
    while True:
        line = await asyncio.wait_for(proc.stdout.readline(), 10)
        assert line, (await proc.stderr.read()).decode()
        result = json.loads(line)
        if result.get("id") == request_id:
            return result


def test_legacy_host_gets_clean_jsonrpc_and_original_report(tmp_path):
    async def check():
        proc = await raw_start(tmp_path)
        try:
            await raw_send(
                proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}
            )
            tools = (await raw_read(proc, 3))["result"]["tools"]
            definition = next(tool for tool in tools if tool["name"] == "kernel_query")
            assert "inputSchema" in definition and "input_schema" not in definition
            assert (
                definition["inputSchema"]
                == mcp_server.BY_NAME["kernel_query"].input_schema
            )
            assert definition["inputSchema"]["properties"]["text"]["type"] == "string"
            await raw_send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "kernel_query",
                        "arguments": {"text": "UB to L1", "max_output_bytes": 65536},
                    },
                },
            )
            wire = (await raw_read(proc, 2))["result"]
            data = json.loads(wire["content"][0]["text"])
            assert (
                data["delivery"] == "complete"
                and data["report"]["context"][0]["id"] == "ub-l1"
            )
            actual = len(
                json.dumps(wire, ensure_ascii=False, separators=(",", ":")).encode()
            )
            normalized = len(
                types.CallToolResult.model_validate(wire)
                .model_dump_json(by_alias=True, exclude_none=True)
                .encode()
            )
            assert (
                actual
                <= normalized
                == data["budget"]["result_bytes"]
                <= data["budget"]["limit_bytes"]
            )
        finally:
            proc.stdin.close()
            await asyncio.wait_for(proc.wait(), 10)

    asyncio.run(check())


def test_process_start_failure_is_a_tool_error(tmp_path):
    bridge = mcp_server.Bridge(workspace=tmp_path, python=tmp_path / "missing-python")
    result = asyncio.run(bridge.call("kernel_query", {"text": "pl.move"}))
    assert result.is_error and payload(result)["delivery"] == "adapter_error"


def test_selected_virtual_environment_is_preserved(tmp_path):
    env = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(env)],
        check=True,
        timeout=20,
    )
    entry = tmp_path / "identity.py"
    entry.write_text("import sys,json\nprint(json.dumps({'prefix':sys.prefix}))\n")
    bridge = mcp_server.Bridge(
        workspace=tmp_path, python=env / "bin/python", entry=entry
    )
    data = payload(asyncio.run(bridge.call("kernel_query", {"text": "probe"})))
    assert data["report"]["prefix"] == str(env)


def test_capture_ceiling_does_not_return_a_partial_json_report(tmp_path):
    entry = tmp_path / "large.py"
    entry.write_text(
        f"import json\nprint(json.dumps({{'payload': 'x' * {mcp_server.CAPTURE_LIMIT + 1024}}}))\n"
    )
    bridge = mcp_server.Bridge(workspace=tmp_path, entry=entry)
    result = asyncio.run(bridge.call("kernel_query", {"text": "probe"}))
    data = payload(result)
    assert data["delivery"] == "capture_limit_exceeded" and data["report"] is None
    assert (
        data["cli_stdout_bytes"] > mcp_server.CAPTURE_LIMIT
        and data["cli_stdout_sha256"] is None
    )
    assert result.is_error


def sleeper(tmp_path):
    script = tmp_path / "sleep.py"
    script.write_text(
        "import os,signal,time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "child=os.fork()\n"
        "Path('child.pid' if child==0 else 'parent.pid').write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    return script


async def started(tmp_path):
    for _ in range(200):
        if (tmp_path / "parent.pid").exists() and (tmp_path / "child.pid").exists():
            return
        await asyncio.sleep(0.02)
    pytest.fail("Owned subprocesses did not start")


async def stopped(tmp_path):
    ps = shutil.which("ps")
    assert ps is not None, "ps is required to check descendant process state"
    ps = os.path.abspath(ps)
    for _ in range(100):
        states = [
            subprocess.run(
                [ps, "-o", "stat=", "-p", (tmp_path / n).read_text()],
                capture_output=True,
                text=True,
            ).stdout.strip()
            for n in ("parent.pid", "child.pid")
        ]
        if all(not s or s.startswith("Z") for s in states):
            return
        await asyncio.sleep(0.02)
    pytest.fail(f"Owned subprocesses still running: {states}")


@pytest.mark.parametrize("stop", ["timeout", "cancel"])
def test_timeout_and_cancellation_kill_owned_descendants(
    bridge, source, tmp_path, stop
):
    script = sleeper(tmp_path)

    async def check():
        task = asyncio.create_task(
            bridge.call(
                "kernel_run",
                {
                    "sources": [source.name],
                    "argv": [sys.executable, str(script)],
                    "timeout_seconds": 2 if stop == "timeout" else 30,
                },
            )
        )
        await started(tmp_path)
        if stop == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            data = payload(await task)
            assert (
                data["delivery"] == "timeout"
                and data["exit_code"] == 124
                and data["report"] is None
            )
        await stopped(tmp_path)
        assert not bridge.processes

    asyncio.run(check())


@pytest.mark.parametrize("stop", ["eof", "signal"])
def test_host_disconnect_or_server_termination_cleans_execution(source, tmp_path, stop):
    script = sleeper(tmp_path)

    async def check():
        proc = await raw_start(tmp_path)
        try:
            await raw_send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "kernel_run",
                        "arguments": {
                            "sources": [source.name],
                            "argv": [sys.executable, str(script)],
                        },
                    },
                },
            )
            await started(tmp_path)
            if stop == "eof":
                proc.stdin.close()
            else:
                proc.terminate()
            await asyncio.wait_for(proc.wait(), 10)
            await stopped(tmp_path)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    asyncio.run(check())


def test_protocol_cancellation_stops_execution_and_keeps_server_usable(
    source, tmp_path
):
    script = sleeper(tmp_path)

    async def check():
        proc = await raw_start(tmp_path)
        try:
            await raw_send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "kernel_run",
                        "arguments": {
                            "sources": [source.name],
                            "argv": [sys.executable, str(script)],
                        },
                    },
                },
            )
            await started(tmp_path)
            await raw_send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": 2, "reason": "test cancellation"},
                },
            )
            await stopped(tmp_path)
            await raw_send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "kernel_query",
                        "arguments": {"text": "pl.insert", "max_output_bytes": 65536},
                    },
                },
            )
            data = json.loads((await raw_read(proc, 3))["result"]["content"][0]["text"])
            assert data["delivery"] == "complete"
        finally:
            proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), 10)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()

    asyncio.run(check())
