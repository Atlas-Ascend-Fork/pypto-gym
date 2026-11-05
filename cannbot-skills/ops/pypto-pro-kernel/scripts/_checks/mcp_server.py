# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Local stdio adapter to the existing JSON CLI, with no semantic/result cache.

Each call uses a fresh CLI interpreter. Native imports and explicit commands run
with the server account's permissions; this adapter is not a filesystem sandbox.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import signal
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import anyio
from jsonschema import Draft202012Validator
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .paths import EXAMPLES_ROOT, PYPTO_ROOT, SKILL_ROOT
from .actions import GOALS
from . import knowledge

DEFAULT_BUDGET = 32768
MAX_BUDGET = 262144
CAPTURE_LIMIT = 8 * 1024 * 1024
STDERR_LIMIT = 8192
ENTRY = SKILL_ROOT / "scripts/check.py"
SCHEMA = "pypto-mcp-result/1"
SERVER_NAME, SERVER_VERSION = "pypto-pro-kernel", "0.1.0"


def _string(description):
    return {"type": "string", "minLength": 1, "description": description}


def _describe_numeric_bounds(schema):
    """Keep numeric limits visible in hosts that render schemas as scalar types."""
    schema = dict(schema)
    if schema.get("type") in ("integer", "number"):
        bounds = [
            f"{key}: {schema[key]}"
            for key in ("minimum", "maximum", "default")
            if key in schema
        ]
        if bounds:
            schema["description"] = " ".join(
                filter(None, (schema.get("description"), "; ".join(bounds) + "."))
            )
    if "properties" in schema:
        schema["properties"] = {
            key: _describe_numeric_bounds(child)
            for key, child in schema["properties"].items()
        }
    if isinstance(schema.get("items"), dict):
        schema["items"] = _describe_numeric_bounds(schema["items"])
    return schema


def _tool(name, description, properties, required, *, read_only):
    properties = {
        **properties,
        "max_output_bytes": {
            "type": "integer",
            "minimum": 2048,
            "maximum": MAX_BUDGET,
            "default": DEFAULT_BUDGET,
            "description": "Maximum serialized MCP result bytes; not tokens. Narrow source scope before increasing.",
        },
    }
    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    schema = _describe_numeric_bounds(schema)
    return types.Tool(
        name=name,
        description=description,
        input_schema=schema,
        annotations=types.ToolAnnotations(
            read_only_hint=read_only,
            destructive_hint=not read_only,
            idempotent_hint=read_only,
            open_world_hint=not read_only,
        ),
    )


_SOURCES = {
    "type": "array",
    "minItems": 1,
    "maxItems": 64,
    "items": _string(
        "Python file or directory; relative to the operation's working directory."
    ),
}
_DETAILS = {
    "type": "boolean",
    "default": False,
    "description": "Expand the existing CLI's identity/descriptor fields.",
}
_GOAL = {
    "type": "string",
    "enum": list(GOALS),
    "description": "Explicit action goal; adds guidance, not execution or acceptance.",
}
_READ_LINES = {
    "type": "integer",
    "minimum": 1,
    "description": "Requested line count. Large requests are paged by the active read policy; follow the returned next continuation.",
}
TOOLS = [
    _tool(
        "kernel_search",
        "Search explicit source paths with bounded files, scanned bytes and reply bytes. A partial search cannot establish absence.",
        {
            "paths": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "items": _string("Explicit file or directory."),
            },
            "pattern": _string(
                "Regular expression by default; use fixed=true for literal code such as 'vf.full('. Omitted only for file listing."
            ),
            "glob": _string("Optional file glob."),
            "files": {"type": "boolean", "default": False},
            "ignore_case": {"type": "boolean", "default": False},
            "fixed": {"type": "boolean", "default": False},
        },
        ["paths"],
        read_only=True,
    ),
    _tool(
        "kernel_read",
        'Read bounded UTF-8 source pages with hashes, raw text and numbered_text (absolute line|source) for citations. start_column_bytes marks partial-line continuations. Use paths, e.g. {"paths":["kernel.py"]}, or requests for per-file ranges; choose one. A changed hash invalidates continuation.',
        {
            "paths": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "items": _string("Explicit source file."),
            },
            "requests": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "items": {
                    "type": "object",
                    "required": ["path"],
                    "additionalProperties": False,
                    "properties": {
                        "path": _string("Explicit file for this range."),
                        "start_line": {"type": "integer", "minimum": 1},
                        "max_lines": _READ_LINES,
                        "fragment": {
                            **_string(
                                "Markdown heading slug or explicit anchor; section-bound continuation."
                            ),
                            "maxLength": 2048,
                        },
                        "offset_bytes": {"type": "integer", "minimum": 0},
                        "source_sha256": _string("Continuation source SHA."),
                    },
                },
            },
            "start_line": {"type": "integer", "minimum": 1, "default": 1},
            "max_lines": _READ_LINES,
            "offset_bytes": {"type": "integer", "minimum": 0},
            "fragment": {
                **_string("Markdown heading slug or explicit anchor; use one path."),
                "maxLength": 2048,
            },
            "source_sha256": _string("Required source hash for byte continuation."),
        },
        [],
        read_only=True,
    ),
    _tool(
        "kernel_status",
        "Boundedly wait on the existing request and checked source; never launch/retry. Process outcome alone proves no native/numerical/performance result.",
        {
            "receipt": _string("Existing controller receipt JSON path."),
            "request_id": _string("Pinned existing request ID."),
            "source": _string("Checked source file."),
            "wait_seconds": {
                "type": "number",
                "minimum": 0,
                "maximum": 30,
                "default": 20,
            },
        },
        ["receipt", "request_id", "source"],
        read_only=True,
    ),
    _tool(
        "kernel_query",
        'Discover typed knowledge and exact public symbols through shared check.py query. Start with a page, e.g. {"text":"pl.DT_FP16","limit":1}, then follow original references. Omit kinds unless the category is known; api excludes constants and enum members. For an official API page use kinds ["document"] and check document_role. Routing conditions are not runtime support; a miss is not API absence.',
        {
            "text": {
                **_string(
                    "Exact symbol/knowledge ID or question, e.g. pl.system.bar_all, pl.DT_FP16, UB to L1."
                ),
                "maxLength": 4096,
            },
            "kinds": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "enum": list(knowledge.KINDS)},
            },
            "filters": knowledge.FILTER_SCHEMA,
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": knowledge.MAX_LIMIT,
                "description": f"Objects per page; default {knowledge.DEFAULT_LIMIT}.",
            },
            "cursor": {
                **_string(
                    "Read-only continuation; repeat text, kinds and filters. Source/query changes require restarting."
                ),
                "maxLength": 4096,
            },
            "reference_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
                "description": "Pin a document/docstring or example kernel ID resolution to its discovered source SHA.",
            },
            "task": _string(
                "pypto-task-context/1 JSON inventory; for a prose question use text instead. No directory discovery."
            ),
            "validate": {
                "type": "boolean",
                "const": True,
                "description": "Read-only knowledge admission instead of discovery or task inventory; validates identity, metadata, relations and existing maintenance gates.",
            },
            "source": _string("Optional Python source for local transfer facts."),
            "details": _DETAILS,
            "goal": _GOAL,
        },
        [],
        read_only=True,
    ),
    _tool(
        "kernel_check",
        "Run check.py rules on explicit sources. Advisory/unknown facts are not numerical acceptance.",
        {"sources": _SOURCES, "details": _DETAILS},
        ["sources"],
        read_only=True,
    ),
    _tool(
        "kernel_native",
        "Run check.py native using a concrete metadata spec. Imports the entry and parses native IR; no device launch. Module imports can have side effects.",
        {
            "spec": _string(
                "Native metadata JSON file; entry paths resolve beside it."
            ),
            "details": _DETAILS,
        },
        ["spec"],
        read_only=False,
    ),
    _tool(
        "kernel_run",
        "Source-gate then execute an explicit argv through check.py run. Preserves exit codes and targeted failure context. Use the original evaluator; exit zero alone proves no numerical/performance result.",
        {
            "sources": _SOURCES,
            "argv": {
                "type": "array",
                "minItems": 1,
                "maxItems": 256,
                "items": {"type": "string"},
            },
            "cwd": _string(
                "Working directory, relative to the server workspace. Defaults to that workspace."
            ),
            "native_spec": _string(
                "Optional native metadata preflight before executing argv."
            ),
            "details": _DETAILS,
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 3600,
                "description": "Whole CLI/child timeout; bounded by the server timeout ceiling.",
            },
        },
        ["sources", "argv"],
        read_only=False,
    ),
    _tool(
        "kernel_suggest",
        "Return bounded conditional optimization candidates via check.py suggest, optionally ranked by source-bound measured task costs.",
        {
            "sources": _SOURCES,
            "measurements": _string("Optional measured-cost JSON file."),
            "limit": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
            "details": _DETAILS,
            "goal": _GOAL,
        },
        ["sources"],
        read_only=True,
    ),
    _tool(
        "kernel_explain",
        "Inspect source-bound contracts, diagnostic logs and work-site differences through check.py explain. Shares execution diagnostic routing; no code execution or authenticated execution result.",
        {
            "source": _string("Candidate Python file."),
            "baseline": _string("Optional baseline Python file."),
            "error_log": _string("Optional diagnostic log."),
            "function": _string("Optional function or qualified nested name."),
        },
        ["source"],
        read_only=True,
    ),
    _tool(
        "kernel_experiment",
        "Read or update the existing best/current experiment state. A valid operation can reject a candidate: inspect phase, best.verified and next_action, not just exit_code.",
        {
            "state": _string("Explicit scratch state JSON file."),
            "action": {
                "type": "string",
                "enum": ["show", "init", "select", "observe", "export_best"],
                "default": "show",
            },
            "input": _string(
                "Plan/selection/observation JSON, or new export directory. Required except for show."
            ),
        },
        ["state"],
        read_only=False,
    ),
    _tool(
        "kernel_claim",
        "Check an explicit performance assertion against source-bound workload/metric evidence through shared experiment gates. Reads attestations; no execution or independent authentication.",
        {"spec": _string("Performance claim JSON specification.")},
        ["spec"],
        read_only=True,
    ),
]
BY_NAME = {tool.name: tool for tool in TOOLS}
BY_NAME["kernel_read"].input_schema["oneOf"] = [
    {"required": ["paths"], "not": {"required": ["requests"]}},
    {
        "required": ["requests"],
        "not": {
            "anyOf": [
                {"required": [key]}
                for key in (
                    "paths",
                    "start_line",
                    "max_lines",
                    "offset_bytes",
                    "source_sha256",
                    "fragment",
                )
            ]
        },
    },
]
BY_NAME["kernel_query"].input_schema["oneOf"] = [
    {
        "anyOf": [{"required": [key]} for key in ("text", "kinds", "filters")],
        "not": {"anyOf": [{"required": ["task"]}, {"required": ["validate"]}]},
    },
    {
        "required": ["task"],
        "not": {
            "anyOf": [
                {"required": [key]}
                for key in (
                    "text",
                    "source",
                    "kinds",
                    "filters",
                    "limit",
                    "cursor",
                    "reference_sha256",
                    "validate",
                )
            ]
        },
    },
    {
        "required": ["validate"],
        "not": {
            "anyOf": [
                {"required": [key]}
                for key in (
                    "text",
                    "source",
                    "task",
                    "kinds",
                    "filters",
                    "limit",
                    "cursor",
                    "reference_sha256",
                    "goal",
                )
            ]
        },
    },
]


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _argument_error(exc):
    detail = {"path": list(exc.absolute_path), "constraint": exc.validator}
    if exc.validator_value is None or type(exc.validator_value) in (
        str,
        int,
        float,
        bool,
    ):
        detail["expected"] = exc.validator_value
    if exc.validator == "additionalProperties" and detail["path"]:
        detail["allowed_fields"] = sorted(exc.schema.get("properties", {}))
    return detail


def _integer_arguments(value, schema):
    """JSON integer values may use floating syntax; preserve their CLI meaning."""
    if schema.get("type") == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict) and "properties" in schema:
        properties = schema.get("properties", {})
        return {
            key: _integer_arguments(item, properties[key])
            if key in properties
            else item
            for key, item in value.items()
        }
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        if len(value) > schema.get("maxItems", len(value)):
            return value
        return [_integer_arguments(item, schema["items"]) for item in value]
    return value


def _encoded(body, error):
    """One JSON text block avoids duplicating a potentially large structured result."""
    result = types.CallToolResult(
        content=[types.TextContent(type="text", text=_json(body))],
        is_error=error,
        meta={
            "io.modelcontextprotocol/serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            }
        },
    )
    return result, len(
        result.model_dump_json(by_alias=True, exclude_none=True).encode()
    )


def _query_budget_retry(report, arguments, size, limit):
    """Describe one optional read-only recovery without executing it."""
    if not isinstance(arguments, dict):
        return None
    returned = report.get("knowledge", {}).get("page", {}).get("returned", 0)
    if returned > 1 and not arguments.get("cursor"):
        patch = {"limit": 1}
        instruction = "Read one object, then follow its original references. A single oversized object requires more reply bytes, not a lower object limit."
    else:
        target = max(limit * 2, ((size + 1024 + 2047) // 2048) * 2048)
        if target > MAX_BUDGET:
            return None
        patch = {"max_output_bytes": target}
        instruction = "Retry this read-only query with the shown byte budget. Lowering limit cannot shrink a single object."
    return {
        "tool": "kernel_query",
        "arguments_patch": patch,
        "preserve_other_arguments": True,
        "instruction": instruction + " No retry was executed.",
    }


def reply(body, limit, *, tool_arguments=None):
    body = copy.deepcopy(body)
    body["schema"] = SCHEMA
    body["budget"] = {"limit_bytes": limit, "result_bytes": 0, "context_tokens": None}
    for _ in range(12):
        error = body.get("exit_code") != 0 or body.get("delivery") != "complete"
        result, size = _encoded(body, error)
        if size > limit:
            if body.get("delivery") == "invalid_arguments":
                if body.get("input_schema") is not None:
                    body.pop("input_schema")
                    body["input_schema_omitted"] = "output_budget"
                    body["next_action"] = (
                        "No CLI process was started. Correct the named arguments using the tool's published input schema."
                    )
                elif body.get("error"):
                    body["error"] = body["error"][: len(body["error"]) // 2]
                    body["error_truncated"] = True
                continue
            if body.get("stderr_tail"):
                body["stderr_tail"] = ""
                body["stderr_truncated"] = True
                continue
            report = body.get("report")
            if report is not None:
                body["report_summary"] = {
                    k: report[k]
                    for k in ("schema", "stage", "status", "phase")
                    if k in report
                }
                if body.get("operation") == "kernel_query":
                    retry = _query_budget_retry(report, tool_arguments, size, limit)
                    if retry:
                        body["read_only_retry"] = retry
                body["report"] = None
            body["delivery"] = "budget_exceeded"
            body["next_action"] = (
                "The operation already ran; the report exceeds this output budget. "
                "For read-only operations narrow the source scope or raise max_output_bytes. "
                "Do not repeat run/state mutations to retrieve output: inspect the command's artifacts or show the current experiment state."
            )
            # A malformed request can contain an arbitrarily long property name.
            if body.get("error"):
                body["error"] = body["error"][:400]
        elif body["budget"]["result_bytes"] == size:
            return result
        body["budget"]["result_bytes"] = size
    raise RuntimeError("Could not fit the MCP response into its declared budget")


class Bridge:
    """Only translates tool parameters and supervises the CLI process tree."""

    def __init__(
        self,
        *,
        workspace,
        pypto_root=PYPTO_ROOT,
        examples_root=EXAMPLES_ROOT,
        python=None,
        timeout_seconds=120,
        entry=ENTRY,
        context_policy=None,
        cards_root=None,
    ):
        self.workspace = Path(workspace).resolve()
        if not self.workspace.is_dir():
            raise ValueError("MCP workspace must be an existing directory")
        if not 0.1 <= timeout_seconds <= 3600:
            raise ValueError("MCP timeout ceiling must be between 0.1 and 3600 seconds")
        self.timeout_seconds = timeout_seconds
        self.entry = Path(entry).resolve()
        self.context_policy = Path(context_policy).resolve() if context_policy else None
        if self.context_policy:
            from .host_io import policy

            policy(self.context_policy)
        # Resolving a venv's Python symlink discards that interpreter's environment.
        self.python = str(Path(python or sys.executable).absolute())
        self.prefix = [
            self.python,
            str(self.entry),
            "--pypto-root",
            str(Path(pypto_root).resolve()),
            "--examples-root",
            str(Path(examples_root).resolve()),
        ]
        if cards_root is not None:
            self.prefix += [
                "--cards-root",
                str(Path(cards_root).expanduser().resolve()),
            ]
        self.processes = set()
        self.mutations = asyncio.Lock()
        self.readers = asyncio.Semaphore(4)

    @staticmethod
    async def terminate(proc):
        """Terminate the owned POSIX process group, including ordinary descendants."""
        with anyio.CancelScope(shield=True):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(proc.wait(), 1)
            except asyncio.TimeoutError:
                pass
            # The leader can exit while a descendant ignores TERM.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()

    def command(self, name, args):
        cwd = (self.workspace / args.get("cwd", ".")).resolve()
        if not cwd.is_dir():
            raise ValueError("Working directory does not exist")

        def path(value):
            if "\x00" in value:
                raise ValueError("Paths cannot contain NUL bytes")
            return str((cwd / value).resolve())

        command = list(self.prefix)
        if name in {"kernel_search", "kernel_read", "kernel_status"}:
            command += [
                {
                    "kernel_search": "search",
                    "kernel_read": "read",
                    "kernel_status": "status",
                }[name]
            ]
            if self.context_policy:
                command += ["--policy", str(self.context_policy)]
            if name == "kernel_status":
                command += [
                    "--receipt",
                    path(args["receipt"]),
                    "--source",
                    path(args["source"]),
                    "--request-id",
                    args["request_id"],
                    "--wait-seconds",
                    str(args.get("wait_seconds", 20)),
                ]
            else:
                for item in args.get("paths", []):
                    command += ["--path", path(item)]
                for request in args.get("requests", []):
                    command += [
                        "--request",
                        _json({**request, "path": path(request["path"])}),
                    ]
                for key in (
                    "pattern",
                    "glob",
                    "start_line",
                    "max_lines",
                    "offset_bytes",
                    "source_sha256",
                    "fragment",
                ):
                    if key in args:
                        command += ["--" + key.replace("_", "-") + "=" + str(args[key])]
                for key in ("files", "ignore_case", "fixed"):
                    if args.get(key):
                        command.append("--" + key.replace("_", "-"))
        elif name == "kernel_query":
            command += ["query", "--format", "json"]
            if args.get("validate"):
                command += ["--validate"]
            if "goal" in args:
                command += ["--goal", args["goal"]]
            if "task" in args:
                command += ["--task", path(args["task"])]
                if self.context_policy:
                    command += ["--policy", str(self.context_policy)]
            if "source" in args:
                command += ["--source", path(args["source"])]
            for kind in args.get("kinds", []):
                command += ["--kind", kind]
            if "filters" in args:
                command += ["--filters", _json(args["filters"])]
            for key in ("limit", "cursor", "reference_sha256"):
                if key in args:
                    command += ["--" + key.replace("_", "-") + "=" + str(args[key])]
            if args.get("details"):
                command += ["--details"]
            if "text" in args:
                command += ["--", args["text"]]
        elif name == "kernel_check":
            command += ["rules", "--format", "json"] + (
                ["--details"] if args.get("details") else []
            )
            command += [path(p) for p in args["sources"]]
        elif name == "kernel_native":
            command += ["native", "--spec", path(args["spec"]), "--format", "json"]
            if args.get("details"):
                command += ["--details"]
        elif name == "kernel_claim":
            command += ["claim", "--spec", path(args["spec"])]
        elif name in {"kernel_suggest", "kernel_run"}:
            command += ["suggest" if name == "kernel_suggest" else "run"]
            for source in args["sources"]:
                command += ["--source", path(source)]
            if args.get("details"):
                command += ["--details"]
            if name == "kernel_suggest":
                command += ["--limit", str(args.get("limit", 3))]
                if "goal" in args:
                    command += ["--goal", args["goal"]]
                if "measurements" in args:
                    command += ["--measurements", path(args["measurements"])]
            else:
                command += ["--format", "json"]
                if "native_spec" in args:
                    command += ["--native-spec", path(args["native_spec"])]
                command += ["--", *args["argv"]]
        elif name == "kernel_explain":
            command += ["explain", "--source", path(args["source"])]
            for key in ("baseline", "error_log"):
                if key in args:
                    command += ["--" + key.replace("_", "-"), path(args[key])]
            if "function" in args:
                command += ["--function=" + args["function"]]
        elif name == "kernel_experiment":
            command += ["experiment", "--state", path(args["state"])]
            action = args.get("action", "show")
            if (action == "show") != ("input" not in args):
                raise ValueError(
                    "input is required for state mutations and must be omitted for show"
                )
            if action != "show":
                command += ["--" + action.replace("_", "-"), path(args["input"])]
        return command, cwd

    async def execute(self, command, cwd, timeout, env):
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            start_new_session=True,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self.processes.add(proc)
        out, err = bytearray(), bytearray()
        counts = [0, 0]

        async def read(stream, buffer, index):
            while chunk := await stream.read(8192):
                counts[index] += len(chunk)
                if index == 0:
                    buffer.extend(chunk[: max(0, CAPTURE_LIMIT - len(buffer))])
                else:
                    buffer.extend(chunk)
                    del buffer[:-STDERR_LIMIT]

        readers = [
            asyncio.create_task(read(proc.stdout, out, 0)),
            asyncio.create_task(read(proc.stderr, err, 1)),
        ]
        pending = asyncio.gather(proc.wait(), *readers)
        timed_out = False
        try:
            await asyncio.wait_for(asyncio.shield(pending), timeout)
        except asyncio.TimeoutError:
            timed_out = True
            await self.terminate(proc)
        except BaseException:
            await self.terminate(proc)
            raise
        finally:
            for reader in readers:
                if not reader.done():
                    reader.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(pending, return_exceptions=True)
            self.processes.discard(proc)
        code = (
            124
            if timed_out
            else (proc.returncode if proc.returncode >= 0 else 128 - proc.returncode)
        )
        delivery, report = "complete", None
        if not timed_out and counts[0] <= CAPTURE_LIMIT:
            try:
                report = json.loads(out)
                if not isinstance(report, dict):
                    report = None
            except ValueError:
                pass
        if timed_out:
            delivery = "timeout"
        elif report is None:
            delivery = (
                "capture_limit_exceeded" if counts[0] > CAPTURE_LIMIT else "cli_error"
            )
        return {
            "exit_code": code,
            "delivery": delivery,
            "report": report,
            "cli_stdout_bytes": counts[0],
            "cli_stderr_bytes": counts[1],
            "cli_stdout_sha256": hashlib.sha256(out).hexdigest()
            if counts[0] <= CAPTURE_LIMIT
            else None,
            "stderr_tail": err.decode("utf-8", errors="replace"),
            "stderr_truncated": counts[1] > len(err),
            "execution_outcome": "unknown; inspect artifacts before retrying mutations"
            if delivery != "complete"
            else "see report and exit_code",
        }

    async def call(self, name, arguments):
        limit = DEFAULT_BUDGET
        try:
            args = arguments if arguments is not None else {}
            if name in BY_NAME:
                args = _integer_arguments(args, BY_NAME[name].input_schema)
            requested = args.get("max_output_bytes") if isinstance(args, dict) else None
            if type(requested) is int and 2048 <= requested <= MAX_BUDGET:
                limit = requested
            if name not in BY_NAME:
                raise ValueError("Unknown kernel tool")
            Draft202012Validator(BY_NAME[name].input_schema).validate(args)
            command, cwd = self.command(name, args)
            timeout = min(
                args.get("timeout_seconds", self.timeout_seconds), self.timeout_seconds
            )
            limiter = (
                self.readers
                if BY_NAME[name].annotations.read_only_hint
                else self.mutations
            )
            async with limiter:
                # A fresh process alone can still load timestamp/size-matched stale pyc.
                with tempfile.TemporaryDirectory(
                    prefix="pypto-mcp-bytecode-"
                ) as bytecode:
                    env = {
                        **os.environ,
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPYCACHEPREFIX": bytecode,
                    }
                    body = await self.execute(command, cwd, timeout, env)
            body["operation"] = name
        except (OSError, ValueError) as exc:
            body = {
                "operation": name[:128],
                "exit_code": 2,
                "delivery": "adapter_error",
                "report": None,
                "error": str(exc)[:2000],
            }
        except Exception as exc:
            # Schema failures belong in tool feedback, not a protocol traceback.
            from jsonschema.exceptions import ValidationError

            if not isinstance(exc, ValidationError):
                raise
            body = {
                "operation": name[:128],
                "exit_code": 2,
                "delivery": "invalid_arguments",
                "report": None,
                "error": exc.message[:2000],
                "argument_error": _argument_error(exc),
                "input_schema": BY_NAME[name].input_schema,
                "argument_names": sorted(BY_NAME[name].input_schema["properties"]),
                "next_action": "Use this tool's declared input_schema; do not guess parameter names. No CLI process was started.",
            }
            if len(exc.message) > 2000:
                body["error_truncated"] = True
        return reply(body, limit, tool_arguments=args)

    async def close(self):
        for proc in list(self.processes):
            await self.terminate(proc)


class _Pipe:
    """Cancellable POSIX text streams for the SDK's explicit-stream interface.

    Blocking readline worker threads can outlive cancellation while a host keeps
    stdin open. Readiness waits let shutdown complete without waiting for EOF.
    """

    def __init__(self, fd):
        self.fd = os.dup(fd)
        self.blocking = os.get_blocking(self.fd)
        os.set_blocking(self.fd, False)
        self.buffer = bytearray()

    def __aiter__(self):
        return self

    async def __anext__(self):
        while b"\n" not in self.buffer:
            await anyio.wait_readable(self.fd)
            try:
                chunk = os.read(self.fd, 8192)
            except BlockingIOError:
                continue
            if not chunk:
                if not self.buffer:
                    raise StopAsyncIteration
                break
            self.buffer.extend(chunk)
            if len(self.buffer) > CAPTURE_LIMIT:
                raise ValueError("MCP input frame exceeds 8 MiB")
        end = self.buffer.find(b"\n") + 1 or len(self.buffer)
        line = bytes(self.buffer[:end])
        del self.buffer[:end]
        return line.decode("utf-8", errors="replace")

    async def write(self, text):
        data = memoryview(text.encode("utf-8"))
        while data:
            await anyio.wait_writable(self.fd)
            try:
                count = os.write(self.fd, data)
            except BlockingIOError:
                continue
            data = data[count:]

    async def flush(self):
        pass

    def close(self):
        os.set_blocking(self.fd, self.blocking)
        os.close(self.fd)


async def serve(bridge):
    if version("mcp") != "2.0.0":
        raise ValueError(
            "Install the pinned scripts/requirements-mcp.txt in the server interpreter"
        )
    if os.name != "posix":
        raise ValueError(
            "The MCP process supervisor currently requires POSIX process groups"
        )

    async def list_tools(context, params):
        return types.ListToolsResult(tools=TOOLS)

    async def call_tool(context, params):
        return await bridge.call(params.name, params.arguments)

    server = Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        instructions="Native PyPTO kernel tools use the existing CLI and maintained skill references. "
        "Use explicit paths on the server host. Read report.status/validation or experiment phase; "
        "exit zero alone is not numerical/performance acceptance. Execution is never retried automatically.",
    )
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    stdin, stdout = _Pipe(0), _Pipe(1)
    try:
        async with stdio_server(stdin=stdin, stdout=stdout) as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())
    finally:
        with anyio.CancelScope(shield=True):
            await bridge.close()
        stdin.close()
        stdout.close()
        loop.remove_signal_handler(signal.SIGTERM)


def run(**options):
    try:
        asyncio.run(serve(Bridge(**options)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
