# MCP access to the kernel CLI

The local stdio server exposes the existing CLI and maintained reference sections. It translates tool arguments into an argv list, starts a fresh `check.py` process, and returns that process's JSON report and exit code. It does not reimplement API contracts, static rules, diagnostics, optimization candidates or experiment acceptance.

## Start on the execution host

An optional performance-card library is selected once at startup with `--cards-root /path/to/pypto-pro-kernel/references/knowledge-cards` before `mcp`; relative paths resolve from the startup working directory. Install `scripts/requirements-cards.txt` into the same interpreter when using it. `kernel_query` and `kernel_suggest` then share the CLI's [card source and relation semantics](optimization.md#optional-performance-card-source), without per-call root overrides or automatic sibling discovery.

The server uses Python 3.10+ and the pinned [MCP dependencies](../../scripts/requirements-mcp.txt). Other CLI commands retain their existing Python/dependency requirements and do not import the MCP SDK. Use an isolated environment for the server; the interpreter selected by `--python` needs [requirements-checks.txt](../../scripts/requirements-checks.txt) and, for native work, the matching PyPTO installation.

```bash
python3 -m venv /scratch/kernel-mcp-env
/scratch/kernel-mcp-env/bin/python -m pip install -r /path/to/skill/scripts/requirements-mcp.txt
/scratch/kernel-mcp-env/bin/python /path/to/skill/scripts/check.py \
  --pypto-root /path/to/matching/pypto \
  --examples-root /path/to/skill/examples \
  mcp --workspace /path/to/workspace --python /path/to/kernel-env/bin/python \
  --timeout-seconds 300
```

Global root options precede `mcp`. Omit `--python` to use the server interpreter; virtual-environment interpreter paths are preserved rather than dereferenced to their base Python. The workspace must exist. Input paths resolve on the server host, relative to that workspace, except `kernel_run` paths, which follow its optional `cwd`; native entry paths and experiment spec paths keep their existing CLI semantics. Absolute paths are also accepted.

Source the pinned framework/CANN environment before starting the server. A local machine path is not a remote machine path. For a device host, an external SSH/launcher script can source the environment and `exec` this command; send setup messages to stderr so stdout remains MCP. Keep machine addresses, accounts, environment paths, devices and locks in ignored local configuration. The server inherits its environment and does not install, select or change a toolchain during a tool call.

The transport currently targets POSIX hosts. It uses the [official Python SDK](https://github.com/modelcontextprotocol/python-sdk) for protocol negotiation, tool discovery and stdio framing, with cancellable pipe I/O for shutdown. Both the SDK's current protocol and the classic `2025-11-25` handshake are tested. The pinned SDK is required; changing its major version is an explicit compatibility task.

## Tools and their CLI owners

The [CLI/MCP mode table](checks.md#cli-and-mcp-modes) maps all eleven tools and owns availability, permission and retry decisions. CLI remains usable when MCP is absent. Tool discovery provides the input schemas; `details` expands reports where supported, and experiment `show` omits `input`. Task inventories, bounded batch reads/search/status and optional fixed `--context-policy` are maintained in [checks](checks.md#bounded-context-io-and-existing-execution-status); assertion consistency is owned by the [claim contract](optimization.md#performance-claim-consistency).

All tools accept `max_output_bytes`. [Checks](checks.md), [candidate/state contracts](optimization.md) and [native timing](../verification/native-timing.md) remain the knowledge owners. For example:

```json
{"text":"UB to L1","source":"candidate.py"}
```

Call `kernel_query` with those arguments, then `kernel_check` with `{"sources":["candidate.py"]}`. Once actual tensor metadata is available, call `kernel_native` with `{"spec":"native-spec.json"}`. To validate the complete implementation, use `kernel_run` with the original evaluator:

```json
{"sources":["candidate.py","test.py"],"native_spec":"native-spec.json","argv":["python","-m","pytest","test.py","-q"],"timeout_seconds":240}
```

For typed discovery, use `kernel_query({"kinds":["contract"],"filters":{"source_memory":"UB","destination_memory":"L1"}})`; exact symbols such as `pl.system.bar_all`, `pl.DT_FP16` and `pl.MemorySpace.Vec` use `text`. A page supplies stable-ID `resolve` queries and an optional `next_cursor`; repeat the original text/kinds/filters when continuing. The [knowledge query contract](checks.md#knowledge-discovery-and-exact-symbols) owns types, applicability, deferred adapters and freshness. These parameters use the same CLI implementation, including non-callable constants and unknown conditions.

For original usage material, `kernel_query({"text":"vf.addc","kinds":["document"]})` discovers official pages, guides, skill topics and source docstrings. Call a returned `resolve` object, preserving its `reference_sha256`; a `document:` ID may append a heading fragment such as `#调用示例`. `docstring:vf.addc` selects the source literal instead. The [document contract](checks.md#official-documents-guides-and-docstrings) owns scope, source roles, partial content and unresolved differences. `kernel_read` accepts `fragment` globally for one Markdown path or inside each batch request, and keeps its byte continuation inside that section. These are shared CLI operations; discovery and source statements do not establish runtime support.

For complete programs, use `kernel_query({"text":"batched transposed GEMM with M tails","kinds":["example"]})` or an API such as `pl.insert` with `kinds:["example"]` and a suitable `limit`. Resolve the returned `example:` ID, then pass its `program.read_requests.reference` or `.test` to `kernel_read({"requests":[...]})`. The [program contract](checks.md#complete-example-programs) owns roles, transitive source dependencies, applicability and source-bound reads. The same query handles all three example groups; no per-group MCP tool or parallel registry is used. Broad cross-kind pages may exceed the unchanged default budget: explicitly request `max_output_bytes:65536` or browse narrower pages. Budget failure is explicit, and test files or source claims never establish current execution acceptance.

For read-only knowledge admission, use `kernel_query({"validate":true})`, equivalent to `query --validate --format json`. This separate mode cannot combine with discovery, task inventory or action guidance. The [admission contract](checks.md#knowledge-admission-and-refresh) owns schema validation, duplicate isolation, relations, routing witnesses and reused maintenance checks. Source growth uses this existing tool and the same CLI implementation; it does not require a tool per document or example.

`argv` is passed literally, without an implicit shell or environment-variable expansion. The caller owns the evaluator, device lock, device selection and timeout. On a shared device, include the configured lock command in the explicit argv or use the existing external execution controller. This server serializes its mutating tools; it does not coordinate other server processes or substitute for the shared-device lock. `native`, `run` and experiment operations are marked as potentially mutating, including module-import effects. Static/query tools are marked read-only. Tool annotations describe behavior, not a sandbox.

The server executes with its process account's access. Workspace/cwd set path interpretation, not access control; explicit native imports and commands can access other resources permitted to that account. Connect this stdio service through the intended host's trust/approval controls. There is no network listener or cloud deployment in this entrypoint.

## Result, budget and failure contract

Each tool returns one JSON text content block with schema `pypto-mcp-result/1`. It deliberately omits a duplicate `structuredContent`; clients can parse `content[0].text`. The [MCP tool result contract](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) permits this content form.

| Field | Meaning |
| --- | --- |
| `report` | The existing CLI JSON object, unchanged when delivery is complete. Feedback identities and validation limits remain intact; experiment `phase` keeps its existing meaning |
| `exit_code` | Original CLI status, including the runner's child status. Signals use `128 + signal`; bridge timeout uses 124. Exit zero is not numerical/performance acceptance |
| `delivery` | `complete`, `budget_exceeded`, `timeout`, `cli_error`, `capture_limit_exceeded`, `invalid_arguments` or `adapter_error` |
| `input_schema`, `argument_names` | Schema-validation failures return the same authoritative schema published by tool discovery and its accepted field names; no CLI process has started. Under a smaller requested budget, `input_schema_omitted: output_budget` retains the field names and error classification; `error_truncated` marks shortened diagnostics |
| `argument_error` | Schema failures identify the argument `path` as field names and array indices, the failed `constraint`, and a scalar `expected` value when available. Nested unknown fields include their local `allowed_fields`. This remains available when the full schema is omitted for the reply budget; it does not modify arguments or execute a retry |
| `read_only_retry` | A query-budget failure may suggest an `arguments_patch`, preserving other arguments. Multiple returned objects can select `limit:1`; a single object or an existing cursor instead receives a concrete larger `max_output_bytes` when the captured result can fit the server ceiling. Lowering object count cannot shrink one object. This is optional caller guidance, never an executed retry; mutations receive no replay suggestion |
| `cli_stdout_bytes`, `cli_stderr_bytes` | Observed byte counts before presentation; stderr is drained continuously with at most an 8 KiB retained tail |
| `cli_stdout_sha256` | Digest of the exact captured CLI stdout, including its newline; null when the 8 MiB capture ceiling is exceeded |
| `stderr_tail`, `stderr_truncated` | Bounded original diagnostics. The tail may be omitted to fit the result budget; missing tail is not evidence that no diagnostic was emitted |
| `budget` | Requested `limit_bytes`, measured `result_bytes`, and `context_tokens: null` because no model tokenizer/host telemetry is inferred |

The default limit is 32 KiB; callers may request 2–256 KiB. `result_bytes` measures the SDK-normalized `CallToolResult` using `model_dump_json(by_alias=True, exclude_none=True)`, including server-identity metadata, JSON escaping, SDK defaults and the budget fields themselves. A legacy transport can omit default fields such as `resultType`; its compact wire result is smaller than this conservative normalized count. Tests check both forms against the budget. JSON-RPC request IDs/envelopes and host-added rendering, tool schemas or prompts are excluded. A client's tokenizer and context accounting can produce a different cost; measure those separately. No character-to-token conversion is asserted.

When the result cannot fit, delivery becomes `budget_exceeded`, `report` is null, and a small `report_summary` retains the CLI's phase/status when available. The full report is not silently clipped. MCP `isError` is true for nonzero CLI status or incomplete delivery, including an otherwise successful operation whose report exceeds budget. Read-only callers can narrow the scope or raise the budget. An execution/state mutation has already run: do not rerun it merely to fetch output. Inspect its explicit artifacts or use the experiment `show` action. There is no hidden disk report or retry queue.

For knowledge discovery, select an object page with `limit` and follow its original references. An official API-page request uses `kinds:["document"]`; check `document_role` before reading so a source docstring or guide remains distinguishable from the requested page. The same selection is available through CLI `query --limit` and `--kind document`. Existing output budgets and explicit caller choices remain unchanged.

A completely delivered query can still report `source_unavailable`, `invalid_request`, `cursor_invalidated` or `source_changed` with a nonzero CLI status. Healthy matches can retain partial source coverage after another owner is quarantined. Preserve those semantic fields and admission findings rather than treating complete delivery as a complete corpus or successful admission. Schema-invalid tool arguments use adapter `invalid_arguments`; [query classifications](checks.md#knowledge-discovery-and-exact-symbols) remain owned by the CLI.

The server drains child pipes with an 8 MiB stdout capture ceiling. A non-JSON/oversized stdout report or timeout leaves acceptance unestablished. On timeout, cancellation, host EOF or graceful SIGTERM, it terminates the owned process group, escalating to SIGKILL for descendants that ignore SIGTERM. Detached processes outside that group, remote work behind an SSH command and externally managed jobs still require their own execution controller and watchdog. Hard-killing the server itself cannot provide a cleanup guarantee. Killing a host process also does not establish device recovery; follow [board-run failure handling](../verification/board-runs.md).

The bridge never retries a tool. A timeout or lost connection can leave a completed state mutation or externally launched work; inspect current state/artifacts before deciding the next action. Host tool timeouts should exceed the server timeout plus cleanup allowance; for example, a 300-second server ceiling with a 330-second host timeout.

## Freshness and parity

There is no query, knowledge, parsed-module or result cache in the adapter. Every call rereads the current CLI, selected framework source, reference owners, examples, inputs and state through a fresh CLI process. The child environment sets `PYTHONDONTWRITEBYTECODE=1` and a new empty temporary `PYTHONPYCACHEPREFIX`; an existing `.pyc` with matching timestamp/size cannot conceal edited Python source. The temporary directory is removed after the call. Framework capability guards and source SHA checks stay in their existing owners. Source/document/example changes with the same size or timestamp must still affect the next result. Updating the transport's own SDK, input schemas or startup configuration requires restarting the server; an existing server process does not hot-reload its transport code.

For parity checks, invoke the CLI with the same interpreter, cwd, roots, options and inputs, then compare its parsed JSON with MCP `report` and its status with `exit_code`. Do not compare different installations or treat the MCP wrapper's delivery metadata as a semantic change. Runtime compilation, numerical results, hardware timing and agent effectiveness remain separate evidence.

Admission shares discovery's source snapshots and final consumed-byte verification. Metadata repairs, duplicate removal and document/program renames affect the next call, including same-size/same-timestamp edits; there is no persistent metadata or graph index to refresh. Existing scan and reply limits still apply. A running server must be restarted to expose a changed `kernel_query` input schema.

## Host configuration

For a generic stdio host, point its server command at the actual interpreter and `check.py mcp` entry above. No new API database, cloud plugin or package copy is required. In Codex, the [official MCP configuration](https://developers.openai.com/codex/mcp/) uses a server table such as:

```toml
[mcp_servers.pypto_kernel]
command = "/path/to/kernel-mcp-env/bin/python"
args = ["/path/to/skill/scripts/check.py", "--pypto-root", "/path/to/matching/pypto", "--examples-root", "/path/to/skill/examples", "mcp", "--workspace", "/path/to/workspace", "--python", "/path/to/kernel-env/bin/python", "--timeout-seconds", "300"]
startup_timeout_sec = 20
tool_timeout_sec = 330
```

Resolve placeholders on the execution host; keep the concrete configuration outside tracked source. Use the host's supported allow/deny and approval settings for the intended workflow. Reconnect after adding or changing a server, then verify discovery and a harmless query in that host before relying on native execution. A standalone SDK client pass does not prove that an unrelated desktop session has loaded the configuration.

Evaluation adapters reuse the published tool schemas and `Bridge.call`; separate parameter caps can reject requests that the CLI accepts. Numeric descriptions derive their bounds from the same schema. JSON integer values such as `5.0` are serialized as CLI integers only for schema-declared integer fields; fractional values in those fields remain invalid, and literal command strings are preserved. A positive `max_lines` requests content under the active read policy, with continuation when a page is incomplete; a page limit is not an additional argument maximum. Before freezing an evaluation, check legal boundary requests, informative schema rejection without a child process, continuation, and an already authorized harmless `kernel_run` through the actual host. These checks establish tool delivery and permission coverage, independently of kernel correctness or device performance.

Tests live in `scripts/_checks/tests/test_mcp.py`; run the full tools suite with the pinned MCP dependency installed for stage-4 acceptance. Without that optional dependency, these MCP-specific tests skip rather than changing ordinary CLI requirements. Tests cover actual stdio negotiation, semantic parity, source/knowledge refresh, budget accounting, literal argv, source gates and process cleanup; device/host observations stay in ignored scratch.
