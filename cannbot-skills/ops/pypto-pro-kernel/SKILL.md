---
name: pypto-pro-kernel
description: Write, verify, debug and optimize native pypto_pro kernels on Ascend NPUs using @pl.jit, VF, SIMT and Cube APIs. Includes runnable examples, static checks and native-task timing. Use for kernel development rather than development of the PyPTO framework itself.
---

# PyPTO Pro Kernel Workflow

Use native `pypto_pro.language` to implement and verify the user's mathematical contract. Resolve links relative to this file; runnable [examples](examples/) are bundled inside this skill. The external framework source checkout and installed PyPTO package are separate roots.

## Constraints and workspace policy

- Keep kernels native: no skill-specific tensor types, decorators or transfer wrappers. Keep compiler semantics and framework diagnostics in the independent framework, and guard internal interfaces against the installed version.
- Preserve the numerical contract, public layouts and tolerances. Report numerical correctness, model diagnostics and hardware performance separately, bound to source and toolchain identity.
- Vector computation uses actual `vf.*` operations inside `@pl.vector_function`; only `pl.sort32`, `pl.mrgsort` and `pl.mrgsort2` may use tile Vector compute. Allocation, movement/padding, synchronization, scalar control and SIMT remain available. Cube matmul starts with `pl.gemm` / `pl.gemm_mx`, subject to the [closed exceptions](references/patterns/gemm-desugar.md). This is authoring policy, not an API-support or speed claim.
- Establish the [supported domain and specialization rules](references/workflows/authoring.md#generalization-and-specialization); one entry and a few passing shapes do not prove generalization.
- Keep official contracts, installed signatures and observed behavior distinct. Before claiming a capability absent, check Python/C++ source, alternate names and a focused probe using the [evidence rules](references/verification/evidence.md).

## Choose the current stage

Start from the user's artifact and best verified baseline. For a new kernel, read the [DSL primer](references/workflows/getting-started.md) unless already in context, then [select a compatible complete example](references/workflows/authoring.md#selecting-an-example). For supplied code, inspect the source and supplied evidence first. Only when the controller supplies a `pypto-task-context/1` JSON inventory, use `query --task manifest.json` to distinguish declared files from inline statements/fixtures; do not search for a separate file representing inline evidence. This checks the inventory; source discovery still uses positional `query TEXT`. A prose task such as `task.txt` is not this inventory. Query a specific unresolved API or capability with positional `query TEXT`; a decision already supported by the workload and source does not need a documentation tour. A validation-only request with a supplied evaluator should obtain its first result before speculative repairs, after checking its loading route. Read any selected section's body; a heading or truncated excerpt does not establish its instructions.

| Current need | Next action |
| --- | --- |
| Clarify requirements or write a requested specification | Resolve the [mathematical contract](references/workflows/requirements.md); use its optional SPEC tools only when the task or selected generator needs that format |
| Locate an existing example or implementation | Query the workload in ordinary words with `query TEXT --kind example --limit 3`, then resolve the selected program for its kernel, independent reference, tests and source hashes. Use catalog/file search for a retrieval miss or an unresolved scope condition |
| Implement or adapt a kernel | Resolve [contract/resources](references/workflows/authoring.md#preflight-decisions) and matching [feature triggers](references/workflows/authoring.md#feature-triggers); validate one complete slice before extending it |
| Fix a parse, compile, numerical or device failure | Read the original diagnostic and query its unresolved operation; use [debugging](references/workflows/debugging.md) when the boundary still needs classification |
| Optimize a correct kernel | For SIMT, select one [mechanism-specific comparison](references/patterns/simt-first.md#choose-a-simt-experiment); otherwise use the [cost-driven workflow](references/workflows/optimization.md#choose-the-implementation-experiment). Consult the general workflow for unresolved budget or acceptance questions; preserve the best verified baseline |
| Micro-tuning stalled or the structure may be wrong | Read [structural review](references/workflows/structural-review.md) before more local changes |
| Optimize with a coverage requirement | Read [coverage mode](references/workflows/optimization-coverage.md); enumeration needs an explicitly supplied card library (`--cards-root`, see [the optional card source](references/tooling/optimization.md#optional-performance-card-source)). Without one, report coverage as incomplete. Enumerate active rows, close each with evidence, then use the cost-driven workflow above |
| Run a supplied evaluator or measure time | Verify its [loading route](references/verification/board-runs.md#integrate-a-supplied-evaluator), then [capture and scoring](references/verification/native-timing.md#verify-capture-before-expanding-performance-coverage) on one complete nondegenerate case; for a score target, do this before any full matrix |
| Per-core, the 7 `op_summary_*` metric groups, or `msprof` / `msopprof` evidence is needed | Collect with [performance collection](references/tooling/performance-collection.md), read [field meanings](references/tooling/performance-fields.md); paired timing still decides speed |
| Deliver an operator package | Select the [delivery target and policy](references/workflows/delivery.md). Python/JIT source organization defaults to [pypto-pro-op-standard](../pypto-pro-op-standard/SKILL.md); explicit user requirements override project rules, and either can disable organization or select an alternative. Preserve native implementation and technical acceptance requirements in every path; CANN offline packages use `pypto-pro-cann-delivery` |

For an explicitly provided performance-card library, enable the [optional card source](references/tooling/optimization.md#optional-performance-card-source). Query the bottleneck, technique or stable card ID, read the original applicability/API gates and follow declared relationships to the owning contract. Check card coverage before treating a miss as complete. Cards add conditional references and relationships; they do not change authoring policy, establish runtime support or activate another skill's workflow.

Open only the rows matching the current uncertainty. The [reference index](references/index.md) routes other patterns and APIs.

## Query, check, execute, decide

CLI is the base interface; MCP is an optional adapter to the same implementation and knowledge. Use exposed, authorized `kernel_*` tools, or invoke `scripts/check.py` directly when MCP is unavailable. The [mode and fallback contract](references/tooling/checks.md#cli-and-mcp-modes) preserves interpreter, paths and permissions. An approval denial is not an availability failure. Use [bounded search/read and existing-request status](references/tooling/checks.md#bounded-context-io-and-existing-execution-status) when available; continue a partial read using its source hash rather than assuming it is complete. For long checks, wait about 20–30 seconds on the existing session or pinned receipt; a running request is still pending. After timeout/disconnection, inspect state and artifacts before any retry.

Shell examples below run from this skill's root; elsewhere use the check script's absolute path. Bundled examples need no path override; select matching framework source with `PYPTO_SRC` or `--pypto-root`. Global `--pypto-root` / `--examples-root` options precede the subcommand.

```bash
python scripts/check.py query 'UB to L1' --source candidate.py
python scripts/check.py search --path /path/to/relevant/module --pattern SYMBOL
python scripts/check.py read --path /path/to/owner.md --start-line 180 --max-lines 40
python scripts/check.py rules candidate.py --format json
python scripts/check.py native --spec /path/to/native-spec.json
python scripts/check.py run --source candidate.py --source test.py -- python -m pytest test.py -q
python scripts/check.py suggest --source candidate.py
```

1. Query the operation at the unresolved boundary, for example `python scripts/check.py query 'vf.reduce_sum' --source candidate.py` from this skill’s root. Use the returned body or conditional route first; broaden to [API lookup](references/api/pypto_pro_language.pyi), a linked owner or [example selection](references/tooling/examples.md) only for facts it leaves unresolved. A missing match is not API absence.
2. Check the changed sources, then use [native preflight](references/tooling/checks.md#scenario-query-and-native-preflight) with concrete metadata. Unknown scope and advisories require review; exit zero does not establish numerical correctness.
3. Gate device commands with `run` and the [board discipline](references/verification/board-runs.md). Read the [feedback](references/diagnostics/messages.md#tool-feedback-contract), including the original exit status and unresolved evidence. For MCP also check `delivery`; incomplete delivery leaves the result unresolved.
4. For optimization, use [suggestions and current experiment state](references/tooling/optimization.md) to choose one testable hypothesis and retain or restore the best source. For a supplied performance assertion or a positive speedup claim, [check its exact evidence scope](references/tooling/optimization.md#performance-claim-consistency); a decision making no speedup claim needs no extra claim-tool tour. A suggestion is conditional; no match does not end the search. Two unsuccessful fixes at one boundary without new evidence require revisiting its contract.

## Reading and maintenance

Reuse context already read. On resuming, retain the contract, selected pattern, current source identity, best verified baseline and next unresolved check. Detailed facts have one owner under `references/`; keep scripts in [scripts](scripts/), kernels in [examples](examples/). Follow the [admission rules](references/index.md#admission-rules); keep this entry short and do not accumulate per-attempt histories.
