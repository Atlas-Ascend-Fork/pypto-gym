---
name: pypto-op-perf-panko-manual
description: Stage 7 PANKO world-model + evaluator contract. Read ONLY by pypto-op-auto-tuner-panko. Defines one search cycle (evaluate, record, reflect, select) over the persisted tree, and the CODE/STOP directive returned to the orchestrator. No API key. Use when the user asks for PANKO search-based auto-tuning in Stage 7.
---

# pypto-op-perf-panko-manual — Stage 7 PANKO (world model + evaluator)

**Read only by `pypto-op-auto-tuner-panko`.** The orchestrator does not read this skill; it only relays
your directive and dispatches the coder. You never call `state_transition` (you are a subagent).

You are the **World Model** (propose δ, reflect, rescore V, Insert/Update/Prune) **and the
Evaluator** (run the frozen `E(x)→(s,p)`). The deterministic core — selection (argmax
`pypto_action_priority`, short symbol `V_pypto` = our composite `w_stage·prior_gain·V`, where
`prior_gain` (g) is learned from every measurement and read optimistically;
V is PANKO's per-node value),
stagnation, `J = s·(P_ref/p)·100`, tree-edit application, stop conditions, state I/O — is
reached through `scripts/panko_harness.py` and implemented in the `scripts/panko/` package
beside it, one module per responsibility (`state`, `measurement`, `front`, `stopping`,
`selection`, `delivery`, `lifecycle`, …). You **call** the CLI; you do not re-implement its
decisions, and nothing outside the package imports its internals.

The world model is the **persisted tree** in `custom/<op>/optimization/search_state.json`, not
this session. Each cycle may resume the same task-scoped optimizer conversation: **read the state → apply one transition →
persist → return a directive.** Therefore every V update / prune MUST carry its reason
(`--edits ... "reasons"`); a later dispatch re-hydrates the belief from your log.

## Two modes — one objective by default, the front when you name a second

`config.objectives` decides which search runs, and the DEFAULT is one objective:

```
"objectives": ["aicore_e2e"]                 # default: single-objective PANKO
"objectives": ["aicore_e2e", "workspace"]    # multi-objective PANKO
```

**Single-objective (the default).** Nothing to trade against: the archive holds one
point, that point IS the global best, and the ratchet, `impl.py` and `J` are what they
have always been. No hypervolume reference is needed, so no `--chip-memory-mib`. The
other readings are still MEASURED and still recorded on `m` — pass `--workspace-b` on
every `record` anyway, kept or not, because that record is how a single-objective run
leaves the trade-offs it saw for whoever reads the campaign later.

**Multi-objective.** Reached by naming the second axis and by nothing else. Then the
archive is a Pareto front, dominance on the measured axis is decided by the bootstrap
rather than by the medians, the next action's base is the point with the largest
hypervolume contribution, and convergence is an OR of the hypervolume gain with the
per-axis records. A byte-valued axis needs a ceiling (below), and `init` refuses
without one rather than running the single-objective search under a front's name.

**Which to run.** Optimising AICore E2E alone — the default — is the single-objective
mode. Optimising AICore E2E *and* the workspace is the multi-objective mode. A
sentence below that quotes a front, a hypervolume or a trade-off is about the
second mode.

## Optimization principles — how to value actions (raise `V`)

Beyond the catalog `prior_gain`, raise an action's **`V`** (the belief factor of
`pypto_action_priority = w_stage · prior_gain · V`) by how strongly its intent serves the
two principles that make a pypto kernel faster and lift utilization — and **propose (insert) new
intents** along these axes:

1. **Minimize data copy_in / copy_out** (Global-Memory traffic). Keep intermediates on-chip
   (L1 / UB) and never round-trip them through GM: fuse producer→consumer ops, reuse already-loaded
   data (L1Reuse), drop redundant loads/stores.
2. **Reduce computation complexity.** Fewer / cheaper ops: compute-once-and-reuse, skip tail
   zero-pad compute (`valid_shape`), simplify algebra, shrink effective work.

**Utilization ratio — overlap:** pipeline **MTE data-movement ∥ Cube compute ∥ Vec compute** so they
run **concurrently** (double-buffer / nbuffer / CubeNBuffer / `submit_before_loop` / stitch), filling
the idle **bubble** and lowering latency.

Each seed's **`prior_gain`** already encodes how **directly** it serves these principles — strong lever
(fuse / keep-on-chip, cut ops, MTE∥Cube∥Vec overlap) ≈ 0.9, moderate ≈ 0.6, weak / general tuning
≈ 0.3 — so principle-aligned actions are prioritized by the harness. Weight `V` the same way, and give
every inserted intent a `prior_gain` by this same principle-directness scale.

## References (load as needed)
- `references/evaluator.md` — the frozen `E(x)→(s,p)` you must run (fixed commands/thresholds).
- `references/search_acceleration.md` — evaluator modes, cache validity and worker lifecycle.
- `references/state_schema.md` — `search_state.json` + `<op>_optimization.md`.
- `references/action_catalog.json` — stage-tagged seed intents (used at INIT).
- `references/tunable_constants.md` — design note: extending INIT normalisation
  from tile calls to the constants that drive loop and view extents. Not
  implemented yet; read it before touching `bayesian_optimization.normalize` or `bayesian_optimization.apply`.

**Two-directional actions.** A `"directive": true` entry names a compiler or
runtime option that is either on the program or absent from it, and every one of
them has a counterpart carrying `"counterpart_of"` that removes it. The removal
is not decorative: on one kernel removing the directive is the larger win and
on another setting it is a heavy regression, so neither direction is safe to
assume. Treat the pair as one axis and let the device pick the direction.

**Held actions.** At INIT the harness splits the catalog into `open` and `held`.
An action is held when the kernel does not currently satisfy its `requires`
(no matmul, no broadcast, no chunked scan, single module). Held actions are
never deleted: after every improvement the harness recomputes the kernel's
structure and returns any that now qualify, so a transformation that introduces
a matmul re-admits the whole cube family. Do not try to work around the split;
if you believe an action should be reachable, propose the transformation that
creates its precondition (F-20 does this for reductions).

**Scope: the kernel, not the call.** Both time objectives are on-device readings of the
`@jit` kernel. `aicore_e2e` IS `p` -- the on-device profiler's "AICore End-to-End Time" line,
so `--p`, `--samples` and `J` all mean it. `mix_aic` is msprof's per-launch MIX_AIC
`aicore_time(us)`, carried on `m`. They are DIFFERENT quantities
and their ratio is not a constant: on a cube-bound kernel they nearly agree, while on a
vector-bound one they can differ by most of an order of magnitude. Which is why naming one
is not naming the other: a candidate can move one without moving the other, and in the
multi-objective mode the front is where that shows. A reading under one is not comparable with a reading under the other, so
numbers are quoted against the instrument that produced them. Neither counts the host launch
path. The host wrapper is
FROZEN for the whole run: `init` records a signature of everything that is not a `@jit`
kernel, and `feasible` and `record` both reject a candidate that changed it. Counting only
the kernel orders two candidates correctly exactly while the uncounted part is identical
between them; move work into the wrapper and `p` improves without the operator getting
faster. Put every δ inside the kernel. Actions that moved work to the host are not in the
catalogue.

**The chip envelope is derived, not assumed.** `init` reads the buffer capacities
(UB / L0A / L0B / L0C / L1) and the cube/vector core counts out of the platform ini
the installed CANN ships — `<cann_home>/<arch>/data/platform_config/<SoC>.ini`, the
SoC resolved from the device — and every static rule in the run validates against
that one envelope. It **refuses and asks you** when it cannot derive them; it does
not fall back to another chip's numbers, because those reject legal tiles before any
device sees them. `--chip-envelope '<json>'` supplies them deliberately and is
recorded as such. The ini path used is recorded in the state and echoed by `init`.

**A byte-valued objective needs a ceiling, and `init` refuses without one.** The
hypervolume's reference point is fixed for the run: the preopt on the primary
objective, and the **chip's memory** on a byte axis such as `workspace`. That comes
from the envelope's `memory_mib` (the SKU's ini, `SoCInfo/memory_size`), or from
`--chip-memory-mib <MiB>` when the ini does not carry it or when the kernel does not
get the whole device -- in deployment it shares memory with weights, the KV cache and
other operators, so the chip's total is a generous bound, not the real one. With more
than one objective declared and no ceiling anywhere, `init` **refuses**
(`hv_reference_unavailable`) rather than starting: without a reference the next
action's base is picked as if there were one objective (least tried, then fastest) and
the convergence test loses its hypervolume half, both silently -- a single-objective
search reporting a multi-objective front. A single-objective run is unaffected and
needs no ceiling. `init` echoes the box it will use as `hv_reference`, in each
objective's own units.

**And re-checked against the device before the first evaluation.** The SoC is resolved
*for* `--device`, not for the box, and the name the envelope was built from is asked
again before any block measures anything. A different chip in the slot **halts and asks
you** — re-initialise here, or point PANKO at the chip it was initialised for. `init`
itself never refuses on this (it must stay runnable with no device attached), but it
reports the mismatch the moment it sees one. Where no runtime can be asked, the run
proceeds and the state records `confirmed: unknown`, so a report can say whether the
numbers were checked against silicon or only against a name.

**`debug_options={"runtime_debug_mode": 1}` is REQUIRED, and is frozen.** The run's
latency objective is `aicore_e2e`, and that reading exists only in the on-device profiler's
stdout record, which is written only at mode 1. At mode 0 the "AICore End-to-End Time" line
never appears and the run has no latency to optimise, so every top-level jit kernel carries
the flag and the baseline is built with it. Mode 1 also supplies `util`, `aic_util` and
`aiv_util`. The `mix_aic` objective and msprof's own columns (`aiv_time`, `aic_mac_ratio`,
`aiv_vec_ratio`, `cube_utilization`) come from `op_summary_*.csv` and do not depend on the
flag; `workspace` comes from the alloc callback and does not either.
`init` records a signature of every kernel's `debug_options` and `feasible` / `record`
reject a candidate that changed it. The reason is comparability, not truth: **an objective
IS its instrument.** A candidate read on a different instrument is not comparable with the
rest of the run, and a candidate whose only change is dropping the flag would score as a
win on the difference between instruments. See `references/evaluator.md`.

**Dependency.** The tile search needs `optuna >= 2.0` (developed against 4.9).
`init` runs a capability preflight and REFUSES when
`bayesian_optimization_tile_search` is on and optuna is not importable: tile values
are the one lever the model cannot propose, so running without it is a different
method rather than a degraded one.

⛔ **That refusal carries `ask_user`, and it is a question for the USER.** Relay
`ask_user.question` with its `options` and stop. Do not install anything, do not
re-run `init` with the tile search switched off, and do not fall back to the
stepwise path on your own — which of those three the run should be is a decision
about what is being measured, and it is not yours. Resume only on the user's
answer.

Every `init` reports `bayesian_optimization: {available, optuna_version, reason}`
and persists it, so a report names the algorithm that ran rather than the one that
was configured.

**Paths resolve from THIS SKILL, not from your working directory.** Let `SKILL_ROOT`
be the directory this file is in — you know it, having just read this file from
there. In a pypto-gym clone that is `cannbot-skills/ops/pypto-op-perf-panko-manual`; in a
project the plugin was installed into it is `.opencode/skills/pypto-op-perf-panko-manual`
(or your client's equivalent), a symlink to the same place. There is no
`cannbot-skills/` tree in that project, so a path spelled from the repo root does
not exist there.

Let `HARNESS = python <SKILL_ROOT>/scripts/panko_harness.py`.

Run INIT/STEP with `python <SKILL_ROOT>/scripts/evaluation_session.py --op-dir <op_dir> --op <op> --op-file <op_file> --device <DEVICE> --test-command '<test_command>' --lint-root <lint_root>`. It runs the full tests, lint, and frozen measurement in a resident worker. Pass `--evaluation-mode resident` to `HARNESS block` to reuse it for BO, keeping paths and commands consistent across dispatches. A terminal `HARNESS stop` closes it. On abort, run `python <SKILL_ROOT>/scripts/evaluation_session.py --close --op-dir <op_dir>`.

For additional reuse, select `--mode cached` (complete results and exact-source static failures) or `--mode prepared` (also guarded deterministic inputs and pure golden references). Before a new INIT's first measurement, call `evaluation_session.py --reset --op-dir <op_dir>` once; never reset a resumed STEP. It starts a new cache generation and removes stale terminal markers while preserving prior evidence. Use the same mode for BO's `--evaluation-mode`. Tests, references, SPEC, lint, configuration, runtime versions, host and environment invalidate the cache. Generated tuning notes do not. Set `PANKO_RUNTIME_SIGNATURE` when replacing runtime binaries without changing their version metadata or changing test dependencies outside the operator directory. Prepared mode requires deterministic tests and a pure `<op>_golden` in `<op>_golden.py`; unsupported fixtures keep fresh preparation. Tensor contents and versions are checked before reuse, including reference outputs.

To retain harness imports and API metadata, set `HARNESS = python <SKILL_ROOT>/scripts/harness_session.py --op-dir <op_dir> --`. Its arguments and stdout remain the original harness contract. After `record`, it runs the original stop check; a real stop with the best source restored runs the original report and writes `optimization/controller-terminal.json`. After recording, require its run_start to match progress._start_ts, its reason to match progress.stop_reason, and its delivered_code_hash to match the current best source before returning STOP. Forced stops do not create it. On abort, call `harness_session.py --op-dir <op_dir> --close` to release both owned workers. Its `--timeout` bounds a complete harness command (default 3600 seconds); set it to the command's BO budget.

If the client exposes a resumable optimizer task/session ID, reuse it only within the same operator directory, role and search configuration; keep the coder's role separate. Always reload `search_state.json` and the current implementation before a resumed cycle. Start a new conversation when that scope changes. Context reuse does not replace the persisted state or bypass any transition.

The action catalogue resolves itself: **omit `--catalog`** and the harness reads
`<SKILL_ROOT>/references/action_catalog.json`, wherever that is. Pass `--catalog`
only to point at a different catalogue. `init` echoes `skill_root` and `catalog`
so what it resolved is on the record.

## Inputs (from the dispatch prompt)
`op`, `op_dir=custom/<op>`, `DEVICE=TILE_FWK_DEVICE_ID`, `op_file`, `test_command`, and either:
- **INIT cycle** (first dispatch — no `candidate_file` given): also `P_ref`, `config`
  (`stagnation_K` / `staged_prior` / limits).
- **STEP cycle** (`candidate_file` + `action_id` given): the candidate the coder just produced.

Detect the mode by whether `candidate_file` is present.

## Code snapshots & greedy accumulation (harness-owned — do NOT restore/snapshot by hand)
Node code lives under `custom/<op>/optimization/nodes/<code_hash>.py`. **The harness enforces greedy
accumulation deterministically**, so every action stacks on the running global best without you copying
any files:
- `init` (pass `--op-file <op_file>`) snapshots the preopt `op_file` → `nodes/<preopt_hash>.py`.
  The hash is **computed from the file**; `--preopt-hash` is optional and, if given, is verified
  against those bytes. `init` **refuses** if `op_file` cannot be read, if a supplied hash disagrees
  with it, or if the snapshot cannot be written — a run whose baseline is missing has nothing for
  later cycles to stack on, and would silently accumulate onto whatever is in the working file.
- `select` **restores the running global best into `op_file` before returning**, so the coder applies
  this action's δ on top of the accumulated best (`restore_code_hash`/`restored` are echoed for logging).
  If that restore fails and the file is not already the best, `select` **refuses** rather than hand out
  an action: the response names the program the δ is to be applied to, and it has to be that program.
- `record` snapshots a **kept** candidate → `nodes/<code_hash>.py`; on a **reverted** candidate it rolls
  `op_file` back to the action's current best (or the global best) automatically.
- `close` leaves `op_file` at the action's best; `stop` restores the global-best `op_file` for delivery
  and **refuses to record a stop it cannot deliver** — `stop_reason` is sticky and the Stage 7 gate
  reads it, so writing it over some other program would complete Stage 7 on a kernel the search
  never chose.

So `best_aicore_e2e_us` trends down monotonically and the best code can never be lost. A candidate slower
than the running best simply regressed **on top of** the best — the harness reverts it. Your only file
action is to have the coder **apply the δ** to `op_file`. Insert new intents with parent defaulting to
`best_node`.

**This is greedy hill climbing, and that is deliberate — say so rather than implying otherwise.**
The tree in `search_state.json` is a tree of *intents*. The reachable program states are not a tree:
they are a single non-decreasing chain. A candidate is kept only if it beats the bar (the global best,
or the action's own best once the action has produced one, which was itself ≥ the global best when it
was set), so **no program that is worse than the running best is ever built on**. The consequence
follows directly: a route that must get *worse* before it gets better is out of reach, and PANKO does
not claim to search one.

Two things do reach combinations, and they are the only two:

- **Within one δ.** An action may apply several changes at once and be judged as one step. `S-8`
  (L1Reuse + CubeNBuffer) is exactly this, and it is the pattern to use whenever two changes are only
  jointly profitable — `F-8` names the `F-7` chunk in its own δ for the same reason. A pair split
  across two catalogue entries is a pair the ratchet can lose: the first half regresses, is reverted,
  and the second half never sees it.
- **Within one tile block.** The tile search proposes all Cube and Vector tile parameters *together*
  (TPE over the joint space), not one axis at a time, so combinations inside that space are reached
  without the ratchet being involved.

Everything else is one change at a time, kept or reverted on its own measurement. If a campaign needs
a genuinely non-monotone search, that is a different optimizer, not a setting here.

## Cycle — INIT
> **No target: leave `--p-ref` out.** `P_ref` is the target for the primary objective, and
> `success` fires when the DELIVERED program reaches it. So `--p-ref` takes a REFERENCE
> KERNEL'S MEASUREMENT and nothing else. Without one — no reference kernel, or this kernel
> already beats it — omit the flag: the preopt becomes the reference, `success_stop` goes off
> with it, and the run ends on the budget, the wall clock or `hv_stagnation`. `J` then reads as
> a percentage of the baseline (`J = 150` is 1.5x the preopt), and the search is unchanged,
> because `J` is monotone in `1/p` for any fixed `P_ref`. **Never put a guess there.** The flag
> was required once, so a campaign without a reference had to invent a number. A reference the
> preopt already beats means `j0` above 100, `success` true before the first candidate, and the
> run over at evaluation 0 and reported as a success. `init` now REFUSES a
> `P_ref` the preopt already meets, names both numbers, and asks which it was. `init` echoes
> `p_ref` and `p_ref_via` (`explicit:cli` or `preopt`), and the state keeps
> `target.P_ref_via`, so a report that quotes a speedup can say what against.
> Report progress by `best_speedup_vs_preopt` / the front, not by `J`.

0. **Normalize the kernel, before anything is measured.** `bayesian_optimization.apply` tunes a tile call only when
   every argument is an integer literal. A kernel written `pypto.set_vec_tile_shapes(TILE_B, TILE_D)`
   is classified `vec_dyn`, `tunable_sites()` returns nothing, and the tile lever produces **zero**
   trials for the whole run — whatever else is configured. Exposing that parameter is what moves the
   tile from your judgment to the core's arithmetic, and it is the one step that cannot be done later:
   submitted as a search candidate a performance-neutral refactor measures the same as the incumbent,
   is reverted as unimproved, and its bytes are burned into the eval cache so it can never be
   re-issued.

   - `cp <op_file> <op_file>.preopt` first. You need the original to check against and to restore.
   - Rewrite so that **every** `set_vec_tile_shapes` / `set_cube_tile_shapes` argument is an integer
     literal. Where one name drives both a tile and a view extent, a loop bound or the host wrapper's
     padding, give the view its own name and put the literal at the tile call. Every binding site
     moves together or none does.
   - `HARNESS normalize --before <op_file>.preopt --after <op_file> --op-dir custom/<op> --op <op>`.
     On `ok: false` the reasons name what failed; fix and re-run, or restore `.preopt` and continue
     without normalizing. **Never** proceed on a failed verdict.
   - Then run `E(op_file)` and confirm it is correct and that its latency matches `E(<op_file>.preopt)`
     within the noise floor. A normalization that made the kernel *faster* changed a value; restore
     `.preopt` and try again.

   - **A kernel written as a FACTORY needs no rewrite — it needs `--config-key`.** When the tile
     values live in a per-case dict that is `**`-expanded into a kernel factory
     (`_KERNEL_CONFIG = {10: dict(mm_l0=64, ...)}` → `_make_static_kernel(..., **cfg)`), the tile
     CALLS carry no literals and never will: normalising them would specialise every case at once.
     Pass `init --config-key 10` instead. It marks that entry `# panko:tune`, inserting an empty
     `10: dict(),` when the table omits the case (a no-op — the factory's defaults then apply), and
     the core tunes the entry's numbers as ordinary sites. Without it the tile lever is DEAD on
     these kernels and every tile value is swept by hand. `init` refuses `--config-key` on a
     kernel that has no such table, rather than look like it is tuning one.

   **The line: normalization exposes parameters and does not change values.** `set_vec_tile_shapes(TILE_B, TILE_D)`
   → `set_vec_tile_shapes(16, 64)` is the whole of it. Changing that 64 to something larger "while we are here",
   flattening 2-D to 1-D, or reordering operations are search actions — they must earn their place
   through the ratchet, and moving them here would corrupt the preopt baseline every speedup in this
   run is divided by. `normalize` refuses all three.

1. Run the frozen eval `E(preopt)` → `(s0, p0)` on the **normalized** kernel (evaluator.md; wrap the
   run in `timeout <eval_timeout_s>`). If `s0 == 0`: return `{"directive":"STOP","reason":"preopt_incorrect"}`.
   - `measure_latency` also reports **`compile_s`**. Pass it to `init` as `--preopt-compile-s <compile_s>`:
     `eval_timeout_s` is derived from it, because the right value is a property of the operator:
     compile time rises with the workspace a kernel allocates, and a candidate grows it further.
     Without it the configured default stands and `config.eval_timeout_source` records that it was
     not derived.
   - **Pass the whole result as `--preopt-measurement`**, exactly as `record` takes
     `--measurement`. The baseline is an archive point like any other, and every declared
     objective reads its value from there — `workspace` above all, which `init` REFUSES to
     start without. Spelling the readings out one flag at a time
     (`--preopt-workspace-b` / `--preopt-workspace-state`, `--preopt-util`, `--preopt-bubble`)
     still works and still overrides it; it is just the way to lose one.
   - It also reports **`samples`** (the 9 warm readings behind the median), and
     `--preopt-measurement` carries those too; `--preopt-samples <comma-separated>` passes them
     alone and overrides it. They do two jobs: they seed the run's own noise estimate
     (`noise.sigma_log`), which is operator-specific — the cv varies widely across operators — and
     is what the margin-based stop reads — without them `noise.source` stays `unmeasured` — and they are
     what lets the FRONT compare the baseline on an interval. A preopt with no samples is a
     permanent front member judged on exact medians while every other pair uses the bootstrap.
2. `HARNESS init --op-dir custom/<op> --op <op> [--p-ref <reference kernel's us>] --device <DEVICE> --config '<json>' --preopt-s <s0> --preopt-p <p0> --preopt-measurement '<measure_latency result as JSON>' --op-file <op_file>`
   (no `--catalog`: it defaults to this skill's own. `--preopt-hash` is optional
   and verified against `op_file` if given.)
   - If `E(preopt)` reported core-util % / bubble %, pass them as `--preopt-util <u0> --preopt-bubble <b0>` — the harness then biases the frontier toward the actions that fix the current bottleneck (deterministic symptom → V boost). The harness re-applies this at every `close` from the measured metrics, so the frontier always points at the *current* bottleneck. **Required whenever `symptom_boost > 1`**: without them the harness cannot infer a symptom and profile anchoring silently does nothing. `init` returns a `warning` field when they are missing.
3. `HARNESS select --op-dir custom/<op>` → action `a` (the harness restores the running best into `op_file`).
4. Return `{"directive":"CODE","op_file":"<op_file>","action_id":"<a.id>","intent":"<a.delta>"}`.
   - If `select` returned `bayesian_optimization.applied == true`, add `"coder": false` and set `intent` to what the
     core already applied (e.g. `set_vec_tile_shapes(32, 256) [core/BO, seed=floor]`). See
     "Tile actions" below.

## Cycle — STEP
0. **Check static feasibility first**: `HARNESS feasible --op-dir custom/<op> --parent <action_id> --op-file <candidate_file>`.
   On `feasible: false` the candidate never reaches the device: it costs no `eval_budget`, the harness has already
   rolled the working file back, and `stop_refine` tells you whether the action is now exhausted. Skip `probe` and
   `record` entirely and propose the next concrete δ (or close the action). Do not argue with the rejection: the
   reason names the rule and the numbers.

   **`retune: true` in the same response.** The δ you just applied moved the tile space: the view's
   extents, the dtype (kernel-wide or at one vec site) or the residency changed, so the tile this
   program inherited was chosen for a space that no longer exists. Run `block --parent <action_id>` **before** `record`, so the
   candidate the ratchet judges is the (structure, tile) **pair**.

   This is not a nicety. A structural change submitted at its stale tile measures worse, is reverted
   as unimproved, and its bytes are burned into the eval cache — so the combination that would have
   won becomes unreachable for the rest of the run. A hand-optimised kernel is typically a structural
   change **and** the tile the new structure wants; neither half wins alone, and a search that judges
   one variable at a time cannot reach it at all.

   A rename, a comment or a new tile value leaves the space alone and `retune` stays false. If you
   believe the optimum moved anyway — a rewrite that changed the access pattern without changing any
   extent — you may ask for a block yourself. That block's outcome decides the action's fate:
   the core owns the mechanical detection, you own the judgement call and you pay for it.
   - **A rejection does NOT advance `n`.** `n` counts measurements that failed to improve, and the gate never ran
     the program, so nothing was learned about the action. Rejections are counted in `n_static`, which is one of
     the tallies behind `n_unmeasured` — every proposal that produced no measurement — bounded by
     `unmeasured_K` (10) so an action whose whole neighbourhood is infeasible still terminates. When it fires,
     the response carries `stop_cause: "unmeasured_K"`.
   - **So do not treat a rejection as a signal to give up on the action.** It is a signal about one number. The
     reason string names the binding constraint — read it and move the candidate to the other side of it. A
     `UB estimate 264KB (view 256KB + 2x tile 4KB) > budget 192KB` says the *view* is what does not fit, so
     shrinking the tile cannot help and enlarging it makes the message worse.
0b. **Probe the cache — MANDATORY, never skip this.** `HARNESS probe --op-dir custom/<op> --code-hash <sha256(candidate_file)>`.
   On `hit`: reuse the returned `(s, p)` and **skip `E`**. On miss: run step 1.
   - `record` now **rejects** a hash it has already seen: no budget is charged, `n` is not
     incremented, and no candidate is registered. Skipping `probe` therefore does not smuggle a
     duplicate through, it only wastes a device measurement before the rejection.
   - A hit means the candidate file is byte-identical to something already measured. The usual
     cause is that the coder was handed the global best by `select` (see `already_measured` in its
     output) or its own local best by a `record` rollback, and changed nothing. **That is evidence
     the action does not apply to this kernel.** Do not re-issue the same δ: either propose a
     materially different concrete δ, or close the action.
1. Run `E(candidate_file)` → `(s, p)` (frozen method, `timeout`) — only on a cache miss.
   `measure_latency.py` runs the measurement under msprof itself; `msprof` must be on PATH.
   **If it returns `p: null` with `p_error`, the objective was not obtained** (msprof missing,
   no MIX_AIC rows, rows that do not map to calls): do not record the candidate as `s = 0`.
   Stop the run (`stop --force objective_unmeasured`) and tell the user what `p_error` says.
   Verify the hash you are about to evaluate differs from `already_measured` returned by `select`.
2. `HARNESS record --op-dir custom/<op> --parent <action_id> --code-hash <sha256(candidate_file)> --s <s> --p <p> --samples <samples> --measurement '<measure_latency result as JSON>'` → `{J, improved, n, stop_refine}`.
   - **Pass `--measurement`.** It carries every reading `measure_latency` produced, and an
     objective reads its value from there. Spelling readings out one flag at a time
     (`--util`, `--workspace-b`, `--metrics`) still works and still overrides it, but a
     reading nobody spells out by hand is simply lost.
   - `--workspace-b` / `--workspace-state` are `measure_latency`'s own fields (the msprof
     workspace, read from the runtime's `alloc` callback). Pass them on EVERY record, kept or not:
     they are how a single-objective run leaves the trade-offs it measured on record. If the state
     is `unavailable`, pass `--workspace-state unavailable` and no size -- never `--workspace-b 0`.
   - `--samples` is `measure_latency`'s `samples`, comma-separated. Optional, but without it the
     measurement does not reach the noise estimate.
   - ⚠️ **If the run hit your `timeout` before the kernel ever compiled, do NOT report `--s 0`.**
     Report `record ... --code-hash <hash> --s 0 --unevaluated compile_timeout` instead. `s = 0` is a
     verdict about the candidate, and a timed-out compile establishes nothing about it — reporting
     it as incorrect charges `eval_budget` for a run that produced no number and teaches every
     later mechanism a false label. Compile time scales with program size, so the mistake is not
     random: it marks the large candidates wrong.
     How to tell: `measure_latency` prints `[panko-progress] first_call_done compile_s=<x>` once the
     first call returns. **Marker absent → `--unevaluated compile_timeout`. Marker present → the
     candidate compiled and then ran long, which IS a verdict, so `--s 0`.**
     The reply carries `n_timeout`, `n_unmeasured` / `unmeasured_K`, and `stop_refine` with
     `stop_cause: unmeasured_K`.
   - A response carrying `duplicate: true` means those bytes were already measured; nothing was
     recorded and `n` is unchanged. Treat it as step 0b's `hit` case.
   - The harness snapshots a kept candidate and rolls a reverted one back to the running best **for you** — no manual file ops. `improved` here means it **beat the running global best** (a candidate that only beats this action's own local best but is still slower than the global best is a regression → reverted, `n++`).
   - **`stop_cause` says why a refinement stopped, and there are three.** `stagnation_K` — this many
     candidates were measured and the readings put them off the front. `undetermined_K` — this many were
     measured and the readings could NOT order them against the front, which is not evidence about the
     action, so they never advanced `n`; the fix is more readings per measurement, not more
     candidates. `unmeasured_K` — this many proposals produced no measurement at all: statically
     infeasible, a semantic no-op, a duplicate, or a compile that ran past the budget. The reply
     carries all three counters every time, and `close` accepts any
     of them (plus `success`, and an action that produced nothing at all).
   - If **not** `stop_refine`: **propose the next concrete candidate δ for this action.**
     `close` will refuse. This is enforced now, not advised: an action abandoned
     after one or two non-improving candidates has not been tested, it has been
     sampled once or twice. If you believe the axis is exhausted, submit its
     remaining values anyway — the duplicate check rejects them without
     spending budget, and "I measured every
     value" is a claim the tree can carry while "I judged it spent" is not. The
     one case `close` still accepts early is an action that produced no candidate
     at all, because the delta turned out to be inexpressible in this build. (specific parameter / config values, e.g. `cube_l1_reuse_setting={-1:2}`; never `<same δ>` or an open-ended "refine more" — picking each concrete variation is your search, not the coder's) and return `{"directive":"CODE","op_file":"<op_file>","action_id":"<action_id>","intent":"<next concrete δ>"}`.
3. If `stop_refine`:
   1. `HARNESS close --op-dir custom/<op> --action <action_id>` (attach CLOSED, update global best).
   2. **Reflect** over the trajectory, then emit tree edits:
      `HARNESS evolve --op-dir custom/<op> --edits '{"insert":[{"parent":..,"stage":..,"delta":..,"V":..,"prior_gain":..}],"update":[{"id":..,"V":..}],"prune":[..],"prune_permanent":[..],"reasons":[".."]}'`
      - `stage` is one of `frontend` / `swimlane` / `incore` — the three the campaign cycles
        through. Anything else is read as `frontend` and the response says so in `notes`.
      - If this evolve would leave the frontier empty, insert at least one new intent in the same
        call. The response carries `expansion_required: true` when it did not.`
      Insert = new promising child intents; Update = rescore frontier V on new evidence;
      Prune = drop infeasible/redundant. **`reasons` is mandatory** (write the learned insight).
      **Calibrate `prior_gain`/`V` of an inserted (non-catalog) action to the catalog's star
      scale** (`references/action_catalog.json`: strong ≈ 0.9 / medium ≈ 0.6 / weak or speculative
      ≈ 0.3), so novel actions mix fairly with seeds — do not assign > 0.9 without hard evidence.
   3. `HARNESS stop --op-dir custom/<op>` → `{stop, reason}`.
      - If `stop` and `verify.required`: **verify before reporting.** Repeat until `steps_left == 0`
        (four steps, ABBA: preopt, delivered, delivered, preopt):
        `HARNESS verify --op-dir custom/<op>` → `{arm, op_file, runs}` (stages that arm's program) →
        run `measure_latency --op-file <op_file> --warmup 1 --runs <runs> --agg median` →
        `HARNESS verify --op-dir custom/<op> --samples <samples>` (the harness puts the delivered
        kernel back after every step). The result is a bootstrap interval on the speedup:
        `confirmed` / `unconfirmed` / `regressed`. Search-time acceptance is a point comparison, so
        the search's own speedup is biased upward; this is where it is checked. If the device goes
        away, `HARNESS verify --op-dir custom/<op> --skip device_unavailable` -- the report then says
        UNVERIFIED rather than presenting the search number as confirmed.
      - Then `HARNESS report --op-dir custom/<op>` (the harness has already restored the global-best
        `op_file` for delivery); return `{"directive":"STOP","reason":"<reason>","best":{...}}`.
      - Else: `HARNESS select --op-dir custom/<op>` → next action `a` (the harness restores the running
        best into `op_file`); return `{"directive":"CODE","op_file":"<op_file>","action_id":"<a.id>","intent":"<a.delta>"}`.
        Add `"coder": false` when `select` returned `bayesian_optimization.applied == true`.


## Tile tuning: ask for a block, get one configuration back

```
HARNESS block --op-dir custom/<op> --parent <action_id> --test-command '<E(x) command>'
              [--reason '<why now>'] [--live-tiles <n>] [--max-trials <n>] [--lint-root <path>]
```

**`eval/test_inputs.py` is the objective's input contract.** `measure_latency`
builds the wrapper's arguments from it before calling the kernel: it wants
`build_inputs(device)` (or `build_inputs()`, or an `INPUTS` sequence). **`device`
is the `--device` value VERBATIM -- the id as a string, `"7"` -- not a torch
device**, so `torch.randn(device=device)` raises `Invalid device string: '7'`.
By then `torch.npu.set_device` has been called and `TILE_FWK_DEVICE_ID` is set,
so a builder can ignore the argument and use `.npu()`, or write
`f"npu:{device}"` if it wants it explicitly. A builder that raises is reported
as `p: None` with `p_error` naming it -- nothing ran, so it is an environment
fault, never a verdict on the candidate -- but it still costs the cycle.

`--test-command` runs **without a shell**. The supported form is
`cd custom/<op> && python3 test_<op>.py`, with optional `NAME=value` prefixes and
further `&&` steps, which stop at the first failure. A pipe, a redirect, a
subshell, a glob or a background `&` is refused by name (exit 126) rather than
silently mis-run — put such a command in a script and name the script.

The core runs its own optimization loop — propose, write, evaluate, learn, repeat
against a fixed base — and hands back **one** configuration, already written to
the op_file. You then `record` it and decide what to do next.

**When to ask.** Two moments, one mechanism.

1. A tile-shape action was selected. The number is the core's; ask for a block.
2. A structural action has landed and the tile it inherited describes a program
   that no longer exists. A flatten, a change of view extents, a fusion that
   changes how many tensors are resident. The core fires a block automatically
   when its derived domain changes; ask explicitly when you believe the optimum
   moved even though the domain did not.

**What comes back.**

```json
{"ran": true, "best_config": {...}, "best_objective_us": 4.02,
 "best_mix_aic_us": 8.1, "best_workspace_b": 524288,
 "best_workspace_state": "measured", "code_hash": "…",
 "n_trials": 21, "n_feasible": 19, "stopped_early": true,
 "device_trials": 21, "charged": 20, "granted": 42}
```

The block ranks its trials on `p`, the same reading `J` uses, so `best_objective_us` is
`p`, `best_mix_aic_us` is the same launch's MIX_AIC, and `best_workspace_b` is the same
launch's workspace. Then:
`record --parent <action_id> --code-hash <code_hash> --s 1 --p <best_objective_us> --metrics '{"mix_aic": <best_mix_aic_us>}' --workspace-b <best_workspace_b> --workspace-state <best_workspace_state>`

**Pass the workspace. Every objective must have a value on every measurement, and
`workspace` is an objective in the multi-objective mode -- so a promotion carrying only
`--p` and `mix_aic` used to HALT the run (exit 4, `objective_unmeasured`) for bytes the
block had measured a moment earlier. The harness now recovers the block's own reading when the promoting
call carries none, so either way works; passing it is the form that does not depend on
that recovery. A promotion is still not a re-measurement: `--samples` is what must not
be passed, because the block already counted every trial's readings. The workspace is
not a reading -- it is one number, compared exactly -- so handing it over counts nothing
twice.**

- **Every objective must have a value on every measurement.** `config.objectives` defaults to
  `["aicore_e2e"]` alone (see Two modes); `workspace` and `mix_aic` are available but not
  default, and their readings are carried on `m` in every run whether or not they are
  objectives -- see `references/evaluator.md` for what `mix_aic` costs as a declared one. Pass `--measurement` on every `record` and
  `--preopt-measurement` at `init` and they are all covered. A declared objective with no
  value HALTS the run (exit 4, `error: objective_unmeasured`) -- it names what was missing and
  where it should have come from. A user-declared objective reads from `m`: record it with
  `--metrics '{"<key>": <value>}'`, and declare `samples_from` if it has readings to be judged
  on an interval rather than on a number.
- **There is no `latency` objective.** It meant `p`, and `p` has been two different
  instruments, so the name did not say what was being optimised. Name the instrument:
  `aicore_e2e` or `mix_aic`. `config.objectives: ["latency", ...]` is refused.
- **`J` is always `aicore_e2e`**, whatever `config.objectives` says, and `aicore_e2e` needs
  `debug_options={"runtime_debug_mode": 1}` on every top-level jit kernel. At mode 0 the
  profiler writes no "AICore End-to-End Time" line, so the reading does not exist and cannot
  be recovered by re-recording; `init` refuses and says so.
- **Dominance on latency is an interval, not a median.** Two points are ordered only when a
  percentile bootstrap of the ratio of their medians excludes 1 (the same estimator and seed
  `verify` uses); otherwise the latencies are a TIE, and the other objectives decide. So two
  points separated only by an unresolved latency both stay on the front, while one that is not
  measurably slower and uses strictly less workspace dominates. Pass `--samples` on every
  `record` — without readings the comparison falls back to the medians. The workspace is
  compared exactly; it has no noise.
- **To settle an undetermined pair, re-measure — it is free.** `record --remeasure --code-hash
  <hash> --samples <...>` adds readings to a point the archive already holds: not charged to
  `eval_budget` (which counts distinct programs measured), no `n`, no ratchet. The
  readings are pooled, `p` becomes their median, and the front is re-derived. It refuses a hash
  the archive does not hold, and refuses without `--samples`.
- **Every selection is recorded.** `select` appends the full candidate list with each node's
  `V`, `g`, `w_stage` and `V_sel` to `optimization/selects.jsonl`, and the log line names the
  runner-up and its score so it says how close the call was. Nothing asks you to do anything
  with it; it is what makes a later question about the search's own decisions answerable.
- **There is ONE search, and it runs on the front from the first evaluation.** A candidate is
  KEPT when no archive point dominates it and it is no worse than the preopt on the primary
  objective (`aicore_e2e`) — so a program that is slower than the preopt is never kept whatever
  it saves elsewhere, and a trade-off that is no slower IS kept and the action continues from it.
  `record` answers with `improved` (accepted), `faster` (the latency question) and `acceptance`
  (the margin, the reference and what it was compared against). With one objective this is the
  old `j > bar`, verdict for verdict.
- **A candidate the readings cannot order is not counted against the action.** `stagnation_K`
  treated every non-accepted candidate alike, and a large share of measured candidates sit inside
  the measurement's own spread. So a rejection advances `n` only when the readings put the
  candidate off the front — a front point dominates it on the interval, or it is slower than the
  preopt by more than the readings explain. Otherwise it advances
  `n_undetermined`, which closes the action at `undetermined_K` (default 7, so an action is
  bounded at 14 candidates) with `stop_cause: undetermined_K`. `acceptance.clarity` on every
  `record` says which it was, and the trajectory row says `undetermined, n unchanged`. The fix
  for an action that only ever produces undetermined candidates is more readings per measurement,
  not more candidates.
  `select` restores the front point the last `close` chose — `restore_code_hash` says which, and
  `base_code_hash` is on every `select` and `close`. That point is the one with the largest
  exclusive hypervolume contribution among the points no worse than the preopt and non-dominated
  on the medians: the emptiest neighbourhood on the front. With one objective only the fastest
  point contributes, so the search accumulates greedily exactly as before.
  A run ends on `success`, the budget, the wall clock, `frontier_exhausted` — or
  `hv_stagnation`: `hv_stagnation_limit` (default 48) measurements since the front last GREW.
  Growth is the OR of two signals, because neither sees every real improvement: the hypervolume
  gain clearing `hv_min_gain` (1%) by more than the readings' own noise, or an axis RECORD moving
  by more than `axis_min_gain` (1%, or `axis_min_gain_bytes` 5% on a byte axis). The hypervolume
  is a volume, so the box's shape weights the axes and an 8% workspace cut is 0.14% of it;
  conversely a point filling the middle of the front moves no record and is only visible as
  volume. The reference is FIXED for the run — the preopt on the primary objective, the chip's
  memory on a byte axis, which is measured as log(bytes) — so `hv` reads as a progress curve and
  a point with a worse workspace no longer inflates the volume of everything already there.
  The stop payload and the report carry `hv`, `hv_stagnation` and the last counted gain;
  `hv_stagnation_limit: null` leaves the measurement and removes the halt.
  `progress.front_stagnation` still counts measurements since the archive last accepted any
  point, reported and not enforced.
  The two-phase run (`mo_phases`, `greedy_stagnation_limit`, `front_stagnation_limit`,
  `progress.phase`) is gone: a phase 1 that pursues latency alone is a single-objective run, and
  the switch never fired on a real campaign anyway.
- **Every front point is verified, not only the delivered one.** After the delivered point's
  ABBA pass finishes, `verify` opens the next front point's pass against the same preopt and
  says so (`status: pending`, `point`, `front_points_left`); keep calling `verify` until it
  reports a verdict with no point left. `impl.py` holds the delivered point throughout, and
  none of it is charged to `eval_budget`. Each point's verdict lands in `frontier.json` and in
  the report's frontier block. `config.verify_front: false` verifies the delivered point alone.
- **The report ends with an Accounting block and a Recommended block.** The accounting says
  where the budget went (tile blocks vs structure, repaired draws, free rejections), how long the
  front has gone without a new point and which point the search is extending, and what the
  archive holds — all of it already in the state, written out because
  a run is reviewed from this file. The recommended block names the best point on each
  objective (a fact about the front) and one `balanced` point under a stated convention (the
  knee: min-max normalised, nearest the ideal corner). With two points there is no knee, and
  the report says so. None of it moves `impl.py`.
- **The frontier is a deliverable.** `stop` and `report` write every non-dominated
  (latency, workspace) point to `optimization/frontier/` with its source and measurements, and the
  report ends with which point `impl.py` holds. That is the best PRIMARY objective among every
  correct measurement whose source is on disk, ties broken by the remaining objectives — read off
  the measurements rather than off `best_node`, which is `J` and only moves at a `close`, and can
  let a run ship a program another front point beat. Do not ask which point to ship:
  the preference belongs to whoever weighs this kernel against the others on the device.

- **Too-big draws are repaired, not rejected.** A draw that overflows a buffer (UB, L0A/B/C, L1,
  or the learned ceiling) is shrunk one ladder rung at a time on the offending axes until it fits,
  and that point is measured. So `n_repaired` trials in the log line cost a device evaluation
  each, where a rejection was free. **The value is filed at the params that were MEASURED**, and
  the drawn point gets the penalty a rejection would have given it — telling the surrogate the
  infeasible region is fine is what made it keep drawing there, and with repair on every such
  draw costs an evaluation. `best_config` is always the measured config.
  `config.bayesian_optimization_repair: false` restores rejection.
- **Blocks may never have taken more than half of what the run has spent**
  (`config.block_budget_share`, 0.5), counted across every block. It is a RUNNING RATIO against
  `evals_used`, not a slice of `eval_budget`, so it applies whether or not the run names a
  budget -- `eval_budget: null` used to remove the guard entirely, because a share of an
  unstated total is undefined. Tile numbers are the one lever the model cannot propose, but an
  action the search never reaches contributes nothing: with the share off, a run can spend
  nearly its whole budget on repeated blocks of a single action and deliver its preopt.
- **One block may take at most `config.block_max_trials` device trials** (12). Separate from the
  share, which bounds the run's total tile spend and so lets a single block take all of it, and
  separate from `failure_limit`, which only brakes a block that cannot search at all. Without
  it a handful of blocks can take half a run's evaluations between them and return almost no
  feasible trials, with neither the share nor `failure_limit` binding any of it. The cap is
  also the FLOOR the share may not push below, so the first block of a run is usable.
- **The run delivers the best program it measured**, closed or not. If the budget or the clock
  ends the run while an action is still open, `stop` promotes that action's best first when it
  beat the global best (`progress.promoted_at_stop`, and a `promote-at-stop` line in the log).

- **Do not dispatch the coder.** The winner is already on disk, and every tile
  binding site moved together. A coder that changes the kernel's tile but not the
  wrapper's padding arithmetic ships a kernel that fails verification; the action
  is then scored unproductive and closed. That mistake has cost real wins.
- **Do not re-measure it.** The block measured every objective on the device.
  Pass them through -- `best_objective_us`, `best_mix_aic_us` and
  `best_workspace_b` -- and pass no `--samples`.
- **Do not propose a tile value of your own,** before or after. Inside tile
  tuning the concrete number is the core's; your search is which action to take
  and what the result implies for the *others*.
- **When an action's intent must name a cube tile anyway, it is `[L0, L1]`.**
  `pypto.set_cube_tile_shapes(m, k, n)` takes the three AXES, each a two-element
  pair with **L0 first**, under `0 < L0 <= L1` and `L1 % L0 == 0`. So `[256, 128]`
  is not "L1 256, L0 128" -- it is an L0 larger than its L1 and not a divisor of
  it, and the lint gate rejects it before the device is touched. Four places
  agree and none of them is this one: the lint gate (citing the pypto API doc),
  `bo_apply`'s writer (verified against the installed pypto), the lever (which
  draws `(L0, L1/L0)` so the constraint holds by construction) and the
  optimization catalogue. An inverted intent costs a cycle: the coder is right
  to refuse to reorder the literal, because reordering it is reinterpreting the
  delta, which is yours. Write the pair in this order and the question does not
  arise.

**`aborted` naming environment faults ends the block. Whether it ends the RUN is
not yours to decide — call `stop` and relay what it says.** The evaluator already
waited the device out before reporting a fault at all, and `stop` then asks the
device a question of its own before ending anything. Two answers come back:

- `device_unavailable` — the probe confirmed it. Return the STOP.
- `harness_fault` — the command never started; the fault is a bug here, not a
  busy device. Return the STOP and quote the marker.
- no reason at all — the device recovered. **Continue.** Select the next action
  normally; the aborted block's observations are kept and a later block on the
  same structure resumes from them.

Deciding this in the optimizer ends a campaign a fraction of the way into its
budget, on faults that had cost no budget at all. Never assert a device is
unavailable from a block that finished some time ago.

`ran: false` or `best_config: null` carries a `reason`. The one that matters:
*"no tunable tile site; the tile call is not written with integer literals"* means
normalization at INIT did not happen or did not succeed, and **no amount of
searching will make the core own this kernel's tile.** Say so in your reflection.

**Budget.** Every trial is a device evaluation. `charged` is what the block spent;
the remaining one is charged by your `record`. `granted` is what the core allowed
after clamping your `--max-trials` against the remaining budget and the run-level
share. A block stops on seven consecutive non-improving trials, so a converged
block hands the rest back.

**`--live-tiles` is a fact, not a bound.** How many tile-shaped tensors are
simultaneously resident is a dataflow property the extractor cannot read after
you have fused or flattened anything, so you may supply it. You may not supply a
domain, a range, or a maximum: those the core derives, and a domain narrowed by
judgment is the one error that leaves no trace anywhere in the run.

**Omitting it is not neutral.** Without it the footprint bound is OFF and the
block checks alignment only, so every tile too large for the buffer costs a
device evaluation instead of being rejected for free. It is easy to forget, so
`block` says so in the trajectory and in its own output.

**A block is worth spending only when the tile space actually moved.** A
structural delta that changes a view extent triggers `retune: true`, but that
flag says the space CHANGED, not that it changed usefully. Raising an extent on
a shape whose loop count is already 1 produces the same single-task program
twice over, so a second block re-searches the space the first had just
converged on. Before accepting a retune, ask what about the tile space is
different; if the answer is "nothing the core can see", the delta is a tile
proposal wearing a structural name, and the core has already searched that axis.

## Tile actions: the core has already written the file

`select` returns an extra `bayesian_optimization` block when the chosen action is one of the four
tile-shape levers (`F-9`, `F-10`, `S-11`, `S-12`) and `bayesian_optimization_tile_search` is on:

```json
"bayesian_optimization": {"applied": true, "config": {"vec#0": {"M": 32, "N": 256}},
       "label": "floor", "scope": "vec", "trial": 2, "sites": 1, "code_hash": "..."}
```

**Each action moves one site kind, and only that one.** `F-9` and `S-11` are Cube
actions and reach Cube sites; `F-10` and `S-12` are Vector actions and reach Vector
sites — `scope` names which, and the config handed back contains nothing outside it, so
a gain is attributable to the action that earned it. A Cube action on a kernel with no
Cube work declines with `no cube tile site` rather than quietly tuning the Vector tiles;
retire it. Searching Cube **and** Vector together is still available and still wanted —
that is `block`, which you ask for deliberately and which is recorded as a block. The
two actions over one kind share one study (per structure), so the second continues the
first's surrogate instead of re-exploring the same space on fresh device evaluations.

`record` returns the same `bayesian_optimization` block whenever the action continues
(`stop_refine: false`) and it is one of the four tile levers. Treat it exactly
as you treat the one from `select`: **the value is already on disk**, so emit
`{"directive":"CODE", ..., "coder": false}` and do not propose a tile value of
your own. Inside a tile action the concrete variation is the lever's to pick,
not yours — that is the whole reason the lever exists. Your search continues to
be the choice of action, and the reflection at `close`.

**`applied: true` means the candidate is already on disk.** The deterministic
core chose the number from its own Bayesian study and rewrote the tile call's
argument list itself. Two consequences, and both are contractual:

1. **Do NOT emit `CODE` for it.** Emit `{"directive":"CODE", ..., "coder": false}`
   so the orchestrator skips the coder dispatch, or evaluate it yourself in this
   same cycle. Handing this to the coder would overwrite the value the core just
   wrote, and the study would receive the measurement of a program it did not
   propose. The search would then learn nothing at all, silently.
2. **Do not choose a tile number.** For these four actions the number is not
   yours. Your job is whether to select the action at all, and what the result
   implies for the *other* actions.

This is the one place where "picking each concrete variation is your search"
does not apply. It applies everywhere else, and this is exactly why: a tile
value is a number the core can compute from the program and the hardware, and
the other forty-four actions are not.

`applied: false` with a `reason` means the lever had nothing to propose (no
tunable tile site, or the space is exhausted). Treat the action as an ordinary
one and pick the δ yourself.

## Return format (to the orchestrator) — exactly one JSON directive, nothing else
- `{"directive":"CODE","op_file":"custom/<op>/<op>_impl.py","action_id":"u13","intent":"<δ>"}`
- `{"directive":"CODE","op_file":"custom/<op>/<op>_impl.py","action_id":"u10","intent":"<δ>","coder":false}`
  — the core already wrote it; the orchestrator must not dispatch the coder.
- `{"directive":"STOP","reason":"success|converged|wall_clock|frontier_exhausted|global_stagnation|eval_budget|preopt_incorrect|device_unavailable|harness_fault","best":{"aicore_e2e_us":..,"mix_aic_us":..,"J":..,"speedup_vs_preopt":..}}`
  (`device_unavailable` and `harness_fault` come from `stop` after a block aborted on environment
  faults. Report whichever it returns; never choose between them yourself.)
  (`converged` = a full frontend→swimlane→incore cycle produced no ≥ `stage_min_gain` improvement.)

## Hard rules
- Read `search_state.json` every dispatch; never rely on session memory. Persist only via the harness.
- Never evaluate a candidate without probing first. Duplicate submissions are rejected by the
  harness and counted in `progress.duplicate_rejected`; a high count means actions are being
  selected that the kernel cannot express, not that the coder is failing.
- `prune` retires an action **against the structure now in the file**, not forever. Your evidence
  is a set of measurements, and measurements are taken on a program: once a later action
  restructures the kernel, the disproof is stale and the action returns to the frontier as an
  ordinary candidate ranked by `pypto_action_priority` like any other. This is not a suggestion to
  revisit it — the harness does it, and the response names what it retired under
  `pruned_on_structure` and the structure hash under `structure`. **Geometry is conditional on
  structure**: a tile sweep or a loop-granularity sweep disproves those values *for the shape the
  program currently has*, and a restructuring that changes a cube extent or a loop trip count
  invalidates it. A tile VALUE is not a restructuring, so a BO trial does not revive anything.
- `prune_permanent` retires an action outright, through any restructuring. Use it only when the
  intent is **inexpressible on this operator** — it would change the public contract, or the
  option does not exist in this build — and say which, from the API or the schema rather than from
  a measurement. A mechanism that merely lost is a `prune`, not a `prune_permanent`.
- Pruning an action that is currently *held* is ignored under either key: unmet preconditions are
  the harness's business, and a later transformation may make that action reachable.
- **Extending the frontier is your job; ending the search is not.** `insert` is the world model:
  when the trajectory teaches you something the catalogue did not contain, add it. An `evolve`
  that leaves the frontier empty without inserting anything is a stop request, and the harness
  will not honour it. `select` then returns `reason: expansion_required` and you must reflect and
  insert. If there is genuinely nothing left, say so on the record with
  `evolve --edits '{"exhausted":"<why>"}'`; only a declared exhaustion permits
  `frontier_exhausted`. Pruning is for retiring intents you have disproved, never for clearing
  the board.
- Do not call `state_transition` (subagent). Do not change the frozen eval method or thresholds.
- Preserve the operator's public contract (name / args / shape / dtype / layout); only the internal
  implementation is tuned. DEVICE is a per-run user parameter — never hard-code it.
