# PANKO state & output schema

Two files live under `custom/<op>/optimization/` (created at runtime; not committed):

- `search_state.json` — machine source of truth: the co-evolving world-model tree.
- `<op>_optimization.md` — human-readable report + append-only log.

The world model is the **persisted tree**, not the optimizer session. A fresh optimizer
dispatch re-hydrates the world model from these files, applies one transition, and persists.

## search_state.json

```json
{
  "op": "<op>",
  "device_id": "<TILE_FWK_DEVICE_ID at run start>",
  "target": { "metric": "aicore_e2e_us", "P_ref": 0.0, "P_ref_via": "explicit:cli",
              "lower_is_better": true },
  "config": {
    "objectives": ["aicore_e2e"],
    "stagnation_K": 7,
    "undetermined_K": 7,
    "unmeasured_K": 10,
    "staged_prior": true,
    "wall_clock_limit_s": 86400,
    "idle_grace_s": 7200,
    "eval_timeout_s": 300,
    "eval_budget": 300,
    "success_stop": true
  },
  "progress": {
    "active_stage": "frontend",
    "stage_stagnation": 0,
    "evals_used": 0,
    "wall_clock_s": 0,
    "best_J": 0,
    "preopt_aicore_e2e_us": null,
    "best_aicore_e2e_us": null,
    "preopt_mix_aic_us": null,
    "best_mix_aic_us": null,
    "best_speedup_vs_preopt": null,
    "stop_reason": null
  },
  "nodes": {
    "closed": [
      { "id": "x12", "parent": "root", "stage": "frontend",
        "code_hash": "<sha256>", "J": 0.0,
        "eval": { "s": 1, "p": 0.0, "m": { "util": 0.0, "bubble": 0.0 } } }
    ],
    "open": [
      { "id": "u13", "parent": "x12", "stage": "swimlane",
        "delta": "<natural-language optimization intent>",
        "V": 0.0, "prior_gain": 0.0, "tried": 0 }
    ]
  },
  "best_node": "x12"
}
```

Field notes:

- `target.P_ref` — the TARGET for the primary objective (µs), normally the reference kernel's
  number. `J = s · (P_ref / p) · 100`, so `J ≥ 100` means the target is met, and that is the
  `success` halt; it is asked of the program the run would DELIVER. **`init` refuses a `P_ref` the
  preopt already meets** (`j0 ≥ 100`, `error: target_already_met`); `config.success_stop: false`
  says the kernel genuinely beats the reference, and the run ends on the budget or the clock.
- `target.P_ref_via` — `explicit:cli` when `--p-ref` was given, `preopt` when it was not.
  **`--p-ref` is optional**: without it `P_ref` is the preopt's own reading, `j0` is 100 by
  construction, `success_stop` is turned off with it (asking for both is refused), and `J` reads
  as a percentage of the baseline.
- `config` — run parameters. The block above is ABRIDGED: `init` writes all of `DEFAULT_CONFIG`,
  which is the one place the defaults live. The ones shown: `objectives` picks the search (one
  name = single-objective, the default; see SKILL.md, Two modes); `wall_clock_limit_s` bounds the
  time the campaign was WORKED, with a gap longer than `idle_grace_s` not counted;
  `eval_timeout_s` is the floor of the per-candidate cap; `eval_budget` caps evaluations (null =
  unlimited); `stagnation_K` / `undetermined_K` / `unmeasured_K` are the three close conditions;
  `success_stop` halts on `J ≥ 100`.
- `progress.stop_reason` — **null until `stop` returns a real halt**; only then does the harness
  write the reason (`success` / `wall_clock` / `frontier_exhausted` / `global_stagnation` /
  `eval_budget` / `converged`). Sole source of truth for the Stage-7 completion gate:
  the orchestrator **cannot `complete_stage(7)` until this is set**, so it must relay until the
  optimizer emits `STOP`. Two further reasons come from the device rather than the search:
  `device_unavailable` (the last block aborted on device faults and a fresh probe agrees) and
  `harness_fault` (it aborted on invocation faults).
- `progress.harness_fault` — why a `harness_fault` halt fired: `markers` (the distinct fault
  markers the aborted block saw), `block`, `env_faults`, and `hint`, one sentence naming what to
  fix. None of these faults were charged to `eval_budget`.
- `verification` — the ABBA pass now running, opened by the first real `stop`
  (`config.end_verify`, default on); `point` says which program it is verifying, `verifications`
  holds the passes already finished and `verify_queue` the front points still to verify
  (`config.verify_front`, default true). Each pass is that point against the SAME preopt, so
  drift cancels within the pair. `status`: `pending` → `confirmed` (95% bootstrap interval of
  median(preopt)/median(delivered) above 1) / `unconfirmed` / `regressed`; or `no_change`,
  `skipped`, `disabled`, `insufficient`. `arms.{preopt,delivered}` hold the readings, measured
  ABBA in `config.verify_runs / 2` per step; `result` holds the interval; `search_speedup` is
  what the search reported. The verdicts reach `frontier.json` as `front[i].verified`.
- `config_key` — which entry of a factory kernel's per-case config table this campaign tunes:
  `{key, action, knobs}`, or null for an ordinary kernel. `action` is `marked` or `inserted`. The
  key ALSO lives in the kernel, as the `# panko:tune` comment `init --config-key` writes, and that
  comment is what every later command reads, so it travels with every snapshot. `config_sites.py`
  holds the admission rule; the knobs become ordinary `cfg:` sites.
- `device_id` — recorded for reproducibility. DEVICE is a **per-run user parameter**, never
  hard-coded.
- `nodes.closed[*]` — visited actions with the best program attached: `code_hash`, `J`, and the
  observation `eval = (s, p, m)` (correctness, latency µs, metadata such as util/bubble).
  Also `action` (the u-id that was refined), `catalog_id` (null for an `insert`ed action) and
  `delta`, so what produced a closed node is readable from the state alone. The log carries the
  same attribution on its `- select[...]` and `close` lines.
- `nodes.open[*]` — the frontier `A(S)`: pending intents `(parent, delta)` with priority
  `V ∈ [0,1]`, the catalog `prior_gain`, and a `tried` counter. Selection is the argmax of
  `V_sel = w_stage * g * V`. Under `config.prior_gain_source: "learned"` (default, when a learned
  table can be read) a seeded node's `prior_gain` is the catalogue star moved by measurement
  (`action_posterior.learned_prior_gain`), the star kept as `prior_gain_catalog`; an `insert`ed
  action's g comes from the table's `insert` class (`prior_gain_model` keeps the model's own).
  g is recomputed at every `select` from this run's `action_stats`, read at the
  `config.prior_gain_quantile` (0.8) of the posterior rather than its mean (Bayes-UCB).
- `best_node` — id of the closed node with the highest `J`.
- `prior_table` — which action prior table this run reads: `{source, origin, sha256, snapshot}`.
  `origin` is `explicit` (`config.action_prior_table`, e.g. an experiment), `learned`
  (`$PANKO_HOME/action_prior_table.<soc>.json`, default `~/.panko`) or `none` -- NO table ships,
  so a fresh machine reads the catalogue stars alone. `init` copies it to
  `optimization/action_prior_table.snapshot.json` and every later read uses the copy, so the table
  never changes under a running campaign.
- `optimization/action_outcomes.jsonl` (+ `$PANKO_HOME/outcomes/<soc>/<campaign>.jsonl`) — one
  line per measured candidate: `catalog_id`, `delta_hash` (the catalogue text it was measured
  under), boolean predicates, `s`, `kept`, `log_gain`, `selection`, `propensity` (thompson only),
  `soc`, `phase` and `front_insert`. The table models how often an action makes the program it is
  applied to FASTER, so `kept` is `faster` and `log_gain` a latency gain; `accepted` and
  `front_insert` record the ratchet's and the archive's verdicts separately. **The store and the
  learned table are per chip**: a run reads and rebuilds only its own chip's rows, dropping the
  rest with counts in the table's meta. After a real stop `report` rebuilds the table from the
  shared store (`scripts/prior_store.py`, weight `0.5 ** (age / config.prior_half_life)`) unless
  `config.prior_auto_update` is false or the table is pinned.
- `action_stats` — this run's measured candidates per action, `{n, kept, sum_log_gain}`, keyed by
  `catalog_id`. Recorded always; read by `config.selection: "thompson"`, which draws each open
  action's kept rate from a Beta posterior and selects the largest `w_stage * p` (`* g` with
  `thompson_use_gain`, off by default). No cell weighs more than `TABLE_CAP` = 50 candidates.
  `progress.select_draws` numbers the draws. Default `selection` is `"product"`.
- `nodes.held[*]` — actions whose structural precondition the kernel does not
  currently satisfy, carrying the same fields as `open` plus `requires`. Held,
  not discarded: `close` recomputes the predicates from the new best program
  and moves any that now qualify back into `open`. A kernel with no matmul can
  acquire one, and deleting the cube actions up front would remove that path.
- `predicates` — what the current best program contains: `has_cube_op`,
  `has_broadcast`, `has_chunked_scan`, `module_count`. Derived from the AST, so
  a mention in a comment is not evidence. Undeterminable values admit the
  action rather than hide it.
- `progress.static_rejected` — candidates refused by the feasibility gate before
  reaching the device. Counts towards an action's stagnation, not towards
  `eval_budget`: no device time was spent.
- `progress.unevaluated_rejected` / `progress.unevaluated_reasons` — candidates that **were never
  measured**, by reason (compile past `eval_timeout_s`, device unavailable, allocation failed).
  Device time was spent, unlike `static_rejected`, but no number was produced, so they are **not
  charged to `eval_budget`** and **not recorded as `s = 0`**. Counted in the refine record's
  `n_timeout`, one of the tallies behind `unmeasured_K`.
- `config.eval_timeout_s` / `config.eval_timeout_source` — the per-candidate cap, **derived** at
  `init` from the preopt's `compile_s` (`eval_timeout_k` x it, clamped to `eval_timeout_floor_s`
  / `eval_timeout_cap_s`). `source` is `derived:preopt`, `explicit:config` or `default`, so a
  campaign's own file says which number it ran under.
- `noise` — the run's own measurement noise, `{sigma_log, latest_sigma_log, recent[<=32],
  measurements, source, protocol}`. `sigma_log` is the stdev of log(latency) over one
  measurement's warm readings, aggregated as the **median** of the last 32. Seeded at `init` from
  `--preopt-samples`, updated by every non-duplicate `record --samples`. `source` is `measured` or
  `unmeasured`, never a transcribed constant; `protocol` names the measurement it was taken under
  (`warmup=1,runs=9,agg=median`).
- `eval_cache[<hash>].m.workspace_b` / `.m.workspace_state` — bytes the runtime requested through
  the jit wrapper's `alloc` (`scripts/workspace_probe.py`), on every measured candidate.
  `measured` = read; `zero` = the kernel ran and `alloc` was never called; `unavailable` = no size
  could be read, and then there is NO `workspace_b` — a failed probe is never a 0.
  `workspace_launches` is an invariant check: a wrapper always invokes its kernel once, so 0 means
  the hook is on the wrong class object. Diagnostic in a single-objective run.
- `refine.n` / `refine.n_undetermined` — the two close counters. A rejected candidate advances `n`
  only when the READINGS put it off the front: an archive point dominates it on the interval
  (`dominates_measured`), or it is slower than the preopt by more than the readings can explain.
  Otherwise it advances `n_undetermined`, which closes the action at `config.undetermined_K`
  (default 7) with `stop_cause: undetermined_K`. An acceptance resets both. With no `--samples`
  on either side every rejection is `clear` (`acceptance.clarity`).
- `block_winner` — the ONE cache entry a block pops, stashed with the trial's `s`, `p`, `m`,
  `samples` and `code_hash`. The pop is required because `eval_cache` doubles as the duplicate
  index; the stash is what `_block_reading` falls back to (the cache first) when the promoting
  `record` carries no `--workspace-b` or `--samples`. A resumed block can return a winner replayed
  rather than measured here; then the promotion must carry its own readings.
- `eval_cache[<hash>].verdict` — what the ratchet did with the candidate (`kept` / `tied` /
  `reverted`), so a discarded trade-off stays distinguishable from a superseded winner.
- `config.objectives` — what the run optimises, IN ORDER; the first is the primary objective
  (`impl.py` holds its best point, and it is the only axis with a reference point). Default
  `["aicore_e2e"]` — one objective, so by default there is no front; naming a second
  (`["aicore_e2e", "workspace"]`) turns the multi-objective mode on (SKILL.md, Two modes). An
  entry is a built-in name (`aicore_e2e` = `p`, the profiler's AICore End-to-End Time; `mix_aic`
  = `m.mix_aic`; `workspace` = `m.workspace_b`) or `{"name": ..., "from": "p" | "m.<key>",
  "samples_from": ..., "direction": "min" | "max"}`, so a user adds an objective by recording its
  value and naming it here. A declared objective with no value HALTS the run (`init` refuses a
  preopt missing one; `record` refuses the candidate and sets `stop_reason:
  objective_unmeasured`).
- the trajectory log's refine line carries the objective vector and what the archive did with the
  candidate (`| latency=… workspace=…MiB | archive: inserted|dominated|tied|duplicate -> kept|reverted`),
  because under more than one objective `reverted` alone cannot say whether the point was a
  discarded trade-off or genuinely dominated. `record` returns the same as `objectives` /
  `archive` / `archive_points`.
- `archive` — the non-dominated set over `config.objectives`, maintained on every measurement in
  EVERY run (the search does not read it yet, so a single-objective run produces it for free).
  Rows are `{code_hash, values, samples_by_objective, mix_aic_us, aicore_e2e_us, workspace_b, J,
  action, catalog_id, eval_no}`, sorted by the secondary objectives. Latency is compared as
  measured (median-of-9), workspace exactly; identical objective values are `tied` and not stored.
  `progress.archive_inserted` / `archive_dropped` / `archive_kept_slower` / `archive_unplaceable`
  count the outcomes.
- `config.snapshot_all_candidates` (default true) — snapshot EVERY measured candidate under
  `nodes/<hash>.py`, not only the ones the ratchet keeps: a front point is a deliverable, and
  without this only the kept candidates survive. `progress.snapshot_misses` counts the failures.
- `optimization/frontier/` — written by `stop` and `report`: one source per archive point
  (`<rank>_<MiB>_<hash8>.py`) plus `frontier.json` with the measurements, `is_impl` marking the one
  `impl.py` holds, and `source: null` for a point whose source was not retained. `impl.py` stays
  the FASTEST point; the report lists the others and ends by saying so. Nobody is asked to choose
  at stop -- the preference lives outside the system.
- Which point that is comes from `_delivery_point`, not from `best_node`: the best PRIMARY
  objective among every correct measurement whose source is on disk, ties broken by the remaining
  objectives in order. When the two disagree, `stop` records
  `progress.delivery_outranked_best_node` and reports `best_node_code_hash` beside
  `best_code_hash`.
- `progress.delivered_code_hash` — the program `stop` put in the working file, frozen there as it
  was restored. Everything after the stop reads it rather than `best_node`: the verification arm,
  the file put back between passes, and the claim the pass is compared against. `--remeasure` can
  move the delivery winner without running the ratchet, so the two can name different kernels.
- `optimization/pareto.json` — written by `HARNESS pareto --op-dir <op_dir>` (read-only on the
  search): the non-dominated (latency, workspace) set over every correct candidate with a
  workspace reading, each labelled `delivered` / `kept` / `reverted` / `tied`. `front_reverted` > 0
  is the point: the single-objective run measured a trade-off and discarded it. `aicore_e2e_us`
  and `mix_aic_us` are the two instruments, each from its own reading; `mix_aic_us` and
  `mix_aic_vs_fastest` are null when the candidate carries no MIX_AIC reading, and
  `front_without_mix_aic` counts those. `schema` is 2; schema 1 wrote `aicore_e2e_us` under both
  names, so a reader that took `mix_aic_us` as latency wants `aicore_e2e_us`.
- `eval_cache[<hash>].samples` / `nodes.closed[root].eval.samples` — the warm readings behind `p`,
  when the caller supplied them. `p` (the median) is still what the ratchet compares.
- `op_file` — the working `<op>_impl.py` the harness snapshots/restores (set from `init --op-file`;
  falls back to `<op_dir>/<op>_impl.py`). **Accumulation is harness-owned**: `init` snapshots the
  preopt, `select` restores the running global best into `op_file`, `record` snapshots a kept
  candidate and rolls a reverted one back, `close`/`stop` leave the best on disk. The agent never
  copies node files by hand.
- `progress.refine` — in-flight local refinement of the current action: `{action, best,
  threshold_J, n}`. `threshold_J` is seeded with the **running global best `J` at the action's
  start**, so a candidate is "kept" (and `n` reset) only if it beats the global best.
  **Three counters close an action, and they are the whole rule** (`_disproved`): `n >=
  stagnation_K` (measured, off the front), `n_undetermined >= undetermined_K` (measured,
  undecidable), `n_unmeasured >= unmeasured_K` (no measurement; the sum of `n_static` / `n_noop`
  / `n_dup` / `n_timeout`, kept per kind for the reader). The record is created at the first
  rejection, not the first measurement. `close` also accepts `success` (`best.J >= 100`).
- `chip_envelope` — the buffer capacities and core counts every static rule validates against,
  derived at `init` from the platform ini: `{ub_kb, l1_kb, l0a_kb, l0b_kb, l0c_kb, cube_cores,
  vector_cores, source, soc, soc_via, confirmed}`. `source` is the ini path, or `explicit:cli`
  when the numbers were supplied. `confirmed` is the check against the device the run measures on:
  `{verdict: ok | mismatch | unknown, live_soc, via, at}`.
- `wrapper_signature` — AST hash of every top-level statement outside the `@jit` functions, taken at
  the baseline and frozen for the run. `p` is kernel time, so moving work into the host wrapper would
  improve it without the operator getting faster; a candidate whose wrapper differs is rejected.
- `archive[i].samples` — the pooled readings behind that point's latency, which is what its
  dominance is judged on (`config.archive_bootstrap`, `config.archive_alpha`). An undetermined
  interval is a TIE, so a point that is not measurably slower and uses strictly less workspace
  DOMINATES, while two points separated only by an unresolved latency both stay. The fastest
  point can therefore leave the FRONT; it never leaves the RUN. `eval_cache[<hash>].remeasured`
  and `progress.remeasurements` count readings added to a point already held, which are NOT
  charged to `eval_budget` and do not touch the ratchet.
- `optimization/selects.jsonl` — one JSON line per `select`: the whole candidate list with `V`,
  `g`, `w_stage` and `V_sel` per node, the chosen id, the stage, the base code hash and (under
  thompson) the draws. Appended, never rewritten; the state keeps only the final `V` of the nodes
  still OPEN, so this is the only record a past selection can be re-derived from.
  `progress.select_log_failed` counts records that could not be written.
- `progress.base_code_hash` — the front point the next action is applied on top of, chosen at
  every close by `_advance_front_base` (None until the first close, which means the global best):
  the largest EXCLUSIVE hypervolume contribution among the points that are on disk, no worse than
  the preopt on the primary objective, and non-dominated on the medians. Ties go to the point
  tried from least (`archive[i].tried_from`), then the primary objective, then the hash. With one
  objective this is greedy accumulation. `progress.base_selection` records the choice.
- `progress.front_stagnation` — measurements since the archive last accepted a point. A
  diagnostic, not a halt: it answers "did the front move" but not "did it move enough".
- `progress.hv` / `progress.hv_stagnation` / `progress.hv_last` / `progress.hv_last_gain` — the
  run's convergence test, the **OR of two signals** because neither sees every real improvement:
  the HYPERVOLUME the front encloses against a reference **fixed for the whole run**
  (`_hv_reference`: the preopt on the primary objective, `chip_envelope.memory_mib` on a
  byte-valued objective, the preopt otherwise, and an objective's own `reference` when it carries
  one; a byte axis measured as `log(bytes)`, `_hv_axis`), counted when the bootstrapped gain's
  lower end clears `config.hv_min_gain` (0.01); and the AXIS RECORDS (`_axis_advance`), the best
  value the front holds on each objective, counted when one moves by more than
  `config.axis_min_gain` (0.01), or `axis_min_gain_bytes` (0.05) on a byte axis.
  `panko/front.py`: `hv_reference_point` and `hv_axis` carry the reasoning. `hv_stagnation` counts
  measurements since the last counted gain, and `config.hv_stagnation_limit` (default 48) halts
  the run with `stop_reason: hv_stagnation`; None removes the halt. With no ceiling for a byte
  axis there is no reference, and the records answer alone.
- `chip_envelope.memory_mib` — the memory this kernel may hold, in MiB, and the hypervolume's
  reference on a byte axis. From the platform ini (`SoCInfo/memory_size`), overridden by
  `--chip-memory-mib` (recorded as `memory_mib_via`). **Optional**, unlike every capacity in the
  envelope: it is a budget rather than a property of the silicon.
- **There are no phases.** Acceptance is non-domination within the reference point and the base
  is the largest HV contribution, from the first evaluation, so there is no `config.mo_phases`,
  `progress.phase` or phase-switch key. Nothing halts a run on stagnation under the shipped
  defaults: `success`, the budget, the wall clock and `frontier_exhausted` end it.
- `debug_options_signature` — AST hash of the `debug_options` on every top-level `@jit` kernel,
  taken at the baseline and frozen with it. The flag is the measurement's configuration, not the
  kernel's cost, so a candidate that changed it is rejected on the same path as wrapper drift.
- `bayesian_optimization_studies` — persisted Optuna trials per `<scope>|<structural_signature>`,
  where scope is `cube` or `vec`. A study's `current` seed is the file's own tile configuration
  recorded as a completed observation, so its value is that program's measurement read from
  `eval_cache` (`incumbent_latency`), never `progress.best_aicore_e2e_us`: `select` restores the
  front base, which need not be `best_node`. With no valid measurement the seed is queued to be
  measured instead. `bayesian_optimization_memory` is the per-domain tile memory;
  `bayesian_optimization_refusals` are compiler refusals with the structures they were observed on.
- `domain_fingerprint` — the tile space the current best was chosen for; a delta that changes it
  is what makes a retune worth a block. `eval_cache` is keyed by `code_hash`, so re-proposing
  identical bytes costs no device time. `code_version` / `code_version_history` stamp which build
  wrote the state, restamped on every write.

## <op>_optimization.md

Human-readable report + append-only log:

- **Baseline** — the preopt's value for EVERY objective in `config.objectives`, each in the unit
  it is read in (`- preopt: mix_aic <us> · workspace <MiB>`), then **P_ref**, `j0`, the device,
  util and bubble. The whole run is quoted against this line.
- **Final best** — `aicore_e2e`, `J`,
  `speedup_vs_preopt = preopt_aicore_e2e_us / best_aicore_e2e_us`, util, bubble.
  (util/bubble are diagnostics — not part of `J`.)
- **Trajectory**, written as SECTIONS — one per selected action, because the only structure
  that matters is which action an event belongs to:

```
## u3 · F-3 · reduce loop count by increasing tile size / chunking

selected at eval <n> · stage frontend · V <v> · g <g> · V_sel <score> · next <action> <score> · of <k> candidates

| eval | cand | mix_aic (us) | workspace (MiB) | ratchet | front | n | note |
| ---: | :--- | ---: | ---: | :--- | :--- | ---: | :--- |
| <n> | `<hash>` | <us> | <MiB> | kept ↑ | inserted | 0 |  |
| <n> | `<hash>` | <us> | <MiB> | reverted | dominated | 1 |  |
| · |  |  |  |  |  |  | tile block: <t> trials (<f> feasible) · best <us> · charged <c> of <t> device trials — <what the winning axis did> |
| · |  |  |  |  |  |  | closed → <node> · best <us> (j <j>) · global best_J <j> ↑ · stagnation 0/<K> |

   evolve +0 insert / <u> update / 0 prune (0 permanent)
   > <the axis this action settled, and what the readings showed>
   > <the pipe imbalance at the optimum, from util and bubble>

— <used>/<budget> evals · best <us> · front <p> points · stagnation <n>/<K> · wall <hh>h<mm>m
```

  **The table is GitHub-flavoured Markdown.** Its columns ARE `config.objectives`, in order, each
  carrying its unit in the header. Anything that happened inside the action and is not a
  candidate — a tile block, a duplicate, a free rejection, a lever that could not propose, the
  close — is a row too, marked `·` in the eval column, so the events stay in the order they
  happened; its `detail` follows an em dash in the same cell, with pipes and line breaks
  stripped. A note written before any action has been opened is an ordinary bullet. The status
  line (`— …`) is written at every close and every phase change, and ends the table. The
  per-trial detail of a block lives in `search_state.json` (`blocks[i].trials`) and every
  selection in `selects.jsonl`: **machines read those, this file is for a person.**

> The **reasons are mandatory**: a fresh optimizer re-hydrates the world model from this log, so
> the tacit belief behind each V/prune must be written down, not just the numbers.
