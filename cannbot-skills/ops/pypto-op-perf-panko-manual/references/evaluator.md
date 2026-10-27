# Frozen evaluator: E(x) → (s, p)

The evaluation **method** is frozen and identical across every run and every candidate, so `J`
stays objective. The optimizer *triggers* E but never changes the method. Only DEVICE varies
per run (user-set `TILE_FWK_DEVICE_ID`).

## What the objectives are

The run's objectives are whatever `config.objectives` names, and the DEFAULT is
`["aicore_e2e"]` alone -- one objective, which is the single-objective search (SKILL.md,
Two modes). Objectives are the FRONT: a user adds to them and takes from them, and the
harness does not rank them against each other. The multi-objective runs this file quotes
declared `["aicore_e2e", "workspace"]`.

`workspace` and `mix_aic` are both outside the default, and for different reasons.
`workspace` is outside it because the second objective is the user's choice of mode: a
run that wants the latency/bytes trade-off asks for it. `mix_aic` is outside it for a
reason that is measured rather than aesthetic. Its readings are carried on `m` in every
run, so a campaign that does not optimise it still records it -- which is how the cost
was established. It is the noisier of the two instruments, so with `undetermined` a tie it
mostly ties and adds little to the front. Against that, a declared objective that cannot
be read HALTS the run, and msprof's per-call accounting breaks down at small workspaces --
which is the region a workspace objective is there to explore. Declare it when a
kernel's two clocks are known to disagree and the disagreement is what the front is for.

`J` is separate and is not configurable. It is the single scalar the greedy phase ratchets
on, it is always **`p`**, and `P_ref` is a target in that same instrument
(`J = 100 * P_ref / p`, and `J >= 100` is the success stop). A run whose objectives do not
include `aicore_e2e` still has a `J`, because `J` is `p` and every measurement has one.

- `aicore_e2e` -- **`p` itself**: the on-device profiler's "AICore End-to-End Time" line,
  written only at `runtime_debug_mode: 1`. The run's primary latency, so `--p`, `--samples`
  and `J` all mean this reading, and `init` refuses a preopt without one.
- `mix_aic` -- `m.mix_aic`, msprof's per-launch MIX_AIC `aicore_time(us)` from
  `op_summary_*.csv`, with its readings under `m.mix_aic_samples`.
- `workspace` -- `m.workspace_b`, the bytes the runtime asked for, read from the alloc
  callback.

Both time readings are **kernel-internal**: they measure the `@jit` kernel, not the call, so
a speedup in either is a speedup of the kernel. They are different quantities and their
ratio is not a constant (see below).

**An objective IS its instrument.** These definitions are what a front, a speedup and a
`verify` in this run mean; a reading is not a claim about any quantity measured another way.
If a user names other objectives, those are the objectives, and nothing in this skill judges
that choice, ranks the objectives against each other, or calls one of them wrong.

That is the scope on purpose: PANKO optimizes what is inside the `@jit` kernel. Reshaping
data in the host wrapper, allocating the output tensor and the rest of the launch path are
not the kernel's cost and are not counted.

**The host wrapper is therefore FROZEN for the whole run, and the harness enforces it.**
Counting only the kernel is a sound way to order two candidates exactly while the uncounted
part is identical between them, because then it cancels in the comparison. Move work across
the boundary and the arithmetic breaks: the kernel gets cheaper, the wrapper gets more
expensive, `p` improves, and the ratchet keeps a candidate that is slower to run — 100 µs of
kernel becoming 80 µs while 30 µs appears in the wrapper scores as a 20 µs win.

- `init` records a signature of everything in `op_file` that is not a `@jit` kernel.
- `feasible` rejects a drifted candidate **before** any device time is spent.
- `record` checks again and discards the measurement, because `feasible` is not mandatory
  and a number taken on a changed wrapper is not comparable with the rest of the campaign.

The signature is taken over the AST, so comments and reformatting outside the kernel are
invisible; any change to what the host actually does is not. Actions that moved work to the
host (F-16, F-21, F-22) are **not in the action catalogue** — with the wrapper frozen they
are unreachable, and offering them would be offering candidates the gates reject.

**`debug_options={"runtime_debug_mode": 1}` is a prerequisite, and it is frozen for the
run.** The latency objective is `aicore_e2e`, and the flag is what produces it: "AICore
End-to-End Time" and "AICore Utilization" are written once per launch at mode 1 and **not at
all at mode 0**, and `tilefwk_L1_prof_data.json` (the source of `aic_util` / `aiv_util`) is
not written at mode 0 either. A mode-0 run has no latency reading at all.

The other two objectives do not depend on the flag. `mix_aic` is read from msprof's MIX_AIC
rows, which `runtime_debug_mode` does not affect: with the flag removed entirely a run still
gives one MIX_AIC row per launch with a numeric `aicore_time`, and msprof's metric columns
match the flag-on run. `workspace`
comes from the alloc callback. They are carried at mode 0; the latency objective is not,
which is why mode 1 is mandatory rather than optional. At mode 0 `util` / `aic_util` /
`aiv_util` are reported as MISSING, never as 0. `bubble` and `pred_stall`
need `bubble_analysis.log`, which the installed `draw_swim_lane.py` does not write in EITHER
mode -- they were already unobtainable, and mode 0 does not make that worse. In their place
the msprof columns above are carried under their own names; they are not substitutes, since
the same run reports pypto's stdout utilisation and msprof's `cube_utilization` as very
different numbers, and the symptom index's thresholds were set on the first.

**The flag is fixed at the baseline.** `init` hashes every top-level jit kernel's
`debug_options` and `feasible` / `record` reject a candidate that changed it. The reason is
comparability, not truth: **an objective IS its instrument**, and this run's objectives are
defined as what these instruments report. Whether a reading corresponds to some quantity
measured another way is a question about the objective's choice, not about a candidate's
score. What must not happen is two candidates read on different instruments being ranked
against each other: the two modes can read apart on the same program, in either direction,
by the size of a plausible win -- so a candidate whose only change is dropping `debug_options`
must not be recordable as an improvement. It would also make the diagnostics appear or vanish between candidates the
symptom index treats as comparable.

## Procedure

```
wrapper_frozen(x)                                        # static, no device
if not frozen:    return REJECTED                        # costs no eval_budget
feasible(x)                                              # static, no device
if not feasible:  return REJECTED                        # costs no eval_budget
s = golden_compare(x) AND layout_check(x)     @ DEVICE   # fixed command + thresholds
if not s:  return (0, None)                              # J = 0; skip perf
p = perf_measure(x)                            @ DEVICE   # fixed method
return (1, p)
```

- **Static feasibility** runs first, on the source, before anything is compiled
  or launched. It rejects only configurations the hardware provably cannot
  accept: tile dimensions that are not 16-aligned, an L0 tile larger than its
  L1 tile, or a unified-buffer footprint over budget. A rejection consumes no
  device time and therefore does not count against `eval_budget`; it does count
  towards the action's stagnation counter, so an action that keeps proposing
  infeasible configurations is still abandoned. The check is deliberately
  conservative and rejects only when the violation is certain: a false
  rejection would delete a reachable optimisation from the search space, while
  a false acceptance costs one evaluation. Disable with
  `config.static_feasibility = false`.

- `s ∈ {0, 1}` — 1 iff the candidate passes **both** correctness (`detailed_tensor_compare`
  against `<op>_golden.py`) **and** the layout / structure check. Fixed command and thresholds
  (tolerances from `SPEC.md`); the same golden and layout check used by the onboard Stage 1–6
  harness (`test_<op>.py`).
- `p` — on-board latency (µs), measured **only when `s = 1`**, via the **frozen** script
  `scripts/measure_latency.py` (do NOT hand-author or vary the timing). One warm process runs the
  kernel `warmup + runs` times and captures the on-device profiler's stdout at the file-descriptor
  level; `p` is the **median of the warm "AICore End-to-End Time" lines**, and `samples` /
  `sigma_log` are those same values. No line means no `p`, which is not a verdict on the
  candidate: a tile block aborts on it (`objective_unmeasured`, invocation fault) and the run
  stops. The line exists only at `runtime_debug_mode: 1`.

  The same process runs as a child of `msprof --ai-core=on --task-time=on --ascendcl=on`, and
  after msprof exits it reads `op_summary_*.csv`: rows with `Task Type == MIX_AIC`, in launch
  order, one per call (k per call are summed), warm median under `m.mix_aic`, with
  `m.mix_aic_samples` / `m.mix_aic_sigma_log`. That is the `mix_aic` objective. A row count
  that does not map to calls or no MIX_AIC rows gives `m.mix_aic_error` and costs that
  objective, not the run. On a cube-bound kernel MIX_AIC runs slightly above
  End-to-End and is the quieter of the two, and the msprof wrap costs a small
  fraction of wall time.

  **`aicore_e2e` and `mix_aic` are different quantities, which is why naming one
  is not naming the other.** They nearly agree on a cube-bound kernel. On a
  vector-bound one -- `aic_mac_ratio` and `aiv_vec_ratio` both near zero,
  `aiv_time` close to `aicore_time` -- the same launch can read most of an order
  of magnitude apart under the two, and a host wall clock with the profiler off
  reports a third figure again; each is what its own instrument measures. The two
  rank the same programs differently often enough to matter, and a program that is
  fastest under one can be off the other's front entirely.

  Two consequences. **A reading under one is not comparable with a reading under
  the other**, the gap is not a constant and no conversion exists, so numbers are
  quoted against the instrument that produced them. And an action that MOVES WORK BETWEEN THE PIPES moves the two
  readings by different amounts -- I-11 names exactly that case -- which is what
  a front over both objectives exists to show. Campaign histories recorded
  before both were objectives carry only one of the two. Params:
  `--warmup 1 --runs 9 --agg median`.
  It was `--runs 3 --agg min`, and `min` was the defect: it reports a distribution's fast mode
  rather than its centre, **which flatters the noisier arm**, so the ratchet preferred programs
  for their VARIANCE. The bias is systematic and grows with the spread, and it propagates -- an
  accepted value the program cannot reproduce becomes the bar every later candidate is measured
  against. The median is unbiased; `runs 9` then halves the standard error
  (`SE(median) ~ 1.2533*sigma/sqrt(n)`) and tolerates four outliers instead of one, which
  matters on a device whose noise this repo calls bimodal. A
  profiled call is a small fraction of an evaluation, so the six extra calls are cheap. `m` (util, bubble) come from the same capture as diagnostics — they never enter `p` / `J`.
- **Timeout / failure** — each candidate run is capped at `config.eval_timeout_s`, which `init`
  **derives** from the preopt's own `compile_s` (`eval_timeout_k` x it, clamped to
  `eval_timeout_floor_s` / `eval_timeout_cap_s`) and records under
  `config.eval_timeout_source`. The right value is a property of the operator, not of the
  method: compile time grows with the program, and candidates grow from the preopt.
  Without a `--preopt-compile-s` the configured value stands and the
  source says `default`.
- **A crashed / compile-failed candidate is `s = 0`** and discarded — those are verdicts about the
  candidate. **A candidate whose COMPILE ran past the budget is not.** Nothing about it was
  established, so it is reported as `unevaluated` instead:
  - it is **not charged to `eval_budget`** (the budget counts numbers; a timeout produced none);
  - it is **not recorded as `s = 0`** (that label propagates into the ratchet's bar, the tile
    study's surrogate, and anything later that reads the measurements — and since compile time
    scales with program size, the label is not random: it marks the large candidates wrong);
  - it is counted under `progress.unevaluated_rejected`, and on the action under `n_timeout`
    against `unmeasured_K`, because neither `stagnation_K` (which counts measurements) nor
    `eval_budget` (not charged) can end an action that only produces slow compiles.
  The split is exact rather than inferred: `measure_latency` writes
  `[panko-progress] first_call_done compile_s=<x>` to the real stderr once the first call (JIT
  compile + one execution) returns, so its **absence** in a killed run's partial output means the
  compile never finished, and its presence means the candidate compiled and then ran long — which
  IS a verdict. A device fault still wins over both.
  On the structural path the caller owns the `timeout`, so the caller reports it:
  `record ... --unevaluated compile_timeout`. The tile path detects it in-process.
- **`compile_s`** — the first call's wall time, reported by `measure_latency` as a diagnostic. It is
  deterministic and low-noise, and it is the quantity `eval_timeout_s` is set against. It never
  enters `p` or `J`.

- **Environment faults** — a command that could not be performed says nothing about the
  candidate, so it is never charged, never cached, and never counted toward a block's
  `failure_limit`. It is written to `faults.jsonl` with its marker and the device's last words,
  counted in `state.faults`, and after `fault_limit` consecutive ones the block aborts with a
  `fault_kind`. Two kinds, two responses:
  - **device** (`MAP_REG_ADDR_FAILED`, `NPU out of memory`, `Device is busy`, …) — another
    process holds the device. The evaluator waits out `fault_backoff_s` and runs the candidate
    again; a run that ends here stops as `device_unavailable`, after asking the device once more.
  - **invocation** (`No module named`, `can't open file`, `ctx is NULL`, …) — the command never
    started, or never obtained a device context. Waiting cannot help, so it raises at once and
    the run stops as `harness_fault`, with `progress.harness_fault.hint` naming what to fix.

  `ctx is NULL! ... rtGetDevMsg execution failed, the context is a null pointer` is the
  device-context case and is worth naming: the device is not busy, it was never opened, and the
  fix is the device id the test command is given (`TILE_FWK_DEVICE_ID` /
  `ASCEND_RT_VISIBLE_DEVICES`). Before it was a marker, these were scored as "this tile is
  incorrect": blocks aborted on `failure_limit`, evaluations were charged, and good tile
  regions went to the sampler as infeasible.

## Not part of the frozen method

- **DEVICE** (`TILE_FWK_DEVICE_ID`) — a per-run user parameter. For an A/B comparison, keep it
  constant across the compared runs (a run-operation responsibility, not baked into E).
- **util / bubble** — diagnostics / acceptance context only; they never enter `J`.

## Objective

`J(x) = s · (P_ref / p) · 100` — success stop when `best_J ≥ 100` (p has reached P_ref). The
harness (`scripts/panko_harness.py`) computes `J`; E only returns `(s, p)`.
