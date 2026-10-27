#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The tile block: a run of device trials the core owns end to end.

Tile numbers are the one lever the model cannot propose, so the study picks
them and the harness writes them, and the study is always told the
measurement of the program it proposed.
"""

import datetime
import hashlib
import os

from .config import FREE, REFUSAL_GLOBAL_AFTER, Refusal, SEARCH_PROTOCOL, bayesian, out
from .state import (_structure, chip_envelope, confirm_chip, load, merge_refusals, op_file_path,
                    refusals_for_structure, save, stored_refusals)
from .measurement import incumbent_latency, observe_noise
from .front import archive_record
from .stopping import _free_rejects, block_budget, budget_reason, fault_markers
from .log import append_fault, log_note


def _block_context(a, st, cfg):
    """Everything the block carries from its setup into its bookkeeping.

    The memory is keyed on the derived tile space, not the program text, so an
    edit that leaves the space alone keeps the study and a genuine reshape starts
    from zero. Without it, repeated blocks on one action each begin with
    `ceiling=None`, re-deriving the same footprint bound the run has already
    paid an evaluation to learn.

    Refusals are scoped to the structure until they have been EARNED on more
    than one. A run can buy the same refusal several times over several program
    shapes -- one oversized `set_vec_tile_shapes` raising PadLocalBuffer.Tensor
    in the block for each view, one device trial each -- and carrying it from
    the first sighting would have saved the rest. It also would have been a
    guess: see `refusals_for_structure`. Two sightings on different structures
    is the evidence, so a run now pays twice and saves the remainder.
    """
    op_file = op_file_path(st, a.op_dir)
    with open(op_file, encoding="utf-8", errors="replace") as f:
        base_src = f.read()
    hw, domain = bayesian.block.hw_for(op_file, a.live_tiles, chip_envelope(st))
    structure = _structure(st, a.op_dir)
    refusal_rows = stored_refusals(st)
    return {
        "op_file": op_file, "base_src": base_src, "hw": hw, "domain": domain,
        "remaining": block_budget(st["progress"], cfg, a),
        "mem_key": bayesian.block.memory_key(a.parent, domain),
        "memories": st.setdefault("bayesian_optimization_memory", {}),
        # The learned ceiling, keyed by the TILE DOMAIN with no action prefix.
        # `bayesian.block.ceiling_key` has the evidence: two blocks can hold
        # the same domain hash under two action ids, and the second then throws
        # the first's ceiling away and re-buys the capacity refusals already
        # paid for.
        "ceiling_key": bayesian.block.ceiling_key(domain),
        "ceilings": st.setdefault("bayesian_optimization_ceilings", {}),
        "structure": structure, "refusal_rows": refusal_rows,
        "refusals": refusals_for_structure(refusal_rows, structure),
        "meter": {"device_trials": 0, "winner_runs": 0},
        # Per-trial diagnostics, persisted. A block reporting "24 trials, 1
        # feasible" used to keep nothing about the other 23, so neither the
        # failure class nor the capacity markers could be checked against what
        # the device actually said.
        "trials": [], "faults": [], "winners": {}, "winner_hash": None, "charged": 0,
        # Wall-clock bounds on the block. The CANN host log is the only other
        # record of a device fault and it is keyed by time; without these there is
        # nothing to correlate it against.
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def _block_search(a, st, ctx, cfg):
    """Hand the tile space to the core and let it own the loop.

    `best_aicore_e2e_us` is a measurement OF A PROGRAM, and the block seeds it as a
    free observation of the tile config now in the file. That is only true while
    the file IS that program. A block is normally fired right after the coder
    applied a delta -- the case it exists for is `retune: true`, i.e. the structure
    just moved -- so passing the number through attaches a latency taken on the
    old program to a structure that has never been measured.

    It surfaces as `best_objective_us` equal to the preopt with `best_was_incumbent:
    true` and `winner_hash: null`, which reads like a clean "the incumbent held"
    and is not one. A recorded run hit exactly this: the u1 block on TILE_M=1024
    returned best_objective_us 48.98, the TILE_M=512 number, while the only config it
    had actually measured on the new structure was 148.46. The optimizer caught it
    by hand, spent another evaluation re-running E(x), and got 47.04 -- so the
    block had reported the action as break-even when it was an improvement.

    Narrower than `memory_key`, deliberately. The memory asks whether an old
    measurement is admissible EVIDENCE, and the answer is the tile domain. This
    asks whether `best_aicore_e2e_us` may be quoted as THIS program's incumbent, and
    that is only true of the exact bytes.
    """
    ev = bayesian.evaluator.RealEvaluator(
        op=st["op"], op_dir=a.op_dir, op_file=ctx["op_file"], device=st.get("device_id", ""),
        test_command=a.test_command, lint_root=(a.lint_root or None),
        eval_timeout_s=cfg.get("eval_timeout_s", 300),
        evaluation_mode=getattr(a, "evaluation_mode", "subprocess"),
        # Overridable so tests can assert the schedule without waiting it out.
        # The default is the production one and campaigns are told to pass
        # nothing else.
        fault_backoff_s=tuple(cfg.get("fault_backoff_s", (30, 90))))
    hw = ctx["hw"]
    return bayesian.block.run(ctx["op_file"], ev, hw, bayesian.block.BlockOptions(
        stagnation_k=cfg.get("stagnation_K", 7),
        max_trials=ctx["remaining"],
        # The block ranks its trials on `p`, so the incumbent it is measured
        # against has to be `p` too, and it has to be THIS program's. It read
        # `best_mix_aic_us` while `p` was MIX_AIC; then the global best, guarded
        # by a comparison against `ratchet_best_hash` -- which gave up the
        # incumbent entirely whenever the front base was not `best_node`.
        current_latency=incumbent_latency(st, ctx["base_src"]),
        on_trial=block_recorder(a, st, ctx), domain=ctx["domain"],
        # `bayesian.block.run` defaults this to 4, while `hw_for` derives the real
        # element width from the kernel and defaults it to 2. Leaving it
        # unthreaded meant every block on a bf16/fp16 kernel computed its 16 KB
        # floor seed and 96 KB ceiling seed at twice the true width, so it
        # searched the bottom half of the band it believed it was searching --
        # while bayesian.domain.verify used the correct width, so the two gates
        # disagreed by exactly 2x.
        dtype_bytes=hw.dtype_bytes,
        # Keep drawing at random until this many FEASIBLE trials are measured
        # before TPE takes over, so its surrogate is fitted on grounded data
        # rather than on the 1-2 feasible points a mostly-rejected block used to
        # hand it.
        startup_min_feasible=cfg.get("bayesian_optimization_startup_min_feasible", 5),
        repair=cfg.get("bayesian_optimization_repair", True),
        memory=ctx["memories"].get(ctx["mem_key"]),
        ceiling=ctx["ceilings"].get(ctx["ceiling_key"]),
        refusals=ctx["refusals"]))


def _block_log(a, st, ctx, result):
    """The block, as one note under the action that asked for it.

    The per-trial detail lives in `search_state.json` (`blocks[i].trials`); what
    belongs here is what the block cost and what it found. `free=` keeps its
    split because a block that rejects most draws before the device reads the
    same either way -- working gates, or gates walled off from its own
    incumbent -- and the second is not diagnosable without it.
    """
    log_note(a.op_dir, st,
             f"tile block: {result.get('n_trials')} trials "
             f"({result.get('n_feasible')} feasible) · best "
             f"{result.get('best_objective_us')} · charged {ctx['charged']} of "
             f"{ctx['meter']['device_trials']} device trials · "
             f"{result.get('n_repaired', 0)} repaired · free {_free_rejects(result)}"
             + (f" · ceiling {result['learned_ceiling']}"
                if result.get("learned_ceiling") else "")
             + (f" · {result.get('env_faults')} env faults"
                if result.get("env_faults") else "")
             + (" · stopped early" if result.get("stopped_early") else "")
             # NO message about `--live-tiles`. It said the footprint bound was
             # OFF without it, and that has not been true since the residency
             # became per-site and read from the source -- `_block_payload` says
             # so where its own warning used to be. The line outlived the
             # warning, and two campaigns read it: both recorded "the bound was
             # off" as the reason their tiles reached the device, which stopped
             # the real cause (the bound was ON and too loose) being looked for
             # at all. `domain_notes` reports what the bound actually did.
             + (f" · ABORTED: {result['aborted']}" if result.get("aborted") else ""),
             detail=(a.reason or result.get("reason") or ""))


def _block_payload(a, ctx, result):
    """What the optimizer reads back from a block."""
    winner_hash = ctx["winner_hash"]
    out({"ran": bool(result.get("n_trials")),
         "best_config": result.get("best_config"),
         "best_objective_us": result.get("best_objective_us"),
         "best_mix_aic_us": ctx.get("winner_mix_aic_us"),
         "best_workspace_b": ctx.get("winner_workspace_b"),
         "best_workspace_state": ctx.get("winner_workspace_state"),
         "code_hash": winner_hash,
         "n_trials": result.get("n_trials"), "n_feasible": result.get("n_feasible"),
         "stopped_early": result.get("stopped_early"),
         "aborted": result.get("aborted"),
         "fault_kind": result.get("fault_kind"),
         "fault_markers": fault_markers(ctx),
         "env_faults": result.get("env_faults"),
         "failures": result.get("failures"),
         "replayed": result.get("replayed"),
         "learned_ceiling": result.get("learned_ceiling"),
         "best_was_incumbent": result.get("best_was_incumbent"),
         "device_trials": ctx["meter"]["device_trials"],
         "charged": ctx["charged"], "granted": ctx["remaining"],
         "reason": result.get("reason"),
         "domain_notes": ctx["domain"].get("notes"),
         "capacity_elems": ctx["domain"].get("capacity_elems"),
         "live_tiles": a.live_tiles,
         # No warning about `--live-tiles` any more, and its absence is not a
         # defect in the caller. The footprint bound stopped depending on it
         # when the residency became per-site and read from the source
         # (`domain.vec_sites`): `space` says so itself where the old
         # `hw.n_live_tiles` rule used to live. The message said the opposite --
         # "the footprint bound is OFF" -- and a campaign that had correctly
         # declined to assert a residency it could not justify read it, believed
         # the harness over the code, and recorded the omission as its own
         # mistake. A message that makes a right action look wrong is worse than
         # no message. `domain_notes` above reports what the bound actually did.
         "warning": None,
         "note": ("the winner is already written to the op_file and NOT yet charged; "
                  "call `record` with this code_hash, s=1, best_objective_us "
                  "as --p, best_mix_aic_us under "
                  "--metrics '{\"mix_aic\": <it>}', and best_workspace_b / "
                  "best_workspace_state under --workspace-b / --workspace-state. "
                  "Do not dispatch the coder and do not re-measure."
                  ) if winner_hash else None})


def cmd_block(a):
    """Run one tile-tuning block: the core owns the loop and returns one config.

    Fired two ways, one mechanism. A tile action is selected, or the optimizer
    asks after a structural change has made the previous optimum describe a
    program that no longer exists. What differs is only what happens afterwards.

    Budget. Every trial is a device evaluation. The losers are charged here and
    their bytes are cached so they can never be paid for twice; the winner is
    deliberately left uncached, so the ordinary `record` path charges it and runs
    the ratchet. The ratchet stays in exactly one place, and what reaches it is
    the pair (structure, tile) rather than a tile on its own.
    """
    st = load(a.op_dir)
    cfg = st["config"]
    if bayesian is None or not cfg.get("bayesian_optimization_tile_search", False):
        out({"ran": False,
             "reason": "bayesian_optimization_tile_search is off, or the "
                       "bayesian_optimization package is absent"})
        return
    op_file = op_file_path(st, a.op_dir)
    if not op_file or not os.path.exists(op_file):
        out({"ran": False, "reason": f"op_file not found: {op_file!r} (cwd={os.getcwd()!r})"})
        return
    ctx = _block_context(a, st, cfg)
    # Before the first evaluation of this block, not after it: see confirm_chip.
    bad = confirm_chip(st, st.get("device_id", ""))
    if bad is not None:
        save(a.op_dir, st)
        out(dict(bad, ran=False, refused=True))
        raise Refusal(3)
    # Kept so `stop` can ask the device a question of its own rather than end the
    # run on an observation a block made some time ago. Nothing else reads it.
    st["test_command"] = a.test_command
    # `_domain_state` reads `st["live_tiles"]`, and nothing ever wrote it. So the
    # fingerprint this block stores is derived WITH the residency count (capacity
    # bound on) while every later `feasible` re-derives it WITHOUT (capacity bound
    # off) -- the two can never match, and `_retune_verdict` reported
    # `capacity 12288 -> None elements` as a real move. Every block that produced
    # a winner therefore made the next `feasible` demand another block.
    if a.live_tiles:
        st["live_tiles"] = a.live_tiles
    result = _block_search(a, st, ctx, cfg)
    # The core knows only that it was granted nothing; this file knows which of
    # the three limits did it. Substituted rather than appended so the one
    # sentence the agent reads is the actionable one.
    if result.get("reason") == "no evaluation budget remaining":
        result["reason"] = budget_reason(st["progress"], cfg, a)
    block_apply_winner(st, ctx, result)
    block_fold(a, st, ctx, result, cfg)
    save(a.op_dir, st)
    _block_log(a, st, ctx, result)
    _block_payload(a, ctx, result)


def _fault_record(st, a, rec):
    """The env-fault log entry: the marker and the device's own last words."""
    return {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "op": st["op"], "action": a.parent,
            "marker": rec.get("reason"), "kind": rec.get("kind"),
            "retries": rec.get("retries", 0), "config": rec.get("config"),
            "detail": (rec.get("detail") or "")[:2000]}


def block_recorder(a, st, ctx):
    """Build the per-trial callback: count what touched the device, and cache the
    bytes it measured.

    An environment fault is not a trial. The device was unusable, so nothing was
    learned about the candidate and nothing may be charged for it -- the first
    recorded block paid five evaluations for `MAP_REG_ADDR_FAILED` and scored them
    as five bad tile shapes.

    Two separate things, and conflating them undercounts. The cache is keyed by
    bytes, so a configuration drawn twice occupies one entry -- but
    `RealEvaluator` ran the test command and the profiler both times, so the
    device was paid for twice. The meter counts device trials; the cache
    deduplicates bytes. A static rejection reached neither and is charged nothing,
    which is the existing rule for `feasible`.
    """
    cache = st.setdefault("eval_cache", {})

    def on_trial(rec):
        """One trial, as the driver saw it."""
        if rec.get("status") == "env_fault":
            # Not charged and not a trial -- but written down, immediately, with
            # the marker and the device's own last words. See `append_fault`.
            f = _fault_record(st, a, rec)
            ctx["faults"].append(f)
            append_fault(a.op_dir, f)
            return
        # Every status the driver returns BEFORE calling the objective. Missing
        # one here is expensive twice over: the trial is charged an evaluation it
        # never spent, and `cache[h] = {"s": 0, "p": None}` records a program that
        # was never run as measured-and-incorrect -- after which `probe` reports a
        # hit and `record` rejects those bytes as a duplicate, so a tile the gate
        # merely guessed about is unreachable for the rest of the run.
        #
        # `over_learned_l1` was missing, which is the whole point of that gate:
        # the same file's `FREE` map already calls it free, so the block logged
        # `charged=N ... free=l1:M` about the trials it had just charged for.
        if rec.get("status") in FREE:
            # Not charged, not cached -- but RECORDED. Those were the same
            # `return` before, so a block could report `free=static:15` in its log
            # line while the state held no trace of why any of the fifteen were
            # rejected. The gate that rejects three quarters of the draws is
            # exactly the one whose reasons have to be auditable: the bound it
            # enforces is the harness's own model of the buffer, not something the
            # compiler reports, and nobody can check a model whose verdicts are
            # not written down.
            ctx["trials"].append({"status": rec.get("status"), "config": rec.get("config"),
                                  "objective_us": None,
                                  "reason": rec.get("reason", "")})
            return
        ctx["meter"]["device_trials"] += 1
        t = {"status": rec.get("status"), "config": rec.get("config"),
             "objective_us": rec.get("objective_us"),
             "mix_aic_us": (rec.get("metrics") or {}).get("mix_aic"),
             "reason": (rec.get("reason") or "")[:400]}
        if rec.get("metrics"):
            t["metrics"] = rec["metrics"]
        ctx["trials"].append(t)
        new_src, n = bayesian.block.apply_best(ctx["base_src"], rec.get("config"))
        if not n:
            return
        h = hashlib.sha256(new_src.encode("utf-8")).hexdigest()
        ctx["winners"].setdefault(h, 0)
        ctx["winners"][h] += 1
        # `p` is what the trial was ranked on, which is `J_OBJECTIVE`'s
        # reading -- the same column every other `record` writes. The launch's
        # MIX_AIC rides on `m` under its own name.
        entry = {"s": 1 if rec.get("objective_us") else 0,
                 "p": rec.get("objective_us"),
                 "m": dict(rec.get("metrics") or {}), "via": "block"}
        if rec.get("samples"):
            # The readings, not only the median. The archive judges dominance on
            # them (`dominates_measured`), and a block point that arrives
            # without them falls back to comparing medians, which on a noisy
            # kernel is not an ordering at all.
            entry["samples"] = list(rec["samples"])
        cache.setdefault(h, entry)
        if rec.get("objective_us"):
            archive_record(st, h, rec["objective_us"],
                           rec.get("metrics") or {}, a.parent)

    return on_trial


def block_apply_winner(st, ctx, result):
    """Write the winning tile config into the op_file, if the block found one."""
    if not result.get("best_config") or result.get("best_was_incumbent"):
        return
    new_src, n = bayesian.block.apply_best(ctx["base_src"], result["best_config"])
    if not n:
        return
    winner_hash = hashlib.sha256(new_src.encode("utf-8")).hexdigest()
    with open(ctx["op_file"], "w", encoding="utf-8") as f:
        f.write(new_src)
    # Left OUT of the cache on purpose: `record` must see it as new, so it charges
    # the one remaining evaluation and runs the ratchet.
    entry = st.setdefault("eval_cache", {}).pop(winner_hash, None)
    # ... but kept, under a key the duplicate index is not. See `_block_reading`.
    if entry:
        st["block_winner"] = dict(entry, code_hash=winner_hash)
    ctx["winner_hash"] = winner_hash
    # The winner's OTHER reading, so the promoting `record` can pass it under
    # `--metrics` without re-measuring. The entry just popped is the trial that
    # measured these bytes, so it is the same launch.
    wm = ((entry or {}).get("m") or {})
    ctx["winner_mix_aic_us"] = wm.get("mix_aic")
    # And the same launch's workspace. `workspace` is a DEFAULT objective, so a
    # promoting `record` that passes only `--p` and `mix_aic` halts the run with
    # `objective_unmeasured` whenever the winner's hash is not already cached --
    # which is exactly what `block_apply_winner` arranges, on purpose, so the
    # evaluation is charged once. The bytes are the trial's own reading, not a
    # re-measurement, so handing them back costs nothing.
    ctx["winner_workspace_b"] = wm.get("workspace_b")
    ctx["winner_workspace_state"] = wm.get("workspace_state")


def block_fold(a, st, ctx, result, cfg):
    """Fold the block's result back into the state: memory, refusals, budget, record."""
    if result.get("memory"):
        ctx["memories"][ctx["mem_key"]] = result["memory"]
    # Lowered only. A block that refused nothing returns `ceiling: None`, and
    # that is the absence of new evidence, not a retraction of what an earlier
    # block on this same domain paid to learn.
    learned = result.get("ceiling")
    if learned:
        held = ctx["ceilings"].get(ctx["ceiling_key"])
        if not held or learned < held:
            ctx["ceilings"][ctx["ceiling_key"]] = int(learned)
    # Truthiness, not `is not None`. `bayesian.block._empty()` returns `refusals: []`
    # for every early return -- no tunable site, no budget left, a source that does
    # not parse -- and "no budget left" is a routine end-of-campaign condition. An
    # `is not None` guard let any of them wipe the whole history.
    if result.get("refusals"):
        # MERGE, never overwrite. `result["refusals"]` is the driver folding its
        # new rows into the ones it was handed, and it was handed only the subset
        # admissible on this structure -- assigning it back would drop every
        # refusal earned on every other structure.
        st["bayesian_optimization_refusals"] = merge_refusals(
            ctx["refusal_rows"], result["refusals"], ctx["structure"],
            cfg.get("bayesian_optimization_refusal_global_after",
                    REFUSAL_GLOBAL_AFTER))
    if ctx["winner_hash"]:
        st["domain_fingerprint"] = ctx["domain"]
    # Each measured trial is a reading of the operator's noise too. The winner
    # is later promoted by a `record` with no --samples, so nothing is counted
    # twice.
    for h in result.get("history") or []:
        if h.get("status") == "ok" and h.get("samples"):
            observe_noise(st, h["samples"], SEARCH_PROTOCOL)
    # Every device trial is charged. One is handed to `record` along with the
    # winner, so the total across block+record equals the trials that ran.
    # Never negative. The winner is left uncached so `record` charges it once, and
    # this subtracts it here so it is not charged twice -- but a RESUMED block can
    # return a winner drawn from its predecessor's replayed observations while
    # every fresh draw was rejected for free, i.e. device_trials == 0 with a winner
    # set. That refunded an evaluation the run had already spent.
    device_trials = ctx["meter"]["device_trials"]
    ctx["charged"] = max(0, device_trials - (1 if ctx["winner_hash"] else 0))
    st["progress"]["evals_used"] += ctx["charged"]
    st["progress"]["block_evals"] = st["progress"].get("block_evals", 0) + device_trials
    st.setdefault("blocks", []).append(
        {"action": a.parent, "reason": a.reason, "live_tiles": a.live_tiles,
         "granted": ctx["remaining"], "n_trials": result.get("n_trials"),
         "n_feasible": result.get("n_feasible"),
         "stopped_early": result.get("stopped_early"),
         "aborted": result.get("aborted"),
         "fault_kind": result.get("fault_kind"),
         "fault_markers": fault_markers(ctx),
         "env_faults": result.get("env_faults"),
         "failures": result.get("failures"),
         "learned_ceiling": result.get("learned_ceiling"),
         "best_objective_us": result.get("best_objective_us"),
         "best_config": result.get("best_config"),
         "device_trials": device_trials,
         "charged_now": ctx["charged"], "winner_hash": ctx["winner_hash"],
         # The source every trial in this block was drawn against. `trials`
         # already carries each config and the compiler's own refusal message,
         # but a config means nothing without the program it was applied to --
         # the tile domain is derived from that source, so re-adjudicating a
         # refusal needs it. It was reachable only through `winner_hash` ->
         # node -> parent, and a block whose winner never became a node (a
         # reverted one, which is most of the refusing blocks) has no such
         # path, so most capacity refusals could not be re-judged offline for
         # want of this one field. One hash makes every future refusal free
         # evidence.
         "base_hash": hashlib.sha256(
             ctx["base_src"].encode("utf-8")).hexdigest(),
         "started_at": ctx["started_at"],
         "ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
         "trials": ctx["trials"]})
