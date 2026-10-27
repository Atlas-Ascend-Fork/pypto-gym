#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""One transition per dispatch: select, record, close, evolve, stop.

The optimizer calls exactly one of these per cycle. Candidate acceptance,
the rollback and the state update are organised here; the layers below are
asked for their verdicts rather than reaching back for them.
"""

import json
import time

import action_posterior
import evaluation_session

from .config import (CATALOG_PRIOR_DEFAULT, CandidateRow, MeasurementVerdict, RatchetMeasurement, Refusal,
                     SEARCH_PROTOCOL, STAGE_ORDER, SelectionContext, _bound, clamp_belief, clamp_prior, clamp_stage,
                     feasibility, out)
from .state import (_base_hash, _domain_state, _gstag, _refine_base, _semantic_noop, _structure, _wrapper_drift,
                    accumulate_wall_clock, chip_envelope, file_hash, live_frontier, load, next_id, npath,
                    op_file_path, preopt_code_hash, ratchet_best_hash, restore, rollback, save, snapshot)
from .measurement import (block_samples, cli_samples, compute_j, median_of, objective_values, objectives_of,
                          observe_noise, record_metrics)
from .front import (_advance_front_base, _refresh_front_point, archive_alpha, archive_record, archive_reprune,
                    demand_hint_text, front_acceptance, front_demand, rederivable)
from .stopping import (_admit, _bayes_free_trials, _refine_for, _refine_stop_cause, _refuse_close,
                       _retire_spent_action, _unmeasured, device_verdict, refuse_early_close, stop_condition)
from .log import _candidate_note, _demand_note, append_log, log_candidate, log_note, log_status
from .selection import (_choose_action, _expansion_demand, _learned_g, _learned_g_on, _log_selection, _observe_action,
                        _prior_table_path, _record_outcome, _retune_verdict, _select_source_payload, _symptom_reboost,
                        _tile_bayes_tell, bayes_next, pypto_action_priority, record_select)
from .delivery import (_deliver_best, _delivered_check, _open_verification, _record_frontier, _record_probe,
                       _verify_directive, export_frontier)
from .init import _refuse_objectives


def _refuse_if_action_open(st):
    """An open refinement has to be finished before another action is chosen.

    Without this, making `close` refuse achieves nothing: the optimizer can
    leave an action at n=1 simply by selecting a different one and never
    coming back, which is the same abandonment the refusal exists to prevent,
    and it also leaves `refine` pointing at an action whose candidates the
    next `record` would be attributed to.
    """
    r = (st["progress"].get("refine") or {})
    if not r.get("action"):
        return
    why = refuse_early_close(st, r["action"])
    if why:
        out({"selected": None, "refused": True, "reason":
             f"action {r['action']} is still open: {why}",
             "open_action": r["action"], "n": r.get("n", 0)})
        raise Refusal(3)


def _out_expansion_required(st, a):
    """The frontier is empty and nothing was inserted: not a stop, a request."""
    # Not a stop. The frontier is the model's to extend.
    demand = front_demand(st, a.op_dir)
    out({"selected": None, "reason": "expansion_required",
         "declared_exhausted": st["progress"].get("expansion_declined"),
         **_expansion_demand(st, demand)})


def _restore_or_refuse(a, restore_hash, op_f):
    """Restore the base program into the working file, or refuse the action.

    Refuse rather than hand out an action. The response says "apply your
    delta on top of restore_code_hash", and the file is not that program,
    so the candidate's gain would be measured against a base nobody
    recorded and its parent link in the tree would be a fiction. Returning
    `restored: false` and continuing made that the caller's problem, and
    nothing downstream checks it.
    """
    restored = restore(a.op_dir, op_f, restore_hash)
    if restored or file_hash(op_f) == restore_hash:
        return restored
    out({"selected": None, "refused": True, "error": "restore_failed",
         "restore_code_hash": restore_hash, "op_file": op_f,
         "snapshot": npath(a.op_dir, restore_hash) if restore_hash else None,
         "reason": ("the base program has no snapshot on disk"
                    if restore_hash else "no base code_hash is recorded"),
         "hint": "optimization/nodes/<code_hash>.py is the search's memory of "
                 "every kept program. Without it the next candidate would be "
                 "built on whatever is in the working file."})
    raise Refusal(3)


def cmd_select(a):
    st = load(a.op_dir)
    _refuse_if_action_open(st)
    active = st["progress"]["active_stage"]
    staged = st["config"]["staged_prior"]
    frontier = live_frontier(st, a.op_dir)
    if not frontier:
        _out_expansion_required(st, a)
        return
    free = _bayes_free_trials(st)
    best, thompson = _choose_action(st, frontier, active, staged, free)
    # greedy accumulation: restore the running global best into the working file NOW, so the
    # optimizer applies this action's delta ON TOP of the accumulated best (deterministic —
    # not left to the agent). restore_code_hash is still returned for visibility/logging.
    # Phase 1: the global best (greedy accumulation). Phase 2: the front point
    # being extended -- `_base_hash` is the only thing that differs.
    restore_hash = _base_hash(st, a.op_dir)
    op_f = op_file_path(st, a.op_dir)
    restored = _restore_or_refuse(a, restore_hash, op_f)
    # `already_measured` is the whole point: the file the coder is about to edit IS
    # the global best and its bytes are already in the cache. Submitting it unchanged
    # is the single most common way budget was wasted (see cmd_record).
    payload = {"selected": {"id": best["id"], "parent": best["parent"], "stage": best["stage"],
                            "delta": best["delta"], "V": best["V"],
                            "pypto_action_priority": pypto_action_priority(best, active, staged, free),
                            "restore_code_hash": restore_hash, "restored": bool(restored),
                            "already_measured": restore_hash,
                            "known_hashes": len(st.get("eval_cache", {}))}}
    payload.update({"thompson": thompson} if thompson else {})
    # Same rule as at `record`: a proposal that cannot be made must not cost
    # the caller the action it was just handed.
    proposal = bayes_next(st, a.op_dir, best["id"])
    if proposal:
        payload["bayesian_optimization"] = proposal
    sel = SelectionContext(active, staged, free, thompson)
    rec = record_select(a.op_dir, st, frontier, best, sel)
    payload.update(_select_source_payload(a.op_dir, rec))
    save(a.op_dir, st)
    _log_selection(a.op_dir, st, best, rec, sel)
    out(payload)


def _record_duplicate(a, st, pr, cache):
    """Byte-identical resubmission.

    Two ways this happens, both observed: the coder is handed back its own local
    best by the roll-back in `_apply_ratchet` and edits nothing, or it is handed
    the global best by `select` and edits nothing. Either way the bytes were
    already measured, so this is not a candidate. Reject it WITHOUT touching
    r["n"] -- counting it as a regression let an action be closed by re-proposing
    its own best three times.

    Returns True when it has emitted a rejection and the caller must stop.
    """
    if a.code_hash not in cache:
        return False
    pr["duplicate_rejected"] = pr.get("duplicate_rejected", 0) + 1
    r0 = _refine_for(pr, a.parent)
    # A run of duplicates is the only honest signal that an action has no
    # unmeasured value left, and it is one the harness observes rather than one
    # the optimizer declares. `close` now refuses while n < stagnation_K, so an
    # action whose axis really is a three-point space would otherwise deadlock:
    # it cannot produce a new candidate, cannot advance n, and cannot be closed.
    # Three consecutive duplicates release it.
    #
    # Deliberately NOT in SKILL.md, and deliberately not reported in this
    # response. A documented "three duplicates ends an action" is a cheap exit the
    # optimizer can take whenever refining gets tedious, which is the behaviour
    # the enforcement exists to stop. It has to be reachable by an action that has
    # genuinely run out, and invisible to one that has not.
    # One cumulative tally, no consecutive streak. The streak reset on every
    # real measurement, so an action that alternates a new program with a re-run
    # of an old one never reached three in a row: one recorded action resubmitted
    # two hashes nine and eighteen times and it never fired. `n` does not move on
    # a duplicate either -- correctly, nothing was measured -- so without a bound
    # that counts these an action can propose indefinitely.
    r0["n_dup"] = r0.get("n_dup", 0) + 1
    # Roll back, exactly as the non-improving branch does. This branch used to
    # return with the working file still holding the rejected trial's bytes -- and
    # a duplicate is by definition a program that was already measured and did NOT
    # become the best, so what stayed on disk was a loser. Every candidate after
    # it was then built on that base. Observed twice in one run: a BO trial
    # rejected as a duplicate left the pre-loop vector tile at the study's (4,64)
    # instead of the incumbent's (1,64), so two "single-variable" candidates in
    # fact changed two sites, and the optimizer's own account of what it had
    # tested was wrong.
    rollback(a, st, r0)
    log_note(a.op_dir, st,
             f"{a.code_hash[:8]} duplicate — already measured, not charged, "
             f"n stays {r0.get('n', 0)} ({pr['duplicate_rejected']} so far)")
    out({"duplicate": True, "improved": False, "cached": cache[a.code_hash],
         "J": cache[a.code_hash].get("J"), "n": r0.get("n", 0), "stop_refine": False,
         "hint": "these bytes were already evaluated -- call `probe` before E(x); "
                 "if the coder cannot change this file, the action does not apply"})
    return True


def _record_semantic_noop(a, st, pr, j):
    """Backstop for the gate in `feasible`.

    That is where a semantic no-op is caught for free; this is where it is stopped
    from being KEPT, for runs that reach `record` without the gate. The
    measurement has already been paid by the time we get here, so this cannot save
    the evaluation -- only the delivered kernel. Same treatment as a
    byte-duplicate, and for the same reason: it is the same program, so it is not
    a candidate and `n` must not move.

    Returns True when it has emitted a rejection and the caller must stop.
    """
    # Only when the file on disk IS the bytes being recorded. `record` is told a
    # `code_hash` and reads a file, and nothing forces the two to agree -- a block
    # winner is promoted through here, `select` restores a snapshot, and the tests
    # drive the state machine with hashes that name no file at all. Judging a
    # candidate by a file that is not it would reject on the strength of whatever
    # happened to be on disk.
    same, why = ((False, "") if file_hash(op_file_path(st, a.op_dir)) != a.code_hash
                 else _semantic_noop(st, a.op_dir, op_file_path(st, a.op_dir), a.parent))
    if not same:
        return False
    pr["semantic_noop_rejected"] = pr.get("semantic_noop_rejected", 0) + 1
    r0 = _refine_for(pr, a.parent)
    # `n_dup`, exactly as the byte-duplicate branch. This IS a duplicate -- the
    # same program, missed by the byte hash -- and without it the branch had no
    # termination path at all: `n` must not move (nothing new was measured), so
    # an action that can only resubmit its own program could neither advance nor
    # be closed.
    r0["n_dup"] = r0.get("n_dup", 0) + 1
    rollback(a, st, r0)
    log_note(a.op_dir, st,
             f"{a.code_hash[:8]} semantic no-op — n stays {r0.get('n', 0)}, "
             f"n_unmeasured {_unmeasured(r0)}/{_bound(st, 'unmeasured_K')}", why)
    out({"semantic_noop": True, "improved": False, "J": j,
         "n": r0.get("n", 0), "stop_refine": False,
         "reason": f"this candidate is the incumbent program: {why}",
         "hint": "call `feasible` before E(x) -- this would have been caught "
                 "without spending the measurement"})
    return True


def _record_wrapper_drift(a, st, pr):
    """`feasible` is the cheap gate and it is not mandatory, so the ratchet checks
    too. A measurement taken on a changed wrapper is not comparable with the rest
    of the campaign and must not be allowed to set the best. Emits and raises."""
    drift = _wrapper_drift(st, op_file_path(st, a.op_dir))
    if not drift:
        return
    pr["wrapper_drift_rejected"] = pr.get("wrapper_drift_rejected", 0) + 1
    restore(a.op_dir, op_file_path(st, a.op_dir), ratchet_best_hash(st))
    save(a.op_dir, st)
    log_note(a.op_dir, st,
             f"{a.code_hash[:8]} wrapper drift — measurement discarded, "
             f"working file rolled back")
    out({"improved": False, "refused": True, "error": "wrapper_drift",
         "reason": drift, "wrapper_frozen": False,
         "hint": "PANKO optimizes inside the @jit kernel. This measurement "
                 "is not comparable with the campaign's and was discarded."})
    raise Refusal(3)


def _apply_ratchet(a, st, op_file, measured):
    """Gate keep + stagnation on the FRONT, not a fresh per-action optimum.

    A candidate is "kept" (and n reset) if `front_acceptance` admits it --
    `accepted` -- and is otherwise reverted + n++. With one objective that is
    exactly "it beats the running best"; with more, a trade-off that no
    archive point dominates and that is no slower than the preopt is kept too,
    and the action continues from it. `accepted=None` (the vector could not be
    formed) falls back to `j > bar`. This is the existing Stage-7
    rule -- keep anything that gains, roll back anything that loses -- and it
    stops an action from burning budget below the global best.

    Both branches touch the snapshot chain, and a failure in either one is
    invisible in the response the optimizer reads. `select` and `stop` refuse on a
    broken chain, but a refinement submits candidate after candidate for the SAME
    action without passing through either, so a failure here would go unnoticed
    until the next select -- with every candidate in between built on the wrong
    base.

    Returns (refine record, improved, bar, chain_error).
    """
    pr = st["progress"]
    r = _refine_for(pr, a.parent)
    # A real measurement. The unmeasured tallies are CUMULATIVE and are not
    # reset here: an action that alternates a new program with a re-run of an old
    # one was exactly the case the consecutive streak could not see, and one
    # recorded action resubmitted two hashes nine and eighteen times without it
    # ever firing.
    bar = r["best"]["J"] if r["best"] is not None else r.get("threshold_J", pr["best_J"])
    improved = bool(a.s) and (measured.j > bar if measured.accepted is None
                              else bool(measured.accepted))
    chain_error = None
    if improved:
        r["n_undetermined"] = 0
        point = {"code_hash": a.code_hash, "J": measured.j, "p": a.p, "m": measured.m_eval}
        # Two records, because front acceptance separates two questions the
        # scalar ratchet answered with one program. `base` is where the action
        # continues from: the last point it accepted, which may be a trade-off
        # slower than an earlier one. `best` is the fastest point it accepted,
        # and is what `close` and `stop` promote and what `success` reads --
        # with one record, an action that found a fast point and then accepted
        # a slower trade-off handed the slower program forward and lost the
        # fast one as the delivered kernel.
        r["base"] = point
        if r["best"] is None or measured.j > r["best"]["J"]:
            r["best"] = point
        r["n"] = 0
        if not snapshot(a.op_dir, op_file, a.code_hash):  # persist the new running best
            # The bytes are still on disk, so nothing is lost yet -- but the next
            # reverted candidate could not be rolled back to them, which is the
            # promise that "the best code can never be lost" rests on.
            chain_error = "snapshot_failed"
        # The kept program is now the one later deltas are judged against, so the
        # space it defines becomes the reference. Without this the verdict would
        # keep firing against the preopt's space long after it was superseded.
        st["domain_fingerprint"] = _domain_state(st, op_file) or st.get("domain_fingerprint")
    elif measured.clarity == "undetermined":
        # Not evidence about the action: the readings cannot order it against
        # the front. `undetermined_K` bounds these on their own, so an action
        # still terminates on a kernel whose noise swallows every candidate.
        r["n_undetermined"] = r.get("n_undetermined", 0) + 1
    else:
        r["n"] += 1
        # roll the working file back to the action's current best (or the phase's
        # base if this action has produced nothing yet), so the next candidate is
        # applied on the right base.
        #
        # `_base_hash`, not `ratchet_best_hash`. In phase 2 the candidates are judged
        # against the GREEDY global best, so essentially none of them improves --
        # that is the phase's whole point -- and this line therefore runs on
        # every candidate but the first. With `ratchet_best_hash` it rolled the file
        # back to the global best G, not to the front point F that `select`
        # restored and that the response still names as the base. Candidate 1
        # extended F; candidates 2..K silently re-ran greedy search from G while
        # every row was labelled `phase: "front"`.
        base_hash = (_refine_base(r)["code_hash"] if _refine_base(r)
                     else _base_hash(st, a.op_dir))
        if not restore(a.op_dir, op_file, base_hash) and file_hash(op_file) != base_hash:
            # The loser is still in the working file. Left unsaid, the next
            # candidate for this action is applied on top of a program the ratchet
            # has already rejected, and its measured gain is against a base
            # nothing recorded.
            chain_error = "restore_failed"
    return r, improved, bar, chain_error


def _halt_on_broken_chain(a, st, payload, chain_error, op_file):
    """The snapshot chain is broken. Emit the measurement and stop the run.

    Called AFTER `save`: the measurement was paid for and belongs in the state and
    the cache whatever happens to the files. The non-zero exit is what stops the
    run, because the next candidate would be built on the wrong program.
    """
    payload.update({"refused": True, "error": chain_error,
                    "code_hash": a.code_hash, "op_file": op_file,
                    "hint": "optimization/nodes/ is the search's memory of "
                            "every program it may return to; the measurement "
                            "is recorded, but the next candidate cannot be "
                            "built until this is repaired."})
    log_note(a.op_dir, st,
             f"{a.code_hash[:8]} HALTED — the snapshot chain is broken", chain_error)
    out(payload)
    raise Refusal(3)


def cmd_record(a):
    """One candidate evaluated inside local refinement of action `parent`."""
    st = load(a.op_dir)
    pr = st["progress"]
    if _record_handled_elsewhere(a, st, pr):
        return
    m_rec = record_metrics(a, st)
    j = compute_j(a.s, a.p, st["target"]["P_ref"])
    _refuse_unmeasured_objectives(a, st, pr, m_rec)
    cache = st.setdefault("eval_cache", {})     # memoize E(x) by code_hash (write-once = deterministic)

    if _record_duplicate(a, st, pr, cache):
        return
    if _record_semantic_noop(a, st, pr, j):
        return
    _record_wrapper_drift(a, st, pr)

    pr["evals_used"] += 1
    # A block winner is promoted through this path deliberately (so the ratchet
    # runs in exactly one place), but the block measured it and the CLI call
    # that promotes it carries no reading. Recover it from the block record
    # rather than writing a zero over a real measurement.
    m_eval = _record_cache_entry(a, st, cache)
    # Only now. Above this line the candidate may have been a byte-identical
    # resubmission, which is not a measurement of anything: telling the study a
    # value for a program it did not propose would poison the surrogate with a
    # point the search never actually visited.
    _tile_bayes_tell(st, a.op_dir, a.parent, a.s, a.p)
    op_file = op_file_path(st, a.op_dir)
    accepted, acceptance = _record_acceptance(a, st, pr, op_file, m_eval)
    measured = RatchetMeasurement(j, m_eval, accepted, (acceptance or {}).get("clarity"))
    r, improved, bar, chain_error = _apply_ratchet(a, st, op_file, measured)
    # The LATENCY question, kept apart from acceptance. The action prior and
    # the outcome store's `kept` / `log_gain` mean "this action made the
    # program faster" (see `_record_outcome`), and a kept trade-off did not.
    # Feeding acceptance in would record negative gains as wins and change
    # what the learned table measures without anyone deciding it should.
    faster = bool(a.s) and bool(bar) and j > bar
    pr["refine"] = r
    stagnation_k = st["config"]["stagnation_K"]
    undetermined_k = _bound(st, "undetermined_K")
    unmeasured_k = _bound(st, "unmeasured_K")
    outcome = MeasurementVerdict(faster, a.s, j, bar)
    arc_verdict = _record_archive_and_outcome(a, st, m_eval, outcome, improved)
    stop_cause = _refine_stop_cause(r, stagnation_k, undetermined_k, unmeasured_k)
    # One ask per measurement, not one per action. Asked only at select, the
    # lever proposes once and the remaining refinements fall back to the coder
    # guessing at tile numbers -- so `stagnation_K` would count the model's
    # guesses rather than BO trials, and six evaluations in seven would be
    # spent on exactly the thing the lever exists to replace.
    verdict = "kept" if improved else ("tied" if (a.s and j == bar) else "reverted")
    # What the ratchet did with it, kept with the measurement: `pareto` needs to
    # tell a discarded trade-off from a superseded winner, and neither survives
    # in `nodes` (only closed actions' bests do).
    cache[a.code_hash]["verdict"] = verdict
    # SAVED BEFORE THE PROPOSAL, deliberately. The measurement is made, charged
    # and ratcheted by this line; everything after it is a convenience for the
    # NEXT candidate. A campaign can lose a whole run's evaluations when a
    # failure while choosing the next tile comes out of `record` before the
    # only `save`.
    save(a.op_dir, st)
    proposal = _record_next_lever_value(a, st, stop_cause)
    save(a.op_dir, st)
    vals, _ = objective_values(objectives_of(st), a.p, m_eval)
    # The objective vector and what the archive did with it, as a row of the
    # action's table: under more than one objective, `reverted` alone hides
    # whether the point was a discarded trade-off or genuinely dominated.
    row = CandidateRow(vals if a.s else {}, verdict + (" ↑" if improved else ""),
                       arc_verdict, r["n"], note=_candidate_note(a, acceptance))
    log_candidate(a.op_dir, st, a.code_hash, row)
    # The lever's next value rides back with the verdict, so the optimizer
    # evaluates it without another select and without the coder.
    record_payload = {"J": j, "improved": bool(improved), "n": r["n"],
                      # `improved` is front acceptance; `faster` is the
                      # latency question it used to be.
                      "faster": bool(faster),
                      "acceptance": acceptance,
                      # What the front did with it, next to what the ratchet did.
                      "objectives": vals, "archive": arc_verdict,
                      "archive_points": len(st.get("archive") or []),
                      "n_static": r.get("n_static", 0),
                      **_refine_close_payload(r, undetermined_k, stop_cause)}
    if proposal:
        record_payload["bayesian_optimization"] = proposal
    if chain_error:
        _halt_on_broken_chain(a, st, record_payload, chain_error, op_file)
    out(record_payload)


def refine_and_bound(st, pr, parent, field):
    """
    Bump one unmeasured-proposal tally on the parent's refine record.

    Returns the record (or None, when the proposal has no parent), the TOTAL
    across every such tally, and the bound that total is measured against. The
    per-kind tallies stay -- a reader needs to know whether the gate refused the
    candidate or the action was resubmitting its own best -- but the close rule
    reads the total, so a run of one kind and a mixture of several are the same
    fact about the action: it is proposing things that produce no measurement.
    """
    r = _refine_for(pr, parent) if parent else None
    if r is not None:
        r[field] = r.get(field, 0) + 1
    return r, _unmeasured(r), _bound(st, "unmeasured_K")


def _reject_semantic_noop(a, st, pr, op_f, retune):
    """The cheaper and more decisive question, asked before the hardware one: is
    this the program it is being judged against?

    The eval cache answers that for identical BYTES; this answers it for
    identical SEMANTICS. A run can need it: a candidate that merely restates a
    parameter's own default -- `submit_before_loop=False`, say -- wins on a low
    draw of the incumbent's own latency distribution, and the delivered kernel
    changes for no reason.
    Rejecting here rather than at `record` is what makes it free: the device is
    never asked.

    Returns True when it has emitted a rejection and the caller must stop.
    """
    same, why = _semantic_noop(st, a.op_dir, op_f, a.parent)
    if not same:
        return False
    pr["semantic_noop_rejected"] = pr.get("semantic_noop_rejected", 0) + 1
    # Its own TALLY, and the shared bound. A semantic no-op is not a hardware
    # infeasibility and the two must stay distinguishable in the state, but
    # `stagnation_K` must not be advanced by either -- neither produced a
    # measurement. `_refine_for` creates the record here rather than waiting for
    # `record`, so the bound also covers an action whose FIRST candidate is a
    # no-op: previously that action had no `refine` to count into at all.
    r, unmeasured, unmeasured_k = refine_and_bound(st, pr, a.parent, "n_noop")
    rollback(a, st, r)
    log_note(a.op_dir, st,
             f"semantic no-op, rejected before the device — n_unmeasured "
             f"{unmeasured}/{unmeasured_k}", why)
    out({"feasible": False, "semantic_noop": True,
         "reason": f"this candidate is the incumbent program: {why}",
         # The hardware gate did NOT run -- this was rejected before it. On the
         # `static_feasibility: false` ablation arm that gate is off entirely,
         # and reporting `checked: true` there would describe a check the arm
         # does not perform.
         "checked": False, **retune,
         # `.get`, because a `refine` dict is not guaranteed to carry `n`: the
         # duplicate branch of `record` can create one before the action's
         # first measurement.
         "n": (r.get("n", 0) if r else 0),
         "n_static": (r.get("n_static", 0) if r else 0),
         "n_noop": (r.get("n_noop", 0) if r else 0),
         "n_unmeasured": unmeasured, "unmeasured_K": unmeasured_k,
         "stop_refine": unmeasured >= unmeasured_k,
         "stop_cause": ("unmeasured_K" if unmeasured >= unmeasured_k else None),
         "hint": "the delta has to change what the program DOES. Passing a "
                 "keyword its own default value is the same call as omitting "
                 "it, so there is nothing here for the device to measure."})
    return True


def _reject_wrapper_drift(a, st, pr, op_f, retune):
    """Before the capacity check and before any device time: a candidate that
    moved work into the wrapper is not a candidate, whatever its tiles do.

    Returns True when it has emitted a rejection and the caller must stop.
    """
    drift = _wrapper_drift(st, op_f)
    if not drift:
        return False
    pr["wrapper_drift_rejected"] = pr.get("wrapper_drift_rejected", 0) + 1
    r, unmeasured, unmeasured_k = refine_and_bound(st, pr, a.parent, "n_static")
    rollback(a, st, r)
    log_note(a.op_dir, st,
             f"wrapper drift, rejected before the device — n_unmeasured "
             f"{unmeasured}/{unmeasured_k}")
    out({"feasible": False, "reason": drift, "checked": True,
         "wrapper_frozen": False, **retune,
         "n": (r["n"] if r else 0),
         "n_static": (r.get("n_static", 0) if r else 0),
         "n_unmeasured": unmeasured, "unmeasured_K": unmeasured_k,
         "stop_refine": unmeasured >= unmeasured_k,
         "stop_cause": ("unmeasured_K" if unmeasured >= unmeasured_k else None),
         "hint": "PANKO optimizes inside the @jit kernel. Edit the kernel, "
                 "not the wrapper."})
    return True


def _reject_infeasible(a, st, pr, why, retune):
    """The capacity gate said no. Count it among the unmeasured proposals, not
    against stagnation.

    `n` is untouched on purpose: see `cmd_feasible`'s docstring. Only the
    separate bound can end a refinement here, and when it does it is a different
    event from stagnation, so it is logged as one. The refine record is created
    here when this is the action's first event, so an action whose opening
    neighbourhood is entirely infeasible can still reach the bound -- it used to
    need a measurement first, and an all-infeasible action never gets one.
    """
    pr["static_rejected"] = pr.get("static_rejected", 0) + 1
    r, unmeasured, unmeasured_k = refine_and_bound(st, pr, a.parent, "n_static")
    stop_refine = unmeasured >= unmeasured_k
    rollback(a, st, r)
    log_note(a.op_dir, st,
             f"infeasible, rejected before the device — n_unmeasured "
             f"{unmeasured}/{unmeasured_k}"
             + (" · closing on unmeasured_K, not stagnation" if stop_refine else ""),
             why)
    out({"feasible": False, "reason": why, "checked": True, **retune,
         "n": (r["n"] if r else 0),
         "n_static": (r.get("n_static", 0) if r else 0),
         "n_unmeasured": unmeasured, "unmeasured_K": unmeasured_k,
         "stop_refine": stop_refine,
         "stop_cause": ("unmeasured_K" if stop_refine else None),
         "static_rejected": pr["static_rejected"]})


def cmd_feasible(a):
    """Static hardware feasibility, checked before the candidate reaches the NPU.

    A rejection here costs no device time, so it does NOT consume eval_budget,
    and it does NOT advance the action's stagnation counter either.

    Charging it to `n` would be wrong twice over. `n` counts *measurements that
    failed to improve*, which is evidence about the action; a static rejection is
    evidence about one candidate, since the gate never ran the program. And the
    two are not independent -- the gate fires hardest exactly where an action is
    pushing a tile or a view against the buffer, which is where the interesting
    candidates are. An action can take rejection after rejection and close on
    the stagnation limit having never reached the device.

    Rejections are counted in `n_static`, against `unmeasured_K`. That bound
    exists only to guarantee termination: an action whose whole neighbourhood is
    infeasible would otherwise loop forever, spending no budget and so never
    hitting eval_budget either. It is set far above anything one action
    accumulates in practice, because a rejection costs one dispatch rather than
    one device run.

    Rejecting is deliberately conservative -- see feasibility.py. A false
    rejection deletes a real optimisation from the search space permanently,
    while a false acceptance costs one evaluation.
    """
    st = load(a.op_dir)
    pr = st["progress"]
    op_f = a.op_file or op_file_path(st, a.op_dir)
    retune = _retune_verdict(st, op_f)

    if _reject_semantic_noop(a, st, pr, op_f, retune):
        return

    if not st["config"].get("static_feasibility", True) or feasibility is None:
        out({"feasible": True, "reason": "", "checked": False, **retune})
        return
    # Same envelope the tile search validates against. `--ub-kb` / `--l1-kb`
    # still override, for a caller measuring a chip the state does not name.
    env = chip_envelope(st)
    hw = feasibility.HW(ub_budget_kb=(a.ub_kb or env["ub_kb"]),
                        l1_budget_kb=(a.l1_kb or env["l1_kb"]),
                        m_l1=a.m_l1, k_l1=a.k_l1)
    if _reject_wrapper_drift(a, st, pr, op_f, retune):
        return
    ok, why = feasibility.check(op_f, hw)
    if ok:
        out({"feasible": True, "reason": "", "checked": True, **retune})
        return
    _reject_infeasible(a, st, pr, why, retune)


def cmd_probe(a):
    """Cache lookup by code_hash. The optimizer calls this BEFORE running the (expensive) frozen
    E(x): a hit means these exact bytes were already evaluated, so it reuses (s,p) and skips
    compile+measure. Kills redundant re-evals of a coder no-op (unchanged bytes) or a re-proposed best.
    """
    st = load(a.op_dir)
    hit = st.get("eval_cache", {}).get(a.code_hash)
    out({"hit": hit is not None, "s": (hit or {}).get("s"), "p": (hit or {}).get("p"),
         "m": (hit or {}).get("m", {})})


def _close_without_candidate(a, st, pr, r):
    """Nothing beat the running best: count a stagnation, retire the action if it
    paid for one, and leave the working file AT the global best (records already
    rolled back; `_retire_spent_action` restores again defensively)."""
    pr["global_stagnation"] += 1
    pr["stage_stagnation"] += 1
    retired, why_not = _retire_spent_action(a, st, r)
    mine = bool(r and r.get("action") == a.action)
    # Both advances used to live only on the WITH-candidate path, so they were
    # reachable only from a close that beat the running best.
    #
    # `_advance_stage`: a run whose gains all come from swimlane actions closes
    # them improving -- but the stage never moved, because the improving close
    # is the same event. The recorded case is the reverse and worse: twenty
    # consecutive closes with no candidate, stage pinned at `frontend`, every
    # swimlane action scoring 0.6x against it for the whole run.
    #
    # `_advance_front_base`: in phase 2 a candidate is judged against the greedy
    # global best, so almost no phase-2 action closes WITH one. `tried_from` was
    # therefore never incremented, `front_base`'s fewest-tried rule had nothing
    # to order by, and phase 2 extended one single front point from start to
    # finish -- the opposite of the "spread the budget over the front" its own
    # docstring states.
    _advance_stage(st, pr)
    _advance_front_base(st, a)
    # Only when the open refinement IS this action's. `close --action <stale id>`
    # used to clear it unconditionally, which threw away a DIFFERENT action's
    # local best -- a program already snapshotted and beating the global best,
    # since `best_J` only moves at that action's own close -- along with its `n`,
    # its margins and its SPRT record. `refuse_early_close` deliberately admits
    # a mismatched id rather than refusing, so nothing else stood in the way.
    # Left in place, the next `select` refuses and names the action still open,
    # which is recoverable; a wiped refinement is not.
    if mine:
        pr["refine"] = None
    save(a.op_dir, st)
    log_note(a.op_dir, st,
             f"closed → no candidate beat the best · stagnation {_gstag(st)}"
             # `r` is None when the action produced nothing at all -- the
             # coder found the delta inexpressible, which is the one escape
             # `refuse_early_close` grants without a counter. Subscripting it
             # raised TypeError here, AFTER the `save` above: the stagnation
             # counters, the stage advance and the retirement were persisted
             # and then the call died, so the escape the close rule depends on
             # could not be taken.
             + ("" if mine or not r else
                f" · left {r.get('action')}'s refinement open")
             + (" · retired against this structure (revives if it changes)"
                if retired else f" · NOT retired: {why_not}"))
    log_status(a.op_dir, st)
    out({"closed": None, "note": "no successful candidate for this action",
         "retired": retired,
         "hint": ("this action is retired against the structure now in the file and "
                  "`select` will not offer it again until a delta changes that "
                  "structure -- do NOT prune it by hand" if retired else
                  "the action stays selectable at reduced priority: retirement "
                  "requires stagnation_K measurements AND a readable structure")})


def promote_new_best(a, st, pr, b, cid):
    """
    This action's best beat the global best. Re-point the frontier at it and
    re-analyse. Returns the symptoms that were re-boosted.
    """
    prev_p = pr["best_aicore_e2e_us"]
    pr["best_J"] = b["J"]
    st["best_node"] = cid
    pr["best_aicore_e2e_us"] = b["p"]
    mix = (b.get("m") or {}).get("mix_aic")
    if mix:
        pr["best_mix_aic_us"] = mix
    pr["best_speedup_vs_preopt"] = (
        round(pr["preopt_aicore_e2e_us"] / pr["best_aicore_e2e_us"], 4)
        if (pr.get("preopt_aicore_e2e_us") and pr.get("best_aicore_e2e_us"))
        else None)
    pr["global_stagnation"] = 0
    for n in st["nodes"]["open"]:   # greedy accumulate: stack subsequent actions on the new best
        n["parent"] = cid
    # re-analysis: the bottleneck shifts as we optimize -> re-point open beliefs (V) at the NEW
    # best kernel's current symptom, reusing the util/bubble we just measured (no extra eval).
    # `.get(k)` -> None when the key is absent, which is the point: an
    # unmeasured node must not be read as an idle one.
    reboosted = _symptom_reboost(st, (b.get("m") or {}).get("util"),
                                 (b.get("m") or {}).get("bubble"),
                                 op_file_path(st, a.op_dir), (b.get("m") or {}))
    admitted = _admit(st, a.op_dir)   # the new best may have new structure
    if admitted:
        log_note(a.op_dir, st,
                 f"preconditions now met — {len(admitted)} action(s) back on the "
                 f"frontier: {', '.join(admitted)}")
    gain = (prev_p - b["p"]) / prev_p if prev_p else 0.0
    if gain >= st["config"].get("stage_min_gain", 0.01):
        pr["stage_stagnation"] = 0
        pr["cycle_productive"] = True   # productive (>=1%): keep mining this stage
    else:
        pr["stage_stagnation"] += 1                                 # marginal (<1%): counts toward advancing
    return reboosted


def _advance_stage(st, pr):
    """Stage progression: advance CYCLICALLY (frontend->swimlane->incore->frontend...)
    when the current stage yields no productive (>=1%) gain; a full cycle with no
    >=1% gain => converged."""
    if pr["stage_stagnation"] < st["config"].get("stage_patience", 2):
        return
    i = (STAGE_ORDER.index(pr["active_stage"]) + 1) % len(STAGE_ORDER)
    if i == 0:  # wrapped incore -> frontend = a full cycle completed
        if not pr.get("cycle_productive", False):
            # Recorded either way; only halts when stage_convergence is on.
            pr["cycle_unproductive"] = pr.get("cycle_unproductive", 0) + 1
            if st["config"].get("stage_convergence", True):
                pr["converged"] = True
        pr["cycle_productive"] = False
    pr["active_stage"] = STAGE_ORDER[i]
    pr["stage_stagnation"] = 0


def cmd_close(a):
    """Attach the refinement's best candidate as a CLOSED node; update global best."""
    st = load(a.op_dir)
    pr = st["progress"]
    _refuse_close(a, st, pr)
    r = pr.get("refine")
    if not r or r.get("action") != a.action or r["best"] is None:
        _close_without_candidate(a, st, pr, r)
        return
    b = r["best"]
    act = next((n for n in st["nodes"]["open"] if n["id"] == a.action), None)
    stage = act["stage"] if act else pr["active_stage"]
    reboosted = []
    cid = next_id(st, "x")
    # `action` / `catalog_id` / `delta`: a closed node used to keep only its own
    # x-id, so what produced it was recoverable only from the log -- and only
    # while the log and the state described the same campaign.
    st["nodes"]["closed"].append({"id": cid, "parent": act["parent"] if act else "root",
                                  "stage": stage, "code_hash": b["code_hash"], "J": b["J"],
                                  "eval": {"s": 1, "p": b["p"], "m": b["m"]},
                                  "action": a.action,
                                  "catalog_id": (act or {}).get("catalog_id"),
                                  "delta": (act or {}).get("delta")})
    remaining = []
    for n in st["nodes"]["open"]:
        if n["id"] != a.action:
            remaining.append(n)
    st["nodes"]["open"] = remaining  # consumed
    if b["J"] > pr["best_J"]:
        reboosted = promote_new_best(a, st, pr, b, cid)
    else:
        # The action accepted a point -- `best` is only ever written by an
        # acceptance -- so it extended the front without beating the fastest
        # program. That is progress for the run's halt (`global_stagnation`
        # counts closes that added nothing), and it is not a latency gain for
        # the stage machinery, which keeps its latency meaning.
        pr["global_stagnation"] = 0
        pr["stage_stagnation"] += 1
    _advance_stage(st, pr)
    _advance_front_base(st, a)
    pr["refine"] = None
    save(a.op_dir, st)
    # leave the working file AT this action's best (= the new running best when it improved).
    restore(a.op_dir, op_file_path(st, a.op_dir), b["code_hash"])
    log_note(a.op_dir, st,
             f"closed → {cid} · best {b['p']} us (j {b['J']}) · global best_J "
             f"{pr['best_J']}{' ↑' if b['J'] >= pr['best_J'] else ''} · "
             f"stagnation {_gstag(st)} · symptoms {','.join(reboosted) or '—'}"
             + f" · next base {str(pr.get('base_code_hash'))[:8]}")
    log_status(a.op_dir, st)
    out({"closed": cid, "best_J": pr["best_J"], "best_aicore_e2e_us": pr["best_aicore_e2e_us"],
         "reboosted_symptoms": reboosted,
         "base_code_hash": pr.get("base_code_hash")})


def _apply_updates(op_nodes, edits):
    """V only. Revising a belief is not an attempt, and counting it as one
    inverted the whole mechanism: in a recorded run the two actions the optimizer
    had repeatedly identified as its most promising carried tried=3 without ever
    having been evaluated, and were outranked by an inapplicable action at
    tried=0. `tried` is incremented where an action is actually attempted -- at
    `close` -- and nowhere else.
    """
    upd = {u["id"]: u for u in edits.get("update", [])}
    for n in op_nodes:
        if n["id"] in upd:
            n["V"] = clamp_belief(upd[n["id"]]["V"])
            n["V_source"] = "model"
    return upd


def _apply_prunes(st, a, op_nodes, edits):
    """(permanent, conditional, ignored_held, unknown, structure).

    Pruning splits by what the optimizer could actually see. A node in `open` had
    its preconditions satisfied and was retired on judgment, so the retirement is
    permanent and `_admit` must honour it. A node in `held` is already governed by
    the deterministic precondition system, and letting the optimizer delete it
    would destroy the path that `held, never deleted` exists to preserve: a later
    action can introduce the structure that makes it applicable.

    `prune` retires an action against the CURRENT structure; `prune_permanent`
    retires it outright. The split exists because two prunes can be different
    in kind and the harness could not tell them apart: one disproved by a tile
    sweep on one particular program, the other inexpressible without changing
    the operator's frozen layout. Only the second is a fact about the operator.
    See `live_frontier`.
    """
    prune = set(edits.get("prune", []))
    forever = set(edits.get("prune_permanent", []))
    open_ids = {n["id"] for n in op_nodes}
    held_ids = {n["id"] for n in st["nodes"].get("held", [])}
    permanent = forever & open_ids
    conditional = (prune & open_ids) - permanent
    ignored_held = (prune | forever) & held_ids
    unknown = (prune | forever) - open_ids - held_ids
    sig = _structure(st, a.op_dir)
    if conditional and sig is None:
        # No signature means no scope, and `live_frontier` skips a node whose `pruned_on`
        # is falsy -- so writing None would report every conditional prune as
        # applied while none of them hid anything. That is the default state on an
        # arm built without `bayesian.lever`, and it fires transiently whenever the
        # file on disk does not parse. Retire them outright and say so, rather
        # than silently discarding what the optimizer asked for.
        permanent |= conditional
        conditional = set()
    for n in op_nodes:
        if n["id"] in conditional:
            # Kept in `open` rather than deleted: the node has to survive for its
            # evidence to expire. `live_frontier` hides it while the structure matches.
            n["pruned_on"] = sig
    st["nodes"]["open"] = [n for n in op_nodes if n["id"] not in permanent]
    st["nodes"]["pruned"] = sorted(set(st["nodes"].get("pruned", [])) | permanent)
    return permanent, conditional, ignored_held, unknown, sig


def _evolve_note(st, pr, live, ignored_held, unknown, bounded):
    """The warnings an evolve appends to its log line, in words."""
    note = ""
    if not live:
        if pr.get("expansion_declined"):
            note += f" | FRONTIER EXHAUSTED, declared: {pr['expansion_declined']}"
        else:
            note += (f" | FRONTIER EMPTY and nothing inserted: expansion required "
                     f"(request {pr.get('expansion_requests', 0)}). Propose new intents, "
                     f"or pass exhausted:\"<reason>\" to declare there are none.")
            note += _demand_note(st, pr)
    if ignored_held:
        note += (f" | {len(ignored_held)} prune(s) ignored: still held by unmet "
                 f"preconditions, not the optimizer's to delete: {sorted(ignored_held)}")
    if unknown:
        note += f" | {len(unknown)} prune(s) named no known node: {sorted(unknown)}"
    if bounded:
        note += f" | {len(bounded)} insert(s) had prior_gain bounded: {'; '.join(bounded)}"
    return note


def cmd_evolve(a):
    """Apply the optimizer's tree edits: {insert:[...], update:[...], prune:[ids]}."""
    st = load(a.op_dir)
    edits = json.loads(a.edits)
    op_nodes = st["nodes"]["open"]
    bounded = apply_inserts(st, op_nodes, edits)
    upd = _apply_updates(op_nodes, edits)
    # Pruning splits by what the optimizer could actually see. A node in `open`
    # had its preconditions satisfied and was retired on judgment, so the retirement
    # is permanent and _admit must honour it. A node in `held` is already governed
    # by the deterministic precondition system, and letting the optimizer delete it
    # would destroy the path that `held, never deleted` exists to preserve: a later
    # action can introduce the structure that makes it applicable.
    # `prune` retires an action against the CURRENT structure; `prune_permanent`
    # retires it outright. The split exists because two prunes can be different
    # in kind and the harness could not tell them apart: one disproved by a tile
    # sweep on one particular program, the other inexpressible without changing
    # the operator's frozen layout. Only the second is a fact about the
    # operator. See `live_frontier`.
    permanent, conditional, ignored_held, unknown, sig = _apply_prunes(st, a, op_nodes, edits)

    # Emptying the frontier is termination by another name, and termination is
    # not the model's decision. Extending the tree IS its job: `insert` is the
    # world model. So an evolve that leaves nothing to select must either offer
    # new intents or state, explicitly, that it has none. It has been observed:
    # a run whose last evolves inserted nothing and pruned steadily, the last of
    # them reasoned as "pruning exhausted actions to focus remaining budget on
    # novel directions" while offering no novel direction. The search halted
    # with most of its budget unspent.
    pr = st["progress"]
    # What `select` will actually see. A conditionally-pruned node stays in
    # `open` so its evidence can expire, so `open` is no longer the frontier and
    # asking it here would report a frontier that select cannot choose from.
    live = live_frontier(st, a.op_dir)
    _record_frontier(pr, edits, live, permanent or conditional,
                     front_demand(st, a.op_dir))
    # (stage progression is handled in cmd_close: cyclic advance on <stage_min_gain diminishing returns)
    save(a.op_dir, st)
    reasons = _evolve_reasons(edits)
    note = _evolve_note(st, pr, live, ignored_held, unknown, bounded)
    append_log(a.op_dir, st["op"],
               f"   evolve +{len(edits.get('insert', []))} insert / {len(upd)} update"
               f" / {len(conditional)} prune ({len(permanent)} permanent)"
               + (f"{note}" if note else "")
               # The reasons are the most re-read thing in this file -- they are
               # where a run says what it LEARNED -- so they get their own lines
               # rather than a tail on a long one.
               + "".join(f"\n   > {' '.join(x.split())}"
                         for x in str(reasons).split(";") if x.strip()))
    out({"open": len(live), "active_stage": st["progress"]["active_stage"],
         "pruned_permanently": sorted(permanent),
         # Retired against this structure only, and named so the optimizer can
         # see that its own evidence has an expiry date.
         "pruned_on_structure": sorted(conditional), "structure": sig,
         "prune_ignored_held": sorted(ignored_held),
         "prune_unknown": sorted(unknown), "pruned_total": len(st["nodes"]["pruned"]),
         # Named, not merely applied. The optimizer wrote a prior the core would
         # not honour, and it has to be able to see that its insert does not sit
         # where it asked for it to sit.
         "prior_gain_bounded": bounded,
         "expansion_required": bool(pr.get("expansion_required")),
         "expansion_declined": pr.get("expansion_declined"),
         # The direction, when the front can name one, beside the request it
         # qualifies. See `front_demand`.
         "demand": pr.get("expansion_demand"),
         **({"demand_hint": demand_hint_text(pr["expansion_demand"], objectives_of(st))}
            if pr.get("expansion_demand") else {})})


def cmd_stop(a):
    st = load(a.op_dir)
    pr = st["progress"]
    # `load` does not accumulate, so without this the halt check is an interval
    # stale.
    accumulate_wall_clock(pr, st.get("config") or {})
    reason, probe = device_verdict(st, a)
    if reason is None:
        reason = stop_condition(st, a)
    if reason is None and getattr(a, "force", ""):
        # No condition fired, but the caller knows the run is over: the agent
        # session ended without ever calling stop. Recording that is strictly
        # better than leaving stop_reason null, which is indistinguishable from
        # a crash and fails the Stage-7 completion gate. Flagged separately so
        # analysis can exclude forced stops from budget accounting.
        reason = a.force
        pr["forced_stop"] = True
    if reason is not None:
        # Before delivering, not after: an action still open may hold the best
        # program the run measured.
        _promote_open_refine(st, a)
        _deliver_best(st, a)
        pr["stop_reason"] = reason   # sticky; the Stage-7 completion gate reads this
        evaluation_session.close_session(a.op_dir)
        _open_verification(st)
    if probe is not None:
        _record_probe(st, a, probe)
    delivered = _delivered_check(st, a.op_dir)
    front = export_frontier(st, a.op_dir) if reason is not None else None
    save(a.op_dir, st)
    out({"stop": reason is not None, "reason": reason,
         "frontier": ({"points": front["points"], "written": front["written"],
                       "sources_missing": front["sources_missing"]} if front else None),
         "device_probe": probe,
         **delivered,
         "best_J": pr["best_J"], "best_aicore_e2e_us": pr["best_aicore_e2e_us"],
         # `wall_clock_s` is worked time, which is what the limit bounds.
         # `elapsed_s` is derived, not stored, and is here because the two differ
         # by every hour nobody was driving the campaign -- a 17 h run that
         # spanned three days should say so.
         "wall_clock_s": pr["wall_clock_s"],
         "elapsed_s": int(time.time() - pr["_start_ts"]),
         "evals_used": pr["evals_used"],
         "static_rejected": pr.get("static_rejected", 0),
         "duplicate_rejected": pr.get("duplicate_rejected", 0),
         "global_stagnation": pr["global_stagnation"],
         "front_stagnation": pr.get("front_stagnation", 0),
         "hv": pr.get("hv"), "hv_stagnation": pr.get("hv_stagnation", 0),
         "hv_last_gain": pr.get("hv_last_gain"),
         "forced": bool(pr.get("forced_stop")),
         "frontier_emptied_by_prune": pr.get("frontier_emptied_by_prune", 0),
         "expansion_requests": pr.get("expansion_requests", 0),
         "expansion_declined": pr.get("expansion_declined"),
         "verify": _verify_directive(st)})


def _measured_objectives(st, code_hash):
    """Whether the cache already holds this program WITH every objective."""
    entry = (st.get("eval_cache") or {}).get(code_hash)
    if not entry or not entry.get("s") or entry.get("p") is None:
        return False
    try:
        _, missing = objective_values(objectives_of(st), entry["p"], entry.get("m") or {})
    except ValueError:
        return False
    return not missing


def _remeasure_target(a, st, cache, samples):
    """The cache entry and archive row `--remeasure` adds readings to."""
    entry = cache.get(a.code_hash)
    point = next((e for e in (st.get("archive") or [])
                  if e["code_hash"] == a.code_hash), None)
    if entry is None or point is None:
        out({"remeasured": False, "refused": True, "error": "not_a_front_point",
             "code_hash": a.code_hash, "in_cache": entry is not None,
             "in_archive": point is not None,
             "hint": "`--remeasure` adds readings to a point the archive already "
                     "holds. A program that was never measured has to be "
                     "recorded normally, which is what charges it."})
        raise Refusal(3)
    if not samples:
        out({"remeasured": False, "refused": True, "error": "no_samples",
             "code_hash": a.code_hash,
             "hint": "pass measure_latency's `samples` (comma-separated). The "
                     "aggregate alone adds no evidence: the interval is "
                     "computed from the readings."})
        raise Refusal(3)
    return entry, point


def _repool_measured_objectives(a, st, entry, objs):
    """Pool each measured objective's readings and re-derive its value."""
    # EVERY measured objective is re-derived, not just `p`. A re-measurement
    # exists to settle a dominance the interval left undetermined, and the axis
    # it is undetermined on may not be `p`'s -- under `J_OBJECTIVE` it usually
    # is not. Pooling one objective's readings while another kept its first
    # median left `progress` and the front row describing the same program with
    # numbers from different measurements.
    entry["m"] = dict(entry.get("m") or {})
    fresh = record_metrics(a, st)
    for o in rederivable(objs):
        sf = o.get("samples_from")
        if not sf or sf == "samples":
            continue
        key = sf[2:]
        new_reads = fresh.get(key)
        if not isinstance(new_reads, (list, tuple)) or not new_reads:
            continue
        pooled_o = list(entry["m"].get(key) or []) + [float(x) for x in new_reads]
        entry["m"][key] = pooled_o
        if o["from"].startswith("m."):
            entry["m"][o["from"][2:]] = median_of(pooled_o)


def _refresh_remeasured_headline(a, st, entry, point):
    """Move the run's headline numbers when the re-measured row owns them."""
    pr = st["progress"]
    # The run's headline numbers are about a PROGRAM, and this is that program
    # when the point re-measured is the one `impl.py` holds. Left alone, the
    # report quoted `progress` while the front row for the same code_hash
    # carried the pooled median -- two different latencies for one kernel, and
    # the speedup was computed from the stale one.
    # The preopt is an archive point like any other and SKILL.md names
    # `--remeasure` as the way to settle an undetermined pair, so it is a
    # documented thing to remeasure -- and early in a run `ratchet_best_hash` returns
    # the preopt's own hash, so the branch below fired and moved
    # `best_aicore_e2e_us` while `preopt_aicore_e2e_us` kept its single original
    # reading. That number is the numerator of `best_speedup_vs_preopt` and of
    # every front point's `search_speedup`, so the run then reported a speedup of
    # the preopt against itself and carried the stale baseline into `verify`.
    # Same defect this function's own comment says it fixed for the best point.
    if a.code_hash == preopt_code_hash(st):
        pr["preopt_aicore_e2e_us"] = entry["p"]
        mix = (entry.get("m") or {}).get("mix_aic")
        if mix:
            pr["preopt_mix_aic_us"] = mix
    if a.code_hash == ratchet_best_hash(st):
        pr["best_aicore_e2e_us"] = entry["p"]
        pr["best_J"] = point["J"]
        mix = (entry.get("m") or {}).get("mix_aic")
        if mix:
            pr["best_mix_aic_us"] = mix
        pr["best_speedup_vs_preopt"] = (
            round(pr["preopt_aicore_e2e_us"] / entry["p"], 4)
            if (pr.get("preopt_aicore_e2e_us") and entry["p"]) else None)


def _cmd_remeasure(a, st):
    """More readings for a program the archive already holds.

    Why it exists: the front's dominance test is an interval (see
    `dominates_measured`), and the way to settle an undetermined pair is more
    readings of the two points -- not another candidate. The budget counts
    DISTINCT programs measured (`eval_budget` against `evals_used`, and the
    duplicate check makes a resubmission free), so charging a re-measurement
    would price the front's own evidence as if it were exploration and let a
    campaign run out of budget while measuring what it already had.

    So: no charge, no `n`, no ratchet, no SPRT, no outcome record. The readings
    are pooled with the ones already held, `p` becomes their median, and the
    front is re-derived from every point it holds -- new readings can widen an
    interval as easily as narrow it, so points that were dropped are not
    recoverable here but points that are now unordered stay.
    """
    pr = st["progress"]
    cache = st.setdefault("eval_cache", {})
    samples = cli_samples(a)
    entry, point = _remeasure_target(a, st, cache, samples)
    pooled = list(entry.get("samples") or []) + samples
    entry["samples"] = pooled
    entry["p"] = median_of(pooled)
    entry["remeasured"] = int(entry.get("remeasured", 0)) + 1
    observe_noise(st, samples, SEARCH_PROTOCOL)
    objs = objectives_of(st)
    _repool_measured_objectives(a, st, entry, objs)
    _refresh_front_point(point, entry, objs, st)
    _refresh_remeasured_headline(a, st, entry, point)
    before = len(st["archive"])
    st["archive"] = archive_reprune(objs, st["archive"], archive_alpha(st))
    kept = {e["code_hash"] for e in st["archive"]}
    pr["remeasurements"] = pr.get("remeasurements", 0) + 1
    still_on = a.code_hash in kept
    save(a.op_dir, st)
    append_log(a.op_dir, st["op"],
               f"- remeasure {a.code_hash[:8]}: +{len(samples)} readings "
               f"(n={len(pooled)}) p={round(entry['p'], 4)} | "
               f"front {before} -> {len(st['archive'])}"
               + ("" if still_on else " | this point is now dominated"))
    out({"remeasured": True, "code_hash": a.code_hash, "charged": False,
         # THIS point's readings, not the run's best.
         "readings": len(pooled), "aicore_e2e_us": entry["p"],
         "mix_aic_us": (entry.get("m") or {}).get("mix_aic"),
         "on_front": still_on, "archive_points": len(st["archive"]),
         "dropped": before - len(st["archive"]),
         "best_aicore_e2e_us": pr["best_aicore_e2e_us"], "best_J": pr["best_J"],
         "evals_used": pr["evals_used"]})


def _record_handled_elsewhere(a, st, pr):
    """True when this `record` call is not a candidate measurement at all."""
    if getattr(a, "remeasure", False):
        # Before everything: this is not a candidate, and every gate below
        # (the cache, the ratchet, the budget) is about candidates.
        _cmd_remeasure(a, st)
        return True
    if getattr(a, "unevaluated", None):
        # Before the cache, deliberately. Memoizing an unevaluated candidate
        # would make the duplicate check answer for a program nobody measured,
        # and the point has to stay submittable once the budget allows it.
        _reject_unevaluated(a, st, pr)
        return True
    return False


def _refuse_unmeasured_objectives(a, st, pr, m_rec):
    """Halt when a correct candidate carries no reading for a declared axis."""
    # BEFORE the cache and before the ratchet: a measurement missing one of the
    # run's objectives ends the run (see `_refuse_objectives`). Only a correct
    # candidate has objectives to read -- an incorrect one is discarded either
    # way, and an unevaluated one never reached this function.
    #
    # A RESUBMISSION is not a measurement. These bytes are already in the cache
    # with their objectives, so a re-record that carries no numbers is the
    # duplicate the check below rejects for free, not a run that measured
    # something and lost an objective. Halting on it ends a campaign on the
    # hash of its own incumbent.
    if a.s and not _measured_objectives(st, a.code_hash):
        objs = objectives_of(st)
        _, missing = objective_values(objs, a.p, m_rec)
        if missing:
            pr["stop_reason"] = pr.get("stop_reason") or "objective_unmeasured"
            pr["objective_unmeasured"] = {"code_hash": a.code_hash, "missing": missing}
            save(a.op_dir, st)
            _refuse_objectives("record", missing, objs,
                               {"code_hash": a.code_hash, "recorded": False,
                                "stop_reason": pr["stop_reason"]})


def _record_cache_entry(a, st, cache):
    """Memoize the candidate's measurement and whatever readings came with it."""
    # The further objectives ride on every measured candidate -- kept AND
    # reverted, since `eval_cache` keeps both.
    m_eval = record_metrics(a, st)
    # Read BEFORE the overwrite below, which is what dropped them.
    taken = block_samples(st, a.code_hash)
    cache[a.code_hash] = {"s": a.s, "p": a.p, "m": m_eval}
    samples = cli_samples(a)
    if samples:
        # Kept with the aggregate, not instead of it: `p` is still what the
        # ratchet compares, and the readings are what end-of-run verification
        # and the margin test need to know how far to trust it.
        cache[a.code_hash]["samples"] = samples
        observe_noise(st, samples, SEARCH_PROTOCOL)
    elif taken:
        # A block winner being promoted. No `observe_noise`: the block already
        # counted every trial's readings, and its own comment says the
        # promotion carries no `--samples` "so nothing is counted twice".
        cache[a.code_hash]["samples"] = taken
    return m_eval


def _record_acceptance(a, st, pr, op_file, m_eval):
    """Snapshot the candidate, then judge it against the front as it stood."""
    # BEFORE the ratchet: it rolls a non-improving candidate back, and the file
    # holding the candidate is the only copy of it.
    if st["config"].get("snapshot_all_candidates", True) and a.s:
        if not snapshot(a.op_dir, op_file, a.code_hash):
            pr["snapshot_misses"] = pr.get("snapshot_misses", 0) + 1
    # BEFORE the ratchet and before `archive_record`: the candidate is judged
    # against the front as it stood when it was measured, not against a front
    # that already contains it.
    return ((front_acceptance(st, a.code_hash, a.p, m_eval))
            if a.s else (None, None))


def _record_archive_and_outcome(a, st, m_eval, outcome, improved):
    """Offer the point to the front, then tell the stores what came of it."""
    pr = st["progress"]
    arc_verdict = archive_record(st, a.code_hash, a.p, m_eval, a.parent) if a.s else None
    # Measurements since the archive last accepted a point. Not a halt
    # condition -- it is the diagnostic C3's convergence test will read, and
    # what the report quotes for "the front stopped growing".
    pr["front_stagnation"] = (0 if arc_verdict == "inserted"
                              else pr.get("front_stagnation", 0) + 1)
    # AFTER the archive, because both of these need its verdict.
    _observe_action(st, a.parent, outcome)
    _record_outcome(st, a, outcome, arc_verdict, accepted=improved)
    return arc_verdict


def _record_next_lever_value(a, st, stop_cause):
    """The lever's next value for the parent action, when it is still open."""
    proposal = None if stop_cause else bayes_next(st, a.op_dir, a.parent)
    if isinstance(proposal, dict) and proposal.get("proposed") is False:
        log_note(a.op_dir, st, f"the tile lever could not propose: "
                 f"{proposal['error']}")
    return proposal


def _refine_close_payload(r, undetermined_k, stop_cause):
    """The close-rule tallies a `record` directive reports back."""
    # Measured, and the readings could not order it against the front. It does
    # not advance `n`.
    return {"n_undetermined": r.get("n_undetermined", 0),
            "undetermined_K": undetermined_k,
            "stop_refine": stop_cause is not None,
            "stop_cause": stop_cause}


def _reject_unevaluated(a, st, pr):
    """The candidate was never measured, so there is nothing to record about it.

    `s = 0` is the right verdict for a compile FAILURE and the wrong one for a
    compile that ran past the budget: nothing about the candidate was
    established. Two things follow.

    It must not be charged. The budget counts evaluations, and an evaluation is
    a number; a timeout produced none -- even though, unlike a static rejection,
    device time was spent.

    It must not be learned from. `s = 0` is a label, and it propagates: into the
    ratchet's bar, into the tile study's surrogate, and into anything later that
    reads the measurements to decide what an action is worth. Compile time
    scales with program size, so the label is not even random -- it marks the
    large candidates wrong.

    Counted among the unmeasured proposals, because that is what it is: no
    measurement was produced. `stagnation_K` counts measurements and there are
    none here, and `eval_budget` is not being charged, so without that bound an
    action producing only slow compiles could not end. It had its own bound of
    4 until C-series, and it never fired.
    """
    pr["unevaluated_rejected"] = pr.get("unevaluated_rejected", 0) + 1
    reasons = pr.setdefault("unevaluated_reasons", {})
    reasons[a.unevaluated] = reasons.get(a.unevaluated, 0) + 1
    r, unmeasured, unmeasured_k = refine_and_bound(st, pr, a.parent, "n_timeout")
    # The working file still holds the candidate nobody could measure, so the
    # next one for this action would be applied on top of it.
    rollback(a, st, r)
    stop_refine = unmeasured >= unmeasured_k
    log_note(a.op_dir, st,
             f"{a.code_hash[:8]} unevaluated ({a.unevaluated}) — not charged, "
             f"n unchanged, n_unmeasured {unmeasured}/{unmeasured_k}")
    out({"recorded": False, "unevaluated": a.unevaluated,
         "reason": ("the candidate was never measured, so this is not a verdict "
                            "about it: not charged to eval_budget and not recorded as s=0"),
         "evals_used": pr["evals_used"],
         "n": (r.get("n", 0) if r else 0),
         "n_timeout": (r.get("n_timeout", 0) if r else 0),
         "n_unmeasured": unmeasured, "unmeasured_K": unmeasured_k,
         "stop_refine": bool(stop_refine),
         "stop_cause": ("unmeasured_K" if stop_refine else None),
         "eval_timeout_s": st["config"].get("eval_timeout_s"),
         "eval_timeout_source": st["config"].get("eval_timeout_source"),
         "hint": ("raise eval_timeout_s, or select an action whose candidates "
                  "compile inside it" if stop_refine else None)})


def apply_inserts(st, op_nodes, edits):
    """Add the model's new intents to the frontier.

    BOTH factors the model supplies here are bounded on the way in -- see
    `clamp_belief` for V and `clamp_prior` for prior_gain. The notes come back so
    `evolve` can REPORT an adjustment instead of making it quietly: a prior the
    harness silently rewrote is a selection the optimizer cannot account for.
    """
    notes = []
    for ins in edits.get("insert", []):
        prior, note = clamp_prior(ins.get("prior_gain", CATALOG_PRIOR_DEFAULT))
        stage, stage_note = clamp_stage(ins.get("stage"))
        node_id = next_id(st, "u")
        for msg in (note, stage_note):
            if msg:
                notes.append(f"{node_id}: {msg}")
        node = {"id": node_id, "parent": ins.get("parent") or "root",
                "stage": stage, "delta": ins.get("delta", ""),
                "V": clamp_belief(ins.get("V", 0.5)),
                "prior_gain": prior, "tried": 0,
                # What the model said, kept for the record. Under learned g it
                # does not set g -- the insert CLASS's measured record does --
                # because the model's opinion of its own intent is already V.
                "prior_gain_model": prior, "V_source": "model"}
        if _learned_g_on(st):
            node["prior_gain"] = _learned_g(
                st, node, action_posterior.load_table(_prior_table_path(st)))
        op_nodes.append(node)
    return notes


def _evolve_reasons(edits):
    """The reason lines an evolve carries, joined into one string."""
    # A caller that passes ONE reason as a string used to have it joined
    # character by character -- `"; ".join("abc")` is `"a; b; c"` -- and the
    # trajectory then carried one letter per line.
    raw_reasons = edits.get("reasons") or []
    if isinstance(raw_reasons, str):
        raw_reasons = [raw_reasons]
    return "; ".join(str(x) for x in raw_reasons) or "(no reason given)"


def _promote_open_refine(st, a):
    """Close out an open action's best before delivering, if it beat the best.

    `best_J` and `best_node` move at `close`, and the budget does not wait for
    one. A campaign can measure a candidate that beats its preopt, record it
    as `kept`, and then run out of evaluations with the action still open -- so
    `stop` delivers the preopt and the verification reads `no_change`. The
    program was measured, snapshotted and judged better; the only thing
    missing was a `close` the budget did not allow.

    So the ratchet is run here on the same terms it runs at `close`: only a
    candidate that beat the global best is promoted, and it becomes a closed
    node like any other.
    """
    pr = st["progress"]
    r = pr.get("refine") or {}
    b = r.get("best")
    if not b or b.get("J") is None or b["J"] <= pr["best_J"]:
        return None
    act = next((n for n in st["nodes"]["open"] if n["id"] == r.get("action")), None)
    cid = next_id(st, "x")
    st["nodes"]["closed"].append(
        {"id": cid, "parent": act["parent"] if act else "root",
         "stage": act["stage"] if act else pr["active_stage"],
         "code_hash": b["code_hash"], "J": b["J"],
         "eval": {"s": 1, "p": b["p"], "m": b.get("m") or {}},
         "action": r.get("action"),
         "catalog_id": (act or {}).get("catalog_id"),
         "delta": (act or {}).get("delta"),
         "closed_at": "stop"})
    st["nodes"]["open"] = [n for n in st["nodes"]["open"] if n["id"] != r.get("action")]
    promote_new_best(a, st, pr, b, cid)
    pr["refine"] = None
    pr["promoted_at_stop"] = {"action": r.get("action"), "node": cid,
                              "J": b["J"], "p": b["p"]}
    append_log(a.op_dir, st["op"],
               f"- promote-at-stop[{r.get('action')}] -> {cid}: j={b['J']} p={b['p']} "
               f"(the action was still open when the run ended)")
    return cid
