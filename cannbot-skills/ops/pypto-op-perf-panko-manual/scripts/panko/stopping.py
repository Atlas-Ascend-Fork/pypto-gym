#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The budget, the clock, and every rule that ends an action or a run.

Including the device's own verdict: a block that aborted on environment
faults ends the run on the device, not on the search.
"""


import evaluation_session

from .config import DEFAULT_CONFIG, FREE, Refusal, UNMEASURED_FIELDS, _bound, bayesian, out, predicates
from .state import _structure, live_frontier, op_file_path, ratchet_best_hash, restore, save
from .measurement import objective_values, objectives_of
from .front import delivery_point


def _bayes_free_trials(st):
    """0 unless the tile lever is on, so the ablation arm is untouched."""
    cfg = st.get("config", {})
    if not cfg.get("bayesian_optimization_tile_search", False):
        return 0
    return int(cfg.get("bayesian_optimization_free_trials", 0) or 0)


def _is_bayes_lever(node):
    """A frontier node the tile BO lever drives, rather than the model."""
    if bayesian is None:
        return False
    return node.get("catalog_id") in getattr(bayesian.lever, "TILE_ACTIONS", ())


def _admit(st, op_dir):
    """Recompute the kernel's structure and move newly-qualified actions from
    `held` back into `open`.

    Called at init and after every improvement, because the structure is not
    fixed: an action that introduces a matmul makes the whole cube family
    reachable, and holding rather than deleting is what preserves that path.
    Returns the ids that were admitted, for the log.
    """
    if predicates is None or not st["config"].get("action_preconditions", True):
        return []
    preds = predicates.compute(op_file_path(st, op_dir) or "", op_dir)
    st["predicates"] = preds
    pruned = set(st["nodes"].get("pruned", []))
    admitted = []
    keep = []
    for n in st["nodes"].get("held", []):
        if n["id"] in pruned:      # the optimizer retired this while it was reachable
            keep.append(n)
            continue
        if predicates.satisfies(n.get("requires"), preds):
            st["nodes"]["open"].append(n)
            admitted.append(n.get("catalog_id") or n["id"])
        else:
            keep.append(n)
    st["nodes"]["held"] = keep
    return admitted


def fault_markers(ctx):
    """The distinct device fault markers this block saw."""
    return sorted({f["marker"] for f in ctx["faults"] if f.get("marker")})


def _free_rejects(result):
    """Counts of the trials rejected without touching the device, by cause."""
    c = {}
    for h in result.get("history", []):
        k = FREE.get(h.get("status"))
        if k:
            c[k] = c.get(k, 0) + 1
    return "/".join(f"{k}:{v}" for k, v in sorted(c.items())) or "0"


def _refine_for(pr, action):
    """The refinement record for `action`, created if this is its first event.

    Every counter that bounds a refinement has to survive the first event that
    moves it, and two of those events happen BEFORE any measurement: a static
    infeasibility and a semantic no-op, both decided by `feasible` without a
    device. The record used to be created only by `record`, and the pre-device
    branches incremented `n_static` / `n_noop` only `if r and r["action"] ==
    parent` -- so an action whose whole opening neighbourhood was rejected had
    nowhere to put them. Both read 0 forever and `unmeasured_K` could never
    fire, which is the opposite of what SKILL.md promises: the optimizer keeps
    selecting an action that has never produced a measurable candidate.

    Reported from a run with `unmeasured_K=2` and three certainly-out-of-bounds
    candidates: `static_rejected` reached 3 while every response still said
    `n_static=0, stop_refine=false`.

    `record` names the action it is recording for, so it always gets a record.
    `feasible` does not: its `--parent` defaults to empty, and such a call
    belongs to no refinement, so those two call sites skip this entirely rather
    than open a record keyed by "".
    """
    r = pr.get("refine")
    if not r or r.get("action") != action:
        # Fully shaped from the start. The duplicate branch used to leave a
        # headless record behind, which every later reader had to repair, and
        # which a pre-device counter would now silently discard.
        r = {"action": action, "best": None, "threshold_J": pr["best_J"],
             "n": 0, "n_undetermined": 0,
             "n_static": 0, "n_noop": 0, "n_dup": 0, "n_timeout": 0}
        pr["refine"] = r
    return r


def _refuse_close(a, st, pr):
    """The toll on the entrance to `close`: an action may not be closed before it
    has been refined. Emits and raises when it refuses."""
    why = refuse_early_close(st, a.action)
    if not why:
        return
    pr["early_close_refused"] = pr.get("early_close_refused", 0) + 1
    save(a.op_dir, st)
    out({"closed": None, "refused": True, "reason": why,
         "n": (pr.get("refine") or {}).get("n", 0),
         "stagnation_K": st["config"].get("stagnation_K", 7)})
    raise Refusal(3)


def _retire_spent_action(a, st, r):
    """Retire an action that paid `stagnation_K` measurements and produced nothing.

    Lowering the priority is not enough. Several actions can each run the full
    `stagnation_K` measurements, every one closed "no successful candidate",
    and all of them stay selectable; the selector then re-picks a spent lever
    with a handful of evaluations left, leaving the last untried lever class
    never tested at all, and the optimizer has to prune them BY HAND.

    So retire it, on exactly the terms `evolve`'s conditional prune uses:
    `pruned_on` scopes the retirement to the structure now in the file, and
    `live_frontier` revives the action once a later delta changes that structure. The
    retirement is never permanent, because the evidence is about geometry that a
    restructure invalidates.

    `refuse_early_close` still guards the entrance, so this cannot skip the
    toll, only record that the toll was paid for nothing. The other cases it
    admits have paid for no retirement and get none: an action with no
    refinement, a `refine` belonging to a different action, the escape hatch for
    a delta the coder found inexpressible at n=0, and the `unmeasured_K`
    release. Retiring on those would record `stagnation_K` measurements that
    never happened.

    Returns (retired, why_not).
    """
    stagnation_k = st["config"].get("stagnation_K", DEFAULT_CONFIG["stagnation_K"])
    mine = r if (r and r.get("action") == a.action) else {}
    paid, disproof = disproved_on_structure(st, mine, stagnation_k)
    # AFTER the roll-back, not before. `_structure` reads the file on disk, and
    # the working file at this point may still hold the last rejected candidate
    # -- which is a program the run has just discarded. Scoping the retirement to
    # that structure retires the action against something no longer on disk, so
    # `live_frontier` shows it again on the very next `select` while the payload has
    # already told the optimizer not to prune it.
    restore(a.op_dir, op_file_path(st, a.op_dir), ratchet_best_hash(st))
    sig = _structure(st, a.op_dir) if paid else None
    retired = False
    for n in st["nodes"]["open"]:
        if n["id"] == a.action:
            n["tried"] = n.get("tried", 0) + 1
            # No signature means no scope. `live_frontier` skips a falsy `pruned_on`, so
            # writing None would report a retirement that hides nothing. Falsy
            # here means `bayesian.lever` is absent or the file could not be read
            # -- a file that does not PARSE yields the string "unparseable",
            # which is a real scope that stops matching as soon as the file
            # parses again. Leave the rest to `tried`.
            if sig:
                n["pruned_on"] = sig
                retired = True
    # Retiring the last live action empties the frontier, and until now only
    # `evolve` could do that -- so the counter that exists to catch a run halting
    # far under budget on an empty frontier could not see this route.
    if retired and not live_frontier(st, a.op_dir):
        st["progress"]["frontier_emptied_by_prune"] = \
            st["progress"].get("frontier_emptied_by_prune", 0) + 1
    # Three distinct reasons, and they were being reported as two. An action that
    # paid in full but is no longer in `open` got "no structural signature" when
    # the signature was fine, and an action whose `refine` belongs to somebody
    # else got that other action's `n` quoted back.
    if not paid:
        why_not = (f"n={mine.get('n', 0)} of stagnation_K={stagnation_k} and no "
                   f"deterministic release, so nothing was disproved")
    elif not sig:
        why_not = "no structural signature to scope the retirement to"
    else:
        why_not = f"{a.action} is not on the open frontier"
    return retired, (why_not if not retired else disproof)


def device_probe(st, run_cmd=None):
    """
    Is the device usable NOW? Asked before any run is ended for it being busy.

    A block's faults are observations from the moment that block ran. Ending a
    campaign on them asserts something about the present from evidence about the
    past, and the cost is plain: a few faults inside one block can end a
    campaign a fraction of the way into its budget, most of it never spent, on a
    device nobody has asked since.

    The probe re-runs the correctness command and looks for a DEVICE marker in
    the output. Whether the kernel passes is beside the point -- a golden
    mismatch is a usable device saying no, which is exactly what the probe wants
    to distinguish from a device that will not answer at all.
    """
    cmd = st.get("test_command")
    if not cmd:
        return {"ran": False, "usable": False,
                "note": "no test_command recorded; the block's observation stands"}
    # The same shell-free runner the evaluator uses, so the probe re-runs the
    # command exactly as the block did -- see `evaluator.command_steps`.
    runner = run_cmd or (lambda c, t: bayesian.evaluator.run_command(c, t)[0])
    try:
        text = runner(cmd, st["config"].get("eval_timeout_s", 300))
    except Exception as e:                       # timeout, OSError, anything
        return {"ran": False, "usable": False, "note": f"probe could not run: {e}"}
    markers = getattr(bayesian.evaluator, "DEVICE_FAULT_MARKERS", ())
    hit = next((m for m in markers if m in text), None)
    return {"ran": True, "usable": hit is None, "marker": hit}


def _unmeasured(r):
    """Proposals from this action that produced no measurement.

    The per-kind tallies are kept and reported -- a reader needs to know whether
    an action was rejected by the gate or was resubmitting its own best -- but
    the close rule reads only the total.
    """
    return sum(int((r or {}).get(k) or 0) for k in UNMEASURED_FIELDS)


def _refine_stop_cause(r, stagnation_k, undetermined_k, unmeasured_k):
    """Which of the three close rules retires the action, or None."""
    # Three questions, and they are the whole rule. `success` is not here: it
    # ends the RUN (`stop_condition`), and an action is not a run.
    undetermined = r.get("n_undetermined", 0)
    unmeasured = _unmeasured(r)
    return ("stagnation_K" if r["n"] >= stagnation_k
            else "undetermined_K" if (undetermined_k
                                      and undetermined >= undetermined_k)
            else "unmeasured_K" if (unmeasured_k
                                    and unmeasured >= unmeasured_k)
            else None)


def refuse_early_close(st, action):
    """None if this action may be closed, else the reason it may not.

    `stagnation_K` is listed among the deterministic terms, and it was not one:
    `record` computed `stop_refine` and `close` never looked at it, so whether an
    action had been tested was the optimizer's call. Two runs of the SAME build
    read the rule in opposite ways -- one wrote "the harness requires one more
    candidate and I do not override its judgment" and closed at seven, the other
    wrote "closed by judgement, not by the counter" and closed six actions at a
    mean of 1.3. An action abandoned after one non-improving candidate has not
    been tested; it has been sampled once against a measurement whose cv is 15%.

    The single exception is an action that produced no candidate at all -- the
    coder found the delta inexpressible in this build, so there is nothing to
    refine and nothing to learn by demanding six more. Refusing that would
    deadlock the run.
    """
    pr = st["progress"]
    r = pr.get("refine") or {}
    if r.get("action") != action:
        return None
    if r.get("best") is None and not r.get("n"):
        return None                                  # no candidate ever recorded
    stagnation_k = st["config"].get("stagnation_K", 7)
    n = r.get("n", 0)
    # Whatever `disproved_on_structure` reads, `close` has to honour. They were two lists
    # that drifted: the harness would report `stop_refine: true` with a cause
    # and then refuse the close, which is a deadlock -- `n` cannot advance on
    # proposals that were not measurements. One function, asked once.
    paid, _why = disproved_on_structure(st, r, stagnation_k)
    if paid:
        return None
    if (st["config"].get("success_stop", DEFAULT_CONFIG["success_stop"])
            and r.get("best") and r["best"].get("J", 0) >= 100):
        # The run has reached `P_ref`. Not in `disproved_on_structure`: that answers "is
        # this action spent", and it drives RETIREMENT -- an action that just
        # hit the target is the last thing to retire. Here it only has to be
        # closeable, so that `stop` can deliver. `J` rather than
        # `target_met` on purpose: this is about the candidate in hand, and
        # the refine record carries its `J` already.
        return None
    return (f"n={n} of stagnation_K={stagnation_k}, "
            f"n_undetermined={r.get('n_undetermined', 0)} of "
            f"{_bound(st, 'undetermined_K')}, "
            f"n_unmeasured={_unmeasured(r)} of {_bound(st, 'unmeasured_K')}: "
            f"this action has not been tested. Propose the next concrete "
            f"candidate for it. An axis you believe is exhausted still has to "
            f"be shown exhausted -- submit the remaining values and let the "
            f"duplicate check reject them for free.")


def disproved_on_structure(st, r, stagnation_k):
    """Has this action been disproved ON THIS STRUCTURE? (bool, one-line reason).

    `stagnation_K` measurements is one way and used to be the only one. The
    others are `undetermined_K` -- measured, and the readings could not order it
    against the front -- and `unmeasured_K`, the proposals that produced no
    measurement at all. Each is evidence about the action on the program in
    front of it: its neighbourhood is statically infeasible, or it can only
    resubmit what it already has, or it cannot compile inside the budget.

    `refuse_early_close` calls THIS function, so the list cannot drift from
    what `close` honours.

    All three retire the action. Gating retirement on `stagnation_K` alone left
    the other two releasing an action at its exact score -- `pypto_action_priority`
    does not read `tried` -- so the deterministic argmax handed it straight back.
    A run can spend its whole wall clock that way: static rejections, close,
    select the same action, more rejections, charging zero evaluations so no
    budget stop ever fires.

    Retiring is safe here precisely because it is SCOPED: `live_frontier` revives the
    action the moment `structural_signature` moves, which is exactly when a
    static infeasibility or an inexpressible delta might stop being true.
    """
    if not r:
        return False, ""
    if r.get("n", 0) >= stagnation_k:
        return True, f"n={r['n']} measurements did not beat the bar"
    und_k = _bound(st, "undetermined_K")
    if und_k and r.get("n_undetermined", 0) >= und_k:
        return True, (f"undetermined_K: {r['n_undetermined']} candidates the "
                      f"readings could not order against the front")
    unm_k = _bound(st, "unmeasured_K")
    unm = _unmeasured(r)
    if unm_k and unm >= unm_k:
        kinds = ", ".join(f"{k[2:]} {r[k]}" for k in UNMEASURED_FIELDS if r.get(k))
        return True, f"unmeasured_K: {unm} proposals reached no measurement ({kinds})"
    return False, ""


def invocation_hint(markers):
    """What to fix, for a run that ended on invocation faults.

    A device-context fault is not a bug in this harness even though it is filed
    as one: the command ran, the device was never opened, and the fix is in the
    environment the command was given rather than in any candidate. Saying so
    here is the difference between a five-minute fix and a re-run that fails the
    same way.
    """
    known = getattr(getattr(bayesian, "evaluator", None),
                    "DEVICE_CONTEXT_MARKERS", ())
    ctx_markers = [m for m in markers if m in known]
    if ctx_markers:
        return ("the process never obtained a device context "
                f"({', '.join(ctx_markers)}). Nothing here is a verdict on any "
                "candidate. Check the device id the test command is given "
                "(TILE_FWK_DEVICE_ID / ASCEND_RT_VISIBLE_DEVICES) and that the "
                "device named by --device exists, then re-run. No evaluations "
                "were charged for these.")
    return ("the command could not be performed at all "
            f"({', '.join(markers) or 'see faults.jsonl'}). This is a bug in how "
            "the campaign was invoked -- check --test-command and --lint-root -- "
            "not a verdict on any candidate.")


def device_verdict(st, a):
    """(reason, probe) for a run whose last block aborted on environment faults.

    A device another process is holding is not a search outcome, and it is the
    one stop the optimizer must not have to argue for: every block started on a
    contended device spends a few evaluations rediscovering it. Read off the
    last block's own abort rather than taken on the agent's word.

    Two things happen before that abort ends the run. A fault whose marker says
    the command never STARTED is a bug in this harness, not a busy device, and
    recording it as contention is how a mistyped path becomes a line in a
    results table. And a device that was busy some minutes ago may not be busy
    now, so it is asked.
    """
    last_block = (st.get("blocks") or [None])[-1]
    if not (last_block and "environment faults" in (last_block.get("aborted") or "")):
        return None, None
    if last_block.get("fault_kind") == "invocation":
        # Named, not just classified. `harness_fault` on its own sends the next
        # session reading this file; the markers say which of the two very
        # different bugs it is -- a path this harness got wrong, or a device
        # context that was never opened -- and the second has a one-line fix
        # the agent can apply without touching any code.
        markers = last_block.get("fault_markers") or []
        st["progress"]["harness_fault"] = {
            "markers": markers,
            "hint": invocation_hint(markers),
            "block": last_block.get("action"),
            "env_faults": last_block.get("env_faults")}
        return "harness_fault", None
    evaluation_session.close_session(a.op_dir)
    probe = device_probe(st)
    return (None if probe["usable"] else "device_unavailable"), probe


def target_met(st, op_dir):
    """Has the program this run would DELIVER reached `P_ref`?

    On the primary objective's own reading, not on `best_J`. `J` is
    `100 * P_ref / p`, so `J >= 100` says exactly this -- but `best_J` only
    moves at an action's `close` and had already been shown to drift from the
    measurements (`delivery_point`, which this reads instead). A run that met
    the target inside an action still open would not have seen it.
    """
    objs = objectives_of(st)
    prim = objs[0]
    p_ref = ((st.get("target") or {}).get("P_ref"))
    if not p_ref:
        return False
    h, _ = delivery_point(st, op_dir)
    entry = (st.get("eval_cache") or {}).get(h) if h else None
    if not entry:
        return False
    vals, missing = objective_values(objs, entry.get("p"), entry.get("m"))
    if missing:
        return False
    v = vals.get(prim["name"])
    if v is None:
        return False
    return v >= float(p_ref) if prim["direction"] == "max" else v <= float(p_ref)


def stop_condition(st, a):
    """The first halt condition that has fired, or None."""
    pr, cfg = st["progress"], st["config"]
    if (cfg.get("success_stop", DEFAULT_CONFIG["success_stop"])
            and target_met(st, a.op_dir)):
        return "success"
    if pr.get("converged"):
        return "converged"
    if pr["wall_clock_s"] >= cfg["wall_clock_limit_s"]:
        return "wall_clock"
    if cfg.get("eval_budget") and pr["evals_used"] >= cfg["eval_budget"]:
        return "eval_budget"
    if (cfg.get("hv_stagnation_limit")
            and pr.get("hv_stagnation", 0) >= cfg["hv_stagnation_limit"]):
        return "hv_stagnation"
    if (not live_frontier(st, a.op_dir) and not pr.get("refine")
            and pr.get("expansion_declined")):
        return "frontier_exhausted"
    if (cfg.get("global_stagnation_limit")
            and pr["global_stagnation"] >= cfg["global_stagnation_limit"]):
        return "global_stagnation"
    return None


def block_budget(pr, cfg, a):
    """How many device trials this block may spend.

    Three limits, and the block gets the smallest: what is left of the run
    budget, the tile SHARE, and the per-block cap.

    The share is a RUNNING RATIO, not a slice of the total: tile blocks may
    never have consumed more than `block_budget_share` of everything this run
    has spent. It used to be `budget * share`, which is undefined without a
    total -- so `eval_budget: null` silently removed the guard altogether. A
    ratio against `evals_used` is the same invariant ("blocks may not crowd out
    structural search") and it is defined either way, which is the point: a
    campaign should not lose a guard by declining to name a budget.

    The floor keeps the first block usable. Early in a run `evals_used` is
    small, so the ratio alone would grant nearly nothing and a block fired at
    eval 3 could never run; `block_max_trials` is both the ceiling on one block
    and the floor the ratio may not push below.
    """
    budget = cfg.get("eval_budget")
    limits = []
    if budget:
        limits.append(max(0, budget - pr["evals_used"]))
    share = cfg.get("block_budget_share", DEFAULT_CONFIG["block_budget_share"])
    cap = cfg.get("block_max_trials", DEFAULT_CONFIG["block_max_trials"])
    if share is not None:
        spent = pr.get("block_evals", 0)
        # The floor is on the TOTAL this run's blocks may spend, not on what is
        # left: `max(remaining, cap)` would top up a block that had already
        # overspent, and the share could then never bind.
        total_allowed = int(share * pr["evals_used"])
        if cap:
            total_allowed = max(total_allowed, cap)
        limits.append(max(0, total_allowed - spent))
    if cap:
        limits.append(cap)
    if a.max_trials:
        limits.append(a.max_trials)
    return min(limits) if limits else None


def budget_reason(pr, cfg, a):
    """Why this block was granted no trials -- which limit bound it.

    `bayesian.block.run` says "no evaluation budget remaining" for all of them,
    which reads as the campaign being over while most of its evaluations are
    still unspent, when what has run out is the tile SHARE.
    They have opposite responses -- raise the budget, raise the share, raise the
    per-block cap, or pass a larger --max-trials -- so the message has to name
    the one that fired.
    """
    budget = cfg.get("eval_budget")
    share = cfg.get("block_budget_share", DEFAULT_CONFIG["block_budget_share"])
    cap = cfg.get("block_max_trials", DEFAULT_CONFIG["block_max_trials"])
    if budget and (budget - pr["evals_used"]) <= 0:
        return (f"the RUN budget is spent: {pr['evals_used']} of {budget} "
                f"evaluations used. The campaign is over; call `stop`.")
    if share is not None:
        spent = pr.get("block_evals", 0)
        total_allowed = int(share * pr["evals_used"])
        if cap:
            total_allowed = max(total_allowed, cap)
        if total_allowed - spent <= 0:
            return (f"the TILE share is spent, not the run budget: tile blocks "
                    f"have used {spent} of the {total_allowed} "
                    f"device trials allowed so far by "
                    f"block_budget_share={share} (a running ratio of the "
                    f"{pr['evals_used']} evaluations this run has spent, not a "
                    f"slice of a total)."
                    + (f" The run budget is not the limit: "
                       f"{pr['evals_used']}/{budget} used, "
                       f"{budget - pr['evals_used']} still unspent."
                       if budget else "")
                    + " Structural actions may still be opened, and every one "
                      "of them raises this allowance; raise "
                      "block_budget_share only if tiles are where this "
                      "kernel's time is going.")
    if a.max_trials:
        return (f"--max-trials {a.max_trials} left no trials for this block")
    if cap:
        return (f"block_max_trials={cap} left no trials for this block")
    return "no evaluation budget remaining"
