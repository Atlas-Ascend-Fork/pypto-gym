#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Which action the search tries next, and the belief behind the choice.

The learned prior and its store, the Thompson draw, the symptom boost, and
the tile lever that proposes numbers the model cannot.
"""

import hashlib
import json
import math
import os
import shutil
import time

import action_posterior
import prior_store

from .config import DEFAULT_CATALOG, DEFAULT_CONFIG, SYMPTOM_ACTIONS, SYMPTOM_THRESH, bayesian, w_stage
from .state import _domain_state, chip_envelope, op_file_path, open_node, read_text, sdir
from .measurement import _occupancies, incumbent_latency, objectives_of
from .front import demand_hint_text
from .log import log_open_action


def pypto_action_priority(node, active_stage, staged_prior, bayesian_optimization_free_trials=0):
    """V_sel = w_stage * g * V -- the selection score, and the only one.

    g is `prior_gain`: the catalogue star updated by every measurement the
    machine has of the action, read optimistically (see
    `action_posterior.learned_prior_gain`). V is the model's belief about THIS
    program. The core owns w_stage and g; the model supplies V.

    `novelty = 1/(1+tried)` used to be a fourth factor. It never acted: a node
    is consumed by its first close, so every live node has tried 0 and novelty
    1, and a factor bounded by one can only push scores down, never lift an
    untried action. Its intended job --
    an action that failed should score lower, an unknown one should get a look
    -- is done by g, which now carries this run's own measurements and its
    uncertainty. The last argument is kept for callers and ignored.
    """
    del bayesian_optimization_free_trials
    belief = float(node.get("V", 0.0))
    pg = float(node.get("prior_gain", 0.0))
    return round(w_stage(node["stage"], active_stage, staged_prior) * pg * belief, 6)


def _pipe_symptoms(m):
    """Which of the two per-pipe symptoms the readings on `m` support.

    Both need BOTH pipes present: a kernel with no `AIC_` rows is not a cube
    kernel starving, it is a vector kernel doing what it is (a pure elementwise
    kernel has no AIC rows at all and a pure matmul none on AIV -- neither is
    defective, and a bare aic/aiv ratio would fire on both).

    The two are mutually exclusive by construction, and that is the point: they
    prescribe opposite things.

        cube_starved     the cube pipe is idle AND the vector pipe is
                         saturated, so the work is on the wrong pipe and
                         overlapping the two buys almost nothing.
        pipe_serialized  neither pipe is saturated and most of the cube's wait
                         is on its predecessor, so the schedule is the cost and
                         overlap is what pays.
        neither          both pipes busy and balanced: no symptom, which is the
                         check that the pair does not fire on a healthy program.

    A ratio alone cannot tell the first two apart: two kernels can have clearly
    different ratios and BOTH sit below the 0.5 line, so the ratio returns the
    same verdict for a kernel on the wrong pipe and a kernel with a serial
    schedule. Whether the busy pipe is actually busy is what separates
    them, and that test is `pipe_saturated` / `pipe_idle` -- the two thresholds
    taken from master, not the two that are ours.
    """
    aic, aiv = (m or {}).get("aic_util"), (m or {}).get("aiv_util")
    if aic is None or aiv is None:
        return []
    try:
        aic, aiv = float(aic), float(aiv)
    except (TypeError, ValueError):
        return []
    if aiv <= 0.0:
        return []
    starved = aic < SYMPTOM_THRESH["cube_starved"] * aiv
    if starved and aiv >= SYMPTOM_THRESH["pipe_saturated"]:
        return ["cube_starved"]
    pred = (m or {}).get("pred_stall")
    if (max(aic, aiv) < SYMPTOM_THRESH["pipe_idle"] and pred is not None
            and float(pred) > SYMPTOM_THRESH["pred_stall"]):
        return ["pipe_serialized"]
    return []


def _symptom_reboost(st, util, bubble, op_file=None, pipes=None):
    """Deterministic symptom->action bias: re-point OPEN-node beliefs (V) at the current bottleneck
    of the best kernel. Only NUDGES matched actions up (bounded to 1.0); never lowers, so it
    complements -- not overwrites -- evolve. Returns the active symptoms (for logging).

    THREE symptoms, and the third is not a measurement. `occupancy` is read off
    the source by the deterministic core: the fraction of the unified buffer the
    vector tiles are estimated to hold. It is here because the other two cannot
    see it. A tile four times smaller than the buffer affords pays for itself on
    every loop iteration, and a profiler trace of that kernel looks unremarkable
    -- utilisation can be perfectly healthy while the buffer sits nearly empty.
    Nothing in `(s, p, m)` says so, which is why a search that reads only latency
    spends evaluations sampling tile shapes it could have computed.

    This is the symptom the recorded runs needed and did not have. Generated
    kernels routinely carry a tile of a few kilobytes against an affordable
    footprint tens of times larger, and the distance from such a tile to the
    best one is not small. INIT logged `seeded_symptoms=[]` on kernels like
    that, because neither profiler metric crosses its threshold on a kernel
    whose buffer is nearly empty.

    Because it comes from the source and not the device, this symptom is
    available BEFORE the first measurement and survives a run whose profiler
    output is missing.
    """
    boost = float(st["config"].get("symptom_boost", 1.0))
    if boost <= 1.0:
        return []
    try:
        util = float(util or 0.0)
        bubble = float(bubble or 0.0)
    except (TypeError, ValueError):
        util = bubble = 0.0

    active = []
    occ, l1 = _occupancies(op_file)
    if occ is not None and occ < SYMPTOM_THRESH["occupancy"]:
        active.append("occupancy")
    if l1 is not None and l1 < SYMPTOM_THRESH["l1_occupancy"]:
        active.append("l1_occupancy")
    if util > 0.0 or bubble > 0.0:             # a real measurement to read
        if bubble > SYMPTOM_THRESH["bubble"]:
            active.append("bubble")
        if 0.0 < util < SYMPTOM_THRESH["util"]:
            active.append("util")
    active += _pipe_symptoms(pipes)
    if not active:
        return []
    ids = set().union(*(SYMPTOM_ACTIONS[s] for s in active))
    for n in st["nodes"]["open"]:
        if n.get("catalog_id") in ids:
            n["V"] = min(1.0, round(float(n.get("V", 0.0)) * boost, 6))
    return active


def _retune_verdict(st, op_file):
    """Has the delta just applied moved the tile space out from under the optimum?

    Change 4, and the whole of it. When a structural action lands -- a flatten, a
    change of view extents, a fusion that changes residency -- the previously
    located tile describes a program that no longer exists, and submitting the
    structure at that stale tile is how a winning pair gets reverted. An
    optimizer wrote this down twice and had nowhere to send the work.

    The obvious test is `bayesian.lever.structural_signature`, and it is the wrong one:
    it hashes the program with tile literals erased, so a renamed variable
    changes it and a block would fire after every non-tile action. What actually
    invalidates the optimum is a change to the inputs of the derivation, so that
    is what is compared. A rename leaves them alone; u50's 1-D flatten does not.

    Deterministic, AST-only, no device. The case it misses -- a rewrite that
    leaves extents and residency alone yet moves the optimum -- is the model's to
    judge, through an explicit request that costs the action a `tried`.
    """
    baseline = st.get("domain_fingerprint")
    now = _domain_state(st, op_file)
    if now is None or baseline is None:
        return {"retune": False}
    moved, why = bayesian.domain.changed(baseline, now)
    if not moved:
        return {"retune": False}
    return {"retune": True, "retune_reason": why,
            "hint": "the tile this program inherited was chosen for a different "
                    "space. Run `block --parent <action_id>` BEFORE record, so the "
                    "candidate the ratchet judges is the (structure, tile) pair. "
                    "Supply --live-tiles if the rewrite changed how many tiles are "
                    "resident."}


def _tile_bayes_tell(st, op_dir, parent_id, s, p):
    """Report the measurement to the study that proposed it. Silent when the
    action is not a BO lever: every other action still goes through the model.
    """
    if bayesian is None or not st["config"].get("bayesian_optimization_tile_search", False):
        return
    node = next((n for n in st["nodes"]["open"] if n["id"] == parent_id), None)
    cid = (node or {}).get("catalog_id")
    if cid not in bayesian.lever.TILE_ACTIONS:
        return
    op_file = op_file_path(st, op_dir)
    if not op_file or not os.path.exists(op_file):
        return
    src = read_text(op_file)
    bayesian.lever.tell(st.setdefault("bayesian_optimization_studies", {}), cid, src, s, p)


def _expansion_demand(st, demand):
    """What the front is asking the model for, and how to answer it."""
    # A DIRECTION, when the front can name one. "Nothing here makes it
    # faster any more" is the only request a single objective can make,
    # and a model answers it with more of the same; "lower the
    # workspace, slower is acceptable" is answerable with a rewrite the
    # catalogue does not contain. See `front_demand`.
    return {"demand": demand,
            "hint": (demand_hint_text(demand, objectives_of(st)) if demand else
                     "reflect over the trajectory and call `evolve` with at least one "
                     "insert. If there is genuinely nothing left to try, call `evolve` "
                     "with exhausted:\"<reason>\"; only then may the search stop.")}


def _choose_action(st, frontier, active, staged, free):
    """The frontier node this selection hands out, with its Thompson record."""
    _refresh_g(st, frontier)
    thompson = None
    if st["config"].get("selection", DEFAULT_CONFIG["selection"]) == "thompson":
        best, thompson = _thompson_select(st, frontier, active, staged)
    else:
        best = max(frontier, key=lambda n: pypto_action_priority(n, active, staged, free))
    return best, thompson


def _select_source_payload(op_dir, rec):
    """What the selection was chosen from, and where that is written down."""
    return {"candidates": len(rec["candidates"]),
            "selects_log": os.path.relpath(_selects_path(op_dir), op_dir)}


def _log_selection(op_dir, st, best, rec, sel):
    """Write the one-line record of the action just handed out."""
    # One line per selection, carrying the catalogue id. Without it the only
    # record of WHICH action a u-id was is the frontier itself, and a consumed
    # node leaves the frontier -- so the u-id -> catalogue mapping had to be
    # reconstructed afterwards from mutually inconsistent seeding orders, and
    # some measurements could not be attributed at all.
    up = runner_up(rec)
    thompson = sel.thompson
    # Only the branch that prints it pays for it, as before the line was split.
    priority = (None if thompson else
                round(pypto_action_priority(
                    best, sel.active, sel.staged, sel.free), 4))
    log_open_action(op_dir, st, best,
                    f"selected at eval {st['progress']['evals_used']} · "
                    f"stage {best['stage']} · V {best['V']} · "
                    f"g {best.get('prior_gain')} · "
                    + (f"thompson p~{thompson['chosen']['p']} "
                       f"mean {thompson['chosen']['mean']} draw {thompson['draw_no']}"
                       if thompson else
                       f"V_sel {priority}")
                    + (f" · next {up['catalog_id'] or up['id']} {up['V_sel']}"
                       if up else "")
                    + f" · of {len(rec['candidates'])} candidates")


def _selects_path(op_dir):
    return os.path.join(sdir(op_dir), "selects.jsonl")


def record_select(op_dir, st, frontier, chosen, sel):
    """
    One JSON line per selection: every candidate with the factors it was
    scored on, and which one won.

    Written because the question "did X change what was selected?" was
    unanswerable. The state keeps only the FINAL V of the nodes that are still
    open, a consumed node leaves the frontier at its first close, and the log
    carried the winner alone -- so no argmax could be recomputed after the
    fact, for the symptom boost, for `g`, or for anything else. This is the
    same fix one level down, for the DECISION rather than the measurement.

    One line per select, the whole candidate list, in the order it was scored.
    A select costs a device evaluation soon after, so the bytes are free by
    comparison, and a jsonl file is appended rather than rewritten -- a
    truncated last line costs one record and not the file.
    """
    rows = []
    for n in frontier:
        rows.append({"id": n["id"], "catalog_id": n.get("catalog_id"),
                     "stage": n["stage"], "source": n.get("V_source"),
                     "V": n.get("V"), "g": n.get("prior_gain"),
                     "g_catalog": n.get("prior_gain_catalog"),
                     "g_model": n.get("prior_gain_model"),
                     "w_stage": w_stage(n["stage"], sel.active, sel.staged),
                     "V_sel": pypto_action_priority(n, sel.active, sel.staged, sel.free),
                     "tried": n.get("tried", 0)})
    rec = {"seq": st.get("_seq"), "eval_no": st["progress"]["evals_used"],
           "active_stage": sel.active, "staged_prior": bool(sel.staged),
           "base_code_hash": st["progress"].get("base_code_hash"),
           "selection": st["config"].get("selection", DEFAULT_CONFIG["selection"]),
           "chosen": chosen["id"], "chosen_catalog_id": chosen.get("catalog_id"),
           "candidates": rows}
    if sel.thompson:
        # Under thompson the winner is a DRAW, so the scores above do not
        # explain the choice on their own; the draws that do are kept with them.
        rec["thompson"] = sel.thompson
    try:
        with open(_selects_path(op_dir), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    except OSError:
        # A record we could not write is a gap in the audit trail, not a reason
        # to refuse an action the search has already chosen.
        st["progress"]["select_log_failed"] = (
            st["progress"].get("select_log_failed", 0) + 1)
    return rec


def runner_up(rec):
    """The second-highest `V_sel` in a select record, for the log line.

    The margin is what makes the line worth reading: a choice made by 0.001
    and a choice made by 0.4 are different claims about the frontier.
    """
    ranked = sorted((r for r in rec["candidates"] if r["id"] != rec["chosen"]),
                    key=lambda r: r["V_sel"], reverse=True)
    return ranked[0] if ranked else None


def _learned_g_on(st):
    return st["config"].get("prior_gain_source", DEFAULT_CONFIG["prior_gain_source"]) == "learned"


def _learned_g(st, node, table):
    return action_posterior.learned_prior_gain(
        node, table, st.get("predicates"), st.get("action_stats") or {},
        st["config"].get("prior_gain_quantile", DEFAULT_CONFIG["prior_gain_quantile"]))


def _refresh_g(st, frontier):
    """Recompute g for every node about to be scored: this run's measurements
    move it (an action that failed here reads lower once it is revived)."""
    if not _learned_g_on(st):
        return
    table = action_posterior.load_table(_prior_table_path(st))
    for n in frontier:
        if "prior_gain_catalog" not in n and n.get("catalog_id"):
            n["prior_gain_catalog"] = n.get("prior_gain")
        n["prior_gain"] = _learned_g(st, n, table)


def _prior_table_path(st):
    """The table THIS run reads: the snapshot `init` took, so it never changes
    under a running campaign. Before the snapshot exists, the source init would
    pick."""
    snap = (st.get("prior_table") or {}).get("snapshot")
    if snap and os.path.exists(snap):
        return snap
    return _prior_table_source(st)[0]


def _soc(st):
    """The chip this campaign was initialised for, as the envelope recorded it."""
    return (st.get("chip_envelope") or {}).get("soc") or None


def _prior_table_source(st):
    """(path, origin): explicit config > this machine's learned table > none.

    The learned table is PER CHIP (`prior_store.learned_table_path`), so a run
    on one chip never reads what another measured.
    """
    explicit = st["config"].get("action_prior_table")
    if explicit:
        return explicit, "explicit"
    learned = prior_store.learned_table_path(soc=_soc(st))
    if st["config"].get("prior_auto_update", DEFAULT_CONFIG["prior_auto_update"]) \
            and os.path.exists(learned):
        return learned, "learned"
    # NO table ships. The prior is built from this machine's own campaigns, so
    # the first run on a fresh machine reads the catalogue stars and nothing
    # else -- which is what `prior_gain_source: catalog` in the report means.
    return None, "none"


def _snapshot_prior_table(st, op_dir):
    """Copy the table into the campaign, record which one and its hash."""
    src, origin = _prior_table_source(st)
    snap = os.path.join(sdir(op_dir), "action_prior_table.snapshot.json")
    info = {"source": src, "origin": origin, "sha256": prior_store.file_sha(src),
            "snapshot": None}
    if action_posterior.load_table(src) is not None:
        try:
            os.makedirs(sdir(op_dir), exist_ok=True)
            shutil.copyfile(src, snap)
            info["snapshot"] = snap
        except OSError:
            pass
    st["prior_table"] = info
    return info


def _campaign_id(st):
    return f"{st['op']}-{int(st['progress']['_start_ts'])}"


def _record_outcome(st, a, outcome, arc_verdict=None, accepted=None):
    """One measured candidate into the campaign's and the machine's outcome store.

    `phase` and `front_insert` are on every row because `kept` answers one
    question -- did this action make the program faster -- and phase 2 asks a
    different one. `prior_store.build` drops the phase-2 rows from the latency
    table rather than counting them as failures; they are written anyway so
    that the reward phase 2 actually optimises can be fitted from real data
    instead of guessed at.
    """
    node = open_node(st, a.parent) or {}
    last = st["progress"].get("last_selection") or {}
    preds = {k: v for k, v in (st.get("predicates") or {}).items() if isinstance(v, bool)}
    row = {"ts": int(time.time()), "campaign": _campaign_id(st),
           "campaign_start": int(st["progress"]["_start_ts"]), "op": st["op"],
           "action": a.parent, "catalog_id": node.get("catalog_id"),
           "delta_hash": prior_store.delta_hash(node.get("delta")) if node else None,
           "predicates": preds, "s": int(bool(a.s)), "kept": bool(outcome.improved),
           "log_gain": (round(math.log(outcome.j / outcome.bar), 6)
                        if (outcome.improved and a.s and outcome.j
                            and outcome.bar and outcome.bar > 0) else 0.0),
           "selection": st["config"].get("selection", DEFAULT_CONFIG["selection"]),
           "propensity": (last.get("propensity") if last.get("action") == a.parent else None),
           "prior_table_sha256": (st.get("prior_table") or {}).get("sha256"),
           # What an action is worth is a property of the hardware, so the row
           # says which chip measured it. Rows written before this field
           # existed carry no `soc` and are dropped from every chip's table
           # unless `config.prior_legacy_soc` claims them.
           "soc": _soc(st),
           # One search since C2, so every row carries the same label. Old
           # rows say "greedy" or "front", and `prior_store.build` still drops
           # the "front" ones: they were measured under a ratchet that judged
           # them against the global best they were never meant to beat.
           "phase": "search",
           # The archive took this point. None when the candidate was
           # incorrect and never offered.
           "front_insert": (None if arc_verdict is None
                            else arc_verdict == "inserted"),
           # What the ratchet did, which since front acceptance is no longer
           # `kept`: a trade-off the action continues from is accepted and
           # not faster. None on rows written by a caller that does not say.
           "accepted": (None if accepted is None else bool(accepted)),
           "workspace_b": ((st.get("eval_cache", {}).get(a.code_hash, {}).get("m") or {})
                           .get("workspace_b")),
           "code_version": st.get("code_version")}
    shared = os.path.join(prior_store.store_dir(soc=_soc(st)),
                          f"{_campaign_id(st)}.jsonl")
    errs = prior_store.append_outcome(
        row, os.path.join(sdir(a.op_dir), "action_outcomes.jsonl"), shared)
    if errs:
        st["progress"]["outcome_store_errors"] = st["progress"].get("outcome_store_errors", 0) + 1


def _rebuild_prior(st):
    """After a real stop: fold every campaign's outcomes into the learned table."""
    cfg = st["config"]
    if not cfg.get("prior_auto_update", DEFAULT_CONFIG["prior_auto_update"]) \
            or cfg.get("action_prior_table"):
        return {"rebuilt": False, "why": "pinned or disabled"}
    try:
        with open(DEFAULT_CATALOG, encoding="utf-8") as f:
            catalog = json.load(f)
        # No shipped base: the rebuild folds in only what this machine measured.
        base = None
        path, meta = prior_store.rebuild(
            None, catalog,
            prior_store.BuildOptions(
                base=base,
                half_life=cfg.get("prior_half_life", DEFAULT_CONFIG["prior_half_life"]),
                soc=_soc(st),
                legacy_soc=cfg.get("prior_legacy_soc", DEFAULT_CONFIG["prior_legacy_soc"])))
        return {"rebuilt": True, "path": path, "rows_used": meta["rows_used"],
                "soc": meta.get("soc"),
                "rows_dropped_other_chip": meta.get("rows_dropped_other_chip"),
                "rows_unlabelled_chip": meta.get("rows_unlabelled_chip"),
                "campaigns": meta["campaigns"],
                "rows_dropped_catalog_text_changed": meta["rows_dropped_catalog_text_changed"]}
    except (OSError, ValueError) as e:
        return {"rebuilt": False, "why": str(e)[:200]}


def _thompson_select(st, frontier, active, staged):
    """(chosen node, what the draw was) -- see action_posterior."""
    cfg = st["config"]
    table = action_posterior.load_table(_prior_table_path(st))
    pr = st["progress"]
    draw_no = pr.get("select_draws", 0)
    pr["select_draws"] = draw_no + 1
    ctx = action_posterior.DrawContext(
        stats=st.get("action_stats") or {}, table=table, predicates=st.get("predicates"),
        seed=cfg.get("selection_seed", DEFAULT_CONFIG["selection_seed"]), draw_no=draw_no,
        v_strength=cfg.get("thompson_v_strength", DEFAULT_CONFIG["thompson_v_strength"]),
        use_gain=cfg.get("thompson_use_gain", DEFAULT_CONFIG["thompson_use_gain"]))
    chosen, draws = action_posterior.select(
        frontier, lambda n: w_stage(n["stage"], active, staged), ctx)
    mine = next(d for d in draws if d["id"] == chosen["id"])
    prop = action_posterior.propensity(
        frontier, lambda n: w_stage(n["stage"], active, staged), ctx, chosen["id"])
    pr["last_selection"] = {"action": chosen["id"], "mode": "thompson", "propensity": prop}
    mine["propensity"] = prop
    return chosen, {"draw_no": draw_no, "table": bool(table), "chosen": mine,
                    "top": draws[:5], "frontier": len(draws)}


def _observe_action(st, parent, outcome):
    """One measured candidate into the run's per-action counts (thompson's
    evidence). Kept for every run, whatever `selection` says, so a product run
    produces the same data a thompson run learns from.

    The arm's reward is "this action made the program it was applied to
    FASTER" -- `faster`, not front acceptance, which is why `cmd_record` passes
    the two separately. Every measurement is evidence about that question now
    that there is one search; the phase-2 skip this had is gone with the phase.
    """
    node = open_node(st, parent)
    if node is None:
        return
    g = (math.log(outcome.j / outcome.bar)
         if (outcome.improved and outcome.s_ok and outcome.j
             and outcome.bar and outcome.bar > 0) else 0.0)
    action_posterior.observe(st.setdefault("action_stats", {}), node, outcome.improved, g)


def _lever_propose_or_report(st, op_dir, node):
    """Propose the lever's next value, or turn a failure into a record."""
    try:
        return tile_bayes_propose(st, op_dir, node)
    except Exception as e:                       # noqa: BLE001
        pr = st["progress"]
        pr["lever_errors"] = pr.get("lever_errors", 0) + 1
        pr["lever_last_error"] = f"{type(e).__name__}: {e}"[:300]
        return {"proposed": False,
                "error": f"{type(e).__name__}: {e}"[:300],
                "note": "the tile lever could not propose a next "
                        "value; the measurement stands. Propose the "
                        "next candidate yourself, or close the action."}


def bayes_next(st, op_dir, action_id):
    """The lever's next value for an action that is continuing.

    Returns None for every action that is not a tile lever -- `tile_bayes_propose`
    already makes that decision -- so the ordinary refinement loop is untouched.

    A PROPOSAL IS NOT A MEASUREMENT. Anything that goes wrong choosing the next
    tile is reported and stepped over: the measurement that asked for it is
    already made, charged and ratcheted, and losing it to a failed suggestion is
    the worst of both. A campaign can die exactly there -- an off-ladder seed
    makes the study replay raise inside `record`, before `save`, so a run's
    evaluations go unpersisted and the working file is left holding a candidate
    nobody has recorded.
    """
    for group in ("open", "held"):
        for n in st["nodes"].get(group) or []:
            if isinstance(n, dict) and n.get("id") == action_id:
                return _lever_propose_or_report(st, op_dir, n)
    return None


def tile_bayes_propose(st, op_dir, node):
    """For a tile action, choose the next value and write it. Returns the record
    to attach to `select`, or None when this action is not a BO lever.

    The core writes the file itself rather than handing an intent to the coder.
    That is not a shortcut, it removes a failure mode: a tile can be bound at
    two sites, the wrapper's padding arithmetic and the kernel, and a coder that
    changes one and not the other ships a kernel that fails verification. The
    action is then scored as unproductive and closed, and the path it had opened
    is lost to a single bookkeeping mistake. `bayesian.apply` rewrites every
    site or none.

    `bayesian.apply` is a script, not an agent, so the rule that only the coder writes
    kernel code is untouched -- the harness already writes files in `snapshot`
    and `restore`.
    """
    if bayesian is None or not st["config"].get("bayesian_optimization_tile_search", False):
        return None
    if node.get("catalog_id") not in bayesian.lever.TILE_ACTIONS:
        return None
    op_file = op_file_path(st, op_dir)
    if not op_file or not os.path.exists(op_file):
        # Loud, not silent. `op_file` is stored at init as a path relative to the
        # server's working directory; if the harness is invoked from anywhere
        # else it resolves to nothing, the lever quietly does not fire, and the
        # run looks completely normal while the core owns none of the numbers.
        return {"applied": False,
                "reason": f"op_file not found: {op_file!r} (cwd={os.getcwd()!r}). "
                          f"bayesian_optimization_tile_search is ON but the lever cannot read the kernel."}
    src = read_text(op_file)
    studies = st.setdefault("bayesian_optimization_studies", {})
    # The same 2x `cmd_block` threads below, in the one path that never got it.
    # `bayesian.space.HW` defaults `dtype_bytes` to 4 and so does `bayesian.lever.ask`, while
    # `hw_for` reads the real operand width off the kernel -- so built bare, the
    # lever sized a bf16 kernel's L0A/L0B/L1 AND its own tile-footprint ceiling
    # at fp32. That is not conservative, it is wrong: it rejects kernels that
    # RAN. A delivered cube site can sit exactly on the capacity the gate is
    # checking -- 64 KB of L0B at 2 B, 128 KB at 4 B -- and still have measured.
    #
    # The cost is not one axis. A draw holds every site but one at the anchor,
    # and the gate is an AND over all of them, so an anchor the gate calls
    # illegal fails every draw that does not move the offending site -- the vec
    # tiles and the other cube sites become unreachable, and the only proposals
    # left are ones that shrink the offending site below its delivered value,
    # which it can then never climb back to. Meanwhile the block path, correct
    # at 2 B, kept working: the same search_state.json says "at 2B" under
    # `blocks` and "at 4B" under `bayesian_optimization_studies`.
    hw, domain = bayesian.block.hw_for(op_file, st.get("live_tiles"), chip_envelope(st))
    cfg, meta = bayesian.lever.ask(
        studies, node["catalog_id"], src, hw,
        bayesian.lever.AskOptions(
            dtype_bytes=domain.get("dtype_bytes", 4),
            vec_dtype_bytes=domain.get("vec_dtype_bytes"),
            domain=domain,
            # The incumbent is the program ON DISK, which `select` has just
            # restored -- the front base, not necessarily the global best. Same
            # clock as the trials it seeds, and the same rule as `_block_search`.
            current_latency=incumbent_latency(st, src)))
    if cfg is None:
        return {"applied": False, "reason": meta.get("reason", "no candidate")}
    new_src, n = bayesian.apply.apply(src, cfg)
    if not n or new_src == src:
        return {"applied": False, "reason": "no site rewritten"}
    with open(op_file, "w", encoding="utf-8") as f:
        f.write(new_src)
    return {"applied": True, "config": cfg, "label": meta.get("label"),
            "trial": meta.get("trial"), "sites": n,
            "code_hash": hashlib.sha256(new_src.encode("utf-8")).hexdigest(),
            "note": "already written to the kernel; do NOT dispatch the coder, "
                    "just evaluate and record"}
