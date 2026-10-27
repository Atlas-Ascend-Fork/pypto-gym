#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""What the run hands over: the program, its verification, and the report.

The ABBA pass, the front written out as source, and the human-readable
summary. Opened by `stop`; nothing here decides what to search next.
"""

import datetime
import hashlib
import json
import os
import shutil
import time

from .config import DEFAULT_CONFIG, PARETO_SCHEMA, Refusal, VERIFY_ORDER, out
from .state import _read_bytes, file_hash, load, npath, op_file_path, ratchet_best_hash, restore, save, sdir
from .measurement import bootstrap_speedup, ci_verdict, median_of, objective_text, objectives_of, parse_samples
from .front import delivery_point, pareto_front
from .log import append_log
from .selection import _rebuild_prior


def _record_frontier(pr, edits, live, pruned_any, demand=None):
    """Emptying the frontier is termination by another name, and termination is
    not the model's decision. Extending the tree IS its job: `insert` is the world
    model. So an evolve that leaves nothing to select must either offer new
    intents or state, explicitly, that it has none. It has been observed: a run
    whose last evolves inserted nothing and pruned steadily, the last of them
    reasoned as "pruning exhausted actions to focus remaining budget on novel
    directions" while offering no novel direction. The search halted with most
    of its budget unspent.
    """
    if live:
        pr["expansion_required"] = False
        pr["expansion_requests"] = 0
        pr.pop("expansion_declined", None)
        pr.pop("expansion_demand", None)
        return
    if pruned_any and not edits.get("insert"):
        pr["frontier_emptied_by_prune"] = pr.get("frontier_emptied_by_prune", 0) + 1
    if edits.get("exhausted"):
        pr["expansion_declined"] = str(edits["exhausted"])
        return
    # Counted, never acted on. An automatic decline after N requests would itself
    # be a heuristic termination, which is the thing this branch exists to
    # remove: a run merely slow to expand would be cut short before B.
    # frontier_exhausted therefore fires only on an explicit declaration. Note
    # the backstop is wall_clock, not eval_budget: an empty frontier evaluates
    # nothing, so evals_used never advances. `expansion_requests` in the state
    # names the cause after the fact.
    pr["expansion_requests"] = pr.get("expansion_requests", 0) + 1
    pr["expansion_required"] = True
    # Recorded at the moment the request is made, so a later reader can tell a
    # run that asked for a direction from one that asked for anything at all.
    pr["expansion_demand"] = demand


def _delivered_check(st, op_dir):
    """Does the file on disk actually hold the program the run reports as best?

    A run can answer `frontier_exhausted` reporting one best and ship another
    kernel: the two differing by one line of `pass_options`, the snapshot of
    the winner intact under `nodes/`, and nothing anywhere saying the delivered
    artefact was not the reported one. The number was right
    and the deliverable was wrong, which is the worst combination -- it survives
    every check that reads the state file and fails only when somebody diffs the
    kernel by hand.

    So the last thing the harness says about a run includes the hash of what it
    actually left on disk, next to the hash it claims to have found.

    `best_code_hash` is what SHOULD be on disk -- `delivery_point`, falling
    back to `best_node`. `best_node_code_hash` is reported beside it whenever
    the two differ, because that difference is the defect C1 exists for and it
    should be visible in the run's own output rather than only in a hand diff.
    """
    want, detail = delivery_point(st, op_dir)
    node = ratchet_best_hash(st)
    f = op_file_path(st, op_dir)
    on_disk = None
    if f and os.path.exists(f):
        on_disk = hashlib.sha256(_read_bytes(f)).hexdigest()
    out_d = {"best_code_hash": want or node, "delivered_code_hash": on_disk,
             "delivered_is_best": bool(want or node) and on_disk == (want or node),
             "delivery": detail}
    if want and node and want != node:
        out_d["best_node_code_hash"] = node
    return out_d


def _deliver_best(st, a):
    """Put the global best in the working file, or refuse the stop.

    `stop_reason` is sticky and the Stage-7 gate reads it, so writing it while the working
    file holds some other program would let Stage 7 complete on a kernel the
    search never chose.

    The program is `delivery_point`'s, not `best_node`'s. When they differ the
    greedy ratchet lost track of its own measurements, which is recorded rather
    than quietly corrected: the number in the report has to describe the kernel
    the report ships.
    """
    best_hash, detail = delivery_point(st, a.op_dir)
    node = ratchet_best_hash(st)
    if best_hash and node and best_hash != node:
        st["progress"]["delivery_outranked_best_node"] = {
            "delivered": best_hash, "best_node": node,
            "delivered_values": detail.get("values"),
            "at_eval": st["progress"]["evals_used"]}
    best_hash = best_hash or node
    op_f = op_file_path(st, a.op_dir)
    if restore(a.op_dir, op_f, best_hash) or file_hash(op_f) == best_hash:
        # Frozen, because everything after the stop has to name THIS program:
        # the verification arm, the file put back between passes, the speedup
        # the pass is compared against. `ratchet_best_hash` is `best_node` and
        # only moves at an action's `close`, so a `--remeasure` that changes
        # the delivery winner leaves it behind -- and verification would then
        # measure one kernel and restore another over the one being shipped.
        st["progress"]["delivered_code_hash"] = best_hash
        return
    out({"stop": False, "refused": True, "error": "restore_failed",
         "best_code_hash": best_hash, "op_file": op_f,
         "reason": ("the global best has no snapshot on disk"
                    if best_hash else "no global best code_hash is recorded"),
         "hint": "the run cannot be closed out on a file that is not the "
                 "winner. Restore optimization/nodes/<best_code_hash>.py "
                 "into the kernel, then call stop again."})
    raise Refusal(3)


def _record_probe(st, a, probe):
    """Persisted whichever way it went. A run that CONTINUED past a block's
    env-fault abort is as much a decision as one that stopped, and neither is
    reconstructable afterwards without this.
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    st["progress"]["device_probe"] = dict(probe, at=now)
    append_log(a.op_dir, st["op"],
               f"- device probe after env-fault abort: usable={probe.get('usable')}"
               + (f" marker={probe['marker']}" if probe.get("marker") else "")
               + (f" ({probe['note']})" if probe.get("note") else "")
               + (" -> run continues" if probe.get("usable") else " -> device_unavailable"))


def cmd_report(a):
    """Write the FINAL best block — once, only at a real stop.

    'Final best' is a completion marker, so it is gated on a real halt (progress.stop_reason set
    by cmd_stop), the same key the Stage-7 completion gate uses. Without this guard the
    optimizer could call `report` at a dispatch/checkpoint boundary and append a misleading
    '## Final best' mid-run (the search is still improving). No stop yet -> no-op.
    """
    st = load(a.op_dir)
    pr = st["progress"]
    if not pr.get("stop_reason"):
        out({"reported": False, "note": "no stop yet — Final best is written only at a real halt"})
        return
    # `report` is the last thing a run says, so it is where the deliverable is
    # checked. A run can report one best and ship a different kernel while the
    # snapshot of the winner sits on disk the whole time and nothing notices.
    d = _delivered_check(st, a.op_dir)
    if not d["delivered_is_best"]:
        restore(a.op_dir, op_file_path(st, a.op_dir), d["best_code_hash"])
        d = _delivered_check(st, a.op_dir)
        d["repaired"] = True
    # The objective vector of the kernel this report SHIPS, which since C1 need
    # not be `best_node`'s. Reporting `progress.best_*` alone is the "number
    # right, deliverable wrong" pairing `_delivered_check` exists to catch.
    shipped = (d.get("delivery") or {}).get("values") or {}
    # Written again here, not only at `stop`: the frontier has to describe the
    # kernel this report delivers, including a repair.
    front = export_frontier(st, a.op_dir)
    append_log(a.op_dir, st["op"],
               f"\n## Final best\n- aicore_e2e: {pr['best_aicore_e2e_us']} us | j: {pr['best_J']} "
               f"| speedup_vs_preopt: {pr['best_speedup_vs_preopt']}x "
               f"| evals: {pr['evals_used']} | worked: {pr['wall_clock_s']}s "
               f"of {int(time.time() - pr['_start_ts'])}s elapsed "
               f"| stop: {pr['stop_reason']}"
               f"\n- rejected without spending budget: {pr.get('static_rejected', 0)} static, "
               f"{pr.get('duplicate_rejected', 0)} duplicate"
               f"\n- code_version: {st.get('code_version')}"
               f"\n- delivered: {str(d['delivered_code_hash'])[:12]} "
               f"(best {str(d['best_code_hash'])[:12]}) "
               + (f"[{objective_text(objectives_of(st), shipped)}] " if shipped else "")
               + (f"— the greedy best_node {str(d['best_node_code_hash'])[:12]} "
                  f"is beaten by it on the measurements "
                  if d.get("best_node_code_hash") else "")
               + ("OK" if d["delivered_is_best"] else "MISMATCH — the kernel on disk is NOT the best found")
               + (" | repaired from the snapshot" if d.get("repaired") and d["delivered_is_best"] else "")
               + f"\n- verification: {_verification_line(st)}"
               + _accounting_lines(st, front)
               + _frontier_lines(front)
               + _recommend_lines(front, objectives_of(st)))
    prior = _rebuild_prior(st)
    if prior.get("rebuilt"):
        append_log(a.op_dir, st["op"],
                   f"- prior table rebuilt: {prior['rows_used']} outcomes over "
                   f"{prior['campaigns']} campaigns -> {prior['path']}")
    picks = recommend(objectives_of(st), front.get("front") or [])
    out({"reported": True, "best_J": pr["best_J"], "stop_reason": pr["stop_reason"],
         **d,
         "verification": _verification_summary(st),
         "frontier": front,
         "recommended": _recommended_payload(picks),
         "prior_table": prior,
         "static_rejected": pr.get("static_rejected", 0),
         "duplicate_rejected": pr.get("duplicate_rejected", 0)})


def _delivered_hash(st):
    """The program this run ships, for every step that follows `stop`.

    `_deliver_best` freezes it in `progress.delivered_code_hash` as it puts it
    in the working file. Not `ratchet_best_hash`: that is `best_node`, a scalar
    on `J` that only moves at an action's `close`, and `--remeasure` moves the
    delivery winner without running the ratchet. The fallback is for a state
    written before the stop recorded it.
    """
    return st["progress"].get("delivered_code_hash") or ratchet_best_hash(st)


def _delivered_speedup(st, code_hash):
    """The search's own claim ABOUT THE DELIVERED PROGRAM, on `J`'s instrument.

    `progress.best_speedup_vs_preopt` tracks `best_node` for the same reason,
    so quoting it here would compare the ABBA interval against a figure that
    describes a different kernel. Read off the delivered program's own
    measurement instead, the way a front point's claim already is
    (`_next_verification`).
    """
    pr = st["progress"]
    entry = (st.get("eval_cache") or {}).get(code_hash) or {}
    pre, p = pr.get("preopt_aicore_e2e_us"), entry.get("p")
    if pre and p:
        return round(pre / p, 4)
    return pr.get("best_speedup_vs_preopt")


def _open_verification(st):
    """Create the verification record at the first real stop. Idempotent:
    `stop` is sticky and may be called again."""
    if "verification" in st:
        return
    pre_hash = st["nodes"]["closed"][0]["code_hash"]
    best_hash = _delivered_hash(st)
    cfg = st["config"]
    if not cfg.get("end_verify", DEFAULT_CONFIG["end_verify"]):
        st["verification"] = {"status": "disabled"}
        return
    if best_hash == pre_hash:
        # Nothing was kept, so there is no claim to verify.
        st["verification"] = {"status": "no_change", "preopt": pre_hash,
                              "delivered": best_hash}
        return
    runs = int(cfg.get("verify_runs", DEFAULT_CONFIG["verify_runs"]))
    per_step = max(1, -(-runs // 2))          # each arm is measured twice
    st["verification"] = {
        "status": "pending",
        "arms": {"preopt": {"code_hash": pre_hash, "samples": []},
                 "delivered": {"code_hash": best_hash, "samples": []}},
        "schedule": list(VERIFY_ORDER), "step": 0,
        "runs_per_step": per_step,
        "protocol": f"warmup=1,runs={per_step},agg=median x{len(VERIFY_ORDER)} ABBA",
        "search_speedup": _delivered_speedup(st, best_hash),
        "point": "delivered",
    }
    st["verify_queue"] = _front_to_verify(st, pre_hash, best_hash)


def _front_to_verify(st, pre_hash, best_hash):
    """The front points that still need an ABBA pass, in front order.

    The delivered point is the one already being verified, and the preopt is the
    other arm of every pass, so neither is queued. A point whose source was not
    retained cannot be verified -- `verify` stages the program from
    `nodes/<hash>.py` -- but it is dropped when the queue is CONSUMED
    (`_next_verification`), not here: a snapshot can arrive between the stop and
    the pass, and a queue filtered at the stop would have already forgotten it.
    """
    if not st["config"].get("verify_front", DEFAULT_CONFIG["verify_front"]):
        return []
    return [e["code_hash"] for e in (st.get("archive") or [])
            if e["code_hash"] not in (pre_hash, best_hash)]


def _next_verification(st, op_dir):
    """Retire the finished pass and open the next front point's, if any.

    Returns the code hash now being verified, or None when the front is done.
    """
    v = st.get("verification") or {}
    queue = [h for h in (st.get("verify_queue") or [])
             if os.path.exists(npath(op_dir, h))]
    st["verify_queue"] = queue[1:]
    if not queue:
        # The last pass stays in `verification`, where every summary reads it.
        return None
    st.setdefault("verifications", []).append(v)
    nxt = queue[0]
    point = next((e for e in (st.get("archive") or []) if e["code_hash"] == nxt), {})
    st["verification"] = {
        "status": "pending",
        "arms": {"preopt": {"code_hash": v["arms"]["preopt"]["code_hash"], "samples": []},
                 # Keyed `delivered` because that is what the ABBA schedule and
                 # `_finish_verification` name the second arm; `point` says which
                 # program it actually is.
                 "delivered": {"code_hash": nxt, "samples": []}},
        "schedule": list(VERIFY_ORDER), "step": 0,
        "runs_per_step": v["runs_per_step"], "protocol": v["protocol"],
        "point": nxt,
        "objectives": point.get("values"),
        # The search's own claim about this point, on `J`'s instrument -- the
        # same one `verify` is about to re-measure it with.
        "search_speedup": (round(st["progress"]["preopt_aicore_e2e_us"]
                                 / point["aicore_e2e_us"], 4)
                           if (point.get("aicore_e2e_us")
                               and st["progress"].get("preopt_aicore_e2e_us"))
                           else None),
    }
    return nxt


def _verify_directive(st):
    v = st.get("verification")
    if not v:
        return None
    if v["status"] != "pending":
        return {"required": False, "status": v["status"]}
    return {"required": True, "status": "pending",
            "steps_left": len(v["schedule"]) - v["step"],
            "runs_per_step": v["runs_per_step"],
            "next": "HARNESS verify --op-dir <op_dir>  (stages the next arm)"}


def _verification_summary(st):
    v = st.get("verification")
    if not v:
        return {"status": "not_started"}
    keep = ("status", "result", "search_speedup", "protocol", "skip_reason",
            "preopt", "delivered", "point")
    return {k: v[k] for k in keep if k in v}


def _verification_line(st):
    v = st.get("verification") or {}
    status = v.get("status", "not_started")
    r = v.get("result")
    if r:
        return (f"{status.upper()} {r['speedup']}x "
                f"[{r['ci_low']}, {r['ci_high']}] at {int(r['level'] * 100)}% "
                f"(search reported {v.get('search_speedup')}x; "
                f"{r['n_preopt']}+{r['n_delivered']} readings)")
    if status == "pending":
        return "NOT VERIFIED -- the reported speedup is a search-time point estimate"
    if status == "skipped":
        return f"SKIPPED ({v.get('skip_reason')}) -- the reported speedup is unverified"
    return status


def _finish_verification(st, v):
    pre = v["arms"]["preopt"]["samples"]
    dele = v["arms"]["delivered"]["samples"]
    ci = bootstrap_speedup(pre, dele)
    if ci is None:
        v["status"] = "insufficient"
        return
    ci.update(n_preopt=len(pre), n_delivered=len(dele),
              preopt_us=median_of(pre), delivered_us=median_of(dele))
    v["result"] = ci
    v["status"] = ci_verdict(ci)


def _verify_short_circuit(a, st, op_f):
    """True when this `verify` call is answered without new readings."""
    v = st["verification"]
    if getattr(a, "skip", ""):
        if v["status"] == "pending":
            v["status"] = "skipped"
            v["skip_reason"] = a.skip
        restore(a.op_dir, op_f, _delivered_hash(st))
        save(a.op_dir, st)
        out({"verified": False, **_verification_summary(st)})
        return True
    if v["status"] != "pending":
        out({"verified": v["status"] in ("confirmed", "unconfirmed", "regressed"),
             "done": True, **_verification_summary(st)})
        return True
    return False


def _verify_stage_arm(a, v, op_f, arm, want):
    """Put one arm's program in the working file and say how to measure it."""
    if not (restore(a.op_dir, op_f, want) or file_hash(op_f) == want):
        out({"verified": False, "error": "restore_failed", "arm": arm,
             "code_hash": want})
        raise Refusal(2)
    out({"stage": v["step"], "arm": arm, "code_hash": want, "op_file": op_f,
         "runs": v["runs_per_step"],
         "measure": f"measure_latency --op-file {op_f} --warmup 1 "
         f"--runs {v['runs_per_step']} --agg median",
         "then": "HARNESS verify --op-dir <op_dir> --samples <samples>"})


def _verify_readings(a, op_f, arm, want):
    """This step's readings, once they are known to be the staged program's."""
    # The readings must belong to the program this step staged. Anything else
    # would silently verify a kernel nobody is shipping.
    if file_hash(op_f) != want:
        out({"verified": False, "error": "wrong_program", "arm": arm,
             "expected": want, "on_disk": file_hash(op_f),
             "hint": "call `verify` without --samples first; it stages the arm"})
        raise Refusal(2)
    samples = parse_samples(a.samples)
    if not samples:
        out({"verified": False, "error": "no_samples", "arm": arm})
        raise Refusal(2)
    return samples


def cmd_verify(a):
    """Re-measure the delivered kernel against the preopt, after `stop`.

    Without --samples: stage the next arm into the working file and say which.
    With --samples:    record that arm's readings, then put the delivered kernel
                       back, so the deliverable is intact between calls.
    """
    st = load(a.op_dir)
    v = st.get("verification")
    if not st["progress"].get("stop_reason") or not v:
        out({"verified": False, "error": "no_stop",
             "note": "verification runs after `stop`, on the programs it hands over"})
        raise Refusal(2)
    op_f = op_file_path(st, a.op_dir)
    if _verify_short_circuit(a, st, op_f):
        return
    arm = v["schedule"][v["step"]]
    want = v["arms"][arm]["code_hash"]
    if a.samples is None:
        _verify_stage_arm(a, v, op_f, arm, want)
        return
    samples = _verify_readings(a, op_f, arm, want)
    v["arms"][arm]["samples"].extend(samples)
    v["step"] += 1
    nxt = None
    if v["step"] >= len(v["schedule"]):
        _finish_verification(st, v)
        line = _verification_line(st)
        append_log(a.op_dir, st["op"],
                   f"- verification[{str(v.get('point'))[:8]}]: {line}")
        # The front is the deliverable, so every point on it gets its own pass.
        nxt = _next_verification(st, a.op_dir)
    restore(a.op_dir, op_f, _delivered_hash(st))
    save(a.op_dir, st)
    summary = _verification_summary(st)
    if nxt:
        summary = {"status": "pending", "point": nxt,
                   "front_points_left": len(st.get("verify_queue") or []) + 1,
                   "note": "the delivered point is verified; this pass is the "
                           "next front point, which is also part of the "
                           "deliverable"}
    out({"recorded": arm, "n": len(samples), "step": st["verification"]["step"],
         "steps_left": (len(st["verification"]["schedule"])
                        - st["verification"]["step"]), **summary})


def _recommended_payload(picks):
    """The front's named picks, in the shape the report reports them."""
    # The extremes are facts about the front; `balanced` is the knee, a
    # stated convention. Neither moves `impl.py`.
    if not picks:
        return None
    return {"balanced": (picks["balanced"]["code_hash"]
                         if picks.get("balanced") else None),
            "best": {k: v["code_hash"] for k, v in picks["best"].items()},
            "convention": picks["convention"]}


def _frontier_dir(op_dir):
    return os.path.join(sdir(op_dir), "frontier")


def _verdicts_by_point(st):
    """code_hash -> what its ABBA pass concluded, for every pass this run ran."""
    out_v = {}
    for v in list(st.get("verifications") or []) + [st.get("verification") or {}]:
        h = (v.get("arms") or {}).get("delivered", {}).get("code_hash")
        if not h:
            continue
        r = v.get("result") or {}
        out_v[h] = {"status": v.get("status"),
                    "speedup": r.get("speedup"), "ci_low": r.get("ci_low"),
                    "ci_high": r.get("ci_high")}
    return out_v


def export_frontier(st, op_dir):
    """
    Write the archive to `optimization/frontier/`: one source per point, and
    `frontier.json` with the measurements.

    The preference is NOT the system's: `impl.py` stays the fastest point, and
    the other points sit here with their numbers for whoever has to weigh
    workspace against latency -- another kernel on the same device, usually.
    Nobody is asked anything at stop.
    """
    points = list(st.get("archive") or [])
    rows, missing = [], 0
    os.makedirs(_frontier_dir(op_dir), exist_ok=True)
    # What `impl.py` holds, which is `delivery_point` -- `is_impl` has to mark
    # the row the file actually carries, not the row `best_node` names.
    best = delivery_point(st, op_dir)[0] or ratchet_best_hash(st)
    objs = objectives_of(st)
    verdicts = _verdicts_by_point(st)
    for i, e in enumerate(points):
        src = npath(op_dir, e["code_hash"])
        # Named by the workspace when it is an objective, so a directory listing
        # already reads as the trade-off; otherwise by rank alone.
        ws_tag = (f"_{int(e['workspace_b'] / 2 ** 20)}MiB"
                  if e.get("workspace_b") is not None else "")
        name = f"{i:02d}{ws_tag}_{e['code_hash'][:8]}.py"
        dst = os.path.join(_frontier_dir(op_dir), name)
        have = os.path.exists(src)
        if have:
            try:
                shutil.copyfile(src, dst)
            except OSError:
                have = False
        if not have:
            missing += 1
        rows.append({"rank": i, "code_hash": e["code_hash"],
                     "mix_aic_us": e.get("mix_aic_us"),
                     "aicore_e2e_us": e.get("aicore_e2e_us"),
                     "workspace_b": e["workspace_b"],
                     "workspace_mib": (round(e["workspace_b"] / 2 ** 20, 2)
                                       if e.get("workspace_b") is not None else None),
                     # The full objective vector, whatever the run declared.
                     "values": e.get("values") or {},
                     "objectives_text": objective_text(objs, e.get("values") or {}),
                     "J": e["J"], "action": e.get("action"),
                     "catalog_id": e.get("catalog_id"), "eval_no": e.get("eval_no"),
                     # `null` when the candidate's source was not retained --
                     # a run with `snapshot_all_candidates` off, or a point a
                     # tile block measured without producing a file.
                     "source": (os.path.relpath(dst, op_dir) if have else None),
                     "is_impl": e["code_hash"] == best,
                     # What an ABBA pass concluded about THIS point. A front is
                     # handed over as a set of choices, so a point nobody
                     # re-measured is a search-time point estimate.
                     "verified": verdicts.get(e["code_hash"])})
    out_obj = {"op": st["op"], "points": len(rows), "sources_missing": missing,
               "impl_code_hash": best, "front": rows}
    path = os.path.join(_frontier_dir(op_dir), "frontier.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=1)
    except OSError:
        return dict(out_obj, written=None)
    return dict(out_obj, written=path)


def _verified_text(v):
    """`confirmed 1.20x [1.15, 1.25]`, or what is missing instead."""
    if not v or not v.get("status"):
        return "not verified"
    if v.get("speedup") is None:
        return str(v["status"])
    return f"{v['status']} {v['speedup']}x [{v['ci_low']}, {v['ci_high']}]"


def recommend(objs, rows):
    """Which points to put in front of a person: one per objective, plus a knee.

    The preference is NOT the system's -- who weighs this kernel against the
    others on the device owns it. What the core can do without borrowing that
    preference is name the EXTREMES (the best point on each objective, which is
    a fact about the front) and one balanced point under a stated convention.

    The convention for `balanced` is the knee: min-max normalise every objective
    across the front, then take the point closest to the ideal corner. It is a
    convention and the report says so; it is not a claim that this trade-off is
    the right one.

    Returns {} for a front of one point -- there is nothing to choose between.
    """
    live = [r for r in rows if all(r.get("values", {}).get(o["name"]) is not None for o in objs)]
    if len(live) < 2:
        return {}
    picks, span = {}, {}
    for o in objs:
        vals = [r["values"][o["name"]] for r in live]
        lo, hi = min(vals), max(vals)
        span[o["name"]] = (lo, hi)
        best = (max if o["direction"] == "max" else min)(
            live, key=lambda r, name=o["name"]: r["values"][name])
        picks[o["name"]] = best

    def _d(r):
        total = 0.0
        for o in objs:
            lo, hi = span[o["name"]]
            x = (r["values"][o["name"]] - lo) / (hi - lo) if hi > lo else 0.0
            total += (1.0 - x if o["direction"] == "max" else x) ** 2
        # Ties break toward the PRIMARY objective, so the pick is deterministic
        # and says which way it leaned.
        prim = objs[0]
        p_val = r["values"][prim["name"]]
        return (round(total, 9), -p_val if prim["direction"] == "max" else p_val)

    out_rec = {"best": picks,
               "convention": "knee: min-max normalised per objective, nearest "
                             "the ideal corner. A convention, not a preference."}
    # With two points there is no knee: each one IS an extreme, and both sit the
    # same normalised distance from the ideal corner. Naming one `balanced`
    # would be inventing a preference rather than reporting a shape.
    if len(live) > 2:
        out_rec["balanced"] = min(live, key=_d)
    return out_rec


def _recommend_lines(front, objs):
    """The picks, as the last thing a person reads."""
    rec = recommend(objs, front.get("front") or [])
    if not rec:
        return ""

    def line(label, r):
        return (f"  {label:<22} {r['code_hash'][:8]}  "
                f"{r.get('objectives_text') or ''}  "
                f"{_verified_text(r.get('verified'))}"
                + (f"  {r['source']}" if r.get("source") else
                   "  (source not retained)"))
    rows = [line("balanced", rec["balanced"])] if rec.get("balanced") else []
    for o in objs:
        rows.append(line(f"best {o['name']}", rec["best"][o["name"]]))
    return ("\n## Recommended\n" + "\n".join(rows)
            + (f"\n- `balanced` is a {rec['convention']}" if rec.get("balanced")
               else "\n- no `balanced` pick: with two points on the front each "
                    "one is an extreme, so choosing between them is the "
                    "preference itself")
            + "\n- impl.py is unchanged by this section: it holds the fastest "
              "point, which is what the ratchet kept.")


def _hv_accounting_line(st, pr):
    """The hypervolume's line of the accounting block."""
    return (f"- hypervolume: {pr.get('hv')} | {pr.get('hv_stagnation', 0)} "
            f"measurement(s) since it last grew by more than "
            f"{st['config'].get('hv_min_gain')} (noise included)"
            + (f", last gain {pr['hv_last_gain'].get('gain')} at eval "
               f"{pr['hv_last_gain'].get('at_eval')}" if pr.get("hv_last_gain") else "")
            + (f" | limit {st['config'].get('hv_stagnation_limit')}"
               if st["config"].get("hv_stagnation_limit") else " | halt off"))


def _accounting_lines(st, front):
    """Where the budget went, what the phases did, what the front holds.

    Everything here is already in the state; it is written out because a run is
    reviewed from this file, and a run's real story -- a budget eaten by one
    action's tile blocks, a measured improvement never delivered -- is
    otherwise visible only to someone who opens the JSON.
    """
    pr = st["progress"]
    used = pr.get("evals_used", 0)
    blocks = pr.get("block_evals", 0)
    repaired = sum(int(b.get("n_repaired") or 0) for b in (st.get("blocks") or []))
    dev = sum(int(b.get("device_trials") or 0) for b in (st.get("blocks") or []))
    lines = [
        "\n## Accounting",
        f"- evaluations: {used} charged | {blocks} in tile blocks "
        f"({round(100 * blocks / used) if used else 0}% of the run, "
        f"share {st['config'].get('block_budget_share')}) | "
        f"{len(st.get('blocks') or [])} blocks, {dev} device trials, "
        f"{repaired} of them repaired draws",
        f"- free rejections: {pr.get('static_rejected', 0)} static, "
        f"{pr.get('duplicate_rejected', 0)} duplicate, "
        f"{pr.get('semantic_noop_rejected', 0)} semantic no-op, "
        f"{pr.get('wrapper_drift_rejected', 0)} wrapper drift, "
        f"{pr.get('unevaluated_rejected', 0)} unevaluated",
        _hv_accounting_line(st, pr),
        f"- front: {pr.get('front_stagnation', 0)} measurement(s) since the "
        f"archive last accepted a point"
        + (f" | base {str(pr['base_selection'].get('code_hash'))[:8]}, "
           f"hv contribution {round(pr['base_selection'].get('hv_contribution') or 0, 4)} "
           f"of {pr['base_selection'].get('candidates')} candidate(s)"
           if pr.get("base_selection") else ""),
        f"- archive: {len(st.get('archive') or [])} points | "
        f"{pr.get('archive_inserted', 0)} inserted, "
        f"{pr.get('archive_dropped', 0)} dropped, "
        f"{pr.get('archive_kept_slower', 0)} kept though slower than the "
        f"fastest known, {pr.get('archive_unplaceable', 0)} unplaceable",
        f"- noise: sigma_log {(st.get('noise') or {}).get('sigma_log')} over "
        f"{(st.get('noise') or {}).get('measurements', 0)} measurements "
        f"({(st.get('noise') or {}).get('protocol')})",
    ]
    if pr.get("promoted_at_stop"):
        p = pr["promoted_at_stop"]
        lines.append(f"- promoted at stop: {p['action']} -> {p['node']} "
                     f"(j={p['J']} p={p['p']}); the action was still open when "
                     f"the run ended")
    if pr.get("remeasurements"):
        lines.append(f"- re-measurements (not charged): {pr['remeasurements']}")
    del front
    return "\n".join(lines)


def _frontier_lines(front):
    """The list the report ends with: every point, then what impl.py is."""
    if len(front.get("front") or []) < 2:
        return ""
    rows = "\n".join(
        f"  {r['rank']:02d}  {r.get('objectives_text') or ''}  "
        f"{r['code_hash'][:8]}  {r.get('catalog_id') or r.get('action') or 'preopt'}"
        + f"  {_verified_text(r.get('verified'))}"
        + ("  <- impl.py" if r["is_impl"] else "")
        + ("  (source not retained)" if not r["source"] else "")
        for r in front["front"])
    return (f"\n## Frontier ({front['points']} non-dominated points)\n"
            f"  rank  objectives  code  action  verified\n{rows}\n"
            f"- sources and measurements: optimization/frontier/\n"
            f"- impl.py is the FASTEST point ({str(front['impl_code_hash'])[:8]}). "
            f"Tell me if you want a different one.")


def _candidate_points(st):
    """Every correct measured candidate with a workspace reading, labelled with
    what the single-objective search did with it."""
    kept = {c.get("code_hash") for c in st["nodes"]["closed"]}
    refine = st["progress"].get("refine") or {}
    for rec in (refine.get("best"), refine.get("base")):
        if (rec or {}).get("code_hash"):
            kept.add(rec["code_hash"])
    best = ratchet_best_hash(st)
    pts, skipped = [], {"no_workspace": 0, "incorrect": 0}
    for h, e in (st.get("eval_cache") or {}).items():
        m = e.get("m") or {}
        if not e.get("s") or not e.get("p"):
            skipped["incorrect"] += 1
            continue
        if m.get("workspace_state") not in ("measured", "zero") or m.get("workspace_b") is None:
            skipped["no_workspace"] += 1
            continue
        # `verdict` is written by `record`; older states and block trials lack
        # it, and for those only "on the kept chain or not" is known.
        role = ("delivered" if h == best else "kept" if h in kept
                else e.get("verdict") or "not_kept")
        pts.append((float(e["p"]), int(m["workspace_b"]),
                    {"code_hash": h, "role": role, "via": e.get("via", "record"),
                     # The OTHER instrument, read from its own field. None when the
                     # candidate carries no MIX_AIC reading, which is a fact about
                     # the measurement and not a value to substitute for.
                     "mix_aic_us": m.get("mix_aic")}))
    return pts, skipped


def cmd_pareto(a):
    """
    Front readout: which measured candidates are non-dominated in
    (latency, workspace), and which of those the single-objective ratchet threw
    away. Read-only on the search; writes optimization/pareto.json.

    Latency is compared as measured (median-of-9 point values), workspace
    exactly. A reverted point on the front is the existence proof that a
    single-objective run measured a trade-off and discarded it.
    """
    st = load(a.op_dir)
    pts, skipped = _candidate_points(st)
    front = pareto_front(pts)
    # `p` is `--p`, which IS aicore_e2e. This file used to write it under BOTH
    # names: `mix_aic_us` was a stale label that the rename missed, and it was
    # then kept as a duplicate for older readers. That is a worse answer than
    # the one it replaced -- the two instruments differ by a wide factor on real
    # kernels, so a kernel measured at 100 us end-to-end and 10 us in MIX_AIC
    # was reported as 100 under both, and the ratios with it. MIX_AIC is read
    # from its own field now (`_candidate_points`) and is null when the
    # candidate has no such reading.
    rows = [{"aicore_e2e_us": p, "workspace_b": w,
             "workspace_mib": round(w / 2 ** 20, 2), **x} for p, w, x in front]
    fastest = min(pts, key=lambda t: t[0]) if pts else None
    fastest_mix = fastest[2].get("mix_aic_us") if fastest else None
    for r in rows:
        if fastest:
            r["aicore_e2e_vs_fastest"] = round(r["aicore_e2e_us"] / fastest[0], 4)
            # Against the SAME reference program, on MIX_AIC's own scale. Null
            # unless both ends carry the reading: a ratio with one end missing
            # is not a smaller number, it is a different quantity.
            r["mix_aic_vs_fastest"] = (round(r["mix_aic_us"] / fastest_mix, 4)
                                       if (r["mix_aic_us"] and fastest_mix) else None)
            r["workspace_vs_fastest"] = (round(r["workspace_b"] / fastest[1], 4)
                                         if fastest[1] else None)
    out_obj = {"op": st["op"], "schema": PARETO_SCHEMA, "points": len(pts),
               "skipped": skipped, "front": rows,
               "front_reverted": sum(1 for r in rows if r["role"] == "reverted"),
               "front_without_mix_aic": sum(1 for r in rows if r["mix_aic_us"] is None),
               "note": ("latency is a point estimate (median-of-9); points whose latency "
                        "differs by less than the run's noise are not resolved here. "
                        "schema 2: mix_aic_us is the MIX_AIC reading, null when the "
                        "candidate has none; in schema 1 it was a copy of aicore_e2e_us, "
                        "so a reader that took it as latency wants aicore_e2e_us")}
    path = os.path.join(sdir(a.op_dir), "pareto.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=1)
        out_obj["written"] = path
    except OSError:
        out_obj["written"] = None
    out(out_obj)
