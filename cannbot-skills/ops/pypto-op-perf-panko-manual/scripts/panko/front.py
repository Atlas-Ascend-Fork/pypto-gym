#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The archive, dominance, and the hypervolume.

Which measured points the run keeps, how two of them are ordered when the
readings cannot separate them, how much volume the set encloses, and which
point the next action is applied on top of.
"""

import math
import os
import random

import mo_margin
import mo_hv

from .config import (DEFAULT_CONFIG, FrontContext, J_OBJECTIVE, OBJECTIVE_ALIASES, Refusal, VERIFY_ALPHA, VERIFY_BOOT,
                     VERIFY_SEED, out)
from .state import _base_hash, live_frontier, npath, open_node, preopt_code_hash
from .measurement import (bootstrap_speedup, compute_j, is_bytes, median_of, objective_values, objectives_of,
                          preopt_values)


def hv_unaxis(o, r):
    """
    `hv_axis` undone: one axis coordinate back in the objective's own units.

    Only for REPORTING. A reader of `hv_reference` or `base_selection` is owed
    the number they passed, and a byte axis is measured as `log(bytes)`, so
    printing the coordinate raw says the chip has 24.6 bytes of memory.
    """
    if o["direction"] == "max":
        return -r
    if is_bytes(o):
        return round(math.exp(r))
    return r


def _hv_reference_report(st, objs):
    """The fixed reference, per objective, in the objective's own units."""
    if len(objs) < 2:
        return None
    ref = hv_reference_point(st, objs)
    if ref is None:
        return None
    return {o["name"]: hv_unaxis(o, r) for o, r in zip(objs, ref)}


def refuse_no_hv_reference(st, objs):
    """More than one objective and no box to measure volume in. Emit and stop.

    `hv_reference_point` returns None when a byte-valued objective has no ceiling --
    the SKU's ini carried no `memory_size` and `--chip-memory-mib` was not
    passed. Nothing then invents a bound, which is right. But what follows is
    SILENT, and one half of it is the mechanism a multi-objective run is for:

      * `front_base` falls back to all-zero contributions, so the next
        action's base is "least tried, then fastest" -- which is what a
        single-objective run does. The front is still built and still
        reported; it just stops deciding where the search goes next.
      * `hv_observe` keeps answering, on the per-axis records alone. The
        convergence test is an OR of the hypervolume gain with those records,
        and it loses the first half -- the half that sees a point trading
        latency for bytes.

    So the run looks normal, reports a front, and has been choosing its base
    the single-objective way the whole time.

    `init` is where it costs nothing to find out, and it is the same class of
    defect as a `P_ref` past the preopt: the numbers are right and the
    conclusion is wrong. So the run is refused with the three ways to give it
    a ceiling, rather than started without one.

    One objective needs no reference: there is no volume, the ratchet ranks on
    the primary objective, and nothing degrades.
    """
    if len(objs) < 2 or hv_reference_point(st, objs) is not None:
        return
    env = st.get("chip_envelope") or {}
    byte_axes = [o["name"] for o in objs if is_bytes(o) and o.get("reference") is None]
    out({"initialized": False, "refused": True,
         "error": "hv_reference_unavailable",
         "objectives": [o["name"] for o in objs],
         "byte_objectives": byte_axes,
         "soc": env.get("soc"), "chip_memory_mib": env.get("memory_mib"),
         "reason": (
             "the hypervolume reference is fixed for the run and a byte-valued "
             f"objective ({', '.join(byte_axes) or 'none named'}) has no "
             "ceiling to use: the chip envelope carries no `memory_mib`. "
             "Without a reference the next action's base is chosen as if there "
             "were one objective (least tried, then fastest) and the "
             "convergence test runs on the per-axis records alone -- both "
             "silently. The run would report a front it never searched by."),
         "ask_user": {
             "question": ("This run declares more than one objective but has no "
                          "ceiling for the byte axis, so the hypervolume cannot "
                          "be measured. Where should the bound come from?"),
             "options": [
                 {"id": "chip_memory",
                  "label": "The chip's memory",
                  "action": "re-run `init` with --chip-memory-mib <MiB>, or with "
                  "a --chip-envelope that carries memory_mib (read "
                  "from the SKU's ini as SoCInfo/memory_size)"},
                 {"id": "axis_reference",
                  "label": "A bound this kernel must stay under",
                  "action": "give the objective its own `reference` in "
                  "config.objectives: [{\"name\": \"workspace\", "
                  "\"from\": \"m.workspace_b\", \"direction\": "
                  "\"min\", \"reference\": <bytes>}]"},
                 {"id": "single_objective",
                  "label": "Optimise one objective",
                  "action": "set config.objectives to the primary objective "
                  "alone; the other readings are still recorded on `m`"}]}})
    raise Refusal(4)


def front_acceptance(st, code_hash, p, m):
    """(accepted, detail): does this measurement extend the front?

    The acceptance rule of the multi-objective search, in place of `j > bar`.
    Two conditions:

      * no archive point dominates it on the medians (`mo_margin`, the sign
        of the multiplicative epsilon indicator). Medians, not the archive's
        bootstrap interval: the interval keeps every point that is not
        SIGNIFICANTLY worse, and accepting on it would let the base drift by
        noise. Significance is settled once, at `verify`, as it already is
        for the single-objective ratchet;
      * the primary objective is no worse than the preopt's. That is the
        reference point, on the primary axis only: a trade-off that is slower
        than the program the run started from is not a deliverable, whatever
        it saves elsewhere. The other axes carry no bound, so no constant is
        needed.

    The candidate's own archive row is excluded, because a block winner is in
    the archive before `record` promotes it, and every point ties with itself.

    With one objective this is `p < every archive point's p`, which is the
    ratchet's `j > bar` against the fastest program measured.

    (None, detail) when the vector cannot be formed; the caller falls back to
    the scalar test rather than accepting on nothing.
    """
    objs = objectives_of(st)
    vals, missing = objective_values(objs, p, m)
    if missing:
        return None, {"why": "objective_missing", "missing": missing}
    others = [e for e in (st.get("archive") or [])
              if e.get("code_hash") != code_hash and e.get("values")]
    on_front, margin = mo_margin.extends_front(
        objs, [e["values"] for e in others], vals)
    prim = objs[0]
    ref = preopt_values(st, objs)
    bound = None if ref is None else ref.get(prim["name"])
    within = True
    if bound is not None:
        v = vals[prim["name"]]
        within = v <= bound if prim["direction"] == "min" else v >= bound
    finite = margin is not None and not math.isinf(margin)
    accepted = bool(on_front and within)
    detail = {
        "front_margin": round(margin, 6) if finite else None,
        "front_unbounded": bool(margin is not None and math.isinf(margin)
                                and margin > 0),
        "on_front": bool(on_front),
        "reference": {prim["name"]: bound},
        "within_reference": bool(within),
        "compared_with": len(others)}
    if not accepted:
        front = FrontContext(objs, others, prim, bound, within)
        detail["clarity"] = _failure_clarity(st, code_hash, p, m, front)
    return accepted, detail


def _failure_clarity(st, code_hash, p, m, front):
    """`clear` or `undetermined`: is this candidate really off the front?

    Acceptance is decided on the medians, deliberately -- the archive's interval
    keeps every point not SIGNIFICANTLY worse, and accepting on it would let the
    base drift by noise. But the CLOSE counter is a different question. Seven
    candidates each clearly slower and seven each inside the measurement's own
    spread are not the same evidence that an action is spent, and `stagnation_K`
    counted them alike.

    So the rejection is re-asked with the readings, through the same
    interval test the archive's own dominance uses (`dominates_measured`, same
    estimator and seed): is some front point significantly better on every
    axis, or is the candidate slower than the preopt by more than the readings
    can explain? If neither, the rejection rests on medians alone and this
    measurement is not evidence about the action.

    `clear` whenever the question cannot be put -- no readings, no preopt
    vector -- so a run that passes no `--samples` keeps counting exactly as it
    did before.
    """
    cand = archive_point(st, code_hash, p, m)
    if cand is None:
        return "clear"
    alpha = archive_alpha(st)
    for e in front.others:
        if dominates_measured(front.objs, e, cand, alpha):
            return "clear"
    if not front.within and front.bound is not None:
        xs = row_samples(front.prim, cand)
        pre = next((e for e in (st.get("archive") or [])
                    if e.get("code_hash") == preopt_code_hash(st)), None)
        ys = row_samples(front.prim, pre) if pre else []
        if not xs or not ys:
            return "clear"            # the medians are all there is to go on
        verdict, _ = measured_order(xs, ys, alpha)
        if front.prim["direction"] == "max" and verdict in ("better", "worse"):
            verdict = "worse" if verdict == "better" else "better"
        return "clear" if verdict == "worse" else "undetermined"
    return "undetermined"


def _refresh_front_point(point, entry, objs, st):
    """Rewrite one archive row from the readings now pooled on its entry."""
    point["samples"] = entry["samples"]
    point["aicore_e2e_us"] = entry["p"]
    point["mix_aic_us"] = ((entry.get("m") or {}).get("mix_aic")
                           or point.get("mix_aic_us"))
    re_vals, _ = objective_values(objs, entry["p"], entry.get("m"))
    point["values"].update(re_vals)
    point["samples_by_objective"] = objective_samples(objs, entry, entry.get("m"))
    for o in objs:
        if o["from"] == "p":
            point["values"][o["name"]] = entry["p"]
    point["J"] = compute_j(1, entry["p"], st["target"]["P_ref"])


def delivery_point(st, op_dir):
    """(code_hash, detail): the program `stop` leaves in the working file.

    The best PRIMARY objective among every correct measurement whose source is
    on disk, ties broken by the remaining objectives in order and then by the
    hash. `impl.py` is still "the fastest point" -- what changes is that it is
    now read off the measurements instead of off `best_node`.

    Not `best_node`: it is `J`, and it only moves at an action's `close`. A run
    can deliver a program another point beats -- slower, at a workspace many
    times larger -- because the deliverable was chosen by a scalar that had
    stopped tracking the measurements.

    Not the archive either: the bootstrap lets a 0.3% slower point with a better
    secondary objective dominate the fastest one, so the fastest program can
    leave the front (see `dominates_measured`). It never leaves `eval_cache`,
    which is why the candidates come from there.

    Nothing here needs a non-domination filter: a point that is strictly best
    on the primary objective cannot be dominated, and when several tie on it
    the remaining objectives order them -- which is exactly the tied case.

    (None, detail) when no measurement can be delivered; the caller falls back
    to `best_node`.
    """
    objs = objectives_of(st)
    rank = []
    for h, e in (st.get("eval_cache") or {}).items():
        if not e.get("s") or not e.get("p"):
            continue
        if not os.path.exists(npath(op_dir, h)):
            continue
        vals, _ = objective_values(objs, e["p"], e.get("m"))
        if vals.get(objs[0]["name"]) is None:
            continue
        key = []
        for o in objs:
            v = vals.get(o["name"])
            # Missing secondary objectives sort last, so a tie on the primary
            # goes to the point whose trade-off is actually known.
            key.append(float("inf") if v is None
                       else (-float(v) if o["direction"] == "max" else float(v)))
        rank.append((tuple(key), h, vals))
    if not rank:
        return None, {"why": "no_deliverable_measurement"}
    rank.sort(key=lambda t: (t[0], t[1]))
    _, h, vals = rank[0]
    return h, {"code_hash": h, "values": vals, "considered": len(rank)}


# --------------------------------------- one search, with no phase split
# An earlier design ran the single-objective search first and spent whatever
# budget was left extending the front. There are no phases now. Acceptance is
# non-domination within the reference point (`front_acceptance`) and the next
# action starts from the largest hypervolume contribution (`front_base`), from
# the first evaluation -- which is what the phase split was for. A first phase
# that pursues `aicore_e2e` alone IS a single-objective run, so under a phase
# split "multi-objective" could only ever mean the leftovers, and the switch
# never fired.
#
# Nothing halts a run on stagnation under the shipped defaults: `success`, the
# budget and the wall clock end it, and `frontier_exhausted` still ends it when
# the optimizer says so.
# `progress.front_stagnation` keeps counting measurements since the archive
# last accepted a point, as the diagnostic an HV convergence test (C3) will
# read.


def base_candidates(st, op_dir):
    """The archive points an action may be applied to.

    Three filters, each for a reason:
      * the source is on disk -- the delta has to be applied to something;
      * the primary objective is no worse than the preopt's -- the reference
        point `front_acceptance` uses, so the search never continues from a
        point it would not have accepted;
      * non-dominated ON THE MEDIANS among those. The archive itself is built
        with the bootstrap interval and keeps every point that is not
        SIGNIFICANTLY dominated -- noise twins of the best included. Spreading
        actions over those would spend the budget on copies of one program.
        With one objective this leaves exactly the fastest point.
    """
    objs = objectives_of(st)
    prim = objs[0]
    ref = preopt_values(st, objs)
    bound = None if ref is None else ref.get(prim["name"])
    pts = []
    for e in (st.get("archive") or []):
        if not e.get("values") or not os.path.exists(npath(op_dir, e["code_hash"])):
            continue
        v = e["values"].get(prim["name"])
        if bound is not None and v is not None:
            if (v > bound) if prim["direction"] == "min" else (v < bound):
                continue
        pts.append(e)
    return [e for e in pts
            if not any(o is not e and dominates(objs, o["values"], e["values"]) for o in pts)]


def hv_axis(o, v):
    """One objective's value on the hypervolume's axis, in minimisation space.

    A byte-valued objective is measured as `log(bytes)`, which is what makes a
    FIXED reference usable on it. On a linear axis the share of the axis a 1%
    reduction occupies collapses as the reference grows -- at the chip's memory
    it is 0.006%, against 5.6% for a 1% latency gain, so one threshold cannot
    serve both axes. In log, "1% smaller" is the same length wherever it
    happens: 0.195% of the axis at the chip's memory, and still 0.13% if the
    reference were fifteen times larger. The ratio between the two axes'
    sensitivities falls from 2679x to 55x.

    `max(v, 1)` because a workspace probe may legitimately report zero bytes
    and log(0) is not a number.
    """
    v = float(v)
    if o["direction"] == "max":
        v = -v
    elif is_bytes(o):
        return math.log(max(v, 1.0))
    return v


def _hv_axis_bound(o, pre, cfg_ref, mib):
    """One axis of the fixed reference point, or None when nothing bounds it."""
    if o["name"] in cfg_ref:
        return hv_axis(o, cfg_ref[o["name"]])
    if is_bytes(o):
        return hv_axis(o, float(mib) * (1 << 20)) if mib else None
    if pre.get(o["name"]) is not None:
        return hv_axis(o, pre[o["name"]])
    return None


def hv_reference_point(st, objs):
    """The hypervolume's reference point, FIXED for the whole run, or None.

    Per axis:
      * the primary objective: the preopt's value. "A program slower than the
        one the run started from is worth nothing" is exactly what the
        reference means, and it is already the bound acceptance uses.
      * a byte-valued objective: the chip's memory. Nothing larger can run, so
        a program at or past it encloses no volume -- which is what a
        constraint violation should look like.
      * anything else: the preopt's value, and a run that wants otherwise
        passes `reference` on that objective.

    Why fixed. A reference recomputed from the current points moves: one with a
    worse workspace pushes it outward and inflates every volume already
    measured. A run improving its latency step by step then reads each step as
    a smaller fraction of gain than it is, falls under the convergence
    threshold, and the improvement disappears.

    None when a byte axis has no ceiling to use -- the SKU's ini carried no
    `memory_size` and nothing was passed. The caller then skips the
    hypervolume rather than inventing a bound.
    """
    pre = preopt_values(st, objs) or {}
    cfg_ref = {}
    for o in objs:
        if o.get("reference") is not None:
            cfg_ref[o["name"]] = float(o["reference"])
    mib = (st.get("chip_envelope") or {}).get("memory_mib")
    ref = []
    for o in objs:
        axis = _hv_axis_bound(o, pre, cfg_ref, mib)
        if axis is None:
            return None
        ref.append(axis)
    return tuple(ref)


def _hv_frame(st, objs, points):
    """(vectors, reference) for `points`, against the run's fixed reference."""
    vecs = [tuple(hv_axis(o, e["values"][o["name"]]) for o in objs)
            for e in points]
    return vecs, hv_reference_point(st, objs)


def front_base(st, op_dir, detail=None):
    """Which front point the next action extends, or None.

    The point with the largest EXCLUSIVE hypervolume contribution among
    `base_candidates` -- the one whose removal would shrink the front most,
    i.e. the one with the emptiest neighbourhood. The search keeps extending it
    until the front changes, which is greedy accumulation generalised: with one
    objective only the fastest point has a contribution, so the base is the
    fastest point.

    Ties -- every contribution zero, as when the only candidate sits on the
    reference -- go to the point tried from least, then to the primary
    objective, then to the hash, so the choice is deterministic.

    `detail`, when given, receives what the choice was made on.
    """
    objs = objectives_of(st)
    points = base_candidates(st, op_dir)
    if not points:
        return None
    vecs, ref = _hv_frame(st, objs, points)
    if ref is None:
        # No ceiling for a byte axis, so no volume to compare. Fall back to the
        # tie-break alone: fewest tried, then the primary objective.
        contrib = [0.0] * len(points)
    else:
        contrib = mo_hv.contributions(vecs, ref)
    # Compared at 9 significant digits of the largest: two contributions that
    # are equal by geometry come out of the subtraction a few ulps apart, and
    # the tie-break below would otherwise be decided by rounding.
    top = max(contrib) or 1.0
    k = min(range(len(points)),
            key=lambda i: (-round(contrib[i] / top, 9),
                           int(points[i].get("tried_from", 0)),
                           vecs[i][0], points[i]["code_hash"]))
    if detail is not None:
        detail.update({"code_hash": points[k]["code_hash"],
                       "hv_contribution": contrib[k],
                       "candidates": len(points),
                       "reference": {o["name"]: hv_unaxis(o, r)
                                     for o, r in zip(objs, ref)}})
    return points[k]


def _advance_front_base(st, a):
    """After any close, count the action against the point it was applied to
    and pick the next point to extend.

    Every phase, not only phase 2: since the ratchet keeps what extends the
    front, where the next action starts is a question about the front in phase
    1 too. The point counted is `_base_hash` -- the program `select` actually
    restored -- so the preopt and a global best reached before any base was
    chosen are counted like any other.
    """
    pr = st["progress"]
    current = _base_hash(st, a.op_dir)
    for e in (st.get("archive") or []):
        if e["code_hash"] == current:
            e["tried_from"] = int(e.get("tried_from", 0)) + 1
    chosen = {}
    base = front_base(st, a.op_dir, chosen)
    pr["base_code_hash"] = base["code_hash"] if base else None
    pr["base_selection"] = chosen or None


# ----------------------------------------------------- the directional demand
# The single-objective reason to look outside the catalogue is a negative one --
# "nothing here makes it faster any more" -- and with a monotone greedy walk and
# 53 catalogued actions, that rarely happens inside the budget: an inserted
# intent is hardly ever selected at all.
#
# A front makes a POSITIVE request possible. When the archive holds no point
# beyond some value of an objective, and every catalogued action aimed at that
# objective has already been tried on this structure, what the search is short
# of is nameable: not "something better", but "something that lowers the
# workspace, and a latency regression is acceptable". That is a question a model
# can answer with a rewrite the catalogue does not contain.
#
# It is not a preference. WHICH point to ship stays the reader's (see
# `recommend`); which region to explore is search control, and always was.
def _moves_objective(node, name):
    """Does this action's catalogue entry say it aims at objective `name`?"""
    return name in ((node or {}).get("moves") or [])


def front_demand(st, op_dir):
    """The objective the front is short of, or None.

    Two conditions, both necessary. The archive must hold points (there is no
    direction without a front), and the catalogue must be SPENT on that
    objective -- every action tagged as aiming at it either tried or gone from
    the live frontier. Asking for a rewrite while a catalogued action for the
    same objective is still selectable would spend the model's attention on
    what the core was about to try anyway.

    The primary objective is excluded: the greedy phase is already the request
    for it, and `best_J` is what answers.
    """
    objs = objectives_of(st)
    archive = st.get("archive") or []
    if len(objs) < 2 or not archive:
        return None
    live = live_frontier(st, op_dir)
    pre = (st["nodes"]["closed"] or [{}])[0].get("eval") or {}
    pre_vals, _ = objective_values(objs, pre.get("p"), pre.get("m") or {})
    for o in objs[1:]:
        name = o["name"]
        seeded = (st.get("objective_actions") or {}).get(name) or []
        if not seeded:
            continue          # the catalogue never claimed to move this one
        if any(_moves_objective(n, name) and not n.get("tried") for n in live):
            continue
        vals = sorted(e["values"][name] for e in archive if name in e["values"])
        if not vals:
            continue
        best = vals[0] if o["direction"] == "min" else vals[-1]
        return {"objective": name, "direction": o["direction"],
                "beyond": best,
                "preopt": pre_vals.get(name),
                # Reported, not used as the trigger: the model can see whether
                # the front is a line with a hole in it or a pair of clusters.
                "largest_gap": _largest_gap(vals),
                "front_points": len(vals),
                # What the catalogue had to say about this objective, all of
                # which has now been said. Derived from what was SEEDED (a
                # consumed node leaves the frontier at its first close, so the
                # tree cannot answer this on its own).
                "catalog_spent": sorted(seeded)}
    return None


def _largest_gap(vals):
    """(lo, hi) of the widest interval between adjacent front values, or None."""
    if len(vals) < 2:
        return None
    pairs = list(zip(vals, vals[1:]))
    lo, hi = max(pairs, key=lambda ab: ab[1] - ab[0])
    return [lo, hi] if hi > lo else None


def demand_hint_text(d, objs):
    """What to ask the model for, in the words the request is made in."""
    unit = " MiB" if d["objective"] == "workspace" else ""
    val = (round(d["beyond"] / 2 ** 20, 1) if d["objective"] == "workspace"
           else round(d["beyond"], 2))
    way = "LOWER" if d["direction"] == "min" else "RAISE"
    primary = objs[0]["name"]
    return (f"the front holds no point past {val}{unit} on `{d['objective']}`, and every "
            f"catalogued action aimed at it has been tried on this structure "
            f"({', '.join(d['catalog_spent']) or 'none was selected'}). Propose a rewrite "
            f"that {way}S `{d['objective']}`. A `{primary}` REGRESSION IS ACCEPTABLE: the "
            f"archive keeps a point that is slower when nothing dominates it, and `impl.py` "
            f"still holds the fastest program either way. This is the one request the "
            f"catalogue cannot answer, so an intent outside it is what is wanted.")


def dominates(objs, a, b):
    """Does point `a` dominate point `b`? Every objective no worse, one better."""
    better = False
    for o in objs:
        x, y = a[o["name"]], b[o["name"]]
        if o["direction"] == "max":
            x, y = -x, -y
        if x > y:
            return False
        if x < y:
            better = True
    return better


def measured_order(xs, ys, alpha=None):
    """`better` / `worse` / `undetermined` for median(xs) against median(ys).

    A percentile bootstrap of the ratio of medians, the same estimator `verify`
    uses, at the same fixed seed -- so the same two sample sets always give the
    same verdict and a front is reproducible from its readings. `undetermined`
    when the interval covers 1, and below two readings per side there is no
    interval to compute and the answer is undetermined rather than assumed.
    """
    alpha = VERIFY_ALPHA if alpha is None else alpha
    ci = bootstrap_speedup(xs, ys, alpha=alpha)
    if ci is None:
        return "undetermined", None
    if ci["ci_high"] < 1.0:
        return "better", ci
    if ci["ci_low"] > 1.0:
        return "worse", ci
    return "undetermined", ci


def row_samples(o, row):
    """This objective's individual readings on an archive ROW, or [].

    Two places, because rows written before objectives carried their own
    readings must keep working: `samples_by_objective` when the row has it,
    and the row's legacy `samples` list for an objective that declared
    `samples_from: "samples"` (that list has always been `p`'s). An objective
    that declares no sample source has none here, and is compared as a number.
    """
    by = (row.get("samples_by_objective") or {}).get(o["name"])
    if by:
        return list(by)
    if o.get("samples_from") == "samples":
        return list(row.get("samples") or [])
    return []


def objective_samples(objs, entry, m):
    """{objective name: readings} for one measurement.

    `entry` is the eval-cache entry (its `samples` are `p`'s); `m` carries a
    list under whatever key an objective's `samples_from` names. Only
    non-empty lists of numbers are kept: an objective with no readings falls
    back to its number rather than to an empty interval.
    """
    by_name, m = {}, (m or {})
    for o in objs:
        sf = o.get("samples_from")
        if not sf:
            continue
        raw = ((entry or {}).get("samples") if sf == "samples"
               else m.get(sf[2:]))
        if not isinstance(raw, (list, tuple)) or not raw:
            continue
        try:
            vals = [float(x) for x in raw]
        except (TypeError, ValueError):
            continue
        if vals:
            by_name[o["name"]] = vals
    return by_name


def rederivable(objs):
    """
    The readings a re-measurement re-derives: every declared objective, plus
    `J_OBJECTIVE` whether or not the run declared it.

    `J` is read from one instrument in every run (see `J_OBJECTIVE`), so a run
    that optimises a front of other objectives still has a `J` -- and a
    re-measurement that left it on its first median would move the front while
    the ratchet's own number stood still.
    """
    out_objs = list(objs)
    if not any(o["from"] == OBJECTIVE_ALIASES[J_OBJECTIVE]["from"] for o in out_objs):
        out_objs.append(dict(OBJECTIVE_ALIASES[J_OBJECTIVE], name=J_OBJECTIVE))
    return out_objs


def dominates_measured(objs, a, b, alpha=None):
    """`dominates`, with the measured objective judged by its readings.

    `a` and `b` are archive ROWS, so the samples travel with the values. An
    objective read from `p` is the measurement and gets the interval; anything
    read from `m` is structural here (the workspace is
    `AlignUp(..., 32KB) * parallelism`, exact) and is compared as a number. A
    measured objective with no samples on either side falls back to the number
    too: refusing to order two points because nobody passed `--samples` would
    fill the front with duplicates.

    An undetermined measured comparison is a TIE: it does not block dominance,
    and it does not count as an improvement either. Dominance asks two things
    -- no worse on every objective, better on at least one -- and
    `undetermined` answers the first, not the second. So a point that is not
    measurably slower and uses strictly less workspace dominates, which is what
    a reader means by it.

    Treating undetermined as blocking keeps on a front a point that is worse
    on BOTH axes than one it sits beside, held only because noisy readings stop
    the interval from ordering the pair. A non-dominated front
    holding a point that loses on every axis cannot be explained to its reader.

    The cost of the other direction is real and is answered elsewhere: a point
    0.3% slower on a 2% kernel now dominates the faster one on its workspace,
    and the FASTEST program can leave the front. It does not leave the RUN --
    `impl.py`, `best_aicore_e2e_us` and the ratchet are not the archive.
    """
    alpha = VERIFY_ALPHA if alpha is None else alpha
    better = False
    for o in objs:
        x, y = a["values"][o["name"]], b["values"][o["name"]]
        xs, ys = row_samples(o, a), row_samples(o, b)
        if xs and ys:
            verdict, _ = measured_order(xs, ys, alpha)
            if o["direction"] == "max" and verdict in ("better", "worse"):
                verdict = "worse" if verdict == "better" else "better"
            if verdict == "worse":
                return False
            if verdict == "better":
                better = True
            # The interval IS this objective's comparison. Falling through to
            # the medians below would compare the same readings twice, and a
            # stored `p` that had drifted from its own samples would silently
            # overturn the interval.
            continue
        if o["direction"] == "max":
            x, y = -x, -y
        if x > y:
            return False
        if x < y:
            better = True
    return better


def _dominates_row(objs, a, b, alpha=None):
    if alpha is None:
        return dominates(objs, a["values"], b["values"])
    return dominates_measured(objs, a, b, alpha)


def archive_alpha(st):
    """The alpha the archive compares at, or None for plain medians."""
    cfg = st["config"]
    if not cfg.get("archive_bootstrap", DEFAULT_CONFIG["archive_bootstrap"]):
        return None
    return float(cfg.get("archive_alpha", DEFAULT_CONFIG["archive_alpha"]))


def archive_insert(objs, archive, point, alpha=None):
    """
    (archive, verdict): insert `point` if nothing dominates it, dropping what
    it dominates. Verdict is `inserted`, `dominated`, `duplicate` or `tied`.

    A different program with IDENTICAL objectives is `tied` and not stored: it is
    formally non-dominated, but the archive is a list of things to deliver and
    two rows a reader cannot choose between are noise.

    `alpha` turns on the bootstrap comparison of the measured objective (see
    `dominates_measured`); None compares medians.
    """
    for e in archive:
        if e["code_hash"] == point["code_hash"]:
            return archive, "duplicate"
        if e["values"] == point["values"]:
            return archive, "tied"
    if any(_dominates_row(objs, e, point, alpha) for e in archive):
        return archive, "dominated"
    kept = [e for e in archive if not _dominates_row(objs, point, e, alpha)]
    kept.append(point)
    # Sorted by the objectives after the primary one, so the front reads from
    # the cheapest secondary cost upward; the primary breaks ties.
    order = [o["name"] for o in objs[1:]] + [objs[0]["name"]]
    kept.sort(key=lambda e: tuple(e["values"][n] for n in order))
    return kept, "inserted"


def archive_reprune(objs, archive, alpha=None):
    """Re-derive the front from the points it holds.

    Needed after a RE-MEASUREMENT: new readings can widen an interval (two
    points become unordered, and one that was dropped would now have stayed) or
    narrow it (a point that was kept is now dominated). Insertion order is the
    archive's own, so the result does not depend on when the re-measurement
    happened -- only on the readings.
    """
    out_a = []
    for e in archive:
        out_a, _ = archive_insert(objs, out_a, e, alpha)
    return out_a


def archive_point(st, code_hash, p, m, action=None):
    """The archive entry for a measurement, or None when a value is missing."""
    objs = objectives_of(st)
    vals, missing = objective_values(objs, p, m)
    if missing:
        return None
    node = open_node(st, action) if action else None
    # The readings behind `p`, so dominance can be judged on them rather than
    # on the median alone. They live on the cache entry this measurement just
    # wrote (or, for the preopt, the one `init` wrote).
    entry = (st.get("eval_cache") or {}).get(code_hash) or {}
    samples = entry.get("samples")
    return {"code_hash": code_hash, "values": vals,
            "samples": list(samples or []),
            # Per objective, so dominance is judged on the readings that
            # belong to the axis being compared rather than on `p`'s for all
            # of them. `samples` above stays for readers that predate this.
            "samples_by_objective": objective_samples(objs, entry, m),
            # Kept next to `values` because every reader wants them: the
            # primary objective's value and the scalar the ratchet uses.
            "aicore_e2e_us": float(p),
            "mix_aic_us": (m or {}).get("mix_aic"),
            "workspace_b": (int(m.get("workspace_b"))
                            if (m or {}).get("workspace_b") is not None
                            else None),
            "J": compute_j(1, p, st["target"]["P_ref"]),
            "action": action, "catalog_id": (node or {}).get("catalog_id"),
            "eval_no": st["progress"]["evals_used"]}


def _hv_vectors(objs, rows, draw=None):
    """Archive rows as vectors on the hypervolume's axes.

    `draw` maps a program's hash to one bootstrap resample of its readings per
    objective; without it the stored medians are used. The resample is keyed by
    PROGRAM, not by row, because `before` and `after` hold the same programs
    and a comparison that redrew them independently would be measuring the
    bootstrap rather than the front.
    """
    vecs = []
    for e in rows:
        vec = []
        for o in objs:
            v = (draw or {}).get(e["code_hash"], {}).get(o["name"])
            if v is None:
                v = e["values"][o["name"]]
            vec.append(hv_axis(o, v))
        vecs.append(tuple(vec))
    return vecs


def _hv_draw(objs, rows, rng):
    """One bootstrap resample: {hash: {objective: median of a resample}}.

    An objective with no readings is absent and keeps its number -- the
    workspace probe returns one value, not a sample, so the interval below
    carries the primary objective's noise and nothing it does not have.
    """
    draw = {}
    for e in rows:
        per = {}
        for o in objs:
            xs = row_samples(o, e)
            if len(xs) >= 2:
                per[o["name"]] = median_of([xs[rng.randrange(len(xs))] for _ in xs])
        if per:
            draw[e["code_hash"]] = per
    return draw


def hv_gain(st, before, after):
    """What this measurement did to the hypervolume, and whether noise could have done it.

    The reference is the run's fixed one, so the two volumes are directly
    comparable and so is the whole series -- `progress.hv` can be read as a
    progress curve, which it could not be while the box moved. The interval is
    a percentile bootstrap of the relative gain at the same seed `verify` and
    the archive's dominance test use.
    """
    objs = objectives_of(st)
    cfg = st["config"]
    ref = hv_reference_point(st, objs)
    if ref is None:
        return None
    rows = {}
    for e in list(before) + list(after):
        if e.get("values"):
            rows.setdefault(e["code_hash"], e)
    gain, hb, ha = mo_hv.relative_gain(_hv_vectors(objs, before),
                                       _hv_vectors(objs, after), ref)
    detail = {"hv": round(ha, 6), "hv_before": round(hb, 6),
              "gain": None if gain is None else round(gain, 6),
              "opened": hb <= 0.0 and ha > 0.0}
    n_boot = int(cfg.get("hv_bootstrap", VERIFY_BOOT))
    if gain is not None and n_boot > 0:
        rng = random.Random(VERIFY_SEED)
        draws = []
        for _ in range(n_boot):
            d = _hv_draw(objs, list(rows.values()), rng)
            if not d:
                break
            g, _hb, _ha = mo_hv.relative_gain(
                _hv_vectors(objs, before, d), _hv_vectors(objs, after, d), ref)
            if g is not None:
                draws.append(g)
        if len(draws) >= n_boot // 2:
            draws.sort()
            alpha = VERIFY_ALPHA
            detail["ci_low"] = round(draws[int(alpha / 2 * len(draws))], 6)
            detail["ci_high"] = round(draws[int((1 - alpha / 2) * len(draws)) - 1], 6)
            detail["n_boot"] = len(draws)
    return detail


def _axis_records(objs, rows):
    """{objective: the best value the front holds on that axis}."""
    picked = {}
    for o in objs:
        vals = [e["values"][o["name"]] for e in rows
                if (e.get("values") or {}).get(o["name"]) is not None]
        if vals:
            picked[o["name"]] = max(vals) if o["direction"] == "max" else min(vals)
    return picked


def _axis_advance(st, objs, before, after):
    """Which axis records improved, and by how much relative to the old one.

    The hypervolume cannot see every real improvement, however the reference is
    chosen, because it is a volume and the box's shape is the weighting between
    the axes. Two cases need this:

      * a workspace reduction of 8% is 0.14% of hypervolume at the chip's
        memory -- real, and under any usable threshold;
      * conversely a point that FILLS THE MIDDLE of the front improves no
        record at all and is only visible as volume.

    So the stop test is the OR of the two, and each threshold is stated in the
    unit a person reasons about: a fraction of the axis's own value.
    """
    old, new = _axis_records(objs, before), _axis_records(objs, after)
    picked = {}
    for o in objs:
        n = o["name"]
        a, b = old.get(n), new.get(n)
        if a is None or b is None or not a:
            continue
        d = (a - b) / abs(a) if o["direction"] == "min" else (b - a) / abs(a)
        picked[n] = round(d, 6)
    return picked


def _axis_floor(st, o):
    cfg = st["config"]
    if is_bytes(o):
        return float(cfg.get("axis_min_gain_bytes",
                             DEFAULT_CONFIG["axis_min_gain_bytes"]))
    return float(cfg.get("axis_min_gain", DEFAULT_CONFIG["axis_min_gain"]))


def hv_observe(st, before, after):
    """Fold one measurement into the run's convergence test.

    A gain counts only when it is BOTH big enough (`hv_min_gain`) and more
    than the readings' own noise -- the lower end of the interval has to clear
    the threshold. With no interval (an objective with fewer than two readings
    on some point) the point estimate decides, which is the pre-C3 behaviour
    rather than a refusal to judge.
    """
    pr, cfg = st["progress"], st["config"]
    objs = objectives_of(st)
    adv = _axis_advance(st, objs, before, after)
    moved = sorted(o["name"] for o in objs
                   if adv.get(o["name"], 0.0) > _axis_floor(st, o))
    d = hv_gain(st, before, after)
    if d is None:
        # No hypervolume to read (a byte axis with no ceiling). The axis
        # records still answer, so the test degrades rather than switching off.
        if not before:
            return None
        d = {"hv": None, "gain": None, "opened": False}
    d["axis_advance"] = adv
    d["axis_records_moved"] = moved
    floor = float(cfg.get("hv_min_gain", DEFAULT_CONFIG["hv_min_gain"]))
    if not before:
        # `init` offering the preopt. The volume it encloses against itself is
        # zero by construction -- the reference IS the preopt on the primary
        # axis -- and the search has not paid for a measurement yet, so this
        # records the baseline without touching the counter.
        pr["hv"] = d["hv"]
        pr["hv_last"] = dict(d, counted_as_gain=None)
        return d
    if d["opened"]:
        grew = True                      # the front came into existence
    elif d["gain"] is None:
        grew = False
    elif "ci_low" in d:
        grew = d["ci_low"] > floor
    else:
        grew = d["gain"] > floor
    # The OR. A record that moved past its own floor is growth the volume may
    # not be able to see.
    grew = bool(grew or moved)
    d["counted_as_gain"] = bool(grew)
    pr["hv"] = d["hv"]
    pr["hv_last"] = d
    if grew:
        pr["hv_stagnation"] = 0
        pr["hv_last_gain"] = {"at_eval": pr.get("evals_used"), "gain": d["gain"],
                              "hv": d["hv"]}
    else:
        pr["hv_stagnation"] = pr.get("hv_stagnation", 0) + 1
    return d


def archive_record(st, code_hash, p, m, action=None):
    """
    Offer one measurement to the archive. Returns the verdict, or None when
    the point has no workspace reading.
    """
    point = archive_point(st, code_hash, p, m, action)
    if point is None:
        st["progress"]["archive_unplaceable"] = (
            st["progress"].get("archive_unplaceable", 0) + 1)
        return None
    before = st.get("archive") or []
    after, verdict = archive_insert(objectives_of(st), before, point, archive_alpha(st))
    st["archive"] = after
    pr = st["progress"]
    # Every offer, inserted or not: a candidate the front refused is a
    # measurement that bought no volume, which is what the test is counting.
    hv_observe(st, before, after)
    if verdict == "inserted":
        pr["archive_inserted"] = pr.get("archive_inserted", 0) + 1
        pr["archive_dropped"] = pr.get("archive_dropped", 0) + (len(before) + 1 - len(after))
        # Non-dominated but SLOWER than the fastest point known: exactly what a
        # single-objective run measures and discards. Compared against the
        # archive rather than `best_aicore_e2e_us`, which only moves at `close`.
        #
        # Against `aicore_e2e_us`, which is what `p` IS. This read `mix_aic_us`,
        # a different instrument on a different clock -- and one that is None
        # whenever msprof produced no usable MIX_AIC, which the default
        # objectives never halt on. The second archived point then evaluated
        # `float > None` and raised TypeError out of `cmd_record` BEFORE its
        # `save`, losing the measurement and the charged evaluation, and the
        # None row stayed in the state so every later `record` raised in the
        # same place. `aicore_e2e_us` is `float(p)` one line up: never None, and
        # the clock `p` is already on.
        fastest = min((e["aicore_e2e_us"] for e in before
                       if e.get("aicore_e2e_us") is not None),
                      default=float("inf"))
        if p > fastest:
            pr["archive_kept_slower"] = pr.get("archive_kept_slower", 0) + 1
    return verdict


# ------------------------------------------------------ the (latency, workspace) front
def pareto_front(points):
    """
    Non-dominated subset of [(p, ws, payload)], both minimised. Ties on both
    objectives keep every tied point; the result is sorted by workspace.
    """
    front = []
    for i, (p, w, x) in enumerate(points):
        dominated = any((q <= p and v <= w) and (q < p or v < w)
                        for j, (q, v, _) in enumerate(points) if j != i)
        if not dominated:
            front.append((p, w, x))
    return sorted(front, key=lambda t: (t[1], t[0]))
