# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run correctness-constrained TPE on measurable configurations within the Pro tile domain."""
import json
import random
from dataclasses import dataclass, replace
from typing import Any
import optuna

from panko_core.bayesian_optimization import driver as _engine, evaluator
from . import domain as domain_mod, policy, space
from .lever import dist

PENALTY = _engine.PENALTY

error_signature = _engine.error_signature


# Limit persisted refusals per site.
SITE_REFUSALS_PER_SITE = 32


def site_delta(cfg, anchor):
    """Return (site_id, shape) only for a single-site change from the anchor."""
    if not anchor or not cfg:
        return None
    moved = [(sid, dict(c)) for sid, c in cfg.items() if dict(c) != dict(anchor.get(sid) or {})]
    return moved[0] if len(moved) == 1 else None


def shape_key(shape):
    return json.dumps(shape, sort_keys=True)


def remember_site_refusal(ctx, cfg, detail):
    """Cache structure-specific, single-site refusals. Capacity failures use separate ceilings."""
    got = None if ctx["state"].get("anchor_incorrect") else site_delta(cfg, ctx.get("anchor"))
    if got is None:
        return
    sid, shape = got
    store = ctx["state"].setdefault("site_refused", {}).setdefault(sid, {})
    if len(store) < SITE_REFUSALS_PER_SITE:
        store.setdefault(shape_key(shape), (detail or "s=0")[:160])


def site_refused(ctx, cfg):
    """Return the recorded reason for a known bad site shape, excluding the incumbent."""
    store = ctx["state"].get("site_refused") or {}
    if not store:
        return None
    anchor = ctx.get("anchor") or {}
    for sid, c in cfg.items():
        known = store.get(sid)
        if not known or dict(c) == dict(anchor.get(sid) or {}):
            continue           # never refuse the program being tuned
        why = known.get(shape_key(dict(c)))
        if why is not None:
            return ("refused_site_shape", "%s: %s was measured and refused -- %s" % (sid, dict(c), why))
    return None


_dedup = _engine.dedup


def config_widths(opts):
    return _engine.site_widths(opts, live_for=domain_mod.live_for)


def _tile_bytes(cfg, dtype_bytes, widths=None):
    return _engine.draw_tile_bytes(cfg, dtype_bytes, widths, include_cube=False)


@dataclass
class SearchSeeds:
    """Pro seed observations and per-draw site limit."""

    warm_start: Any = None
    observed: Any = None
    sites_per_draw: int = 0


@dataclass
class SearchOptions(_engine.SearchSettings, SearchSeeds):
    """Shared TPE controls following the original Pro seed arguments."""


def _in_seed_phase(ctx):
    """Use the draw cursor to identify queued seeds."""
    return ctx.get("_calls", 0) <= int(ctx.get("seeds_queued", 0) or 0)


def _clusters(ctx):
    """Cache the site groups used by the round-robin cursor and sweep budget."""
    if ctx.get("_clusters") is None:
        opts = ctx["opts"]
        ctx["_clusters"] = space.clusters(ctx["sites"], opts.groups, opts.tie)
    return ctx["_clusters"]


def _next_move(ctx):
    """Which site ids the next draw may move, or None for all of them."""
    k = int(getattr(ctx["opts"], "sites_per_draw", 0) or 0)
    ctx["_calls"] = ctx.get("_calls", 0) + 1
    if k <= 0 or _in_seed_phase(ctx):
        return None
    clusters = _clusters(ctx)
    if k >= len(clusters):
        return None
    cursor = ctx.get("_cursor", 0)
    ctx["_cursor"] = cursor + k
    return [sid for i in range(k) for sid in clusters[(cursor + i) % len(clusters)]]


_emit = _engine.emit_record


def static_reason(ctx, cfg):
    refusal = _engine.candidate_refusal(ctx, cfg, config_widths, _tile_bytes, space, domain_mod)
    return refusal or site_refused(ctx, cfg)


def _try_repair(ctx, cfg):
    """The nearest feasible config below `cfg`, or None. See `space.repair`."""
    if not ctx["opts"].repair:
        return None, 0

    def check(c):
        bad = static_reason(ctx, c)
        return (True, "") if bad is None else (False, bad[1])
    return space.repair(cfg, check, ctx["anchor"], ctx["opts"].domain, ctx["opts"].tie, movable=_movable(ctx),
                        groups=ctx["opts"].groups, hw=ctx["hw"])


def _draw_options(ctx, move):
    """Build the sampler view shared by a draw and its repair."""
    opts = ctx["opts"]
    return space.DrawOptions(groups=opts.groups, anchor=ctx["anchor"], domain=opts.domain,
                             tie=opts.tie, hw=ctx.get("hw"), move=move)


def _movable(ctx):
    """Return the sites eligible for repair in this draw, or None for a joint draw."""
    return space.resolve_move(ctx["sites"], _draw_options(ctx, ctx.get("_move")))


def _static_verdict(ctx, cfg, key, trial):
    """Record a static rejection without charging a device trial."""
    del key
    rejected = static_reason(ctx, cfg)
    if rejected is None:
        return None
    status, reason = rejected
    _emit(ctx, {"config": cfg, "status": status, "reason": reason, "objective_us": None}, trial, status, reason)
    return PENALTY


def config_bytes(cfg, opts):
    return _tile_bytes(cfg, opts.dtype_bytes, config_widths(opts))


UNEVALUATED_LIMIT = _engine.UNEVALUATED_LIMIT


def _draw_candidate(ctx, trial):
    cfg = space.suggest(trial, ctx["sites"], _draw_options(ctx, _remember_move(ctx)))
    if _in_seed_phase(ctx):
        trial.set_user_attr("seed_phase", True)
    return cfg


def trial_objective(ctx, trial):
    return _engine.trial_objective(ctx, trial, policy=_engine.TrialPolicy(
        _draw_candidate, space.flatten, static_reason, _try_repair,
        _static_verdict, _evaluate_candidate, measured_params=True,
        repaired_duplicate_penalty=True))


_abort = _engine.abort


make_study = _engine.make_study


def _distributions(params, sites, opts, anchor):
    """Rebuild the same categorical domains used by shape suggestion."""
    out = {key: dist(key, sites, opts.groups, anchor, domain=opts.domain) for key in params}
    for key, choices in space.grouped_domains(opts.groups, sites, anchor, opts.domain, tie=opts.tie).items():
        if key in out:
            out[key] = optuna.distributions.CategoricalDistribution(choices)
    return out


def _queue_error(params, sites, opts, anchor):
    """Return a reason when fixed seed coordinates are absent from the categorical distributions used by
    the sampler."""
    if not params:
        return "no shape parameters"
    try:
        distributions = _distributions(params, sites, opts, anchor)
        for key, value in params.items():
            distributions[key].to_internal_repr(value)
    except (ValueError, TypeError, RuntimeError) as exc:
        return "outside sampler domain: %s" % exc
    return None


def _warm_start(study, sites, opts, memory, skipped=None):
    """Queue representable seeds, skip replayed points, and record why rejected seeds were omitted."""
    seen = set()
    for index, flat in _engine.unreplayed_seeds(sites, opts, memory, space.flatten):
        key = shape_key(flat)
        if key in seen:
            continue
        why = _queue_error(flat, sites, opts, opts.anchor)
        if why:
            if skipped is not None:
                skipped.append({"seed": index, "params": flat, "why": why})
            continue
        study.enqueue_trial(flat)
        seen.add(key)
    return len(seen)


# Bound retries per cluster before sampling the next cluster.
COVER_TRIES = 8


def spread(clusters):
    """Visit clusters in bit-reversed order so a short prefix spans the search space."""
    clusters = list(clusters or [])
    width = (len(clusters) - 1).bit_length()
    return [clusters[i] for i in sorted(range(len(clusters)), key=lambda index: int(f"{index:0{width}b}"[::-1], 2))]


def _draw_cover_cluster(ctx, group, rng, site_ids):
    """Perturb one whole cluster through the same ladders used by sampling."""
    from . import seed as seed_mod
    opts, anchor, sites = ctx["opts"], ctx["anchor"], ctx["sites"]
    trial_cfg = {sid: dict(c) for sid, c in anchor.items()}
    for sid in group:
        if sid not in site_ids:
            continue
        incumbent = dict(anchor.get(sid) or {})
        drawn = seed_mod.rand_vec(sid, incumbent, opts.domain, rng)
        if drawn and any(incumbent.get(field) != value for field, value in drawn.items()):
            trial_cfg[sid] = dict(incumbent, **drawn)
    return space.unflatten(space.flatten(trial_cfg, sites, opts.groups, opts.tie), sites, opts.groups, opts.tie)


def _cover_candidate(ctx, group, rng, site_ids):
    """Find one admissible, non-anchor cover point within the bounded attempts."""
    opts, sites, anchor = ctx["opts"], ctx["sites"], ctx["anchor"]
    why = ""
    for _ in range(COVER_TRIES):
        trial_cfg = _draw_cover_cluster(ctx, group, rng, site_ids)
        if trial_cfg == anchor:
            why = "every rung this cluster offers is the value it already has"
            continue
        flat = space.flatten(trial_cfg, sites, opts.groups, opts.tie)
        why = _queue_error(flat, sites, opts, anchor)
        if why:
            continue
        bad = static_reason(ctx, trial_cfg)
        if bad is None:
            return flat, why
        why = bad[1]
    return None, why


def cover_points(ctx, limit):
    """Seed uncovered clusters with feasible, unobserved non-anchor configurations."""
    opts, anchor = ctx["opts"], ctx.get("anchor")
    sites, memory = ctx["sites"], ctx.get("memory") or {}
    skipped = ctx.setdefault("cover_skipped", [])
    if not anchor:
        return []
    # Reserve at least half the budget for sampler draws informed by the covering seeds.
    if limit is not None:
        limit //= 2
        if limit <= 0:
            return []
    rng = random.Random(opts.seed)
    known = {shape_key(t.get("params")) for t in memory.get("trials", [])}
    site_ids = {s.id for s in sites}
    out = []
    for grp in spread(space.clusters(sites, opts.groups, opts.tie)):
        if limit is not None and len(out) >= limit:
            break
        # Test the whole cluster because tied sites share extents and memory.
        flat, why = _cover_candidate(ctx, grp, rng, site_ids)
        if flat is None:
            skipped.append({"cluster": list(grp), "why": (why or "")[:160]})
            continue
        key = shape_key(flat)
        if key in known:
            continue
        known.add(key)
        out.append(flat)
    return out


def cover_clusters(study, ctx, limit):
    """Enqueue `cover_points`. Returns how many were queued."""
    points = cover_points(ctx, limit)
    for flat in points:
        study.enqueue_trial(flat)
    return len(points)


def _prepare_search(sites, opts):
    sites = policy.shape_sites(sites)
    groups = policy.shared_shape_groups(sites, opts.groups, opts.domain)
    opts = replace(opts, groups=groups, n_trials=opts.n_trials if sites else 0,
                   tie=policy.expanded_ties(sites, opts.tie, groups),
                   anchor=policy.shape_config(opts.anchor, sites) or None,
                   warm_start=[policy.shape_config(c, sites) for c in (opts.warm_start or [])],
                   observed=[(policy.shape_config(c, sites), value) for c, value in (opts.observed or [])])
    memory = dict(opts.memory or {})
    expected = set(space.flatten(opts.anchor or {}, sites, opts.groups, opts.tie))
    # Replay only complete shape parameters owned by TPE.
    rows = []
    for row in memory.get("trials", []):
        params = row.get("measured") or row.get("params") or {}
        if (policy.shape_params(params, sites) and (not expected or set(params) == expected)):
            rows.append(dict(row, params=params))
    memory["trials"] = rows
    return sites, opts, memory


def _seed_search(study, sites, opts, anchor, memory):
    opts = replace(opts, anchor=anchor)
    anchor_key = (json.dumps(space.flatten(anchor, sites, opts.groups, opts.tie), sort_keys=True) if anchor else None)
    replayed = _engine.replay(study, sites, opts, anchor, memory, distributions=_distributions)
    observed = _add_observed(study, sites, opts)
    skipped = []
    queued = _warm_start(study, sites, opts, memory, skipped)
    extra = {"warm_queued": queued, "memory": memory, "anchor_key": anchor_key, "cover_skipped": [],
             "warm_skipped": skipped, "observed": observed, "seeds_queued": queued}
    return opts, replayed, extra


def _prepare_draws(ctx):
    opts = ctx["opts"]
    ctx["seeds_queued"] += cover_clusters(ctx["study"], ctx,
                                          None if opts.n_trials is None else opts.n_trials - ctx["seeds_queued"])
    ctx["state"]["seen"].update(ctx["observed"])


def run_bo(objective, sites, hw, opts=None):
    """Search Pro shape domains with shared TPE trial management."""
    return _engine.run_bo(objective, sites, hw, opts, policy=_engine.SearchPolicy(
        SearchOptions, _prepare_search, make_study, _seed_search, initial_state,
        _prepare_draws, _optimize, _result))


# Bound sampler attempts per granted device trial; free draws spend no eval.
DRAWS_PER_TRIAL = 16

# Consecutive free-draw floor, scaled to cover one site-cluster sweep.
FREE_RUN_LIMIT = 24


_ran = _engine.ran


def _exhausted(ctx, drawn, budget, why):
    """Explain why drawing stopped before the device budget was spent."""
    return {"draws": drawn, "device_trials": ctx["state"].get("device", 0),
            "budget": budget, "free_run": ctx.get("_free_run", 0), "why": why,
            "note": "the draws that measured nothing were rejected by the "
                    "capacity gate or already known. The neighbourhood of this "
                    "program is closed at this tile domain, which is a fact "
                    "about the kernel rather than about the budget."}


def _optimize(study, ctx, opts):
    return _engine.optimize(study, ctx, opts, trial_objective, stopper, device_budget=True,
                            free_run_limit=free_run_limit, exhausted=_exhausted, draws_per_trial=DRAWS_PER_TRIAL)


def _remember_move(ctx):
    """Restrict repair to the sites selected for this draw."""
    ctx["_move"] = _next_move(ctx)
    return ctx["_move"]


def charge_failure(ctx, cfg):
    """Reset on the first refusal in a new cluster. Count repeats and unattributed draws."""
    state = ctx["state"]
    seen = state.setdefault("failed_clusters", set())
    got = site_delta(cfg, ctx.get("anchor"))
    who = got[0] if got else None
    if who is not None and who not in seen:
        seen.add(who)
        state["failures"] = 0
        return
    state["failures"] += 1


def record_refusal(ctx, cfg, key, outcome, trial):
    return _engine.record_refusal(ctx, cfg, key, outcome, trial, footprint=config_bytes,
                                 remember_site=remember_site_refusal)


def register_measured(ctx, cfg, p, trial):
    return _engine.register_repaired(ctx, p, lambda params: _distributions(
        params, ctx["sites"], ctx["opts"], ctx["anchor"]))


def record_ok(ctx, cfg, key, p, trial):
    return _engine.record_ok(ctx, cfg, key, p, trial, register=register_measured,
                             warmup=getattr(ctx["objective"], "last_warmup", None))


def _evaluate_candidate(ctx, cfg, key, trial):
    """Measure one admitted configuration and preserve outcome accounting."""
    ctx["state"]["device"] = ctx["state"].get("device", 0) + 1
    s, p, kind, detail = _engine.measure_trial(ctx, cfg, trial, evaluator.EnvironmentFault)
    ctx["state"]["dry"] = ctx["state"].get("dry", 0) + 1
    if s:
        ctx["state"].update(failures=0, unevaluated=0, dry=0, failed_clusters=set())
        return record_ok(ctx, cfg, key, p, trial)
    if kind == "unevaluated":
        _engine.record_unevaluated(ctx, cfg, detail, trial)
        raise optuna.TrialPruned()
    ctx["state"]["unevaluated"] = 0
    charge_failure(ctx, cfg)
    if key and key == ctx.get("anchor_key"):
        # Check the incorrect incumbent before spending another tile trial.
        ctx["state"]["anchor_incorrect"] = (kind, detail)
    return record_refusal(ctx, cfg, key, (kind, detail), trial)


def stop_on_failed_measurement(ctx, study):
    """Apply fault, correctness and missing-measurement brakes in priority order."""
    state, opts = ctx["state"], ctx["opts"]
    if state["faults"] >= opts.fault_limit:
        _abort(ctx, study, f"{state['faults']} consecutive environment faults "
                           f"({state['fault_kind']})", state["fault_kind"])
        return True
    if state.get("anchor_incorrect"):
        kind, detail = state["anchor_incorrect"]
        _abort(ctx, study, "the incumbent itself measured incorrect (s=0%s): no "
                           "tile value can repair it, because every draw holds "
                           "the other sites at this program's extents and what "
                           "is wrong with it is not an extent%s"
                           % (", %s" % kind if kind else "",
                              " -- %s" % detail[:160] if detail else ""))
        return True
    if opts.failure_limit and state["failures"] >= opts.failure_limit:
        _abort(ctx, study,
               f"{state['failures']} consecutive candidates ran and were "
               f"rejected by the correctness gate (s=0) at clusters this block "
               f"had already failed on "
               f"({', '.join(sorted(state.get('failed_clusters') or ())[:6])}): "
               f"the search is not moving off them. A first failure at a new "
               f"cluster is a discovery and does not count here -- see "
               f"`charge_failure`")
        return True
    if (opts.stagnation_limit and state.get("dry", 0) >= opts.stagnation_limit):
        _abort(ctx, study,
               f"{state['dry']} device trials since the last feasible "
               f"measurement, which is a whole sweep of this kernel's "
               f"{opts.stagnation_limit} clusters: no extent the search can "
               f"reach is legal on this program")
        return True
    if state["unevaluated"] >= UNEVALUATED_LIMIT:
        _abort(ctx, study, f"{state['unevaluated']} consecutive compiles past "
                           f"eval_timeout_s")
        return True
    return False


def stopper(ctx, study, trial):
    """Apply Pro measurement brakes and exclude duplicate or seed misses from stagnation."""
    if stop_on_failed_measurement(ctx, study):
        return
    attrs = trial.user_attrs or {}
    if attrs.get("status") != "duplicate_config":
        _engine.update_stagnation(ctx, study, trial.value, count_feasible=n_feasible,
                                 count_miss=not attrs.get("seed_phase"), penalty=PENALTY)


def n_feasible(study):
    """Count feasible measurements acquired in this block, excluding cache hits and supplied observations."""
    return sum(t.value is not None and t.value < PENALTY and not (t.user_attrs or {}).get("observed")
               and (t.user_attrs or {}).get("status") != "duplicate_config" for t in study.trials)


def _add_observed(study, sites, opts):
    """Register prior measurements against this study's distributions and return the admitted values."""
    filed = {}
    for cfg, val in (opts.observed or []):
        flat = None if val is None else space.flatten(cfg, sites, opts.groups, opts.tie)
        if not flat:
            continue
        try:
            study.add_trial(optuna.trial.create_trial(params=flat,
                                                      distributions=_distributions(flat, sites, opts, opts.anchor),
                                                      value=float(val), user_attrs={"status": "ok", "observed": True}))
        except (ValueError, TypeError, RuntimeError):
            continue
        filed[json.dumps(flat, sort_keys=True)] = float(val)
    return filed


def initial_state(memory, anchor, opts):
    state = _engine.initial_state(memory, anchor, opts, footprint=config_bytes)
    state.update(failed_clusters=set(), dry=0,
                 site_refused={sid: dict(rows) for sid, rows in (memory.get("site_refusals") or {}).items()})
    return state


def reached(ctx):
    """Return clusters that can be attributed to a measured single-site change."""
    deltas = (site_delta(rec.get("config") or {}, ctx.get("anchor")) for rec in ctx.get("history") or []
              if rec.get("status") in ("ok", "incorrect", "incorrect_capacity"))
    return {got[0] for got in deltas if got}


def result_memory(feasible, memory, state, ceiling):
    """Carry measured observations and structure-scoped refusals forward."""
    return {
        "parameter_scope": policy.PARAMETER_SCOPE,
        "trials": _dedup([{"params": t.user_attrs.get("measured_params", t.params),
                           "value": t.value}
                          for t in feasible], memory.get("trials", [])),
        "ceiling": ceiling,
        # Invalidate these entries when the structural key changes.
        "site_refusals": {sid: dict(rows) for (sid, rows) in (state.get('site_refused') or {}).items() if rows},
    }


def _result(study, ctx, memory, replayed):
    summary = _engine.summarize_trials(study, ctx, memory, replayed, unflatten=space.unflatten,
                                      trial_memory=result_memory)
    queued = ctx.get("seeds_queued")
    summary.update(n_feasible=n_feasible(study), draws_exhausted=ctx["stag"].get("draws_exhausted"),
                   draws=ctx.get("draws"), seeds_queued=queued, cover_skipped=list(ctx.get("cover_skipped") or []),
                   warm_skipped=list(ctx.get("warm_skipped") or []),
                   cover_queued=max(0, int(queued or 0) - int(ctx.get("warm_queued") or 0)),
                   clusters_reached=len(reached(ctx)))
    return summary


def free_run_limit(ctx):
    """Allow one sweep of free draws, with a minimum set by the backend."""
    return max(FREE_RUN_LIMIT, len(_clusters(ctx) or ()))


_record_fault = _engine.record_fault
record_unevaluated = _engine.record_unevaluated
engine = _engine
