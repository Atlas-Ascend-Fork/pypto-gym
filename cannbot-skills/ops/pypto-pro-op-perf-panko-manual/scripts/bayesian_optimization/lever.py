# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Rebuild the selected action’s Optuna study from campaign state on each dispatch."""
import hashlib
import json
from panko_core.bayesian_optimization import lever as core
from panko_core.bayesian_optimization.lever import AskOptions


from . import apply
from . import domain as tile_domain
from . import policy, space

SCOPE_OF = {"F-1": "cube", "S-1": "cube", "F-2": "vec", "S-2": "vec"}
TILE_ACTIONS = set(SCOPE_OF)


# Map shape actions to tile sites by memory space. Mat, Left, Right and Acc serve cube
# operations. Vec serves vector operations.
SCOPE_PREFIXES = {"cube": ("mat", "left", "right", "acc"), "vec": ("vec",)}


def in_scope(site_id, scope):
    """Match a site to its memory-space scope. A scope of None includes every site.
    """
    return scope is None or site_id.split("#", 1)[0].split(":", 1)[-1] in SCOPE_PREFIXES.get(scope, ())


def _scoped(sites, scope):
    """Restrict a tile action to its eligible sites. An unscoped block searches their union."""
    # Unknown site kinds cannot own a recorded action delta.
    return [s for s in sites if in_scope(s.id, scope)]


def structural_signature(src, extra_consts=None):
    """Address relocation and tile edits belong to the same structural study."""
    try:
        spans = apply.erasable_spans(src, extra_consts)
    except SyntaxError:
        return "unparseable"
    return core.signature_from_spans(src, spans)


def lever_key(catalog_id, src, extra_consts=None):
    """Share observations within an action scope and structure. Unknown action IDs retain separate studies."""
    scope = SCOPE_OF.get(catalog_id)
    # Include fixed sites in the key because their shapes affect measured latency.
    try:
        held = {sid: value for sid, value in apply.current_config(src, extra_consts).items()
                if _is_tile(value) and not in_scope(sid, scope)}
    except SyntaxError:
        held = {}
    held_key = hashlib.sha256(json.dumps(held, sort_keys=True).encode()).hexdigest()[:12]
    constants = apply.source_constants(extra_consts)
    context_key = hashlib.sha256(json.dumps(constants, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return (f"{policy.PARAMETER_SCOPE}|{scope or catalog_id}|"
            f"{structural_signature(src, constants)}|{held_key}|{context_key}")


def _is_tile(c):
    return any(k.startswith("d") for k in c)


def _shape_domain(index, arity, value):
    return space.admit(space.SHAPE_LADDER, value)


def _family(cur, dtype_bytes, widths=None):
    return core.seed_family(cur, dtype_bytes, widths, apply.MAX_VEC_ARITY, vector_domain=_shape_domain)


def seed_points(src, hw, dtype_bytes=4, current_latency=None, scope=None, widths=None, dom=None):
    """Return (configuration, measured latency or None, label) for the incumbent and footprint seeds."""
    cur = apply.current_config(src, getattr(hw, "extra_consts", None))
    cur = {k: v for k, v in cur.items() if _is_tile(v) and in_scope(k, scope)}
    options = core.SeedOptions(dtype_bytes, current_latency, scope, widths, dom)
    return core.seeds_from_anchor(cur, hw, options, _family, feasible=space.static_feasible, verify=tile_domain.verify)


def dist(name, sites, groups, anchor, held=None, domain=None):
    """Build the aligned categorical domain, retaining the incumbent and historical values."""
    import optuna
    sid, field = name.rsplit(".", 1)
    grouped = space.grouped_domains(groups, sites, anchor, domain)
    if name in grouped:
        return optuna.distributions.CategoricalDistribution(core.with_held(grouped[name], held))
    a = (anchor or {}).get(sid, {})
    if not (field.startswith("d") and field[1:].isdigit()):
        raise ValueError(f"not a tile dimension: {name}")
    base = space.vec_domain(sid, field, space.vec_fields(a), a, domain)
    return optuna.distributions.CategoricalDistribution(core.with_held(space.admit(base, a.get(field)), held))


def draw_one_site(study, ctx):
    rec, sites, anchor, groups = ctx["rec"], ctx["sites"], ctx["anchor"], ctx.get("groups")
    clusters = space.clusters(sites, groups)
    cluster = clusters[rec.get("cursor", 0) % len(clusters)]
    moving = [site for site in sites if site.id in cluster]
    return core.draw_cluster(study, ctx, moving, cluster[0], space, lambda name,
                             held: dist(name, sites, groups, anchor, held, ctx.get("domain")), groups)


def _gate(cfg, hw, dtype_bytes, widths=None, dom=None):
    return core.gate_draw(cfg, hw, dtype_bytes, widths, dom, feasible=space.static_feasible, verify=tile_domain.verify)


def ask(state_bo, catalog_id, src, hw, opts=None):
    opts = opts or AskOptions()
    constants = getattr(hw, "extra_consts", None)
    key = lever_key(catalog_id, src, constants)
    rec = state_bo.setdefault(key, {"trials": [], "seeded": False, "pending": None})
    scope = SCOPE_OF.get(catalog_id)
    try:
        sites = _scoped(policy.shape_sites(apply.tunable_sites(src, constants)), scope)
    except SyntaxError:
        return None, {"reason": "unparseable"}
    if not sites:
        reason = f"no {scope} tile site" if scope else "no tunable tile site"
        return None, {"reason": reason, "scope": scope}

    groups = policy.shared_shape_groups(sites, domain=opts.domain)
    rec["shape_groups"] = groups
    anchor = apply.current_config(src, constants)

    def flatten(cfg, sites):
        return space.flatten(cfg, sites, groups)

    if not rec["seeded"]:
        points = seed_points(src, hw, opts.dtype_bytes, opts.current_latency, scope, opts.vec_dtype_bytes, opts.domain)

        def checked_params(cfg, label):
            params = flatten(cfg, sites)
            try:
                _validate_seed_parameters(params, sites, groups, anchor, opts.domain)
            except (ValueError, TypeError, RuntimeError) as exc:
                rec.setdefault("seed_skipped", []).append({
                    "label": label, "params": params, "why": f"outside sampler domain: {exc}"})
                return None
            return params

        core.fill_seeds(rec, sites, points, checked_params=checked_params)
    queued = core.next_seed(rec, sites, scope, flatten)
    if queued:
        if rec.get("seed_skipped"):
            queued[1]["seed_skipped"] = list(rec["seed_skipped"])
        return queued
    ctx = {"rec": rec, "sites": sites, "anchor": anchor, "groups": groups, "domain": opts.domain}
    study = core.replay_study(rec, opts.seed, lambda name, held: dist(name, sites, groups, anchor, held, opts.domain),
                              lambda params: policy.shape_params(params, sites))
    return core.draw_until_feasible(study, ctx, scope, {s.id for s in sites}, hw, opts,
                                    draw_one_site, _gate, lambda trial, cfg: flatten(cfg, sites))


def tell(state_bo, catalog_id, src, s, p, extra_consts=None):
    rec = state_bo.get(lever_key(catalog_id, src, extra_consts))
    if not rec or not rec.get("pending"):
        return False
    sites = _scoped(policy.shape_sites(apply.tunable_sites(src, extra_consts)), SCOPE_OF.get(catalog_id))
    groups = policy.shared_shape_groups(sites, rec.get("shape_groups"))
    actual = space.flatten(apply.current_config(src, extra_consts), sites, groups)
    if actual != rec["pending"].get("params"):
        return False
    return core.record_pending(rec, s, p)


def _validate_seed_parameters(params, sites, groups, anchor, domain):
    """Reject seeds outside the sampler's per-parameter distributions."""
    for name, value in params.items():
        dist(name, sites, groups, anchor, domain=domain).to_internal_repr(value)
