# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Construct Pro tile shape domains and check candidate placement against the supplied chip envelope."""
import logging
from dataclasses import dataclass
from dataclasses import replace
from operator import itemgetter
from typing import Any, Dict, Mapping, Optional

from panko_core.bayesian_optimization import space as core_space
from panko_core.bayesian_optimization.space import _lead_key, _lower, _rep, global_knob

from memory import onchip_layout
from source_analysis import tile_sites
from .policy import group_roots

logger = logging.getLogger(__name__)

# Sampler rungs use vector granularity and geometric steps; admit also includes the incumbent.
SHAPE_LADDER = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512,
                768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]

# Agent-owned settings supported by source editing and launch validation.
JIT_KNOBS = ("preload", "auto_mutex")
# Launch overrides go through measurement, never through kernel source edits.
# Profile block counts must agree with the requested override.
LAUNCH_KNOBS = ("block_dim",)


# Map each per-core capacity to its Pro memory space.
HW_FIELD_SPACE = (
    ("ub_budget_kb", "Vec"),
    ("l1_budget_kb", "Mat"),
    ("l0a_kb", "Left"),
    ("l0b_kb", "Right"),
    ("l0c_kb", "Acc"),
)


@dataclass
class HW(core_space.HW):
    """Check placement of known tiles even when their memory space has no capacity limit."""
    m_l1: Optional[int] = 64
    k_l1: Optional[int] = 64
    l1_budget_kb: Optional[int] = None
    ub_budget_kb: Optional[int] = None
    l0a_kb: Optional[int] = None
    l0b_kb: Optional[int] = None
    l0c_kb: Optional[int] = None
    # A zero core count means that the platform could not report that kind of core.
    cube_cores: int = 0
    vector_cores: int = 0
    sites: Any = ()                # tile_sites.TileSite tuple
    author_sites: Any = None       # tile_sites.TileSite tuple, or None
    extra_consts: Any = None       # names read_sites cannot resolve from the text
    alignment: int = onchip_layout.DEFAULT_ALIGNMENT
    width: Any = None              # dtype TEXT -> bytes; None = onchip_layout's table


    def budget(self) -> Dict[str, Optional[int]]:
        """Return per-space byte ceilings. None leaves a space ungated; zero gives it no capacity."""
        out: Dict[str, Optional[int]] = {}
        for field_name, space in HW_FIELD_SPACE:
            kb = getattr(self, field_name, None)
            out[space] = int(kb) * 1024 if kb else None
        return out

    def width_fn(self):
        """dtype TEXT -> bytes. `onchip_layout`'s one table unless overridden."""
        return self.width or onchip_layout.width_of_text


def _hw_for(hw, sid=None):
    """Choose the program envelope. For a per-site mapping, use its first value."""
    if not isinstance(hw, dict):
        return hw
    if sid is not None and sid in hw:
        return hw[sid]
    for value in hw.values():
        return value
    return HW()


def _tie_rep(sites, tie, groups=None):
    """Map each site to the representative of its leading extent. Unlisted sites represent
    themselves."""
    rep = _rep(sites, groups)
    links = ([rep[sid] for sid in group if sid in rep] for group in tie or ())
    roots = group_roots(links, rep.values())
    return {sid: roots[head] for sid, head in rep.items()}


def admit(base, val):
    """Include the incumbent value in a categorical domain."""
    return core_space.admit(base, val, key=_rung_key)


def _rung_key(value):
    """Order numeric rungs by value and other rungs by text for stable Optuna distributions."""
    if isinstance(value, bool):
        return (1, 0.0, str(value))
    if isinstance(value, (int, float)):
        return (0, float(value), "")
    return (2, 0.0, str(value))


def vec_fields(site_config):
    """Return the shape-axis field names for a site, in axis order."""
    fs = sorted((k for k in site_config if k.startswith("d") and k[1:].isdigit()),
                key=lambda k: int(k[1:]))
    return fs or ["d0", "d1"]


def vec_domain(site_id, field, fields, anchor_cfg, domain=None):
    """Use the derived axis extents when bounded, otherwise the default ladder."""
    from . import domain as domain_mod
    cap = domain_mod.vec_ladder_cap(domain, site_id, field, fields)
    ladder = core_space.capped_ladder(domain, site_id, field, SHAPE_LADDER, cap)
    floor = domain_mod.valid_floor(domain, site_id, field, fields)
    if floor:
        pruned = [v for v in ladder if v >= floor]
        if pruned:                 # never empty the ladder
            ladder = pruned
    # Apply the scoped row-stride requirement only to eligible trailing axes.
    # Body-consumed windows may pin an axis to the incumbent; keep that legal
    # anchor instead of manufacturing an empty ladder.
    if domain_mod.pinned_axis(domain, site_id, field, fields):
        return admit([], (anchor_cfg or {}).get(field)) or ladder
    step = domain_mod.cols_block_multiple(domain, site_id, field, fields)
    if step:
        pruned = [v for v in ladder if v % step == 0]
        if pruned:                 # never empty the ladder
            ladder = pruned
    return admit(ladder, (anchor_cfg or {}).get(field))


def grouped_domains(groups, sites, anchor, domain, tie=None):
    """Intersect ladders for shared shape expressions, retaining incumbents for replay."""
    rep = _rep(sites, groups)
    lead = _tie_rep(sites, tie, groups)
    choices, anchors = {}, {}
    for site in sites:
        if site.kind != "vec":
            continue
        a = (anchor or {}).get(site.id) or {}
        fields = vec_fields(a)
        for field_name in fields:
            key = _lead_key(lead, rep[site.id], field_name, fields)
            values = set(vec_domain(site.id, field_name, fields, a, domain))
            choices[key] = choices[key] & values if key in choices else values
            if a.get(field_name) is not None:
                anchors.setdefault(key, set()).add(a[field_name])
    return {key: sorted(values | anchors.get(key, set()), key=_rung_key)
            for key, values in choices.items()}


@dataclass
class DrawOptions(core_space.DrawOptions):
    hw: Any = None
    move: Any = None


def resolve_move(sites, opts):
    """Expand movable sites through groups and ties. Sites without an anchor are always movable."""
    if opts.move is None:
        return None
    rep = _rep(sites, opts.groups)
    lead = _tie_rep(sites, opts.tie, opts.groups)
    wanted = set(opts.move)
    keys = {rep[s.id] for s in sites if s.id in wanted}
    leads = {lead[s.id] for s in sites if s.id in wanted}
    anchor = opts.anchor or {}
    return {s.id for s in sites if rep[s.id] in keys or lead[s.id] in leads or (not anchor.get(s.id))}


def clusters(sites, groups=None, tie=None):
    """Group site IDs that must move together. The block budget and the sampler use the same groups."""
    rep = _rep(sites, groups)
    lead = _tie_rep(sites, tie, groups)
    buckets = {}
    for s in sites:
        # Tied sites share their leading extent and may vary other axes independently.
        key = lead[s.id] if lead[s.id] != s.id else rep[s.id]
        buckets.setdefault(key, []).append(s.id)
    return [sorted(v) for _, v in sorted(buckets.items())]


def _draw_vec(trial, r, a, ctx):
    """Sample each shape axis. Sites tied on the leading axis share its sampler parameter."""
    lead = ctx["lead"]
    fields = vec_fields(a)
    return {f: trial.suggest_categorical(
        _lead_key(lead, r, f, fields),
        ctx["shape_domains"][_lead_key(lead, r, f, fields)])
        for f in fields}


def _draw_site(trial, site, r, ctx):
    """Draw tile dimensions; agent-owned scalar settings are not TPE parameters."""
    opts = ctx["opts"]
    a = (opts.anchor or {}).get(r, {})
    if site.kind == "vec":
        return _draw_vec(trial, r, a, ctx)
    raise ValueError(f"unsupported PyPTO-Pro site kind: {site.kind}")


def _apply_tie(cfg, tie):
    core_space.apply_tie(cfg, tie, vec_fields)


def suggest(trial, sites, opts=None):
    """Draw movable sites subject to shared dimensions and keep the rest at their incumbent values."""
    opts = opts or DrawOptions()
    rep = _rep(sites, opts.groups)
    ctx = {"opts": opts, "lead": _tie_rep(sites, opts.tie, opts.groups), "hw": opts.hw,
           "shape_domains": grouped_domains(opts.groups, sites, opts.anchor, opts.domain, opts.tie)}
    movable = resolve_move(sites, opts)
    cfg = core_space.draw_grouped(trial, sites, rep, ctx, _draw_site,
                                  movable=movable, anchor=opts.anchor or {})
    # Apply ties only to movable sites.
    _apply_tie(cfg, [[sid for sid in grp if movable is None or sid in movable]
                     for grp in (opts.tie or [])])
    return cfg


def flatten(site_configs, sites, groups=None, tie=None):
    return core_space.encode_sites(site_configs, sites, _rep(sites, groups),
                                   _tie_rep(sites, tie, groups), vec_fields, partial=True)


def unflatten(params, sites, groups=None, tie=None):
    return core_space.decode_sites(params, sites, _rep(sites, groups), _tie_rep(sites, tie, groups))


def _value_sane(sid, c):
    """Validate numeric extents and depths before checking chip capacity."""
    if not isinstance(c, Mapping):
        return True, ""
    for key, value in c.items():
        if key == "value":
            # A scalar site: preload may be zero; a depth may not.
            valid_depth = isinstance(value, int) and not isinstance(value, bool) and value >= 1
            if sid.startswith("depth:") and not valid_depth:
                return False, f"depth must be an int >= 1, got {value!r}"
            continue
        if key.startswith("d") and key[1:].isdigit():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                return False, f"a tile extent must be an int >= 1, got {key}={value!r}"
    return True, ""


def _shaped(site, cand):
    """Return this site's candidate shape, falling back to its declared extents."""
    out = []
    cfg = (cand or {}).get(site.site_id) or {}
    for axis, entry in enumerate(site.shape):
        value = cfg.get(f"d{axis}", entry.value)
        out.append(int(value) if value is not None else None)
    return tuple(out)


def _depth_of(site, cand):
    drawn = ((cand or {}).get(f"depth:{site.site_id}") or {}).get("value")
    if isinstance(drawn, int) and not isinstance(drawn, bool) and drawn >= 1:
        return drawn
    depth = site.depth
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise onchip_layout.LayoutInvalid(f"{site.site_id}: buffer depth is unresolved")
    return depth


def author_owned(sites):
    from .apply import addrs_ints
    return onchip_layout.author_owned(sites, addrs_of=addrs_ints)


def _candidate_allocation(site, cand):
    """Resolve every retained allocation before it can contribute to a layout."""
    from . import apply as apply_mod
    space = site.space
    if space is None:
        raise onchip_layout.LayoutInvalid(f"{site.site_id}: allocation memory space is unresolved")
    shape = _shaped(site, cand)
    if not shape or any(d is None for d in shape):
        raise onchip_layout.LayoutInvalid(f"{site.site_id}: allocation shape is unresolved")
    depth = _depth_of(site, cand)
    addresses = apply_mod.addrs_ints(site)
    if not addresses or any(a is None for a in addresses):
        raise onchip_layout.LayoutInvalid(f"{site.site_id}: allocation addresses are unresolved")
    return shape, onchip_layout.dtype_of(site), depth


def regions_for(cand, sites, author_sites=None):
    """Strictly resolve on-chip candidates and preserve their original address ownership."""
    from .apply import addrs_ints
    sites = [site for site in sites if site.space not in tile_sites.OFF_CHIP]
    regions, _ = onchip_layout.regions_from_sites(
        sites, author_sites, dimensions=lambda site: _candidate_allocation(site, cand),
        addrs_of=addrs_ints, pinned_base=itemgetter(0))
    return regions


def _sizes_unchanged(cand, sites):
    """Compare resolved tile extents with the author's sizes, including explicit redraws of the
    incumbent."""
    for site in sites:
        drawn = _shaped(site, cand)
        if not drawn or any(d is None for d in drawn):
            continue
        written = [e.value for e in site.shape]
        if any(v is None for v in written):
            continue
        if list(drawn) != [int(v) for v in written]:
            return False
        if _depth_of(site, cand) != (site.depth or 1):
            return False
    return True


def as_written(allocation_regions, sites):
    """Return regions pinned to declared addresses, or None if any address is unresolved."""
    from . import apply as apply_mod
    by_id = {s.site_id: s for s in sites}
    out = []
    for r in allocation_regions:
        if r.unchecked or r.alias_of is not None:
            out.append(r)
            continue
        site = by_id.get(r.name)
        if site is None:
            return None
        written = apply_mod.addrs_ints(site)
        base = written[0] if written else None
        if base is None:
            return None
        out.append(replace(r, fixed_addr=base))
    return out


def layout_for(cand, hw, keep_author_map=None):
    """Preserve author-owned placements and repack eligible resized tiles."""
    sites = getattr(hw, "sites", ()) or ()
    author_sites = getattr(hw, "author_sites", None)
    groups = onchip_layout.allocation_groups(sites)
    if len(groups) > 1:
        layouts = []
        for group in groups:
            ids = {s.site_id for s in group}
            author_group = (tuple(s for s in author_sites if s.site_id in ids)
                            if author_sites is not None else None)
            layouts.append(layout_for(cand, replace(hw, sites=group,
                                      author_sites=author_group), keep_author_map))
        return onchip_layout.combine_layouts(layouts)
    allocation_regions = regions_for(cand, sites, author_sites)
    if keep_author_map is None:
        keep_author_map = _sizes_unchanged(cand, sites)
    return place_regions(allocation_regions, sites, keep_author_map, hw.budget(),
                         alignment=hw.alignment, width=hw.width_fn())


def place_regions(regions, sites, keep_author_map, budget, *, width,
                  alignment=onchip_layout.DEFAULT_ALIGNMENT):
    """Try the authored placement when requested, then repack within the same bounds."""
    pinned = as_written(regions, sites) if keep_author_map else None
    if pinned is not None:
        try:
            return onchip_layout.plan(pinned, budget, alignment=alignment, width=width)
        except (onchip_layout.LayoutOverlap, onchip_layout.LayoutRejected,
                onchip_layout.LayoutInvalid) as exc:
            logger.debug("repacking an invalid authored placement: %s", exc)
    return onchip_layout.plan(regions, budget, alignment=alignment, width=width)


def author_sites_of(src, consts=None):
    """Read baseline sites with the case constants. An unreadable baseline yields no author reference."""
    if not src:
        return None
    try:
        return tile_sites.read_sites(src, consts or {})
    except (SyntaxError, ValueError):
        return None


def _blame(space, cand, sites):
    """Name the largest site in the memory space that exceeded capacity."""
    widest, blamed = -1, None
    for site in sites:
        if (site.space or "Vec") != space:
            continue
        shape = _shaped(site, cand)
        if any(d is None for d in shape):
            continue
        n = 1
        for d in shape:
            n *= d
        if n > widest:
            widest, blamed = n, site.site_id
    return blamed or space


def static_feasible(site_configs, hw):
    """Check shape dependencies and memory placement before measurement. Return (ok, reason)."""
    for sid, c in (site_configs or {}).items():
        ok, reason = _value_sane(sid, c)
        if not ok:
            return False, f"{sid}: {reason}"

    hw_one = _hw_for(hw)
    sites = getattr(hw_one, "sites", ()) or ()
    if not sites:
        return True, ""

    try:
        layout_for(site_configs, hw_one)
    except onchip_layout.LayoutRejected as exc:
        return False, "%s: %s" % (_blame(exc.space, site_configs, sites), exc)
    except onchip_layout.LayoutOverlap as exc:
        # An overlap from the planner is an internal error. Attribute it to the first site.
        return False, "%s: LayoutOverlap: %s" % (
            getattr(sites[0], "site_id", "layout"), exc)
    except onchip_layout.LayoutInvalid as exc:
        return False, "%s: LayoutInvalid: %s" % (
            getattr(sites[0], "site_id", "layout"), exc)
    return True, ""


# Repair a capacity failure by shrinking one extent at a time to the first feasible neighbor.
REPAIR_MAX_STEPS = 64


def is_capacity(why):
    """Return whether shrinking a tile may resolve the reported capacity error. Invalid
    dtypes and overlapping regions require separate handling.
    """
    if " needs " in why and " budget is " in why:
        return True
    return core_space.is_capacity(why)


def shrinkable_sites(cur, anchor, domain, movable=None):
    """Return movable axes with lower legal rungs, including alternatives to a site at its floor."""
    out = []
    for sid, c in cur.items():
        if movable is not None and sid not in movable:
            continue
        if not isinstance(c, Mapping) or "value" in c:
            continue
        fields = vec_fields(c)
        for f in fields:
            v = c.get(f)
            if not isinstance(v, int) or isinstance(v, bool):
                continue
            rungs = vec_domain(sid, f, fields, (anchor or {}).get(sid) or {}, domain)
            nv = _lower(rungs, v)
            if nv is not None:
                out.append((v, sid, f, nv))
    return out


def _linked_sites(links):
    roots = group_roots(links, (sid for group in links for sid in group))
    members = {}
    for sid, head in roots.items():
        members.setdefault(head, set()).add(sid)
    return {sid: members[head] for sid, head in roots.items()}


def _shared_repair_rungs(cur, members, field_name, anchor, domain):
    """Intersect legal rungs so every tied member accepts the same repair."""
    shared = None
    for g in members:
        rungs = set(vec_domain(g, field_name, vec_fields(cur[g]),
                               (anchor or {}).get(g) or {}, domain))
        shared = rungs if shared is None else shared & rungs
    return shared


def _repair_space_sites(blamed, why, hw):
    """Capacity repair can free bytes only in the reported memory space."""
    if hw is None or why.startswith("largest tile"):
        return None  # The learned ceiling is global, not a space capacity.
    sites = getattr(_hw_for(hw), "sites", ()) or ()
    spaces = {s.site_id: s.space for s in sites}
    space_name = why.split(" needs ", 1)[0] if " needs " in why else spaces.get(blamed)
    return ({sid for sid, space in spaces.items() if space == space_name}
            if space_name in spaces.values() else None)


def repair(cfg, check, anchor=None, domain=None, tie=None, max_steps=REPAIR_MAX_STEPS,
           movable=None, groups=None, hw=None):
    """Return (configuration, steps) after legal shrinking, or (None, steps) if repair fails."""
    cur = {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in cfg.items()}
    groups = list(groups or ())
    linked = _linked_sites(groups)
    leading = _linked_sites(groups + list(tie)) if tie else linked
    for step in range(max_steps + 1):
        ok, reason = check(cur)
        if ok:
            return cur, step
        blamed, _, why = (reason or "").partition(": ")
        if not is_capacity(why):
            return None, step
        affected = _repair_space_sites(blamed, why, hw)
        moves = []
        for value, sid, field_name, nv in shrinkable_sites(cur, anchor, domain, movable):
            members = (leading if field_name == "d0" else linked).get(sid, {sid})
            if affected is not None and not members & affected:
                continue
            if any(g not in cur or field_name not in cur[g]
                   or (movable is not None and g not in movable) for g in members):
                continue
            shared = _shared_repair_rungs(cur, members, field_name, anchor, domain)
            nv = _lower(shared, value)
            if nv is not None:
                moves.append((value, sid, field_name, nv, members))
        if not moves:
            return None, step
        _, sid, field_name, nv, members = max(moves, key=lambda m: (m[0], m[1], m[2]))
        for g in members:
            cur[g][field_name] = nv
    return None, max_steps
