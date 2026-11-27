# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The numeric parameters owned by TPE, shared by both search entry points."""

SHAPE_KINDS = ("vec",)
PARAMETER_SCOPE = "tile_shapes_v3"


def shape_sites(sites):
    """Pro uses kind ``vec`` for shapes in every on-chip memory space."""
    return [site for site in sites if site.kind in SHAPE_KINDS]


def shape_config(config, sites):
    """Drop agent-owned dimensions from an externally supplied seed or anchor."""
    allowed = {site.id for site in shape_sites(sites)}
    result = {}
    for sid, value in (config or {}).items():
        if sid in allowed and isinstance(value, dict):
            result[sid] = {field: number for field, number in value.items()
                           if field.startswith("d") and field[1:].isdigit()}
    return result


def shape_params(params, sites):
    """Whether a persisted observation contains only this search's parameters."""
    known = {site.id for site in shape_sites(sites)}

    def owned(name):
        sid, separator, field = name.rpartition(".")
        return bool(separator) and sid in known and field.startswith("d") and field[1:].isdigit()

    return bool(params) and all(map(owned, params))


def group_roots(groups, order):
    """Map each known site to the first representative retained by its linked groups."""
    parents = {sid: sid for sid in order}

    def root(sid):
        while parents[sid] != sid:
            sid = parents[sid]
        return sid

    for group in groups:
        members = [sid for sid in group if sid in parents]
        for sid in members[1:]:
            parents[root(sid)] = root(members[0])
    return {sid: root(sid) for sid in parents}


def components(groups, order):
    rows = {}
    roots = group_roots(groups, order)
    for sid in order:
        rows.setdefault(roots[sid], []).append(sid)
    return [members for members in rows.values() if len(members) > 1]


def shared_shape_groups(sites, groups=None, domain=None):
    """Bind all users of a writable source shape span to one coordinate tuple."""
    sites = shape_sites(sites)
    spans = {}
    for site in sites:
        start, end = getattr(site, "start", None), getattr(site, "end", None)
        if isinstance(start, int) and isinstance(end, int) and end > start:
            spans.setdefault((site.kind, start, end), []).append(site.id)
    # An offset-free, non-converting move between equal full shapes imposes
    # equality on the *tile parameters*. The move instruction is not a knob.
    # Do not equate ratio locks, partial axes, or mode-converting transfers.
    kinds = {site.id: site.kind for site in sites}
    locks = domain.get("move_locks", ()) if (domain or {}).get("enforce_move_locks") else ()
    equal = [lock["pair"] for lock in locks if _equal_vector_move(lock, kinds)]
    return components(list(groups or []) + list(spans.values()) + equal, [site.id for site in sites])


def expanded_ties(sites, tie, groups):
    """A leading-axis tie reaches every user of a shared full shape."""
    return components(list(groups or []) + list(tie or []), [site.id for site in shape_sites(sites)])


def _equal_vector_move(lock, kinds):
    pair, shapes = lock.get("pair", ()), lock.get("author", ())
    if lock.get("mode") or len(pair) != 2 or len(shapes) != 2:
        return False
    if not all(kinds.get(sid) == "vec" for sid in pair):
        return False
    return bool(shapes[0]) and shapes[0] == shapes[1] and all(type(n) is int and n > 0 for n in shapes[0])
