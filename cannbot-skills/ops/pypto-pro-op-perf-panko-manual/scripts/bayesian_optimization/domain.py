# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Derive legal Pro tile extents, move dependencies and the fingerprint that determines when to retune."""
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Mapping

from memory import onchip_layout
from source_analysis import tile_sites

logger = logging.getLogger(__name__)

DTYPE_BYTES = onchip_layout.DTYPE_BYTES

# Zero disables the residency bound when live tile count is unknown.
DEFAULT_LIVE_TILES = 0

ENFORCE_MOVE_LOCKS = True


def resolve_extra_consts(src, op_file=None, extra_consts=None):
    """Resolve integer constants imported from sibling modules through the source reader."""
    from source_analysis.case_context import validate_context
    validate_context(extra_consts)
    resolved = tile_sites.sibling_constants(src, op_file)
    resolved.update(extra_consts or {})
    return resolved


def _width(dtype_text):
    return onchip_layout.width_of_text(dtype_text)


def site_report(sites):
    """Per-site facts the rest of the domain is computed from."""
    out = {}
    for site in sites:
        out[site.site_id] = {
            "space": site.space,
            "dtype": onchip_layout.dtype_of(site),
            "dtype_bytes": _width(onchip_layout.dtype_of(site)) or None,
            "rank": len(site.shape),
            "shape": [e.value for e in site.shape],
            "depth": site.depth,
            "depth_source": site.depth_source,
            "depth_tunable": False,  # Buffer count is fixed during shape TPE.
            "depth_editable": site.depth_span is not None,
            "valid_shape": list(getattr(site, "valid_shape", ()) or ()),
            "addrs": site.addrs_text,
            "resolvable": bool(site.resolvable),
            "unresolved": list(site.unresolved),
        }
    return out


def _site_widths(report, notes):
    """Resolve known widths and report sites that cannot be sized."""
    widths = {sid: cell["dtype_bytes"] for sid, cell in report.items() if cell["dtype_bytes"]}
    unknown = sorted(sid for sid, cell in report.items() if not cell["dtype_bytes"])
    if unknown:
        notes.append(
            "no element width for %s: the dtype is written as an expression, so "
            "these tiles cannot be sized and are refused rather than guessed at"
            % (", ".join(unknown),))

    return widths


def _unreferenced_sites(src, sites, notes):
    """Report unused allocations without removing their reserved storage."""
    dead = tile_sites.unreferenced(src, sites) if sites else ()
    if dead:
        notes.append(
            "%s allocate on-chip space that nothing reads: their tile group is "
            "never mentioned again. The byte objective pays for them in full and "
            "the latency objective gains nothing. Reported, not removed -- a "
            "declaration may be read through an alias this cannot see, or exist "
            "to reserve an address another group's literal depends on"
            % (", ".join(dead),))

    return dead


def _body_constraints(program, notes):
    """Read consumer windows, retaining declaration-only bounds on failure."""
    try:
        body = program.body_constraints() if program and program.active_sites else {}
    except (SyntaxError, ValueError) as exc:
        body = {}
        notes.append("the kernel body could not be read for consume windows "
                     "(%s), so only the `valid_shape=` kwargs bound the "
                     "extents" % (exc,))
    return body


def _merge_floors(report, body_windows):
    """Combine declared and consumed windows into one floor per axis."""
    floors = {sid: list(cell["valid_shape"]) for sid, cell in report.items()
              if any(v is not None for v in cell["valid_shape"])}
    # Keep the largest declared window on each axis.
    for sid, window in body_windows.items():
        tile_sites.merge_floors(floors.setdefault(sid, []), window)
    floors = {sid: v for sid, v in floors.items() if any(x is not None for x in v)}

    return floors


def _pinned_axes(report, floors, notes):
    """Find axes whose consumed window fixes the incumbent extent."""
    # Pin the axis when its consumer requires the entire incumbent extent.
    pinned = {}
    for sid, cell in report.items():
        cur = cell.get("shape") or []
        window = floors.get(sid) or []
        axes = [i for i, v in enumerate(window) if v is not None and i < len(cur) and cur[i] == v]
        if axes:
            pinned[sid] = axes
    if pinned:
        notes.append(
            "%d of %d sites are PINNED on at least one axis -- the window the "
            "body consumes equals the extent the kernel runs (%s), so every "
            "other rung on that axis breaks the consumer and the axis is drawn "
            "at its current value only"
            % (len(pinned), len(report),
               listed("%s.d%s" % (sid, "/d".join(str(i) for i in axes))
                      for sid, axes in sorted(pinned.items()))))
    return pinned


def _derive_move_locks(report, moves, body):
    """Attach author shapes and transfer direction to offset-less move pairs."""
    # Store declared extents so diagnostics and verification derive the same move ratio.
    move_locks = []
    detail = {tuple(d["pair"]): d for d in (body.get("move_detail") or ())}
    for pair in moves:
        shapes = [list((report.get(sid) or {}).get("shape") or ()) for sid in pair]
        if len(pair) != 2 or not all(shapes):
            continue
        d = detail.get(tuple(pair)) or {}
        move_locks.append({"pair": list(pair), "author": shapes,
                           # Sorted pair keys lose transfer direction. Preserve it for asymmetric constraints.
                           "src": d.get("src"), "dst": d.get("dst"),
                           "mode": d.get("mode")})
    return move_locks


def _note_move_locks(moves, move_locks, notes):
    """Describe enforced ratios and unresolved pairs without omitting either."""
    if moves:
        rated = [(p, move_lock_ratio(p["author"][0], p["author"][1])) for p in move_locks]
        held = [(p, r) for p, r in rated if r]
        blind = ([p["pair"] for p, r in rated if not r]
                 + [list(m) for m in moves
                    if not any(list(m) == p["pair"] for p in move_locks)])
        notes.append(
            "%d `pl.move` pairs carry no offset, so their two tiles are locked "
            "to the ratio the kernel runs them at: %s.%s %s"
            % (len(moves),
               listed("%s=%s %d:%d" % (p["pair"][0], p["pair"][1], n, d)
                      for p, r in held for i, n, d in r[:1]) or "none resolved",
               (" %d of them (%s) read an extent this derivation cannot "
                "resolve, so there is no ratio to hold and they are not "
                "checked."
                % (len(blind), listed("%s=%s" % tuple(b) for b in blind)))
               if blind else "",
               "ENFORCED: a draw that breaks one is refused before the build"
               if ENFORCE_MOVE_LOCKS else
               "Derived but NOT enforced, so a draw that breaks one still "
               "reaches the device and is a build spent to be told `pl.move: "
               "dst != src`"))


def _note_layout_constraints(sites, floors, notes):
    """Describe window floors and the sites subject to column alignment."""
    if floors:
        notes.append(
            "%s declare a `valid_shape=` window, so their extents have a FLOOR: "
            "the window has to fit inside the tile. Rungs below it are pruned "
            "statically rather than drawn and refused by the parser"
            % (", ".join(sorted(floors)),))

    exempt = sorted(site.site_id for site in sites if not cols_block_binds(site.layout, bool(site.fractal), site.space))
    if sites:
        notes.append(
            "%d of %d sites are bound by `RowMajor + NoneBox => Cols * sizeof(T) "
            "%% 32 == 0` (pto_tile.hpp:1444), so their TRAILING extent is drawn "
            "from multiples of 32 bytes%s"
            % (len(sites) - len(exempt), len(sites),
               "; exempt: %s" % (", ".join(exempt),) if exempt else ""))


def _derive_constraints(program, sites, report, notes):
    """Derive shape relations and retain their source evidence together."""
    body = _body_constraints(program, notes)
    windows = body.get("windows") or {}
    moves = [list(pair) for pair in (body.get("moves") or [])]
    floors = _merge_floors(report, windows)
    pinned = _pinned_axes(report, floors, notes)
    unresolved = body.get("unresolved_window_axes") or {}
    for sid, axes in unresolved.items():
        pinned[sid] = sorted(set(pinned.get(sid, [])) | set(axes))
    if unresolved:
        notes.append(
            "unresolved runtime valid-shape axes retain their current extents: %s. "
            "Resolve the consumer bounds from case evidence or change the tile and "
            "its consumers together as a structural edit; correctness alone does "
            "not prove that a dynamic window fits the declared allocation."
            % listed("%s.d%s" % (sid, "/d".join(map(str, axes)))
                     for sid, axes in sorted(unresolved.items())))
    locks = _derive_move_locks(report, moves, body)
    _note_move_locks(moves, locks, notes)
    _note_layout_constraints(sites, floors, notes)
    return {"valid_floor": floors, "pinned_axes": pinned, "body_windows": windows, "unresolved_window_axes": unresolved,
            "move_ties": moves, "move_locks": locks, "enforce_move_locks": ENFORCE_MOVE_LOCKS,
            "layout": {site.site_id: (site.layout, bool(site.fractal)) for site in sites}}


def derive(op_file, live_tiles=None, ub_budget_kb=192, extra_consts=None):
    """Derive per-site bounds from code, case constants and Vec capacity. Dynamic extents have unknown capacity."""
    _, domain = analyze_file(op_file, live_tiles, ub_budget_kb, extra_consts)
    return domain


def analyze_file(op_file, live_tiles, ub_budget_kb, extra_consts):
    """Keep the declarations that produced the public search domain."""
    src = Path(op_file).read_text(encoding="utf-8", errors="replace") if op_file and os.path.exists(op_file) else ""
    notes = []
    extra = resolve_extra_consts(src, op_file, extra_consts)
    try:
        program = tile_sites.TileProgram(src, extra) if src else None
    except SyntaxError as exc:
        notes.append("kernel does not parse: %s" % (exc,))
        program = None
    sites = program.active_sites if program else ()
    report = site_report(sites)
    widths = _site_widths(report, notes)
    trailing = {sid: cell["shape"][-1] for sid, cell in report.items()
                if cell["shape"] and cell["shape"][-1] is not None}
    dead = _unreferenced_sites(src, sites, notes)
    constraints = _derive_constraints(program, sites, report, notes)
    live = DEFAULT_LIVE_TILES if live_tiles in (None, "") else int(live_tiles)
    ub_bytes = int(ub_budget_kb) * 1024 if ub_budget_kb else None
    if ub_bytes is None:
        notes.append("no Vec ceiling was supplied, so no ladder cap is derived; "
                     "the allocator is the only bound on a tile extent")
    return program, {"sites": report, "site_ids": [site.site_id for site in sites], "per_site": {},
                     "dtype_bytes": min(widths.values()) if widths else 4, "vec_dtype_bytes": widths,
                     "vec_trailing": trailing, **constraints, "unreferenced": list(dead),
                     "vec_residency": {sid: live for sid in report} if live else {}, "live_tiles": live,
                     "ub_bytes": ub_bytes, "capacity_elems": None, "extra_consts": extra, "notes": notes}


def vec_ladder_cap(domain, site_id, field, fields):
    """Return the axis limit with other extents fixed, or None for an unknown bound."""
    if not domain:
        return None
    cell = (domain.get("sites") or {}).get(site_id)
    if not cell or cell.get("space") != "Vec":
        return None
    budget = domain.get("ub_bytes")
    width = cell.get("dtype_bytes")
    shape = cell.get("shape") or []
    if not budget or not width or (not shape):
        return None
    if any((d is None for d in shape)):
        return None
    try:
        axis = list(fields).index(field)
    except ValueError:
        return None
    if axis >= len(shape):
        return None
    others = 1
    for i, extent in enumerate(shape):
        if i != axis:
            others *= max(1, int(extent))
    depth = max(1, int(cell.get("depth") or 1))
    residency = max(1, int((domain.get("vec_residency") or {}).get(site_id) or 1))
    per_element = width * others * depth * residency
    if per_element <= 0:
        return None
    cap = budget // per_element
    return cap if cap >= 1 else None


#: Maximum entries listed before a diagnostic explicitly marks truncation.
NOTE_LIST_MAX = 8


def listed(items, total=None):
    """Join diagnostic items and explicitly mark any truncation."""
    items = list(items)
    total = len(items) if total is None else total
    shown = items[:NOTE_LIST_MAX]
    text = ", ".join(shown)
    if total > len(shown):
        text += ", ... (%d of %d shown)" % (len(shown), total)
    return text


COLS_BLOCK_BYTES = 32


def cols_block_binds(layout, fractal, space):
    """Check whether the RowMajor, non-fractal tile must have 32-byte-aligned columns."""
    if fractal:
        return False
    if layout:
        return layout == "RowMajor"
    return (space or "Vec") == "Vec"


def cols_block_scope(domain, site_id):
    """Does the column-alignment predicate bind this site? See `cols_block_binds`."""
    layout, fractal = ((domain or {}).get("layout") or {}).get(site_id, ("", False))
    site = ((domain or {}).get("sites") or {}).get(site_id) or {}
    return cols_block_binds(layout, fractal, site.get("space"))


def cols_block_multiple(domain, site_id, field, fields):
    """
    Return the trailing-extent alignment multiple, or None when unbound.

    RowMajor with NoneBox requires Cols * sizeof(T) to be a multiple of 32 bytes.
    Cols is the last shape axis, so the element multiple is 32 // gcd(32, width).
    The rule is scoped by cols_block_scope and stays unset for unknown widths.
    space.admit retains the incumbent extent even when it is outside this grid.
    """
    if not domain or not cols_block_scope(domain, site_id):
        return None
    flds = list(fields or [])
    if not flds or field != flds[-1]:
        return None
    width = ((domain.get("vec_dtype_bytes") or {}).get(site_id) or domain.get("dtype_bytes"))
    if not width:
        return None
    step = COLS_BLOCK_BYTES // math.gcd(COLS_BLOCK_BYTES, int(width))
    return step if step > 1 else None


def pinned_axis(domain, site_id, field, fields):
    """Whether the consumed window constrains this axis to its incumbent extent."""
    if not domain:
        return False
    axes = (domain.get("pinned_axes") or {}).get(site_id)
    if not axes:
        return False
    try:
        return list(fields).index(field) in axes
    except ValueError:
        return False


def valid_floor(domain, site_id, field, fields):
    """Return the valid_shape floor. Missing, unresolved and -1 entries leave the axis free."""
    if not domain:
        return None
    floors = (domain.get("valid_floor") or {}).get(site_id)
    if not floors:
        return None
    try:
        axis = list(fields).index(field)
    except ValueError:
        return None
    if axis >= len(floors):
        return None
    value = floors[axis]
    return int(value) if isinstance(value, int) and value > 1 else None


def live_for(domain, site_id):
    """Return the simultaneous live tile count, or zero if unknown."""
    if not domain:
        return 0
    per_site = domain.get("vec_residency") or {}
    if site_id in per_site:
        return int(per_site[site_id] or 0)
    return int(domain.get("live_tiles") or 0)


def _tile_extent_ok(domain, sid, cfg, field_name, value):
    """Check one integer extent against its explicit domain and consume floor."""
    per_site = (domain or {}).get("per_site") or {}
    allowed = per_site.get(sid) or {}
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        return False, ("%s: a tile extent must be an int >= 1, got %s=%r" % (sid, field_name, value))
    axis = list(_fields_of(cfg)).index(field_name)
    if axis in ((domain or {}).get("unresolved_window_axes") or {}).get(sid, []):
        shape = ((domain.get("sites") or {}).get(sid) or {}).get("shape") or []
        if axis >= len(shape) or value != shape[axis]:
            return False, ("%s: %s cannot change while its runtime valid-shape "
                           "window is unresolved" % (sid, field_name))
    choices = allowed.get(field_name)
    if choices and value not in choices:
        return False, ("%s: %s=%s is outside the derived domain %s" % (sid, field_name, value, sorted(choices)))
    # Include the declared window name in the refusal.
    floor = valid_floor(domain, sid, field_name, _fields_of(cfg))
    if floor and value < floor:
        return False, ("%s: %s=%s is below the `valid_shape=` window "
                       "this site declares (%d), which has to fit "
                       "inside the tile"
                       % (sid, field_name, value, floor))
    return True, ""


def verify(domain, config, dtype_bytes=4, move_locks=None):
    """Check positive extents, depths, legal rungs and move-lock ratios before measurement."""
    if move_locks is None:
        move_locks = bool((domain or {}).get("enforce_move_locks"))
    for sid, cfg in (config or {}).items():
        if not isinstance(cfg, Mapping):
            continue
        ok, why = _site_values_ok(domain, sid, cfg)
        if not ok:
            return False, why
    if move_locks:
        # Check cross-site locks after collecting all extents.
        ok, why = _move_locks_ok(domain, config)
        if not ok:
            return False, why
    return True, ""


def move_lock_ratio(author_a, author_b):
    """Preserve resolved per-axis extent ratios for offset-free moves."""
    out = []
    for i, (x, y) in enumerate(zip(author_a or (), author_b or ())):
        if not isinstance(x, int) or not isinstance(y, int) or x < 1:
            continue
        if y < 1:
            continue
        g = math.gcd(x, y)
        out.append((i, x // g, y // g))
    return out


def _mode_move_lock_ok(lock, cfgs, fields):
    """Keep a mode-converting move's source at its author-written extent."""
    pair, author = lock["pair"], lock["author"]
    n = pair.index(lock["src"])
    cell, names = cfgs[n], fields[n]
    if cell is None:
        return True, ""
    for i, written in enumerate(author[n]):
        if i >= len(names) or not isinstance(written, int):
            continue
        got = cell.get(names[i])
        if isinstance(got, int) and got != written:
            return False, (
                "%s: d%d=%s moves the SOURCE of the `pl.move` pair "
                "%s=%s, whose `%s=` conversion fixes it at %s. Every "
                "recorded candidate that moved this side failed to "
                "compile; the destination is free and is not judged "
                "here"
                % (lock["src"], i, got, pair[0], pair[1],
                   lock["mode"], written))
    return True, ""


def _ratio_move_lock_ok(lock, cfgs, fields):
    """Check offset-less move ratios, using author extents for omitted sites."""
    pair, author = lock["pair"], lock["author"]
    for i, num, den in move_lock_ratio(author[0], author[1]):
        vals = []
        for n in (0, 1):
            if cfgs[n] is not None and i < len(fields[n]):
                vals.append(cfgs[n][fields[n][i]])
            elif i < len(author[n]):
                vals.append(author[n][i])
            else:
                vals.append(None)
        if None in vals or not all(isinstance(v, int) for v in vals):
            continue
        if vals[0] * den != vals[1] * num:
            return False, (
                "%s: d%d=%s breaks the `pl.move` pair %s=%s, which carries "
                "no offset and holds %d:%d there (the kernel runs %s:%s). "
                "Moving one side without the other is `pl.move: dst tile "
                "shape != src tile shape` -- a compile and a device slot "
                "spent to be told what the body already says"
                % (pair[0], i, vals[0], pair[0], pair[1], num, den,
                   author[0][i], author[1][i]))
    return True, ""


def _move_locks_ok(domain, config):
    """Check offset-free move pairs using source extents for omitted sites."""
    for lock in (domain or {}).get("move_locks") or ():
        pair = lock.get("pair") or ()
        author = lock.get("author") or ()
        if len(pair) != 2 or len(author) != 2:
            continue
        cfgs, fields = [], []
        for sid in pair:
            cell = (config or {}).get(sid)
            cell = cell if isinstance(cell, Mapping) else None
            cfgs.append(cell)
            fields.append(_fields_of(cell or {}))
        if cfgs[0] is None and cfgs[1] is None:
            continue
        if lock.get("mode") and lock.get("src") in pair:
            ok, why = _mode_move_lock_ok(lock, cfgs, fields)
        else:
            ok, why = _ratio_move_lock_ok(lock, cfgs, fields)
        if not ok:
            return False, why
    return True, ""


def _fields_of(cfg):
    """Return shape parameter names in axis order."""
    return sorted((k for k in (cfg or {}) if k.startswith("d") and k[1:].isdigit()), key=lambda k: int(k[1:]))


def _addrs_form(text):
    """Classify addrs syntax as name, int, missing or other, independently of its resolved value."""
    from . import apply as apply_mod
    return apply_mod.addrs_kind(text)


def fingerprint(domain):
    """Hash fixed depth, address ownership and domain constraints across TPE shape edits."""
    cells = (domain or {}).get("sites") or {}
    order = (domain or {}).get("site_ids") or sorted(cells)
    rows = []
    for sid in order:
        cell = cells.get(sid) or {}
        rows.append([sid, cell.get("space"), cell.get("dtype"), cell.get("dtype_bytes"), cell.get("rank"),
                     cell.get("depth_source"), bool(cell.get("depth_tunable")), _addrs_form(cell.get("addrs") or ""),
                     bool(cell.get("resolvable")), sorted(cell.get("unresolved") or ())])
    payload = {
        "sites": rows,
        "fixed_depths": sorted((sid, cell.get("depth")) for sid, cell in cells.items()),
        "dtype_bytes": (domain or {}).get("dtype_bytes"),
        "live_tiles": (domain or {}).get("live_tiles"),
        # Track depth per site so changes remain visible when total residency is constant.
        "vec_residency": sorted(((domain or {}).get("vec_residency") or {}).items()),
        "extra_consts": sorted(((domain or {}).get("extra_consts") or {}).items()),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def changed(before, after):
    """Compare domains or stored fingerprints and report per-site changes."""
    fp_before = before if isinstance(before, str) else fingerprint(before)
    fp_after = after if isinstance(after, str) else fingerprint(after)
    if fp_before == fp_after:
        return False, ""

    try:
        a = json.loads(fp_before)
        b = json.loads(fp_after)
    except ValueError:
        return True, ("the derived tile domain changed (%s -> %s)" % (_digest(fp_before), _digest(fp_after)))

    rows_a = {row[0]: row for row in a.get("sites") or []}
    rows_b = {row[0]: row for row in b.get("sites") or []}
    reasons = []
    gone = sorted(set(rows_a) - set(rows_b))
    new = sorted(set(rows_b) - set(rows_a))
    if gone:
        reasons.append("tile site(s) removed: " + ", ".join(gone))
    if new:
        reasons.append("tile site(s) added: " + ", ".join(new))
    labels = ("id", "space", "dtype", "dtype_bytes", "rank", "depth_source",
              "depth_tunable", "addrs_form", "resolvable", "unresolved")
    for sid in sorted(set(rows_a) & set(rows_b)):
        for i, label in enumerate(labels):
            if i and rows_a[sid][i] != rows_b[sid][i]:
                reasons.append("%s %s: %r -> %r" % (sid, label, rows_a[sid][i], rows_b[sid][i]))
    for key in ("dtype_bytes", "live_tiles", "vec_residency", "extra_consts", "fixed_depths"):
        if a.get(key) != b.get(key):
            reasons.append("%s: %r -> %r" % (key, a.get(key), b.get(key)))
    if not reasons:
        reasons.append("the derived tile domain changed (%s -> %s)" % (_digest(fp_before), _digest(fp_after)))
    return True, "; ".join(reasons)


def _site_values_ok(domain, sid, cfg):
    """Validate one site's scalar depth and extent domains."""
    for field_name, value in cfg.items():
        if field_name == "value":
            if not sid.startswith("depth:"):
                continue
            invalid_depth = not isinstance(value, int) or isinstance(value, bool) or value < 1
            if invalid_depth:
                return False, "%s: depth must be an int >= 1, got %r" % (sid, value)
            continue
        if not (field_name.startswith("d") and field_name[1:].isdigit()):
            continue
        ok, why = _tile_extent_ok(domain, sid, cfg, field_name, value)
        if not ok:
            return False, why
    return True, ""
