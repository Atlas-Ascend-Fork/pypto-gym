# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Edit Pro tile shapes and replan addresses. Structural and launch settings belong to the agent."""

__all__ = ["ApplyConflict", "Edit", "MAX_VEC_ARITY", "Site", "UNSET", "apply", "apply_edits", "current_config",
           "discover_sites", "launch_params", "relayout", "split_launch", "rewritable", "set_addrs", "set_depth",
           "set_shape", "tie_groups", "tunable_sites"]

import ast
import logging
import re
from dataclasses import dataclass
from typing import List, MutableSequence, Optional, Sequence, Tuple

# Use the reader’s UTF-8 column offsets to splice the same source spans.
from panko_core.bayesian_optimization.apply import (MAX_VEC_ARITY, UNSET, Site, _read_site, loop_spans, splice_edits,
                                                    tie_groups as loop_groups)
from panko_core.pypto_ast import jit_decorators, keyword_value
from source_analysis.tile_sites import Offsets, TileProgram, read_sites

from . import space

logger = logging.getLogger(__name__)


def source_constants(extra_consts=None):
    """Copy bindings for this program; omitted bindings never inherit another run."""
    return dict(extra_consts or {})


class ApplyConflict(ValueError):
    """The edits do not describe a rewrite: they overlap, miss, or change nothing."""


@dataclass(frozen=True)
class Edit:
    """Replace `src[start:end]` with `text`. `what` is for the diff and the log."""

    start: int
    end: int
    text: str
    what: str

    @property
    def is_insert(self) -> bool:
        return self.start == self.end

    def describe(self) -> str:
        return "%s: [%d,%d) -> %r" % (self.what, self.start, self.end, self.text)


def apply_edits(src: str, edits: Sequence[Edit]) -> str:
    """Apply original-source edits right to left, rejecting overlaps, invalid spans and no-ops."""
    if not edits:
        return src

    n = len(src)
    for edit in edits:
        if not isinstance(edit, Edit):
            raise ApplyConflict("not an Edit: %r" % (edit,))
        if edit.start < 0 or edit.end < edit.start or edit.end > n:
            raise ApplyConflict("%s: span [%d,%d) is out of range for a %d-byte source" % (edit.what, edit.start,
                                                                                           edit.end, n))
        if src[edit.start:edit.end] == edit.text:
            raise ApplyConflict(
                "%s: span [%d,%d) already reads %r; a no-op edit means the caller "
                "thinks it changed something it did not"
                % (edit.what, edit.start, edit.end, edit.text)
            )

    ordered = sorted(edits, key=lambda e: (e.start, e.end))
    for earlier, later in zip(ordered, ordered[1:]):
        if later.start < earlier.end or later.start == earlier.start:
            raise ApplyConflict("overlapping edits: %s and %s" % (earlier.describe(), later.describe()))

    return splice_edits(src, ((e.start, e.end, e.text) for e in reversed(ordered)))


def _address_literal(text):
    stripped = (text or "").strip()
    if not stripped:
        return "missing", None
    try:
        node = ast.parse(stripped, mode="eval").body
    except SyntaxError:
        return "other", None
    if isinstance(node, ast.Name):
        return "name", None
    if _nonnegative_address(node):
        return "int", [node.value]
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts and all(_nonnegative_address(e) for e in node.elts):
        return "list", [e.value for e in node.elts]
    return "other", None


def addrs_kind(text: str) -> str:
    """Distinguish a scalar base, an explicit address list and a symbolic address."""
    return _address_literal(text)[0]


def addrs_ints(site):
    """Use the declaration's resolved addresses, falling back to literal syntax."""
    values = getattr(site, "addrs_values", None)
    if values and all(v is not None for v in values):
        return [int(v) for v in values]
    return _address_literal(getattr(site, "addrs_text", ""))[1]


def _address_kind(site):
    """Return the resolved address form, preserving scalar versus per-buffer list syntax."""
    text = (getattr(site, "addrs_text", "") or "").strip()
    if not text:
        return "missing"
    values = addrs_ints(site)
    if not values or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        return "other"
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return "other"
    is_list = (isinstance(node, (ast.List, ast.Tuple)) or getattr(site, "addrs_is_sequence", False) or len(values) > 1)
    return "list" if is_list else "int"


def _hex(addr: int) -> str:
    """Render an address with uppercase hexadecimal digits."""
    return "0x%X" % addr


def rewritable(site) -> Tuple[bool, str]:
    """Return eligibility and reasons for refusing a tile rewrite."""
    reasons: List[str] = []

    if getattr(site, "space", None) is None:
        reasons.append("memory space unknown (its TileType was not found)")

    depth = getattr(site, "depth", None)
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        reasons.append("buffer depth is not a resolved positive integer; specialize depth or mutex_ids")

    if not getattr(site, "resolvable", False):
        unresolved = ", ".join(getattr(site, "unresolved", ()) or ()) or "?"
        reasons.append("shape is not resolvable (unresolved: %s)" % unresolved)
    elif len(site.shape) > MAX_VEC_ARITY:
        reasons.append(
            "shape has rank %d; this module tunes up to MAX_VEC_ARITY=%d axes and "
            "will not write some of a shape while leaving the rest"
            % (len(site.shape), MAX_VEC_ARITY))

    kind = _address_kind(site)
    if kind == "missing":
        reasons.append("no addrs= argument to rewrite")
    elif kind == "other":
        reasons.append(
            "addrs= is not statically resolved to non-negative integers "
            "(%r); specialize the binding or variant selection before rewriting"
            % (site.addrs_text,)
        )
    elif kind == "list":
        # Explicit addresses require one entry per buffer depth.
        listed = len(addrs_ints(site) or ())
        if isinstance(depth, int) and listed != depth:
            reasons.append(
                "addrs= lists %d address(es) but depth is %d; pypto_pro requires "
                "len(addrs) == depth, so this kernel does not parse"
                % (listed, depth)
            )

    return not reasons, "; ".join(reasons)


def set_shape(site, index: int, value: int) -> Edit:
    """Rewrite one entry of this site's `shape=[...]`."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ApplyConflict("%s shape[%s]: a tile extent must be an int >= 1, got %r" % (site.site_id, index, value))
    shape = site.shape
    if index < 0 or index >= len(shape):
        raise ApplyConflict("%s has %d shape entries; no shape[%d]" % (site.site_id, len(shape), index))
    entry = shape[index]
    return Edit(entry.start, entry.end, str(value), "%s shape[%d]" % (site.site_id, index))


def set_depth(site, value: int) -> Edit:
    """Rewrite explicit depth. Inferred depth follows the mutex rotation."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ApplyConflict("%s depth: must be an int >= 1, got %r" % (site.site_id, value))
    if site.depth_span is None:
        raise ApplyConflict(
            "%s has no depth= argument: depth lives in mutex_ids for this site "
            "(depth_source=%r, mutex_ids=%s). Change the length of that list, or "
            "add depth= by hand; this function will not guess."
            % (site.site_id, site.depth_source, site.mutex_ids_text)
        )
    start, end = site.depth_span
    return Edit(start=start, end=end, text=str(value), what="%s depth" % site.site_id)


def set_addrs(site, addr: int, stride: Optional[int] = None) -> Edit:
    """Preserve scalar or list address syntax and space list entries by stride."""
    if not isinstance(addr, int) or isinstance(addr, bool) or addr < 0:
        raise ApplyConflict("%s addrs: must be an int >= 0, got %r" % (site.site_id, addr))
    kind = _address_kind(site)
    if kind == "missing":
        raise ApplyConflict("%s has no addrs= argument to rewrite" % (site.site_id,))
    if kind == "other":
        raise ApplyConflict(
            "%s writes unresolved addrs=%s; replacing it with %s could delete "
            "a variant selection"
            % (site.site_id, site.addrs_text, _hex(addr))
        )
    if kind == "list":
        listed = addrs_ints(site) or []
        if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
            raise ApplyConflict(
                "%s writes addrs= as a list of %d address(es), so rewriting it needs "
                "the per-tile stride (Layout.stride); got %r"
                % (site.site_id, len(listed), stride)
            )
        text = "[" + ", ".join(_hex(addr + i * stride) for i in range(len(listed))) + "]"
    else:
        text = _hex(addr)
    start, end = site.addrs_span
    return Edit(start=start, end=end, text=text, what="%s addrs" % site.site_id)


def _jit_decorators(tree: ast.AST) -> List[Tuple[str, ast.AST]]:
    functions = (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    return [(fn.name, decorator) for fn in functions for decorator in jit_decorators(fn)]


def _jit_insert_at(src, dec, offsets):
    arguments = [*dec.args, *(kw.value for kw in dec.keywords)]
    if arguments:
        return max(offsets.span(node)[1] for node in arguments)
    return src.index("(", offsets.span(dec.func)[1]) + 1


def _relayout_note(site_id, reason, skipped):
    """Record address skips, keeping unchanged addresses below warning level."""
    level = logging.DEBUG if reason.startswith("already at") else logging.WARNING
    logger.log(level, "relayout skipped %s: %s", site_id, reason)
    if skipped is not None:
        skipped.append((site_id, reason))


def _address_edit(layout, site, addr):
    """Write a changed scalar or address progression, preserving no-op bytes."""
    written = addrs_ints(site)
    is_list = _address_kind(site) == "list"
    if written is None or not is_list:
        if written is not None and written[0] == addr:
            return None, "already at %s" % _hex(addr)
        return set_addrs(site, addr), None
    stride = None
    getter = getattr(layout, "stride", None)
    if callable(getter):
        try:
            stride = getter(site.site_id)
        except KeyError:
            stride = None
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        return None, ("addrs= is a list of %d and this layout gives no per-tile "
                      "stride for it" % (len(written),))
    planned = [addr + i * stride for i in range(len(written))]
    if written == planned:
        return None, "already at %s" % _hex(addr)
    return set_addrs(site, addr, stride=stride), None


def _relayout_edit(layout, site):
    """Find a site's planned address without rewriting author-owned windows."""
    ok, reason = rewritable(site)
    if not ok:
        return None, reason
    try:
        addr = layout.addr(site.site_id)
    except KeyError:
        return None, "no placement named %r in this layout" % (site.site_id,)
    placements = getattr(layout, "placements", None)
    pinned = (any(p.name == site.site_id and p.unchecked for p in placements)
              if placements is not None else
              site.space in (getattr(layout, "author_owned", ()) or ()))
    if pinned:
        return None, ("already at %s (%s keeps the author's addresses)" % (_hex(addr), site.space))
    return _address_edit(layout, site, addr)


def relayout(src: str, layout, sites, skipped: Optional[MutableSequence] = None) -> List[Edit]:
    """Emit changed placements and append skipped (site_id, reason) pairs when requested."""
    edits: List[Edit] = []
    for site in sites:
        edit, reason = _relayout_edit(layout, site)
        if reason is not None:
            _relayout_note(site.site_id, reason, skipped)
        else:
            edits.append(edit)
    return edits


def _shape_span(site):
    """Return the source span from the first shape entry through the last."""
    if not site.shape:
        return site.call_span[0], site.call_span[0]
    if not getattr(site, "shape_is_sequence", True) and site.shape_span is not None:
        return site.shape_span
    return site.shape[0].start, site.shape[-1].end


def _tile_sites(sites, loops):
    """Collect tile sites with their enclosing loops for lexical ties."""
    out = []
    for site in sites:
        start, end = _shape_span(site)
        enclosing = [(a, b) for a, b in loops if a <= site.call_span[0] < b]
        span = min(enclosing, key=lambda ab: ab[1] - ab[0]) if enclosing else None
        out.append(Site(id=site.site_id, kind="vec", ordinal=0, func_text="",
                        start=start, end=end, loop_depth=0, loop_span=span))
        if site.depth_span is not None and site.depth is not None:
            out.append(Site(id="depth:%s" % site.site_id, kind="depth", ordinal=0, func_text="",
                            start=site.depth_span[0], end=site.depth_span[1], loop_depth=0, loop_span=span))
    return out


# Templates insert missing JIT keywords; existing keywords have editable value spans.
JIT_INSERT = {"auto_mutex": ", auto_mutex={}", "preload": ", pipeline=pl.pipeline.PipelineConfig(preload={})"}


def _preload_value_node(dec):
    """Return the editable value, None for no pipeline, or False for a protected expression."""
    pipeline = keyword_value(dec, "pipeline")
    if pipeline is None:
        return None
    if not isinstance(pipeline, ast.Call) or pipeline.args:
        return False
    if any(kw.arg != "preload" for kw in pipeline.keywords):
        return False
    return keyword_value(pipeline, "preload") or False


def _jit_sites(src, tree, offsets):
    """Expose writable auto_mutex and preload keywords only when the source has one unambiguous JIT."""
    found = _jit_decorators(tree)
    if len(found) != 1:
        return []
    _, dec = found[0]
    if not isinstance(dec, ast.Call):
        # A bare @pl.jit has no keyword insertion site.
        return []

    insert_at = _jit_insert_at(src, dec, offsets)
    out = []
    staged = "pl.pipeline.stage" in src or "pipeline.stage" in src
    for knob in space.JIT_KNOBS:
        if knob == "preload":
            if not staged:
                logger.info("preload not offered as a site: this kernel has no "
                            "`@pl.pipeline.stage` function, and `pipeline=` is "
                            "refused without one (PRO_PIPELINE F00007)")
                continue
            node = _preload_value_node(dec)
            if node is False:
                logger.info("preload not offered as a site: @pl.jit carries a "
                            "pipeline= this module will not edit a field of")
                continue
        else:
            node = keyword_value(dec, knob)
        if node is None:
            out.append(Site(id="g:%s#0" % knob, kind="global", ordinal=0, func_text=JIT_INSERT[knob], start=insert_at,
                            end=insert_at))
        else:
            start, end = offsets.span(node)
            out.append(Site(id="g:%s#0" % knob, kind="global", ordinal=0, func_text="", start=start, end=end))
    return out


def _launch_sites():
    """Expose legacy launch controls as zero-width sites for readers. Shape search never edits
    the wrapper."""
    return [Site(id="g:%s#0" % knob, kind="launch", ordinal=0, func_text="",
                 start=0, end=0)
            for knob in space.LAUNCH_KNOBS]


def discover_sites(src, extra_consts=None):
    """Resolve case constants and enumerate supported controls in source order."""
    return _search_sites(TileProgram(src, source_constants(extra_consts)), writable_only=False)


def _search_sites(program, *, writable_only):
    sites = program.active_sites
    if writable_only:
        refused = _refusals(sites)
        if refused:
            logger.info("%d of %d tile site(s) refused: %s", len(refused), len(sites),
                        "; ".join("%s (%s)" % kv for kv in sorted(refused.items())))
        sites = [site for site in sites if site.site_id not in refused]
    tree, offsets = program.bindings.tree, program.offsets
    return (_tile_sites(sites, loop_spans(tree, offsets.pos))
            + _jit_sites(program.src, tree, offsets) + _launch_sites())


def tile_program(src, extra_consts=None):
    if isinstance(src, TileProgram):
        return src
    return TileProgram(src, source_constants(extra_consts))


def tunable_sites(src, extra_consts=None):
    return _search_sites(tile_program(src, extra_consts), writable_only=True)


def _refusals(sites):
    out = {}
    for site in sites:
        ok, why = rewritable(site)
        if not ok:
            out[site.site_id] = why
    return out


def refusals(src, extra_consts=None):
    return _refusals(tile_program(src, extra_consts).active_sites)


def erasable_spans(src, extra_consts=None):
    """Strip only TPE-owned shapes and derived addresses from the structural signature."""
    out = []
    for site in read_sites(src, source_constants(extra_consts)):
        ok, _ = rewritable(site)
        if ok:
            # Erase the whole shape expression in both alias and literal forms.
            # Erasing the alias definition instead would discard agent-owned
            # constants and change the key when materialization expands the alias.
            shape_span = getattr(site, "shape_span", None) or _shape_span(site)
            out.append((shape_span[0], shape_span[1], "vec"))
            out.append((site.addrs_span[0], site.addrs_span[1], "addrs"))
    return out


def tie_groups(sites, anchor):
    return loop_groups(sites, anchor, space.vec_fields, match_leading=True)


def _requested_shape(site, cfg, allow_unset=True):
    values = []
    for axis, entry in enumerate(site.shape):
        value = cfg.get(f"d{axis}", entry.value)
        values.append(entry.value if allow_unset and value == UNSET else value)
    return values


def shape_edits(site, cfg):
    """Edit changed axes only, preserving unchanged constants and shared spans."""
    if not getattr(site, "shape_is_sequence", True):
        values = _requested_shape(site, cfg)
        if values == [e.value for e in site.shape]:
            return []
        span = getattr(site, "shape_span", None)
        if span is None or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in values):
            raise ApplyConflict("%s: the shape alias cannot be materialized" % site.site_id)
        return [Edit(span[0], span[1], repr(values), "%s shape" % site.site_id)]
    out = []
    for axis, entry in enumerate(site.shape):
        value = cfg.get("d%d" % axis)
        if value is None or value == UNSET:
            continue
        if entry.value == value:
            continue
        out.append(set_shape(site, axis, value))
    return out


def _jit_edits(src, by_knob):
    """Merge keyword insertions in sorted order for a reproducible source hash."""
    edits, inserts = [], {}
    for sid, (site, value) in sorted(by_knob.items()):
        knob = space.global_knob(sid)
        if site.start == site.end:
            inserts.setdefault(site.start, {})[knob] = value
            continue
        text = str(value)
        if src[site.start:site.end] == text:
            continue                       # already set to this; nothing to write
        edits.append(Edit(start=site.start, end=site.end, text=text, what="@pl.jit(%s=)" % knob))
    for pos, knobs in inserts.items():
        text = "".join(JIT_INSERT[k].format(v) for k, v in sorted(knobs.items()))
        edits.append(Edit(start=pos, end=pos, text=text, what="@pl.jit(+%s)" % ", ".join(sorted(knobs))))
    return edits


def launch_params(site_configs):
    """Return the explicit measurement-time launch overrides."""
    return split_launch(site_configs)[1]


def split_launch(site_configs):
    """Keep launch keys out of the source edits, including unset overrides."""
    edits, launch = {}, {}
    for sid, cfg in (site_configs or {}).items():
        key = str(sid)
        knob = space.global_knob(key) if key.startswith("g:") else None
        if knob not in space.LAUNCH_KNOBS:
            edits[sid] = cfg
        elif isinstance(cfg, dict) and cfg.get("value") not in (None, UNSET):
            launch[knob] = cfg["value"]
    return edits, launch


def _site_edits(site, sid, cfg):
    """Produce a single site's shape or depth changes after rewrite admission."""
    ok, _ = rewritable(site)
    if not ok:
        return []
    if sid.startswith("depth:"):
        value = cfg.get("value", UNSET)
        if value == UNSET or value == site.depth:
            return []
        return [set_depth(site, value)]
    return shape_edits(site, cfg)


def _candidate_edits(program, site_configs):
    """Collect source edits and count every changed logical site exactly once."""
    src = program.src
    by_id = {s.site_id: s for s in program.active_sites}
    edits, jit = [], {}
    jit_sites = None
    resized_sites = set()
    n_sites = 0
    for sid, cfg in (site_configs or {}).items():
        if not isinstance(cfg, dict):
            continue
        if sid.startswith("g:"):
            value = cfg.get("value", UNSET)
            if value == UNSET or space.global_knob(sid) not in JIT_INSERT:
                continue
            if jit_sites is None:
                jit_sites = {s.id: s for s in _jit_sites(src, program.bindings.tree, program.offsets)}
            if sid in jit_sites:
                jit[sid] = jit_sites[sid], value
            continue
        if sid.startswith(("const:", "cfg:")):
            continue
        target = sid.split(":", 1)[1] if sid.startswith("depth:") else sid
        site = by_id.get(target)
        if site is None:
            continue
        site_edits = _site_edits(site, sid, cfg)
        if site_edits:
            edits.extend(site_edits)
            resized_sites.add(site.site_id)
            n_sites += 1
    if jit:
        edits.extend(_jit_edits(src, jit))
        n_sites += sum(src[site.start:site.end] != str(value) for site, value in jit.values())
    return edits, resized_sites, n_sites


def unique_edits(edits, conflict="conflicting shapes share source span"):
    """Coalesce shared spans and reject inconsistent requests before writing."""
    unique = {}
    for edit in edits:
        key = (edit.start, edit.end)
        previous = unique.get(key)
        if previous is not None and previous.text != edit.text:
            raise ApplyConflict(f"{conflict} {key}")
        unique[key] = edit
    return list(unique.values())


def mismatched_shape(after, sites, site_configs, *, allow_unset=True):
    """Verify every user of a shared shape expression received its request."""
    after = {s.site_id: s for s in after}
    for site in sites:
        wanted = (site_configs or {}).get(site.site_id, {})
        expected = _requested_shape(site, wanted, allow_unset)
        got = after.get(site.site_id)
        if got is None or [e.value for e in got.shape] != expected:
            return site.site_id
    return None


def apply(src, site_configs, extra_consts=None, relayout_addresses=True):
    """Return source and edit count. Replan unowned addresses and return launch overrides via launch_params."""
    program = TileProgram(src, source_constants(extra_consts))
    return apply_to_program(program, site_configs, relayout_addresses)


def apply_to_program(program, site_configs, relayout_addresses=True):
    src = program.src
    sites = program.active_sites
    edits, resized_sites, n_sites = _candidate_edits(program, site_configs)
    launch = launch_params(site_configs)
    if launch:
        logger.info("Launch overrides apply at measurement time: %s", launch)
    edits = unique_edits(edits)
    tuned = apply_edits(src, edits) if edits else src
    if not resized_sites:
        return tuned, n_sites
    consts = program.bindings.context
    after = read_sites(tuned, consts)
    changed = mismatched_shape(after, sites, site_configs)
    if changed:
        raise ApplyConflict("%s: a shared shape binding changed outside its requested "
                            "configuration; update all users together" % changed)
    if relayout_addresses:
        tuned = _replan(tuned, consts, after, sites, resized_sites)
    return tuned, n_sites


def _replan(src, consts, sites, author_sites, resized_sites):
    """Replan every allocation sharing the resized sites' memory groups."""
    try:
        groups = space.onchip_layout.allocation_groups(sites)
        touched = (group for group in groups if any(s.site_id in resized_sites for s in group))
        sites = tuple(s for group in touched for s in group)
        ids = {s.site_id for s in sites}
        author_sites = tuple(s for s in author_sites if s.site_id in ids)
        unsized = [s.site_id for s in sites if not s.resolvable or s.space is None]
        if unsized:
            raise ApplyConflict("cannot re-plan unresolved tile sites: %s" % ", ".join(unsized))
        hw = space.HW(sites=sites, extra_consts=consts, author_sites=author_sites)
        layout = space.layout_for({}, hw, keep_author_map=False)
    except (SyntaxError, ValueError, RuntimeError) as exc:
        raise ApplyConflict("addresses could not be re-planned: %s" % exc) from exc
    skipped: List[Tuple[str, str]] = []
    edits = relayout(src, layout, sites, skipped=skipped)
    incomplete = [(sid, why) for sid, why in skipped if not why.startswith("already at")]
    if incomplete:
        raise ApplyConflict("address materialization incomplete: %s" % incomplete)
    if not edits:
        return src
    return apply_edits(src, edits)


def _current_site_config(src, site, sites):
    """Decode one discovered knob without treating Boolean text as a number."""
    if site.kind == "vec":
        tile = sites[site.id]
        return _read_site("vec", (entry.value for entry in tile.shape))
    if site.kind == "depth":
        return {"value": sites[site.id.split(":", 1)[1]].depth}
    if site.start == site.end:
        return {"value": UNSET}
    nums = [int(x) for x in re.findall(r"-?\d+", src[site.start:site.end])]
    if nums:
        return _read_site(site.kind, nums)
    text = src[site.start:site.end].strip()
    return {"value": text == "True" if text in ("True", "False") else text}


def current_config(src, extra_consts=None):
    """Read the incumbent configuration, preserving unset launch overrides."""
    return incumbent_config(tile_program(src, extra_consts))


def incumbent_config(program):
    sites = {s.site_id: s for s in program.active_sites}
    return {site.id: _current_site_config(program.src, site, sites)
            for site in _search_sites(program, writable_only=True)}


def _nonnegative_address(node):
    if not isinstance(node, ast.Constant):
        return False
    value = node.value
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
