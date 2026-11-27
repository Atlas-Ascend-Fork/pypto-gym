# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check that INIT rewrites preserve lexical ownership, control flow and declaration values."""

import ast
import json
from pathlib import Path

from memory import onchip_layout
import pro_init_handoff as init_handoff
import pro_init_repair as init_repair
from source_analysis import kernel_boundary, tile_sites
from source_analysis.case_context import validate_context
from source_analysis.source_bindings import UNKNOWN, Binding


def _declaration_evidence(program):
    """Add control-flow and mutex evidence to the parsed tile declarations."""
    resolver = program.bindings
    controls = {}

    def contexts(node, path=()):
        if tile_sites.is_named_call(node, "make_tile_group"):
            controls[id(node)] = path
        for field, value in ast.iter_fields(node):
            branch = path
            if isinstance(node, ast.If) and field in ("body", "orelse"):
                branch += (("if", ast.dump(node.test), field),)
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)) and field in ("body", "orelse"):
                condition = node.test if isinstance(node, ast.While) else node.iter
                branch += ((type(node).__name__, ast.dump(condition), field),)
            elif isinstance(node, ast.Try) and field in ("body", "handlers", "orelse", "finalbody"):
                branch += (("try", field),)
            for child in value if isinstance(value, list) else (value,):
                if isinstance(child, ast.AST):
                    contexts(child, branch)

    contexts(resolver.tree)
    result = {}
    for call in program.calls:
        type_binding = program.type_binding(call)
        space = None
        if type_binding is not None:
            binding = program.keyword_binding(type_binding, "target_memory")
            resolved = resolver.resolve(binding)
            if resolved is not None or binding is None:
                space = tile_sites.space_of(resolved.node if resolved else None)
        mutex_node = tile_sites.keyword_value(call, "mutex_ids")
        mutex_value = (resolver.value(Binding(mutex_node, resolver.environment(call)))
                       if mutex_node is not None else None)
        result[program.offsets.span(call)] = {"space": space, "control": controls.get(id(call), ()),
                                              "mutex": None if mutex_value is UNKNOWN else mutex_value,
                                              "mutex_known": mutex_value is not UNKNOWN,
                                              "mutex_text": ast.dump(mutex_node) if mutex_node is not None else ""}
    return result


def _site_facts(program, include_inactive):
    out = []
    evidence = _declaration_evidence(program)
    for site in (program.sites if include_inactive else program.active_sites):
        proof = evidence[site.call_span]
        tokens = onchip_layout.dtype_tokens(site.dtype_text)
        out.append((site.site_id, {
            "identity": (site.scope, site.group_var, proof["control"]),
            "space": proof["space"] if proof["space"] is not None else site.space,
            "dtype": site.dtype_resolved or (tokens[0] if len(tokens) == 1 else None),
            "shape": tuple(e.value for e in site.shape),
            "rank": len(site.shape),
            "rank_known": site.shape_rank_known,
            "depth": site.depth,
            "unresolved": list(site.unresolved),
            "addrs": site.addrs_values,
            "addrs_sequence": site.addrs_is_sequence,
            "addrs_rank_known": (site.addrs_is_sequence or
                                 bool(site.addrs_values) and
                                 all(x is not None for x in site.addrs_values)),
            "mutex": proof["mutex"],
            "mutex_known": proof["mutex_known"],
            "mutex_text": proof["mutex_text"],
        }))
    return out


def _writable_count(sites):
    from .apply import rewritable
    return sum(1 for site in sites if rewritable(site)[0])


_IMMUTABLE = (("space", "memory space"), ("dtype", "dtype"), ("shape", "resolved shape"), ("rank", "shape rank"),
              ("depth", "effective depth"), ("addrs", "resolved addresses"))


def _readable_changed(old, new):
    """Whether a readable component of the original is missing from, or different in, the rewrite."""
    return any(x is not None and (i >= len(new) or x != new[i]) for i, x in enumerate(old))


def _context_conflicts(src, extra_consts, contextual, label):
    """External case bindings may fill holes, never override known source facts."""
    if not extra_consts:
        return []
    reasons = []
    local_by_identity = {facts["identity"]: facts for _, facts in _site_facts(tile_sites.TileProgram(src, None), False)}
    for sid, bound in contextual:
        local = local_by_identity.get(bound["identity"])
        if local is None:
            continue
        for key in ("space", "dtype", "depth", "shape", "addrs"):
            old, new = local[key], bound[key]
            mismatch = _readable_changed(old, new) if isinstance(old, tuple) else old is not None and old != new
            if mismatch:
                reasons.append("%s %s: external constants conflict with source %s %r -> %r"
                               % (label, sid, key, old, new))
    return reasons


def _compare_values(sid, key, label, before, after, report):
    """Preserve every readable component while allowing newly exposed facts."""
    old, new = before[key], after[key]
    if old == new:
        return
    if key in ("shape", "addrs"):
        rank_known = before["rank_known" if key == "shape" else "addrs_rank_known"]
        if _readable_changed(old, new) or rank_known and len(old) != len(new):
            report["reasons"].append(
                "%s: %s changed %r -> %r; a normalisation exposes a tile, it does not retune one"
                % (sid, label, old, new))
            return
        unreadable = [i for i, x in enumerate(old) if x is None]
        if not rank_known or unreadable:
            names = ", ".join(sorted(before.get("unresolved", []))) or "?"
            report["caveats"].append(
                "%s: original %s was not readable at axis/slot %s (%s); confirm exposed %r "
                "against recorded specialization evidence and the golden gate"
                % (sid, label, unreadable or "unknown rank", names, new))
            return
    elif old is None and new is not None:
        report["caveats"].append(
            "%s: original %s was not readable; confirm exposed %r against recorded "
            "specialization evidence and the golden gate" % (sid, label, new))
        return
    elif key == "rank" and not before["rank_known"] and after["rank_known"]:
        report["caveats"].append(
            "%s: original shape rank was not readable; confirm exposed rank %s "
            "against recorded specialization evidence and the golden gate" % (sid, new))
        return
    report["reasons"].append(
        "%s: %s changed %r -> %r; a normalisation may not change what the kernel computes"
        % (sid, label, old, new))


def _compare_sites(before, after, report):
    for (sid, b), (_, a) in zip(before, after):
        for key, label in _IMMUTABLE:
            _compare_values(sid, key, label, b, a, report)
        if b["addrs_rank_known"] and b["addrs_sequence"] != a["addrs_sequence"]:
            report["reasons"].append("%s: address representation changed between a base and a slot sequence" % sid)
        if b["mutex_known"] and (not a["mutex_known"] or b["mutex"] != a["mutex"]):
            report["reasons"].append("%s: resolved mutex ids changed %r -> %r" % (sid, b["mutex"], a["mutex"]))
        elif not b["mutex_known"] and b["mutex_text"] != a["mutex_text"]:
            report["caveats"].append(
                "%s: original mutex ids were not readable; confirm the exposed ids "
                "against recorded specialization evidence and the golden gate" % sid)


def check(before_src, after_src, extra_consts=None, *, after_extra_consts=None):
    """
    Is `after_src` an admissible normalisation of `before_src`?

    Returns {"ok": bool, "reasons": [...], "before": {...}, "after": {...}}. Every
    reason names what failed. `after_extra_consts` supplies the proposed file's
    independently resolved sibling/case context; omission reuses `extra_consts`.
    A supplied empty dict means the proposed file has no external context.
    """
    report = {"ok": False, "reasons": [], "caveats": []}
    if after_extra_consts is None:
        after_extra_consts = extra_consts

    programs = []
    for label, src, consts in (("normalised", after_src, after_extra_consts), ("original", before_src, extra_consts)):
        try:
            program = tile_sites.TileProgram(src, consts)
            programs.append((program, _site_facts(program, True)))
        except (SyntaxError, ValueError) as e:
            report["reasons"].append(f"{label} source does not parse: {e}")
            return report
    (new, after), (old, before) = programs

    n_before = _writable_count(old.active_sites)
    n_after = _writable_count(new.active_sites)
    report["before"] = {"tile_calls": len(before), "tunable": n_before}
    report["after"] = {"tile_calls": len(after), "tunable": n_after}

    if len(before) != len(after):
        report["reasons"].append(
            f"tile site count changed: {len(before)} -> {len(after)}; a "
            f"normalisation may not add or remove a pl.make_tile_group")
        return report
    for i, ((b_id, b), (a_id, a)) in enumerate(zip(before, after)):
        if b["identity"] != a["identity"]:
            report["reasons"].append(
                "tile site %s declaration identity changed: %r -> %r (%s -> %s)"
                % (i, b["identity"], a["identity"], b_id, a_id))

    report["reasons"].extend(_context_conflicts(before_src, extra_consts, before, "original"))
    report["reasons"].extend(_context_conflicts(after_src, after_extra_consts, after, "normalised"))

    _compare_sites(before, after, report)

    if n_after <= n_before:
        report["reasons"].append(
            f"tunable tile sites did not increase ({n_before} -> {n_after}); "
            f"the search can see no more than it could before")

    report["ok"] = not report["reasons"]
    return report


def arguments(sub):
    _parser_init_repair(sub)
    for name in ("normalize", "init"):
        sub.choices[name].add_argument("--case-context", default="")
    init = sub.choices["init"]
    init.add_argument("--mode", choices=["npu"])
    for flag in ("--test-command", "--op-name", "--normalization-before", "--normalization-evidence"):
        init.add_argument(flag)
    init.add_argument("--normalization-tolerance", type=float, default=0.03)


def _inputs(a, before, after, before_path, after_path):
    from .domain import resolve_extra_consts

    context = read_context(a.case_context)
    return context, check(before, after, resolve_extra_consts(before, before_path, context),
                          after_extra_consts=resolve_extra_consts(after, after_path, context))


def command(a):
    from panko.config import out, Refusal

    before, after = (Path(p).read_bytes().decode() for p in (a.before, a.after))
    _, report = _inputs(a, before, after, a.before, a.after)
    init_handoff.original_source(a.op_dir, before)
    out(report)
    if not report["ok"]:
        raise Refusal(1)


def check_init(a, contract):
    original = init_handoff.original_source(a.op_dir)
    after = Path(a.op_file).read_bytes().decode()
    if a.normalization_before:
        before = Path(a.normalization_before).read_bytes().decode()
        if original is not None and original != before:
            raise ValueError("recorded normalization input differs")
    elif original is not None:
        before = original
    elif a.normalization_evidence:
        raise ValueError("normalization evidence requires an original source")
    else:
        return
    if before == after:
        return
    context, static = _inputs(a, before, after, a.normalization_before or a.op_file, a.op_file)
    evidence = json.loads(Path(a.normalization_evidence).read_bytes()) if a.normalization_evidence else None
    measured = init_handoff.check(before, after, context, evidence, a.normalization_tolerance,
                                  evaluation=dict(contract, op_dir=a.op_dir))
    owned = kernel_boundary.discover_owned_bindings(before, context)
    old, new = ast.parse(before), ast.parse(after)
    if (kernel_boundary.shell_dump(old, context, owned) != kernel_boundary.shell_dump(new, context, owned)
            or kernel_boundary.measurement_dump(old) != kernel_boundary.measurement_dump(new)):
        raise ValueError("normalization changed wrapper or measurement options")
    if not static["ok"] or not measured["ok"]:
        raise ValueError("; ".join(static["reasons"] + measured["reasons"]))


def _parser_init_repair(sub):
    q = sub.add_parser("init-repair", help="relay an agent repair only after ordinary INIT/normalize fails")
    q.set_defaults(fn=repair_command)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--op-file", required=True)
    q.add_argument("--action", choices=("propose", "status", "reject"), default="status")
    q.add_argument("--request", default="", help="unchanged original INIT dispatch JSON")
    q.add_argument("--diagnostic", default="", help="JSON refusal from ordinary INIT or normalize")
    q.add_argument("--intent", default="", help="one concrete semantics-preserving preparation change")
    q.add_argument("--original", default="", help="original source if earlier normalization already changed op_file")
    q.add_argument("--case-context", default="")
    q.add_argument("--max-attempts", type=int, default=3, help="repair limit, fixed on the first proposal")
    q.add_argument("--reason", default="", help="failed check to record before rejecting and restoring")


def read_context(path):
    context = json.loads(Path(path).read_bytes()) if path else {}
    validate_context(context)
    return context


def repair_command(a):
    from panko.config import out, Refusal
    try:
        if a.action == "propose":
            if not a.request or not a.diagnostic:
                raise ValueError("propose requires the original request and refusal")
            result = init_repair.propose(a.op_dir, a.op_file, json.loads(Path(a.request).read_bytes()),
                                         json.loads(Path(a.diagnostic).read_bytes()), a.intent,
                                         a.original, read_context(a.case_context), a.max_attempts)
        elif a.action == "reject":
            result = init_repair.reject(a.op_dir, a.op_file, a.reason)
        else:
            saved = init_repair.load(a.op_dir, a.op_file)
            result = {"phase": "init_repair", "state": saved,
                      "init_repair": init_repair.receipt(a.op_dir, saved) if saved else None}
        out(result)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        out({"refused": True, "error": "init_repair_blocked", "reason": str(exc)})
        raise Refusal(4) from exc
