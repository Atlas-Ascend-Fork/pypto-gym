#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
List tile-shape boundaries in a PyPTO kernel implementation.

Performs a pure-AST scan (no shape inference, no pypto import) to produce a
closed checklist of every cube→vector boundary, vector→cube boundary (view
feeding a matmul), view site, reshape site, assemble site, F-17 shape-change
point (reduce/dtype-cast ops), and V-segment in the impl.  The output feeds
the A4 TileShape audit (tune-frontend SKILL.md §A4) as the per-boundary
optimization agenda.

DETECTION ONLY: this tool never modifies the impl and never sets tiles —
it flags sites, tags the applicable F-rule, and (where statically derivable)
suggests a tile FORMULA as 建议首步.  Actual tile setting happens exclusively
in Phase C ITER execution (measured → precision-verified → adopted/rolled back).

Usage:
    python3 list_tile_boundaries.py <impl.py> [--out PATH] [--emit-agenda PATH]

Output:
    A markdown table of the call sequence (line, op, classification) plus a
    flagged-boundaries table (cube→vector with F-20 R1 vs R3 distinction,
    vector→cube with F-20 R4, per-site view rows with Rule 4 Step 1a/1b
    formula, reshape, assemble, F-17 shape-change points, V-segments with
    per-segment op-class counts) and a summary.  Segments containing no
    vec/cube/reduce/reshape/assemble/view ops (config-only regions) are NOT
    boundaries.

    --out PATH        save the report to PATH with an impl-sha256 header
                      (the artifact gated at the S3→S4 handoff).
    --emit-agenda PATH
                      write the A4 agenda skeleton: one PENDING row per
                      flagged site.  A4 is complete iff no PENDING remains
                      and row count matches the boundary report.

Exit codes:
    0  — analysis completed
    1  — parse error, file not found
"""

import argparse
import ast
import datetime
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any


def _reject_errors(record: logging.LogRecord) -> bool:
    """Keep stdout handler from double-emitting ERROR records."""
    return record.levelno < logging.ERROR


_out_handler = logging.StreamHandler(sys.stdout)
_out_handler.setLevel(logging.INFO)
_out_handler.addFilter(_reject_errors)
_err_handler = logging.StreamHandler(sys.stderr)
_err_handler.setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[_out_handler, _err_handler])

# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

_TILE_SETTERS_VEC = {"set_vec_tile_shapes"}
_TILE_SETTERS_CUBE = {"set_cube_tile_shapes"}
_CUBE_OPS = {"matmul", "scaled_mm", "transposed_batchmatmul"}
_REDUCES = {
    "amax", "amin", "sum", "prod", "argmax", "argmin", "var",
    "cumsum", "cumprod",
    "topk", "argsort", "sort32", "mrgsort",
    "scatter_", "scatter", "scatter_update",
    "index_add", "index_add_", "index_add__ub", "index_add_ub",
    "atomic_add",
    "online_softmax", "online_softmax_update",
}
_SCATTER_OPS = {"scatter_", "scatter", "scatter_update",
                "index_add", "index_add_", "index_add__ub", "index_add_ub",
                "atomic_add"}
_RESHAPE_OPS = {"reshape", "unsqueeze", "pad", "fillpad", "gathermask"}
_ASSEMBLE_OPS = {"assemble"}
_VIEW_OPS = {"view"}
_CAST_OPS = {"cast"}
_LAYOUT_OPS = {"permute", "transpose", "pack", "unpack", "concat",
               "interleave", "deinterleave", "expand_clone", "one_hot"}
_CONFIG_OPS = {"set_pass_options", "set_operation_options", "set_cache_policy",
               "set_host_options", "set_codegen_options", "set_verify_options",
               "set_debug_options", "set_print_options", "set_semantic_label",
               "set_matrix_size", "set_build_static", "reset_options", "options"}
_LOOP_OPS = {"loop", "loop_unroll"}
# Max sequential ops between matmul and reduce for R1 classification. If a
# reduce appears beyond this many ops downstream of a matmul (without an
# intervening matmul), the chain is cut as R3 (reduce too far to drive R1
# tile maximization). Named constant — do not inline.
_R1_R3_CHAIN_MAX_OPS = 10
# Op classes that constitute real tensor work; a segment without at least one
# of these is NOT a boundary (config-only regions are skipped).
_REAL_OP_CLASSES = {"vector-op", "reduce-op", "reshape", "assemble", "view", "layout"}
_REAL_OP_LABELS = {"vector-op": "vec", "reduce-op": "reduce",
                   "reshape": "reshape", "assemble": "assemble", "view": "view",
                   "layout": "layout"}

# Rule 4 Step 1a/1b 建议首步 formula (text only — this tool never applies tiles).
_RULE4_FORMULA = (
    "Rule 4 Step 1a (whole≤UB → set_vec_tile_shapes(whole_rows, col)) / "
    "Step 1b (whole>UB → rows = UB_budget // (col × dtype_bytes); "
    "UB_size per platform, see npu-memory-arch.md §一)"
)


def _dotted_name(node: ast.AST) -> str:
    """Extract a dotted attribute chain (e.g. pypto.set_vec_tile_shapes)."""
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _short_name(dotted: str) -> str:
    """Return the last component (e.g. matmul from pypto.matmul)."""
    return dotted.rsplit(".", 1)[-1] if dotted else ""


def _classify(dotted: str) -> str:
    short = _short_name(dotted)
    if short in _CUBE_OPS:
        return "cube-op"
    if short in _TILE_SETTERS_VEC:
        return "vec-tile-set"
    if short in _TILE_SETTERS_CUBE:
        return "cube-tile-set"
    if short in _REDUCES:
        return "reduce-op"
    if short in _RESHAPE_OPS:
        return "reshape"
    if short in _ASSEMBLE_OPS:
        return "assemble"
    if short in _VIEW_OPS:
        return "view"
    if short in _LAYOUT_OPS:
        return "layout"
    if short in _CONFIG_OPS:
        return "config"
    if short in _LOOP_OPS:
        return "loop"
    if dotted.startswith("pypto.") or dotted.startswith("pypto_experimental."):
        return "vector-op"
    return "other"


def _literal_int_args(call: ast.Call) -> bool:
    """True if every positional/keyword arg is an integer literal (statically resolvable)."""
    if not call.args and not call.keywords:
        return False
    return all(
        isinstance(a, ast.Constant) and isinstance(a.value, int)
        for a in list(call.args) + [kw.value for kw in call.keywords]
    )


def _call_args_src(call: ast.Call) -> str:
    """Unparsed positional args of a call (for reporting active tile values)."""
    try:
        return ", ".join(ast.unparse(a) for a in call.args)
    except Exception:
        return ""


def _view_shape_src(call: ast.Call) -> str:
    """Shape argument (2nd positional) of a view call, e.g. '[S2, Dk]'."""
    if len(call.args) >= 2:
        try:
            return ast.unparse(call.args[1])
        except Exception:
            return "?"
    return "?"


def _matmul_trans_flags(call: ast.Call) -> tuple[str, str]:
    """(a_trans, b_trans) kwargs of a cube-op call ('True'/'False'), or 'absent'."""
    flags = {"a_trans": "absent", "b_trans": "absent"}
    for kw in call.keywords:
        if kw.arg in flags:
            try:
                flags[kw.arg] = ast.unparse(kw.value)
            except Exception:
                flags[kw.arg] = "?"
    return flags["a_trans"], flags["b_trans"]


def _real_ops(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter a segment's ops down to real tensor work (vec/reduce/reshape/assemble)."""
    return [op for op in ops if op["class"] in _REAL_OP_CLASSES]


def _exclusivity_warning(types: set[str]) -> str:
    """Build the non-matmul consumer warning if the view has mixed consumers."""
    extras = types - {"matmul-matrix"}
    if extras and "matmul-matrix" in types:
        return (
            f"⚠ view also consumed by non-matmul ops ({', '.join(sorted(extras))}) "
            f"— R4 tile alignment may conflict with vector consumer tile needs"
        )
    return ""


def _origins_at(mapping: dict[str, list[tuple[int, set[int]]]],
                var: str, line: int) -> set[int]:
    """Return origins from the last assignment to *var* at or before *line*.

    Position-aware lookup: when the same var name is reused for multiple
    matmul outputs (common in looped code), only the assignment visible at
    *line* (the most recent one) contributes its origins.
    """
    result: set[int] = set()
    for entry_line, origins in mapping.get(var, []):
        if entry_line <= line:
            result = origins
        else:
            break
    return result


def _latest_view_line(view_lines: dict[str, list[int]],
                      var: str, before_line: int) -> int | None:
    """Return the latest view assignment to *var* at or before *before_line*.

    Position-aware lookup: handles var reuse across loop iterations where the
    same name (e.g. ``q_sub``) is assigned by multiple ``view`` calls.
    """
    result: int | None = None
    for vl in view_lines.get(var, []):
        if vl <= before_line:
            result = vl
        else:
            break
    return result


def _flush_v_segment(boundaries: list[dict[str, Any]],
                     v_segment_ops: list[dict[str, Any]],
                     v_segment_start: int | None,
                     last_vec_tile_line: int | None) -> None:
    """Emit a V-segment boundary only if the segment contains real tensor work.

    Config/loop-only regions are skipped: they are not boundaries.  The detail
    carries a per-class op count so structural applicability predicates
    (F-17/F-20/F-15/...) can be evaluated mechanically from this line alone.
    """
    real = _real_ops(v_segment_ops)
    if not real:
        return
    counts: dict[str, int] = {}
    for op in real:
        counts[op["class"]] = counts.get(op["class"], 0) + 1
    parts = [f"{counts[k]} {label}" for k, label in _REAL_OP_LABELS.items() if k in counts]
    start = v_segment_start if v_segment_start is not None else real[0]["line"]
    since_txt = (
        f"since last vec-tile-set (L{last_vec_tile_line})"
        if last_vec_tile_line is not None
        else "before any set_vec_tile_shapes"
    )
    boundaries.append({
        "type": "V-segment",
        "location": f"L{start}-L{real[-1]['line']}",
        "detail": f"{len(real)} real ops ({', '.join(parts)}) {since_txt}",
        "f_rule": "F-17",
    })


def _is_jit_decorated(func: ast.FunctionDef) -> bool:
    for dec in func.decorator_list:
        if isinstance(dec, ast.Name):
            name = dec.id
        elif isinstance(dec, ast.Attribute):
            name = _dotted_name(dec)
        elif isinstance(dec, ast.Call):
            name = _dotted_name(dec.func) if isinstance(dec.func, (ast.Attribute, ast.Name)) else ""
        else:
            name = ""
        if name and "jit" in name:
            return True
    return False


def _source_segment(source: str, node: ast.AST) -> str:
    try:
        seg = ast.get_source_segment(source, node)
        if seg:
            return seg.strip().replace("\n", " ")
    except Exception as e:
        logging.debug("source segment extraction failed: %s", e)
    return ""


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _collect_calls(func: ast.FunctionDef) -> list[ast.Call]:
    """Collect all Call nodes in function, sorted by source position."""
    calls = [node for node in ast.walk(func) if isinstance(node, ast.Call)]
    calls.sort(key=lambda n: (n.lineno, n.col_offset))
    return calls


def _build_row(idx: int, call: ast.Call, source: str) -> dict[str, Any]:
    """Classify a single call into a row dict for analysis."""
    dotted = _dotted_name(call.func) if isinstance(call.func, (ast.Attribute, ast.Name)) else ""
    classification = _classify(dotted)
    snippet = _source_segment(source, call)
    if len(snippet) > 120:
        snippet = snippet[:117] + "..."
    a_trans = b_trans = operand_vars = None
    if classification == "cube-op":
        a_trans, b_trans = _matmul_trans_flags(call)
        operand_vars = {
            "A": call.args[0].id if call.args and isinstance(call.args[0], ast.Name) else None,
            "B": call.args[1].id if len(call.args) > 1 and isinstance(call.args[1], ast.Name) else None,
        }
    return {
        "idx": idx,
        "line": call.lineno,
        "call": dotted or "<unknown>",
        "short": _short_name(dotted),
        "class": classification,
        "snippet": snippet,
        "arg_vars": [
            name
            for a in list(call.args) + [kw.value for kw in call.keywords]
            for name in _collect_names(a)
        ],
        "tile_literal": (
            _literal_int_args(call)
            if classification in ("vec-tile-set", "cube-tile-set") else None
        ),
        "args_src": (
            _call_args_src(call)
            if classification in ("vec-tile-set", "cube-tile-set") else None
        ),
        "shape_src": (
            _view_shape_src(call) if classification == "view" else None
        ),
        "a_trans": a_trans,
        "b_trans": b_trans,
        "operand_vars": operand_vars,
    }


def _build_rows(calls: list[ast.Call], source: str) -> list[dict[str, Any]]:
    """Build classified row dicts from sorted call list."""
    return [_build_row(idx, call, source) for idx, call in enumerate(calls, 1)]


def _build_view_var_line(func: ast.FunctionDef) -> dict[str, list[int]]:
    """Pre-pass: var name → sorted list of view assignment lines."""
    view_var_line: dict[str, list[int]] = {}
    for node in ast.walk(func):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        vd = _dotted_name(node.value.func) if isinstance(node.value.func, (ast.Attribute, ast.Name)) else ""
        if _short_name(vd) not in _VIEW_OPS:
            continue
        for t in node.targets:
            if isinstance(t, ast.Name):
                view_var_line.setdefault(t.id, []).append(node.value.lineno)
    for _, lines in view_var_line.items():
        lines.sort()
    return view_var_line


def _build_var_origin(func: ast.FunctionDef) -> dict[str, list[tuple[int, set[int]]]]:
    """Pre-pass: var name → list of (line, matmul origins) entries."""
    var_origin: dict[str, list[tuple[int, set[int]]]] = {}
    assign_nodes = sorted(
        (n for n in ast.walk(func)
         if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)),
        key=lambda n: n.lineno,
    )
    for stmt in assign_nodes:
        call = stmt.value
        short = _short_name(_dotted_name(call.func)) if isinstance(call.func, (ast.Attribute, ast.Name)) else ""
        targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
        arg_vars = [
            name
            for a in list(call.args) + [kw.value for kw in call.keywords]
            for name in _collect_names(a)
        ]
        origins: set[int] = set()
        for v in arg_vars:
            origins.update(_origins_at(var_origin, v, stmt.lineno))
        if short in _CUBE_OPS:
            origins.add(stmt.lineno)
        for t in targets:
            var_origin.setdefault(t, []).append((stmt.lineno, origins))
    return var_origin


def _classify_view_consumer(
    var: str,
    vl: int,
    live_end: float,
    rows: list[dict[str, Any]],
) -> tuple[set[str], list[tuple[int, str, str]]]:
    """Collect consumer op classes and details within a view's live range."""
    consumers: set[str] = set()
    consumer_list: list[tuple[int, str, str]] = []
    for r in rows:
        rline = r["line"]
        if rline <= vl or rline >= live_end:
            continue
        if var not in r["arg_vars"]:
            continue
        rcls = r["class"]
        if rcls == "cube-op":
            op_vars = r.get("operand_vars") or {}
            if var == op_vars.get("A") or var == op_vars.get("B"):
                consumers.add("matmul-matrix")
            else:
                consumers.add("matmul-other")
        else:
            consumers.add(rcls)
        consumer_list.append((rline, r["short"], rcls))
    return consumers, consumer_list


def _build_view_consumers(
    rows: list[dict[str, Any]],
    view_var_line: dict[str, list[int]],
) -> tuple[dict[int, set[str]], dict[int, list[tuple[int, str, str]]]]:
    """Pre-pass: for each view line, collect consumer op classes and details."""
    view_consumer_types: dict[int, set[str]] = {}
    view_consumer_lines: dict[int, list[tuple[int, str, str]]] = {}
    for var, vlines in view_var_line.items():
        for i, vl in enumerate(vlines):
            live_end = vlines[i + 1] if i + 1 < len(vlines) else float("inf")
            consumers, consumer_list = _classify_view_consumer(var, vl, live_end, rows)
            view_consumer_types[vl] = consumers
            view_consumer_lines[vl] = consumer_list
    return view_consumer_types, view_consumer_lines


def _annotate_view_consumer(
    vline: int,
    view_row_by_line: dict[int, dict[str, Any]],
    view_consumer_types: dict[int, set[str]],
    view_consumer_lines: dict[int, list[tuple[int, str, str]]],
) -> None:
    """Annotate a view boundary row with its full consumer set."""
    vr = view_row_by_line.get(vline)
    if vr is None or "consumer: pending" not in vr["detail"]:
        return
    clines = view_consumer_lines.get(vline, [])
    if not clines:
        vr["detail"] = vr["detail"].replace(
            "consumer: pending",
            "consumer: (no consumer found in function)",
        )
        return
    parts = []
    for cline, cshort, ccls in clines:
        label = _REAL_OP_LABELS.get(ccls, ccls)
        parts.append(f"'{cshort}' ({label}) at L{cline}")
    consumer_txt = ", ".join(parts)
    types = view_consumer_types.get(vline, set())
    warning = _exclusivity_warning(types)
    if warning:
        vr["detail"] = vr["detail"].replace(
            "consumer: pending",
            f"consumer: {consumer_txt}; {warning}",
        )
    else:
        vr["detail"] = vr["detail"].replace(
            "consumer: pending",
            f"consumer: {consumer_txt}",
        )


def _handle_vec_tile_set(
    r: dict[str, Any], st: dict[str, Any],
) -> None:
    """Process a vec-tile-set call: flush V-segment, update tile state."""
    _flush_v_segment(st["boundaries"], st["v_ops"], st["v_start"], st["last_vec_tile"])
    st["v_ops"] = []
    st["last_vec_tile"] = r["line"]
    st["last_vec_args"] = r.get("args_src")
    st["v_start"] = r["line"]


def _handle_cube_tile_set(
    r: dict[str, Any], st: dict[str, Any],
) -> None:
    """Process a cube-tile-set call: flush V-segment, update tile state."""
    _flush_v_segment(st["boundaries"], st["v_ops"], st["v_start"], st["last_vec_tile"])
    st["v_ops"] = []
    st["last_cube_tile"] = r["line"]
    st["last_cube_args"] = r.get("args_src")
    st["v_start"] = r["line"]


def _handle_cube_op(
    r: dict[str, Any], st: dict[str, Any],
    view_var_line: dict[str, list[int]],
    view_consumer_types: dict[int, set[str]],
    view_active_tile: dict[int, tuple[int | None, str]],
) -> None:
    """Process a cube-op: detect vector→cube boundaries, start R1/R3 chain."""
    ln = r["line"]
    _flush_v_segment(st["boundaries"], st["v_ops"], st["v_start"], st["last_vec_tile"])
    batched = r["short"] == "transposed_batchmatmul"
    for role in ("A", "B"):
        var = (r.get("operand_vars") or {}).get(role)
        if not var:
            continue
        vline = _latest_view_line(view_var_line, var, ln)
        if vline is None:
            continue
        _emit_vector_to_cube(r, ln, var, vline, role, batched, st, view_consumer_types, view_active_tile)
    st["v_ops"] = []
    if ln not in st["pending_chains"]:
        st["pending_chains"][ln] = {
            "type": "cube→vector",
            "location": "pending (no downstream consumer yet)",
            "detail": f"matmul at L{ln}; chain extends to next matmul/end/<{_R1_R3_CHAIN_MAX_OPS} ops",
            "f_rule": "F-20 R3",
        }
        st["chain_counts"][ln] = 0
    st["v_start"] = ln


def _emit_vector_to_cube(
    r: dict[str, Any], ln: int, var: str, vline: int, role: str,
    batched: bool, st: dict[str, Any],
    view_consumer_types: dict[int, set[str]],
    view_active_tile: dict[int, tuple[int | None, str]],
) -> None:
    """Emit a vector→cube boundary for a view feeding a matmul operand."""
    last_vec = st["last_vec_tile"]
    tile_state = (
        f"tile-set present at L{last_vec} after the view — verify it matches Step 1a/1b"
        if last_vec is not None and last_vec > vline
        else "no tile-set after the view — insert per Step 1a/1b"
    )
    a_line, a_args = view_active_tile.get(vline, (None, None))
    active_txt = f"L{a_line}: ({a_args})" if a_args else "none recorded"
    last_cube = st["last_cube_tile"]
    last_cube_args = st["last_cube_args"]
    cube_txt = (
        f"L{last_cube}: ({last_cube_args})"
        if last_cube_args else "none before matmul"
    )
    if batched:
        operand_txt = f"{role} operand, no trans flags"
        align_txt = "batched op — orientation fixed: A (M,B,K), B (B,K,N); 2D trans map N/A"
    elif role == "A":
        operand_txt = f"{role} operand, a_trans={r.get('a_trans') or 'absent'}"
        align_txt = (
            "view vec_tile must == matmul L1 tile per a_trans map "
            "(True: view dim0↔k_l1, dim1↔m_l1; False: dim0↔m_l1, dim1↔k_l1)"
        )
    else:
        operand_txt = f"{role} operand, b_trans={r.get('b_trans') or 'absent'}"
        align_txt = (
            "view vec_tile must == matmul L1 tile per b_trans map "
            "(True: view dim0↔n_l1, dim1↔k_l1; False: view dim0↔k_l1, dim1↔n_l1)"
        )
    types = view_consumer_types.get(vline, set())
    warning = _exclusivity_warning(types)
    excl_warning = f"; {warning}" if warning else ""
    st["boundaries"].append({
        "type": "vector→cube",
        "location": f"L{vline}→L{ln}",
        "detail": (
            f"view '{var}' at L{vline} feeds matmul at L{ln} ({operand_txt}); "
            f"vec tile active before view: {active_txt}; {tile_state} — {_RULE4_FORMULA}; "
            f"cube tile before matmul: {cube_txt}; "
            f"⛔ alignment check (评估信号): {align_txt} — "
            f"mismatch → evaluate F-20 R4/Rule 7 for this site, not a verdict"
            f"{excl_warning}"
        ),
        "f_rule": "F-20 R4",
    })


def _handle_view(
    r: dict[str, Any], st: dict[str, Any],
    line_var_map: dict[int, list[str]],
    view_consumer_types: dict[int, set[str]],
    view_consumer_lines: dict[int, list[tuple[int, str, str]]],
    view_row_by_line: dict[int, dict[str, Any]],
    view_active_tile: dict[int, tuple[int | None, str]],
) -> None:
    """Process a view call: emit per-site view boundary row."""
    ln = r["line"]
    var = next(iter(line_var_map.get(ln, [])), None)
    shape_src = r.get("shape_src") or "?"
    last_vec = st["last_vec_tile"]
    last_vec_args = st["last_vec_args"]
    if last_vec is not None:
        tile_before = f"active vec tile before view (L{last_vec}): ({last_vec_args})"
    else:
        tile_before = "active vec tile before view: none (no set_vec_tile_shapes before this view)"
    st["boundaries"].append({
        "type": "view",
        "location": f"L{ln}",
        "detail": (
            f"{r['call']}{' (' + var + ')' if var else ''} — view shape {shape_src}; "
            f"{tile_before}; "
            f"tile != view shape → evaluate F-20 R4 (set matching tile per Step 1a/1b) / "
            f"R3 (vec-follows-cube) / F-10 for this site — "
            f"评估信号（决定该站点做哪个优化），不是'已/未应用'判定; consumer: pending"
        ),
        "f_rule": "F-20 R4",
    })
    if var:
        view_row_by_line[ln] = st["boundaries"][-1]
        view_active_tile[ln] = (last_vec, last_vec_args)
        _annotate_view_consumer(ln, view_row_by_line, view_consumer_types, view_consumer_lines)
    st["v_ops"].append(r)


def _handle_reshape(r: dict[str, Any], st: dict[str, Any]) -> None:
    """Process a reshape call: emit reshape boundary."""
    st["boundaries"].append({
        "type": "reshape",
        "location": f"L{r['line']}",
        "detail": f"{r['call']} — vec tile must match source shape before & target shape after",
        "f_rule": "F-17/F-4",
    })
    st["v_ops"].append(r)


def _handle_assemble(r: dict[str, Any], st: dict[str, Any]) -> None:
    """Process an assemble call: emit assemble boundary."""
    st["boundaries"].append({
        "type": "assemble",
        "location": f"L{r['line']}",
        "detail": f"{r['call']} — vec tile must match target (assemble destination) shape",
        "f_rule": "F-17",
    })
    st["v_ops"].append(r)


def _handle_vector_reduce_layout(
    r: dict[str, Any], st: dict[str, Any],
    var_origin: dict[str, list[tuple[int, set[int]]]],
) -> None:
    """Process vector/reduce/layout op: R1/R3 chain dataflow + F-17 points."""
    ln = r["line"]
    cls = r["class"]
    touched_matmuls: set[int] = set()
    for var in r["arg_vars"]:
        touched_matmuls.update(_origins_at(var_origin, var, ln))
    _update_r1_r3_chains(r, ln, cls, touched_matmuls, st)
    if cls == "reduce-op" or r["short"] in _CAST_OPS:
        _emit_f17_point(r, ln, cls)
    st["v_ops"].append(r)


def _update_r1_r3_chains(
    r: dict[str, Any], ln: int, cls: str,
    touched_matmuls: set[int], st: dict[str, Any],
) -> None:
    """Update pending R1/R3 chains for a downstream consumer op."""
    pending_chains = st["pending_chains"]
    chain_counts = st["chain_counts"]
    for ml in list(pending_chains.keys()):
        if ml not in touched_matmuls:
            continue
        if chain_counts[ml] == 0:
            pending_chains[ml]["location"] = f"L{ml}→L{ln}"
            pending_chains[ml]["detail"] = (
                f"matmul at L{ml} followed by {cls.replace('-op', '')} op "
                f"'{r['short']}' at L{ln}; chain extends to next matmul/end/"
                f"<{_R1_R3_CHAIN_MAX_OPS} ops"
            )
        chain_counts[ml] += 1
        if cls == "reduce-op":
            pending_chains[ml]["f_rule"] = "F-20 R1"
            pending_chains[ml]["detail"] += f"; reduce at L{ln} → R1"
            st["boundaries"].append(pending_chains.pop(ml))
            chain_counts.pop(ml)
        elif chain_counts[ml] >= _R1_R3_CHAIN_MAX_OPS:
            pending_chains[ml]["detail"] += (
                f"; chain cutoff at {chain_counts[ml]} ops "
                f"(≥{_R1_R3_CHAIN_MAX_OPS}) → R3"
            )
            st["boundaries"].append(pending_chains.pop(ml))
            chain_counts.pop(ml)


def _emit_f17_point(r: dict[str, Any], ln: int, cls: str) -> None:
    """Emit an F-17 shape-change point boundary for reduce/cast ops."""
    if r["short"] in _SCATTER_OPS:
        detail = (
            f"'{r['short']}' at L{ln} (scatter/index axis is non-split) — "
            f"col >= non-split axis length; constraint scoped to the segment "
            f"containing this op (F-10 scatter rule); "
            f"verify segment tile satisfies it, document N-A if satisfied"
        )
    else:
        kind = "reduce output [M,N]→[M,1]" if cls == "reduce-op" else "dtype cast"
        detail = (
            f"'{r['short']}' at L{ln} ({kind}) — shape/dtype change point: "
            f"per-point retile decision required "
            f"([M,1]-matched tile before reduce consumers, restore [M,N]-matched after; "
            f"cast-matched tile across dtype boundary); "
            f"evaluate once under current ambient tile, document N-A if no niche"
        )
    _f17_boundaries.append({
        "type": "F-17 point",
        "location": f"L{ln}",
        "detail": detail,
        "f_rule": "F-17",
    })


_f17_boundaries: list[dict[str, Any]] = []


def _detect_boundaries(
    rows: list[dict[str, Any]],
    view_var_line: dict[str, list[int]],
    var_origin: dict[str, list[tuple[int, set[int]]]],
    view_consumer_types: dict[int, set[str]],
    view_consumer_lines: dict[int, list[tuple[int, str, str]]],
    line_var_map: dict[int, list[str]],
) -> list[dict[str, Any]]:
    """Run the main boundary detection loop over classified rows."""
    boundaries: list[dict[str, Any]] = []
    st: dict[str, Any] = {
        "boundaries": boundaries,
        "last_vec_tile": None, "last_vec_args": None,
        "last_cube_tile": None, "last_cube_args": None,
        "pending_chains": {}, "chain_counts": {},
        "v_ops": [], "v_start": None,
    }
    view_row_by_line: dict[int, dict[str, Any]] = {}
    view_active_tile: dict[int, tuple[int | None, str]] = {}
    for r in rows:
        cls = r["class"]
        if cls == "vec-tile-set":
            _handle_vec_tile_set(r, st)
        elif cls == "cube-tile-set":
            _handle_cube_tile_set(r, st)
        elif cls == "cube-op":
            _handle_cube_op(r, st, view_var_line, view_consumer_types, view_active_tile)
        elif cls == "view":
            _handle_view(r, st, line_var_map, view_consumer_types,
                         view_consumer_lines, view_row_by_line, view_active_tile)
        elif cls == "reshape":
            _handle_reshape(r, st)
        elif cls == "assemble":
            _handle_assemble(r, st)
        elif cls in ("vector-op", "reduce-op", "layout"):
            _handle_vector_reduce_layout(r, st, var_origin)
        else:
            st["v_ops"].append(r)
    _flush_v_segment(boundaries, st["v_ops"], st["v_start"], st["last_vec_tile"])
    _finalize_chains(st["pending_chains"], st["chain_counts"], boundaries)
    _finalize_views(view_row_by_line)
    boundaries.extend(_f17_boundaries)
    _f17_boundaries.clear()
    return boundaries


def _finalize_chains(
    pending_chains: dict[int, dict[str, Any]],
    chain_counts: dict[int, int],
    boundaries: list[dict[str, Any]],
) -> None:
    """Finalize remaining R1/R3 chains at end of graph → R3."""
    for ml, pb in pending_chains.items():
        if chain_counts[ml] == 0:
            pb["detail"] = f"matmul at L{ml} with no downstream consumer found → R3"
        else:
            pb["detail"] += "; end of graph → R3"
        boundaries.append(pb)
    pending_chains.clear()
    chain_counts.clear()


def _finalize_views(view_row_by_line: dict[int, dict[str, Any]]) -> None:
    """Catch any leftover views with unresolved consumers."""
    for vr in view_row_by_line.values():
        if "consumer: pending" in vr["detail"]:
            vr["detail"] = vr["detail"].replace(
                "consumer: pending", "consumer: (no consumer found in function)"
            )


def _build_summary(rows: list[dict[str, Any]], boundaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the summary dict from rows and boundaries."""
    return {
        "matmul_calls": sum(1 for r in rows if r["class"] == "cube-op"),
        "vec_tile_set_calls": sum(1 for r in rows if r["class"] == "vec-tile-set"),
        "cube_tile_set_calls": sum(1 for r in rows if r["class"] == "cube-tile-set"),
        "reduce_calls": sum(1 for r in rows if r["class"] == "reduce-op"),
        "layout_calls": sum(1 for r in rows if r["class"] == "layout"),
        "reshape_calls": sum(1 for r in rows if r["class"] == "reshape"),
        "assemble_calls": sum(1 for r in rows if r["class"] == "assemble"),
        "view_calls": sum(1 for r in rows if r["class"] == "view"),
        "non_literal_tile_sets": sum(
            1 for r in rows
            if r["class"] in ("vec-tile-set", "cube-tile-set") and r["tile_literal"] is False
        ),
        "cube_to_vector_boundaries": sum(1 for b in boundaries if b["type"] == "cube→vector"),
        "vector_to_cube_boundaries": sum(1 for b in boundaries if b["type"] == "vector→cube"),
        "view_site_rows": sum(1 for b in boundaries if b["type"] == "view"),
        "f17_shape_change_points": sum(1 for b in boundaries if b["type"] == "F-17 point"),
        "v_segments": sum(1 for b in boundaries if b["type"] == "V-segment"),
        "f21_hoist_sites": sum(1 for b in boundaries if b["type"] == "F-21 hoist"),
        "total_boundaries": len(boundaries),
    }


# ---------------------------------------------------------------------------
# F-21 dependency-depth analysis
# ---------------------------------------------------------------------------

def _collect_names(node: ast.AST) -> list[str]:
    """Recursively extract all Name identifiers from an AST expression.

    Handles arithmetic (BinOp), call args, subscripts, attribute bases —
    anything that references a variable transitively.
    """
    names: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.append(child.id)
    return names


def _loop_name_kwarg(iter_call: ast.Call) -> str | None:
    """Extract the 'name' keyword value from a loop call, if present."""
    for kw in iter_call.keywords:
        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value)
    return None


def _loop_target_vars(target: ast.expr) -> tuple[str, ...]:
    """Extract idx variable names from a For target (Name or Tuple unpack).

    Target may be a single Name or a Tuple (loop_unroll unpacking
    e.g. `for bs_offset, tile_bs in pypto.loop_unroll(...)`) — every element
    is a loop variable the body can reference.
    """
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, ast.Tuple):
        return tuple(
            elt.id for elt in target.elts
            if isinstance(elt, ast.Name)
        )
    return ("?",)


def _find_loop_nesting(func: ast.FunctionDef) -> list[dict[str, Any]]:
    """Find pypto.loop / pypto.loop_unroll For statements with nesting depth.

    Returns list of {"name", "idx_var", "line", "end_line", "depth"} sorted
    by line.  idx_var is the For TARGET variable — loop-body code references
    it (idx_name kwarg is a trace label only, kept in "name" for display).
    end_line covers the full loop body (used for enclosure tests).
    """
    loops: list[dict[str, Any]] = []

    def _visit(node: ast.AST, depth: int) -> None:
        if (isinstance(node, ast.For) and isinstance(node.iter, ast.Call)):
            dotted = (_dotted_name(node.iter.func)
                      if isinstance(node.iter.func, (ast.Attribute, ast.Name)) else "")
            if _short_name(dotted) in _LOOP_OPS:
                idx_vars = _loop_target_vars(node.target)
                loop_name = idx_vars[0]
                name_kw = _loop_name_kwarg(node.iter)
                if name_kw:
                    loop_name = name_kw
                loops.append({"name": loop_name, "idx_var": idx_vars,
                              "line": node.lineno,
                              "end_line": node.end_lineno or node.lineno,
                              "depth": depth})
                for s in node.body:
                    _visit(s, depth + 1)
                return
        for stmt in ast.iter_child_nodes(node):
            _visit(stmt, depth)

    _visit(func, 0)
    loops.sort(key=lambda loop: loop["line"])
    return loops


def _build_loop_var_deps(
    func: ast.FunctionDef,
    loop_nesting: list[dict[str, Any]],
) -> dict[str, list[tuple[int, set[str]]]]:
    """Propagate loop-variable dependencies through assignments.

    Returns var → list of (line, {loop idx vars}) — position-aware, same
    pattern as _build_var_origin.  Function params carry empty dependency
    sets; loop idx variables depend on themselves; each assignment inherits
    the union of Name dependencies of its value expression.
    """
    idx_vars = {name for loop in loop_nesting for name in loop["idx_var"]}
    deps: dict[str, list[tuple[int, set[str]]]] = {}
    assign_nodes = sorted(
        (n for n in ast.walk(func) if isinstance(n, ast.Assign)),
        key=lambda n: n.lineno,
    )
    for stmt in assign_nodes:
        value_names = _collect_names(stmt.value)
        current: set[str] = set()
        for name in value_names:
            if name in idx_vars:
                current.add(name)
                continue
            prior = _loop_deps_at(deps, name, stmt.lineno)
            current.update(prior)
        targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
        for t in targets:
            deps.setdefault(t, []).append((stmt.lineno, frozenset(current)))
    return deps


def _loop_deps_at(
    deps: dict[str, list[tuple[int, set[str]]]], var: str, line: int
) -> set[str]:
    """Position-aware lookup: loop deps visible for var at line (latest assignment)."""
    result: set[str] = set()
    for entry_line, entry_deps in deps.get(var, []):
        if entry_line <= line:
            result = set(entry_deps)
        else:
            break
    return result


def _subscript_base_name(target: ast.expr) -> str | None:
    """Return the base Name id of a Subscript store target, or None."""
    if not isinstance(target, ast.Subscript):
        return None
    base = target.value
    while isinstance(base, ast.Subscript):
        base = base.value
    return base.id if isinstance(base, ast.Name) else None


def _record_mutated_targets(
    targets: list[ast.expr], lineno: int,
    mutated: dict[str, list[int]],
) -> None:
    """Record Subscript-store targets as mutated vars in *mutated*."""
    for t in targets:
        base_id = _subscript_base_name(t)
        if base_id:
            mutated.setdefault(base_id, []).append(lineno)


def _build_mutated_vars(func: ast.FunctionDef) -> dict[str, list[int]]:
    """Find vars that are Subscript-store targets (``buf[:] = ...``).

    These are loop-carried persistent buffers (F7 pattern): their VALUE at a
    read site depends on prior loop iterations even when the NAME carries no
    loop-var dependency.  Reads of mutated vars must be treated conservatively
    (dependent on all enclosing loop vars) to avoid false F-21 hoist flags.
    """
    mutated: dict[str, list[int]] = {}
    for node in ast.walk(func):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            _record_mutated_targets(targets, node.lineno, mutated)
    return mutated


def _detect_f21_sites(
    rows: list[dict[str, Any]],
    loop_var_deps: dict[str, list[tuple[int, set[str]]]],
    loop_nesting: list[dict[str, Any]],
    mutated_vars: dict[str, list[int]] | None = None,
) -> list[dict[str, Any]]:
    """Classify operations inside pypto.loop nesting by dependency depth (F-21).

    For each operation, compare its loop-var dependency set against the set of
    ALL enclosing pypto.loop idx variables at its position:
      - deps == enclosing  → fully dependent (stays in loop — not a site)
      - 0 < deps < enclosing → partially dependent → F-21 site (precompute)
      - deps == {} → fully static → F-21 site (hoist before all loops)

    Dependency resolution per arg var:
      - var is itself a loop idx var → depends on itself
      - var is a Subscript-mutation target (loop-carried buffer) →
        conservatively depends on ALL enclosing vars (value carried across
        iterations; static name analysis cannot see [:]= dataflow)
      - otherwise → assignment-derived deps (_loop_deps_at)
    """
    if not loop_nesting:
        return []
    mutated_vars = mutated_vars or {}
    idx_vars_all = {name for loop in loop_nesting for name in loop["idx_var"]}
    boundaries: list[dict[str, Any]] = []
    for r in rows:
        if r["class"] in ("loop", "config", "vec-tile-set", "cube-tile-set", "other"):
            continue  # loop statements / option+tile setters / host builtins are not hoistable operations
        ln = r["line"]
        enclosing = [loop for loop in loop_nesting
                     if loop["line"] < ln <= loop["end_line"]]
        if not enclosing:
            continue
        enclosing_vars: set[str] = set()
        for loop in enclosing:
            enclosing_vars.update(loop["idx_var"])
        op_deps: set[str] = set()
        for var in r["arg_vars"]:
            if var in idx_vars_all:
                op_deps.add(var)
            elif var in mutated_vars:
                op_deps.update(enclosing_vars)
            else:
                op_deps.update(_loop_deps_at(loop_var_deps, var, ln))
        if op_deps == enclosing_vars:
            continue
        depth = len(enclosing)
        dep_names = sorted(op_deps)
        missing = sorted(enclosing_vars - op_deps)
        if not op_deps:
            detail = (
                f"'{r['short']}' at L{ln} depends on 0/{depth} loop vars "
                f"(enclosing: {', '.join(sorted(enclosing_vars))}) — fully static, "
                f"hoist before all loops (F-21 core rule)"
            )
        else:
            detail = (
                f"'{r['short']}' at L{ln} depends on {len(op_deps)}/{depth} loop vars "
                f"({', '.join(dep_names)}; missing: {', '.join(missing)}) — "
                f"partially dependent, precompute per missing-var level "
                f"(independent pypto.loop → GM buffer → view in main loop)"
            )
        boundaries.append({
            "type": "F-21 hoist",
            "location": f"L{ln}",
            "detail": detail,
            "f_rule": "F-21",
        })
    return boundaries


def analyze_function(func: ast.FunctionDef, source: str) -> dict[str, Any]:
    """Analyze one function: collect calls, detect boundaries, return structured data."""
    calls = _collect_calls(func)
    rows = _build_rows(calls, source)
    view_var_line = _build_view_var_line(func)
    var_origin = _build_var_origin(func)
    view_consumer_types, view_consumer_lines = _build_view_consumers(rows, view_var_line)
    line_var_map: dict[int, list[str]] = {}
    for var, vlines in view_var_line.items():
        for vl in vlines:
            line_var_map.setdefault(vl, []).append(var)
    boundaries = _detect_boundaries(
        rows, view_var_line, var_origin,
        view_consumer_types, view_consumer_lines, line_var_map,
    )
    loop_nesting = _find_loop_nesting(func)
    loop_var_deps = _build_loop_var_deps(func, loop_nesting)
    mutated_vars = _build_mutated_vars(func)
    boundaries.extend(
        _detect_f21_sites(rows, loop_var_deps, loop_nesting, mutated_vars)
    )
    summary = _build_summary(rows, boundaries)
    return {
        "function": func.name,
        "jit_decorated": _is_jit_decorated(func),
        "call_sequence": rows,
        "boundaries": boundaries,
        "summary": summary,
    }


def analyze_file(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    # Analyze ALL functions: the JIT-decorated entry point is often a thin
    # wrapper that delegates to a plain `_kernel_impl` function containing the
    # actual matmul/tile calls. Tag each with JIT status for the LLM to focus.
    results = [analyze_function(f, source) for f in funcs]
    return {"file": str(path), "functions": results}


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _render_summary_lines(s: dict[str, Any]) -> list[str]:
    """Render the per-function summary block."""
    lines: list[str] = []
    lines.append("### Summary")
    lines.append("")
    lines.append(f"- matmul calls: **{s['matmul_calls']}**")
    non_lit = s['non_literal_tile_sets']
    lines.append(
        f"- set_vec_tile_shapes calls: **{s['vec_tile_set_calls']}** "
        f"(non-literal args: **{non_lit}** — resolve at trace time)"
    )
    lines.append(f"- set_cube_tile_shapes calls: **{s['cube_tile_set_calls']}**")
    lines.append(f"- reduce ops (amax/sum/...): **{s['reduce_calls']}**")
    lines.append(f"- layout ops (permute/transpose/concat/...): **{s['layout_calls']}**")
    lines.append(f"- reshape calls: **{s['reshape_calls']}**")
    lines.append(f"- assemble calls: **{s['assemble_calls']}**")
    lines.append(f"- view calls: **{s['view_calls']}**")
    lines.append(f"- cube→vector boundaries: **{s['cube_to_vector_boundaries']}** (F-20 R1/R3)")
    lines.append(
        f"- vector→cube boundaries: **{s['vector_to_cube_boundaries']}** "
        f"(F-20 R4 — view feeds matmul)"
    )
    lines.append(f"- per-site view rows: **{s['view_site_rows']}** (Rule 4 Step 1a/1b decision each)")
    lines.append(f"- F-17 shape-change points: **{s['f17_shape_change_points']}** (reduce/cast)")
    lines.append(f"- V-segments: **{s['v_segments']}**")
    lines.append(f"- F-21 hoist sites: **{s['f21_hoist_sites']}** (static/partial loop-var dependency)")
    lines.append(f"- total flagged boundaries: **{s['total_boundaries']}**")
    lines.append("")
    return lines


def render_markdown(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Tile Boundary Report: {data['file']}")
    lines.append("")

    for fn in data["functions"]:
        jit_tag = " (@pypto.frontend.jit)" if fn["jit_decorated"] else ""
        lines.append(f"## Function: `{fn['function']}`{jit_tag}")
        lines.append("")

        # call sequence table
        lines.append("### Call sequence")
        lines.append("")
        lines.append("| # | Line | Call | Classification | Snippet |")
        lines.append("|---|------|------|----------------|---------|")
        for r in fn["call_sequence"]:
            cls_cell = r["class"]
            if r["class"] in ("vec-tile-set", "cube-tile-set") and r["tile_literal"] is False:
                cls_cell += " ⚠ non-literal args (resolve at trace time)"
            lines.append(f"| {r['idx']} | L{r['line']} | `{r['call']}` | {cls_cell} | {r['snippet']} |")
        lines.append("")

        # boundaries table
        lines.append("### Flagged boundaries (closed checklist for A4 agenda)")
        lines.append("")
        if fn["boundaries"]:
            lines.append("| # | Type | Location | Detail | F-rule |")
            lines.append("|---|------|----------|--------|--------|")
            for i, b in enumerate(fn["boundaries"], 1):
                lines.append(f"| {i} | {b['type']} | {b['location']} | {b['detail']} | {b['f_rule']} |")
        else:
            lines.append("*(no boundaries flagged)*")
        lines.append("")

        # summary
        lines.extend(_render_summary_lines(fn["summary"]))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A4 agenda
# ---------------------------------------------------------------------------


def _impl_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_BENEFIT_LINES = {
    "F-20 R1": "- F-20 R1 (cube): larger nL0 → fewer tasks → less AIC compute + fewer cube→vec sends",
    "F-20 R2": "- F-20 R2 (cube): fewer tasks → better AIC load balance",
    "F-20 R3": "- F-20 R3 (vec): vec=cube tile → fewer CV handoffs → less AIC wait",
    "F-20 R4": "- F-20 R4 (vec): whole-shape tile → fewer AIV tasks → fewer vec→cube sends + less AIC wait",
    "F-21": "- F-21: hoist partial-static ops → less redundant compute per iteration",
}


def _render_benefit_header(rules: set[str]) -> list[str]:
    """Render per-rule benefit lines for the rules detected in the impl."""
    applicable = [
        text for text in (_BENEFIT_LINES.get(r) for r in sorted(rules))
        if text
    ]
    if not applicable:
        return []
    return ["## Rule benefits (applicable in this impl)", ""] + applicable + [""]


def render_agenda(data: dict[str, Any], impl_path: Path) -> str:
    """A4 agenda skeleton: one PENDING row per flagged boundary site.

    Completeness criterion is mechanical: agenda is done when no PENDING
    remains in any decision column. Rows correspond 1:1 to script-detected
    sites — deleting rows or batch-writing 'no applicable points' is a
    detectable violation (row count must equal the boundary report).
    """
    lines: list[str] = []
    lines.append("# A4 分段 TileShape 优化议程（骨架 — list_tile_boundaries.py 自动生成）")
    lines.append("")
    lines.append(f"- impl: `{impl_path}`")
    lines.append(f"- impl sha256: `{_impl_sha256(impl_path)[:16]}…`")
    lines.append(
        f"- generated: {datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec='seconds')}"
    )
    lines.append("")
    lines.append(
        "> 完成判据（机械）：所有行的 决策列 不再含 PENDING。"
        "每行须给出 适用优化点（结构依据）/ 优先级 / "
        "证据（实测或假设，标注来源）/ 建议首步，"
        "或 N/A + 结构事实。⛔ 禁止删除行、禁止整批写 '无适用点'；"
        "行数必须与 boundary report 的站点数一致。"
        "view→matmul 对齐站点另须给出 a_trans/b_trans 映射核对结论（写入证据列）。"
    )
    lines.append("")
    rules = {b["f_rule"] for fn in data["functions"] for b in fn["boundaries"]}
    lines.extend(_render_benefit_header(rules))
    for fn in data["functions"]:
        lines.append(f"## Function: `{fn['function']}`")
        lines.append("")
        if not fn["boundaries"]:
            lines.append("*(no boundaries flagged)*")
            lines.append("")
            continue
        lines.append(
            "| # | 站点（类型+位置） | F-rule | "
            "适用优化点 | 优先级 | 证据（实测/假设） | 建议首步 | 状态 |"
        )
        lines.append("|---|------------------|--------|-----------|--------|-------------------|----------|------|")
        for i, b in enumerate(fn["boundaries"], 1):
            site = f"{b['type']} {b['location']}"
            lines.append(
                f"| {i} | {site} | {b['f_rule']} | PENDING | PENDING | PENDING | PENDING | PENDING |"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_report(data: dict[str, Any], impl_path: Path, out_path: Path, total: int) -> None:
    """Write the boundary report to --out path with impl sha256 header."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"<!-- impl: {impl_path} | impl sha256: {_impl_sha256(impl_path)[:16]}… "
        f"| generated: {datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec='seconds')} -->\n"
    )
    out_path.write_text(header + render_markdown(data), encoding="utf-8")
    logging.info("boundary report written to %s (%d sites)", out_path, total)


def _write_agenda(data: dict[str, Any], impl_path: Path, agenda_path: Path, total: int) -> None:
    """Write the A4 agenda skeleton to --emit-agenda path."""
    agenda_path.parent.mkdir(parents=True, exist_ok=True)
    agenda_path.write_text(render_agenda(data, impl_path), encoding="utf-8")
    logging.info("A4 agenda skeleton written to %s (%d PENDING rows)", agenda_path, total)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List tile-shape boundaries in a PyPTO kernel impl "
                    "(AST scan, no shape inference).")
    parser.add_argument("impl", help="path to the impl .py file")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="write the boundary report to PATH (impl sha256 recorded in header)")
    parser.add_argument("--emit-agenda", default=None, metavar="PATH",
                        help="write the A4 agenda skeleton (one PENDING row per site) to PATH")
    args = parser.parse_args(argv)

    path = Path(args.impl)
    if not path.is_file():
        logging.error("error: file not found: %s", path)
        return 1

    try:
        data = analyze_file(path)
    except (SyntaxError, UnicodeDecodeError) as e:
        logging.error("error: cannot analyze %s: %s", path, e)
        return 1
    except OSError as e:
        logging.error("error: I/O failure on %s: %s", path, e)
        return 1

    total = sum(len(fn["boundaries"]) for fn in data["functions"])

    if args.out:
        _write_report(data, path, Path(args.out), total)
    if args.emit_agenda:
        _write_agenda(data, path, Path(args.emit_agenda), total)
    if not args.out and not args.emit_agenda:
        logging.info("%s", render_markdown(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
