#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""predicates.py --- what structure does this kernel actually contain?

Answers a handful of yes/no questions about a candidate implementation so the
harness can hold back catalogue actions whose precondition the kernel does not
satisfy. Held, never deleted: the predicates are recomputed after every
accepted candidate, so a transformation that introduces a matmul re-admits the
cube actions that were unreachable a moment earlier.

That distinction is the whole design. A kernel without a matmul today may be
much faster with one tomorrow: on Ascend the Cube pipe has far higher
throughput than Vector, so rewriting a reduction as a matmul against a ones
vector is a real and large optimisation. Deleting the cube actions up front
would remove exactly that possibility.

Detection uses the AST, not text search, so a mention in a comment or a
docstring never counts as evidence.

CONSERVATIVE BY CONSTRUCTION: when a predicate cannot be determined it is
reported True, which admits the action. Wrongly holding an action back removes
a reachable optimisation; wrongly admitting one costs at most a few
evaluations, and the static feasibility gate catches most of those anyway.

Usage:
    preds = compute("custom/<op>/<op>_impl.py", op_dir="custom/<op>")
    python3 predicates.py <impl.py> [--op-dir <dir>]
"""

from dataclasses import dataclass
from typing import Callable

import argparse
import ast
from importlib import import_module
import os
import re
import sys

# Structural event analysis is independent of the DSL. Delay only the DSL
# dependencies for the namespaced library import; the Classic entry keeps its
# established module bindings.
if __package__:
    jsonio = import_module(__package__ + ".jsonio")
    pypto_roots = import_module(__package__ + ".pypto_ast").pypto_roots
else:
    import feasibility
    from pypto_ast import pypto_roots
    import jsonio

# Calls that can only exist if the kernel really has a Cube/matmul operation.
CUBE_CALLS = {
    "matmul",
    "set_cube_tile_shapes",
    "cube_tile",
    "batch_matmul",
    "grouped_matmul",
    "quant_matmul",
}
# Calls that materialise a broadcast/expand of a low-rank tensor.
BROADCAST_CALLS = {
    "expand",
    "expand_clone",
    "broadcast",
    "broadcast_to",
    "combine_axis",
}
# Calls that write a tile back out to global memory.
WRITEBACK_CALLS = frozenset(("assemble", "copy_out", "store", "scatter"))
# Layout work the kernel does per tile that the host could do once. A pypto
# reshape of a view-generated tensor is forced inplace=False -- a physical copy
# on every tile -- and the others are the same trade in a different shape.
# Only pypto calls count: the same file holds the torch wrapper, where doing
# this work is the fix rather than the problem.
LAYOUT_CALLS = {"reshape", "expand", "transpose", "permute", "concat", "cat"}


def _read_text(path, errors="replace"):
    """The file's text, with the handle closed before the caller sees it."""
    with open(path, encoding="utf-8", errors=errors) as f:
        return f.read()


def pypto_calls(tree, roots=None):
    """Every PyPTO call name in the file, attribute or bare.

    PyPTO only. `<op>_impl.py` holds the kernel AND the host-side torch wrapper,
    and collecting by attribute name alone cannot tell them apart: a wrapper's
    `torch.matmul` was recorded as `matmul` and set `has_cube_op`, a
    `torch.expand` counted as evidence of an in-kernel broadcast. Both open
    actions whose delta has nowhere to land, and both make the kernel's search
    space depend on how somebody wrote the wrapper around it.

    `_is_pypto_call` was already the boundary test in this file -- the layout
    and residency predicates use it -- and this is the one place that did not.
    """
    roots = pypto_roots(tree) if roots is None else roots
    return {
        _call_name(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _is_pypto_call(node, roots)
    }


def _chunk_assign(node):
    """An assignment to a chunk-size constant that a loop then uses."""
    return any(
        isinstance(t, ast.Name) and re.fullmatch(r"BT|CHUNK(_SIZE)?|chunk_size", t.id)
        for t in node.targets
    )


def _chunk_constant(tree, chunk_loop):
    """Is there a chunk-shaped loop or constant? Necessary, not sufficient.

    Three pieces of evidence are accepted, and anything weaker is treated as
    absent: see `_chunk_loop` for the two loop forms, and `_chunk_assign` for the
    named constant.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and chunk_loop(node):
            return True
        if isinstance(node, ast.Assign) and _chunk_assign(node):
            return True
    return False


def _target_events(t, out):
    """Assignment-target events, in evaluation order.

    A subscript or attribute target MUTATES the object the base name refers to,
    and that is a store of the name for the purpose of carrying state -- even
    though the `Name` node inside `last_state[:] = ...` carries `ctx=Load`,
    because Python is loading the object in order to write through it. Reading
    the ctx flags alone misses every in-place write, and that is how this repo
    spells carried state: `last_state[:] = cur_state` at
    `chunked_gated_delta_rule_impl.py:425`. The index expression is a genuine
    load and is emitted as one.
    """
    if isinstance(t, ast.Name):
        out.append(("store", t.id))
    elif isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            _target_events(e, out)
    elif isinstance(t, (ast.Subscript, ast.Attribute)):
        if isinstance(t, ast.Subscript):
            _events(t.slice, out)
        base = t
        while isinstance(base, (ast.Subscript, ast.Attribute)):
            base = base.value
        if isinstance(base, ast.Name):
            out.append(("store", base.id))
        else:
            _events(base, out)
    else:
        _events(t, out)


def _events_assign(node, out):
    """`a = expr`: the right-hand side is evaluated before the target is bound."""
    _events(node.value, out)
    for t in node.targets:
        _target_events(t, out)


def _events_augassign(node, out):
    """`x += y` READS x before it writes it.

    The `Name` in an AugAssign target carries `ctx=Store`, so nothing else here
    would see the read.
    """
    if isinstance(node.target, ast.Name):
        out.append(("load", node.target.id))
    _events(node.value, out)
    _target_events(node.target, out)


def _events_annassign(node, out):
    """`a: T = expr`, whose value is optional."""
    if node.value is not None:
        _events(node.value, out)
    _target_events(node.target, out)


def _events_for(node, out):
    """`for t in it:` -- the iterable is evaluated before the target is bound."""
    _events(node.iter, out)
    _target_events(node.target, out)
    for s in list(node.body) + list(node.orelse):
        _events(s, out)


def _events_test_first(node, out):
    """`while`/`if`: the test is evaluated before either branch."""
    _events(node.test, out)
    for s in list(node.body) + list(node.orelse):
        _events(s, out)


def _events_comprehension(node, out):
    """The generators bind before the element is evaluated, but the AST holds
    `elt` first.

    Left to the fallback, that emits the comprehension variable's LOAD before its
    STORE and reports a recurrence in `ys = [g(t) for t in xs]` -- the same
    evaluation-order class of mistake the `AugAssign` and `Assign` cases exist to
    avoid, one level down.
    """
    for gen in node.generators:
        _events(gen.iter, out)
        _target_events(gen.target, out)
        for cond in gen.ifs:
            _events(cond, out)
    for part in ("key", "value") if isinstance(node, ast.DictComp) else ("elt",):
        _events(getattr(node, part), out)


def _events_walrus(node, out):
    """`(acc := acc + 1)` reads before it writes, and `NamedExpr` holds the target
    first. Left alone this MISSES a recurrence rather than inventing one, but it is
    the same bug facing the other way."""
    _events(node.value, out)
    _target_events(node.target, out)


def _events_name(node, out):
    """A bare name is one event, and its context says which."""
    out.append(("store" if isinstance(node.ctx, ast.Store) else "load", node.id))


# The forms where the AST's child order disagrees with evaluation order. Anything
# absent falls through to AST child order, which matches evaluation order for
# ordinary expressions.
_EVENT_ORDER = {
    ast.Assign: _events_assign,
    ast.AugAssign: _events_augassign,
    ast.AnnAssign: _events_annassign,
    ast.For: _events_for,
    ast.While: _events_test_first,
    ast.If: _events_test_first,
    ast.ListComp: _events_comprehension,
    ast.SetComp: _events_comprehension,
    ast.GeneratorExp: _events_comprehension,
    ast.DictComp: _events_comprehension,
    ast.NamedExpr: _events_walrus,
    ast.Name: _events_name,
}


def _events(node, out):
    """('load'|'store', name) in EVALUATION order, not in source order.

    The distinction is the whole of it. Python evaluates an assignment's
    right-hand side BEFORE binding its target, so in `acc = acc + x[i]` the load
    of `acc` happens first while the store is written further left on the line.
    Ordering by `(lineno, col_offset)` gets that backwards and reports the
    canonical accumulator as having no dependency at all.
    """
    handler = _EVENT_ORDER.get(type(node))
    if handler is not None:
        handler(node, out)
        return
    for ch in ast.iter_child_nodes(node):
        _events(ch, out)


def _loop_carried_dependency(tree):
    """Does some loop read a value it also writes -- a recurrence?

    This is what makes a scan a scan. A chunked MAP writes a disjoint slice of
    the output each trip and never reads back what it wrote; a chunked SCAN
    carries state from trip i into trip i+1, and that state is the thing whose
    serial depth the chunk size trades against.

    The rule: a name that is stored somewhere in the loop body and LOADED before
    any store of it in the same iteration. A value produced and consumed within
    one trip (`t1 = silu(x); out[i] = mul(t1, y)`) is stored first, so it does
    not count; an output buffer that is only written (`res[off:off+CHUNK] = ...`)
    is never loaded, so it does not count either.

    Checked against the real kernels rather than against synthetic ones. Combined
    with `_chunk_constant`, `has_chunked_scan` now holds for exactly the four
    gated-delta-rule kernels (`chunked_gdr/`, `qwen3_5_9b/`, `qwen3_6_27b/`,
    `qwen3_next/`) and stays quiet on the `chunk_size = 2` RoPE interleave in
    `deepseek_v4/compressor_impl.py` and on `QkvRmsNormRopeCache`, which chunk an
    axis and carry no state. Before this it held for none of the four and for
    both of the false positives.

    LIMIT, stated because it bounds every claim above: the analysis is lexical
    and intraprocedural. A recurrence written across a helper function -- an
    update chain factored into a function of its own, say -- is not visible to
    it. Missing one admits F-19 nowhere it
    should not be; it does not invent a scan.
    """
    for loop in ast.walk(tree):
        if not isinstance(loop, (ast.For, ast.While)):
            continue
        ev = []
        for stmt in loop.body:
            _events(stmt, ev)
        stored = {n for k, n in ev if k == "store"}
        seen = set()
        for kind, name in ev:
            if kind == "load" and name in stored and name not in seen:
                return True
            if kind == "store":
                seen.add(name)
    return False


def _module_count(op_dir):
    """From DESIGN.md if it says so, else from the modules directory.

    Returns None when neither is available, which admits every action that
    depends on it.
    """
    if not op_dir:
        return None
    design = os.path.join(op_dir, "DESIGN.md")
    if os.path.exists(design):
        txt = _read_text(design)
        m = re.search(r"module_count\s*[:=]\s*(\d+)", txt)
        if m:
            return int(m.group(1))
    mods = os.path.join(op_dir, "modules")
    if os.path.isdir(mods):
        n = len([f for f in os.listdir(mods) if f.endswith("_impl.py")])
        if n:
            return n
    return None


def call_name(call):
    fn = call.func
    return fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")


_call_name = call_name


def is_pypto_call(call, roots=frozenset({"pypto"})):
    """Is this a call on the pypto module?

    The distinction matters: the same file holds the tile-level kernel and the
    host-side torch wrapper. ``out = torch.empty_like(x)`` followed by a launch
    and a return reads ``out`` twice, which would otherwise look exactly like a
    reused on-chip intermediate. Only pypto values are tiles.

    `roots` comes from `pypto_roots`; the default keeps a caller that has no
    tree to hand working on the unaliased form.
    """
    node = call.func
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id in roots


_is_pypto_call = is_pypto_call


def _materializes_large_intermediate(tree, roots=None, writebacks=WRITEBACK_CALLS):
    """Does the kernel hold a value it could instead recompute or not store?

    Two independent pieces of evidence, either sufficient:

    1. A tile-valued name that is READ more than once. A straight-line
       chain (view -> op -> assemble) reads each name exactly once and
       materialises nothing worth trading; a name read twice is a value the
       kernel is keeping alive, which is exactly what store-versus-recompute
       is about.
    2. More than one write-back call. A kernel with a single output writes
       once; additional writes are intermediates staged through global memory.

    Calibrated against a pure elementwise kernel whose loop body is
    ``x_tile = pypto.view(...); out_tile = pypto.<op>(x_tile);
    pypto.assemble(out_tile, ...)``. Each name is read once and there is one
    assemble, so this returns False and F-17 is correctly held: there is no
    intermediate to trade. Such a kernel used to cost a real evaluation to
    discover the same thing at run time.
    """
    roots = pypto_roots(tree) if roots is None else roots
    produced = tile_producers(tree, roots, is_call=_is_pypto_call, writebacks=writebacks)
    loads = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in produced
        ):
            loads[node.id] = loads.get(node.id, 0) + 1
    if any(c >= 2 for c in loads.values()):
        return True

    def _is_writeback(node):
        return (
            isinstance(node, ast.Call)
            and _is_pypto_call(node, roots)
            and _call_name(node) in writebacks
        )

    return sum(1 for node in ast.walk(tree) if _is_writeback(node)) > 1


def _has_nested_pypto_loop(tree, loop_for):
    """Is there a pypto.loop with another pypto.loop inside it?

    Necessary for any transform that folds an outer axis into an inner loop: a
    kernel with one loop level has nothing to fold. Necessary, not sufficient --
    `requires` gates on what makes an action reachable at all, the way
    has_cube_op does.
    """
    for node in ast.walk(tree):
        if not loop_for(node):
            continue
        if any(
            loop_for(inner)
            for stmt in node.body
            for inner in ast.walk(stmt)
        ):
            return True
    return False


@dataclass(frozen=True)
class PredicateRules:
    roots: Callable
    cube_calls: set
    broadcasts: set
    writebacks: set
    is_loop: Callable
    declares_cube: Callable = None


def compute(impl_path, op_dir=None, *, rules=None, gate=None, hw=None):
    """Evaluate catalog preconditions; unreadable source leaves actions eligible."""
    if gate is None:
        gate = import_module(__package__ + ".feasibility") if __package__ else feasibility
    options = {"hw": hw} if hw is not None else {}
    try:
        src = _read_text(impl_path)
        tree = ast.parse(src)
    # RecursionError because `_events` walks expressions recursively with no
    # depth guard. No file in this repo comes close to Python's default limit,
    # but predicates are computed inside `init` and `select`, and the failure
    # mode of "cannot analyse this source" is to admit everything, not to end
    # the run.
    except (OSError, SyntaxError, RecursionError):
        # Cannot tell anything: admit everything.
        return {
            "has_cube_op": True,
            "has_broadcast": True,
            "has_chunked_scan": True,
            "has_inkernel_layout_op": True,
            "has_nested_pypto_loop": True,
            "materializes_large_intermediate": True,
            "module_count": None,
            "ub_occupancy": None,
            "l1_occupancy": None,
            "determined": False,
        }

    if rules is None:
        rules = PredicateRules(pypto_roots, CUBE_CALLS, BROADCAST_CALLS, WRITEBACK_CALLS,
                               lambda call, roots: _call_name(call) == "loop")
    roots = rules.roots(tree)
    calls = pypto_calls(tree, roots)

    def chunk_loop(call):
        return rules.is_loop(call, roots) and chunk_shape(call)

    def loop_for(node):
        return (isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
                and rules.is_loop(node.iter, roots))

    return {
        "has_cube_op": bool(calls & rules.cube_calls) or bool(rules.declares_cube and rules.declares_cube(src)),
        "has_broadcast": bool(calls & rules.broadcasts),
        "has_chunked_scan": _chunk_constant(tree, chunk_loop) and _loop_carried_dependency(tree),
        "has_inkernel_layout_op": bool(calls & LAYOUT_CALLS),
        "has_nested_pypto_loop": _has_nested_pypto_loop(tree, loop_for),
        "materializes_large_intermediate": _materializes_large_intermediate(tree, roots, rules.writebacks),
        "module_count": _module_count(op_dir),
        # How much of the unified buffer the vector tiles are estimated to hold.
        # Unlike the booleans above this is a ratio, and unlike everything the
        # profiler reports it is available before the kernel is ever run. A tile
        # four times smaller than the buffer affords costs latency on every
        # iteration and says nothing about itself in a trace: the kernel is
        # simply, quietly, slow. None when it cannot be estimated.
        "ub_occupancy": gate.ub_occupancy(impl_path, **options),
        # The cube counterpart, over L1 rather than the unified buffer. None for
        # a kernel with no matmul, which is the honest answer: a vector operator
        # has no L1 symptom rather than a bad one.
        "l1_occupancy": gate.l1_occupancy(impl_path, **options),
        "determined": True,
    }


def satisfies(requires, preds):
    """Does the kernel satisfy an action's requires list?

    Grammar is deliberately tiny: a bare predicate name, or
    'module_count >= N'. An unrecognised or undeterminable requirement admits
    the action.
    """
    for req in requires or []:
        req = req.strip()
        m = re.fullmatch(r"module_count\s*>=\s*(\d+)", req)
        if m:
            mc = preds.get("module_count")
            if mc is not None and mc < int(m.group(1)):
                return False
            continue
        if req in preds:
            if preds[req] is False:
                return False
            continue
        # unknown requirement: admit rather than silently hide the action
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("impl")
    ap.add_argument("--op-dir", default=None)
    a = ap.parse_args()
    jsonio.emit(compute(a.impl, a.op_dir or os.path.dirname(a.impl)), indent=2)


def chunk_shape(node):
    """Detect chunk naming or a symbolic/non-unit forward step."""
    for kw in node.keywords or []:
        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
            if re.search(r"chunk|_bt\b|\bbt\b", str(kw.value.value), re.I):
                return True
    if len(node.args) < 3:
        return False
    # A SYMBOL, or an integer other than 1. Every real chunk step in this repo is
    # a name (`l`, `L`, `params.token_tile`); requiring one rather than "anything
    # but 1" keeps out `pypto.loop(n, 0, -1)`, which is reverse iteration and
    # parses as a UnaryOp rather than a Constant.
    step = node.args[2]
    named_step = isinstance(step, (ast.Name, ast.Attribute))
    literal_step = isinstance(step, ast.Constant) and isinstance(step.value, int)
    return bool(named_step or (literal_step and step.value != 1))


def tile_producers(tree, roots, *, is_call, writebacks):
    """Names assigned the result of a pypto call that is not a write-back.

    Host-side torch/numpy locals are not tiles, and a write-back produces no
    value to keep alive, so neither is counted.
    """
    produced = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not is_call(node.value, roots):
            continue
        if _call_name(node.value) in writebacks:
            continue
        for t in node.targets:
            if isinstance(t, ast.Name):
                produced.add(t.id)
    return produced


if __name__ == "__main__":
    main()
