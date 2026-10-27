# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""What the core can say about a tile, and what it must be told.

A tile is bounded by the hardware, not by the program's views: in PyPTO a tile
shape and a view shape are independent quantities. So there are two rules, and
only two.

    trailing dimension is 16-aligned        (in `space`; a compiler rule,
                                             recorded in ERR_CONFIG_ALIGNMENT)
    footprint x dtype x live tiles <= UB    (here)

Neither was enforced. `vec_feasible` checked the alignment and stopped; the
harness constructed `HW(calibrated=False)`, which switched off even the checks
that existed. A ceiling seed larger than the buffer can hold at the kernel's
live-tile count then reaches the device, returns s=0, and records a PENALTY at
the region of interest -- teaching the surrogate to avoid a neighbourhood that
was never actually bad.

The one fact the core cannot read
--------------------------------
Live tile count. How many tile-shaped tensors are simultaneously resident is a
dataflow property, and after the model has fused, flattened or reordered
anything it is a property of a program the extractor has never seen. On a
fused chain the optimizer may derive four live tiles where `ub_occupancy`
assumes two.

The model may report that number, with its reasoning, and may report nothing
else. It may not narrow a range, name a maximum, or otherwise touch the edges.
Without it the footprint bound is OFF -- stated in the notes rather than guessed
at, because a guess here is a guess in an unknown direction.

Which direction to fail
-----------------------
Wide. A bound that is too generous is caught per candidate: every configuration
is re-checked and an impossible one is rejected before it reaches the device. A
bound that is too tight cannot be detected at all -- BO behaves optimally inside
whatever box it is given, and an optimum outside the box leaves no trace anywhere
in the run.

This module previously ignored its own advice. It derived per-dimension domains
from the view's extents, on the assumption that a tile dimension must divide the
view it is cut from. That assumption is false, and the kernels say so: a tile's
leading dimension can be many times the view's. The rule collapsed such an axis
to a single element and capped another far below the extent the hand-optimised
version runs at. A search inside those boxes would have converged cleanly and
reported nothing wrong. It was removed rather than narrowed.
"""

import ast
import os
import sys

# `pypto_ast` and `feasibility` are siblings of this package, one directory up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# One home, shared with `predicates`: see pypto_ast. Not optional -- every tile
# this module counts depends on knowing which calls are pypto's.
from pypto_ast import pypto_roots  # noqa: E402

try:
    import feasibility  # noqa: E402
except Exception:  # pragma: no cover
    feasibility = None

from . import apply  # noqa: E402


# One table, shared with `feasibility`, which gets it from `pypto.bytes_of` when
# pypto can be imported and from its own literal fallback otherwise. Resolved
# LAZILY: `feasibility.dtype_bytes_table()` loads the runtime's shared objects on
# its first call, and this module is imported by an offline test suite that must
# not pay for that. The fallback here exists only so the module keeps parsing
# when the extractor is unavailable.
_DTYPE_FALLBACK = {
    "DT_FP32": 4,
    "DT_FLOAT": 4,
    "DT_FP16": 2,
    "DT_BF16": 2,
    "DT_INT8": 1,
    "DT_UINT8": 1,
    "DT_INT32": 4,
}


def dtype_bytes_table():
    """`{dtype name: width}`, from `feasibility` when it can be reached."""
    table = getattr(feasibility, "dtype_bytes_table", None)
    if callable(table):
        got = table()
        if got:
            return got
    return _DTYPE_FALLBACK


# Unsupplied means the capacity rule is OFF, not that residency is two. A guess
# here is a guess in an unknown direction: too high and the bound is too tight,
# which narrows the domain -- the one error that leaves no trace.
#
# Since the divisor rules turn out to constrain only the axes where the view
# genuinely exceeds the tile, capacity is the bound that does most of the work,
# and it is off until the optimizer supplies `--live-tiles`. A run whose blocks
# never carry that number is a run where the core is checking alignment and
# little else; `block` reports `capacity_elems: null` so it is visible rather
# than inferred from a disappointing latency.
DEFAULT_LIVE_TILES = 0


# Calls that are NOT vector ops governed by a `set_vec_tile_shapes` directive.
# A matmul is a cube op (its footprint lives in L0, checked separately); a view
# is an addressing construct, not a tile allocation; a loop / range is control
# flow. An assignment whose right-hand side calls one of these does not add a
# vec tile to the current site's residency.
_NON_VEC_CALLS = frozenset(
    (
        "matmul",
        "view",
        "loop",
        "range",
        "assemble",
        "load",
        "store",
        "set_cube_tile_shapes",
        "set_vec_tile_shapes",
    )
)

# Methods that produce a SCALAR, not a tile: `(M - off).min(TILE_M)`, `.max()`,
# `x.shape[0]`. An assignment calling one of these is index / trip-count
# arithmetic on SymbolicScalars, not a vector op, so it adds no UB tile.
_SCALAR_METHODS = frozenset(("min", "max", "ceil", "floor", "shape", "item"))


def _read_text(path, errors="replace"):
    """The file's text, with the handle closed before the caller sees it."""
    with open(path, encoding="utf-8", errors=errors) as f:
        return f.read()


def _rhs_calls(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            yield f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


def _produces_a_tile(rhs):
    """Does this right-hand side create a tile-shaped tensor (as opposed to a
    SymbolicScalar)? A `pypto.<op>(...)` call that is not an index/scalar helper
    does: view, matmul, and every vector op return a tile. `(M-off).min(...)`,
    `x.shape[0]` and bare arithmetic do not.
    """
    if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Attribute):
        return rhs.func.attr not in _SCALAR_METHODS
    return False


def _is_pypto_call(call, roots):
    node = call.func
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id in roots


def _is_jit(fn):
    """A `@...jit(...)`-decorated def -- the kernel, as opposed to the wrapper."""
    for d in getattr(fn, "decorator_list", []):
        node = d.func if isinstance(d, ast.Call) else d
        while isinstance(node, ast.Attribute):
            if node.attr == "jit":
                return True
            node = node.value
        if isinstance(node, ast.Name) and node.id == "jit":
            return True
    return False


def _is_vector_op(rhs, tiles, roots=frozenset({"pypto"})):
    """Does this right-hand side allocate a VECTOR tile in UB?

    Two forms, and the discriminator against scalar arithmetic is `tiles` -- the
    set of names already known to hold a tile (results of view / matmul / vector
    ops). A blind operand count cannot tell `gate_tile * sigmoid_g` from
    `idx * TILE_M`: both are BinOps. The tile set can -- the first references a
    tile, the second references only loop scalars.

      * a `pypto.<vecop>(...)` call -- an Attribute call ON PYPTO that is neither a
        cube / addressing op (`_NON_VEC_CALLS`) nor a scalar helper
        (`_SCALAR_METHODS`);
      * a BinOp / UnaryOp that references at least one name in `tiles`.

    Over-counting residency is the dangerous direction -- it shrinks the per-site
    cap and rejects legal tiles -- so anything not positively a tile op is out.
    The call test used to check the attribute name alone, so a host-side
    `torch.cat` or `torch.add` was a vector op holding UB it never touches.
    """
    if isinstance(rhs, ast.Call):
        f = rhs.func
        if isinstance(f, ast.Attribute) and _is_pypto_call(rhs, roots):
            return f.attr not in _SCALAR_METHODS and f.attr not in _NON_VEC_CALLS
        return False
    if isinstance(rhs, (ast.BinOp, ast.UnaryOp)):
        return any(isinstance(n, ast.Name) and n.id in tiles for n in ast.walk(rhs))
    return False


def _rhs_tensor_operands(node, tiles=None):
    """Distinct TILE names read on the right-hand side of a vector op, + result.

    `(operands + 1)` is the `tensor_count` the design docs give: 2 for a unary op,
    3 for a binary one. Module refs (`pypto` in `pypto.cast`) and `pypto.DT_*`
    attributes are not operands, so `sigmoid(gate)` is one operand and
    `gate * sigmoid` is two. Called only for statements `_is_vector_op` has
    already accepted.

    `tiles` is the set of names known to hold a tile-shaped tensor -- the same
    set `_is_vector_op` uses, and for the same reason. Without it this counted
    every `Name` on the right-hand side, so a scalar passed BY NAME was charged a
    tile: `pypto.muls(a, SCALE)` came out at 3 live and `pypto.muls(a, 3)` at 2,
    for identical programs. Loop indices, extents and tile constants are
    routinely named, so the inflated count was the common case -- and it halves
    the site's UB budget in `verify` and halves its ladder in `vec_ladder_cap`,
    which is the narrowing this module's header says leaves no trace anywhere in
    the run. `_is_vector_op`'s own docstring already states the rule: over-counting
    residency is the dangerous direction.

    `tiles=None` keeps the old behaviour for a caller that has no tile set.
    """
    if tiles is None:
        bases = {
            a.value.id
            for a in ast.walk(node)
            if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name)
        }
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        return len(names - bases)
    return len(
        {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in tiles}
    )


# ast.walk does not preserve statement order across nesting, so collect the
# setters and assignments with their positions and sort once.
def _ordered_statements(scopes):
    """The tile setters and the assignments, in source order.

    `ast.walk` does not preserve statement order across nesting, so they are
    collected with their positions and sorted once.
    """
    stmts = []
    for scope in scopes:
        for node in ast.walk(scope):
            if isinstance(node, ast.Call) and _func_attr(node) == "set_vec_tile_shapes":
                stmts.append((node.lineno, node.col_offset, "setter", node))
            elif isinstance(node, ast.Assign):
                stmts.append((node.lineno, node.col_offset, "assign", node))
    stmts.sort(key=lambda t: (t[0], t[1]))
    return stmts


def _assigned_names(node):
    """The plain names this assignment binds."""
    out = set()
    for tgt in node.targets:
        if isinstance(tgt, ast.Name):
            out.add(tgt.id)
    return out


def _called_names(fn):
    """Plain function names this function calls: `helper(...)`, not `pypto.op(...)`."""
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            out.add(n.func.id)
    return out


def _is_nested_def(fn, i, spans):
    """Whether this def's line span lies inside another def's span."""
    for j, (a, b) in enumerate(spans):
        if j != i and a <= fn.lineno and fn.end_lineno <= b:
            return True
    return False


def _kernel_scopes(tree):
    """The functions whose body is kernel code, in source order.

    The `@jit` def is the entry, but it is routinely NOT where the tiles are. The
    generated node files put the work in plain helpers and call them from inside
    the single jit body -- most of a node's `set_vec_tile_shapes` calls can sit
    in a helper and only a few in the jit function -- and the library kernels do
    the same: a jit wrapper around a helper that holds every site.

    Restricting the walk to jit defs alone therefore did not exclude the torch
    wrapper; it excluded the kernel. On that node it saw four sites of eleven
    and numbered them vec#0..3, which are apply's vec#7..10 -- so `verify` read
    one site's residency under another site's name, and the seven it never saw
    fell back to a residency of one.

    So the scope is the jit defs plus every module-level function reachable from
    them by call name. The torch wrapper stays out because nothing in a kernel
    calls it -- the edge runs the other way.
    """
    funcs = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.setdefault(fn.name, fn)
    roots = [fn for fn in funcs.values() if _is_jit(fn)]
    if not roots:
        return [tree]
    seen, stack, out = set(), list(roots), []
    while stack:
        fn = stack.pop()
        if fn.name in seen:
            continue
        seen.add(fn.name)
        out.append(fn)
        for name in _called_names(fn):
            callee = funcs.get(name)
            if callee is not None and callee.name not in seen:
                stack.append(callee)
    # A def nested inside another selected def is already walked with its parent;
    # keeping it as its own scope would count its statements twice.
    spans = [(fn.lineno, fn.end_lineno) for fn in out]
    out = [fn for i, fn in enumerate(out) if not _is_nested_def(fn, i, spans)]
    out.sort(key=lambda fn: fn.lineno)
    return out


def _vec_site_ids(src):
    """{byte offset of a `set_vec_tile_shapes` call: the id `apply` gives it}.

    The ids have to come from `apply`, not from a second count here. `apply`
    numbers `vec` and `vec_dyn` on SEPARATE counters -- a dynamic site
    (`set_vec_tile_shapes(v1_tile[0], v1_tile[1])`) is discovered but never
    rewritten, so it does not consume a `vec#` ordinal -- while a local count in
    source order does not skip it. On any kernel carrying a `vec_dyn` site the
    two numberings disagree from that point on.
    """
    try:
        sites = apply.discover_sites(src)
    except SyntaxError:
        return {}
    return {s.start: s.id for s in sites if s.kind.startswith("vec")}


def _dt_width(node, env):
    """The width a `DT_*` expression names, or None. `env` resolves `dtype = DT_X`."""
    table = dtype_bytes_table()
    if isinstance(node, ast.Attribute) and node.attr in table:
        return table[node.attr]
    if isinstance(node, ast.Name):
        if node.id in table:
            return table[node.id]
        return env.get(node.id)
    return None


def _declared_width(rhs, env, widths):
    """The element width this right-hand side produces, or None when unreadable.

    Two readings, and no third:
      * an explicit dtype argument -- `matmul(a, b, pypto.DT_FP32)`,
        `cast(x, pypto.DT_BF16)`, `view(y, dtype=pypto.DT_FP32)`. This is what
        carries the answer through an antiquant kernel: the INT8 cache is read
        once and everything downstream of the matmul is the fp32 the matmul was
        asked for.
      * failing that, the widest operand already known -- an elementwise op
        returns what it was given.
    A name whose width no statement declares stays absent rather than guessed.
    """
    if isinstance(rhs, ast.Call):
        for a in list(rhs.args) + [k.value for k in rhs.keywords]:
            w = _dt_width(a, env)
            if w:
                return w
    operands = [
        widths.get(n.id)
        for n in ast.walk(rhs)
        if isinstance(n, ast.Name) and n.id in widths
    ]
    return max(operands) if operands else None


def _vec_widths_seen(rhs, produced, widths):
    """The element widths one vector statement's result and operands declare."""
    seen = [produced] if produced else []
    for n in ast.walk(rhs):
        if isinstance(n, ast.Name) and n.id in widths:
            seen.append(widths.get(n.id))
    return seen


def _vec_scope_walk(scope, ids, off, roots, res):
    """Fold one kernel scope's statements into the per-site residency map."""
    active = None
    tiles = set()  # names known to hold a tile-shaped tensor
    widths = {}  # names whose element width the source declares
    env = {}  # names bound to a DT_* constant
    for _, _, kind, node in _ordered_statements([scope]):
        if kind == "setter":
            active = ids.get(off(node.lineno, node.col_offset))
            if active is not None:
                res.setdefault(active, {"live": 1, "dtype_bytes": None})
            continue
        rhs = node.value
        w = _dt_width(rhs, env)
        if w:
            for name in _assigned_names(node):
                env[name] = w
            continue
        # A statement contributes to residency only if it is a vector op. It
        # contributes a tile NAME regardless, so a later BinOp can tell a tile
        # operand from a loop scalar.
        is_vec = active is not None and _is_vector_op(rhs, tiles, roots)
        produced = _declared_width(rhs, env, widths) if _produces_a_tile(rhs) else None
        if is_vec:
            cell = res[active]
            cell["live"] = max(cell["live"], _rhs_tensor_operands(rhs, tiles) + 1)
            seen = _vec_widths_seen(rhs, produced, widths)
            if seen:
                cell["dtype_bytes"] = max(cell["dtype_bytes"] or 0, max(seen))
        if is_vec or _produces_a_tile(rhs):
            names = _assigned_names(node)
            tiles |= names
            if produced:
                widths.update(dict.fromkeys(names, produced))
    return res


def vec_sites(src):
    """Per-`set_vec_tile_shapes` site: how many tiles it holds and how wide they are.

    Returns `{site_id: {"live": int, "dtype_bytes": int or None}}`, keyed by the
    ids `apply` assigns.

    `live` is the largest `operands + 1` over the elementwise ops the site
    governs -- the number of tile-shaped tensors the UB must hold at once for
    that directive. This replaces the single `--live-tiles` number the optimizer
    used to supply by judgement, which drifted between 4, 6 and 8 across runs of
    the same kernel and set the search space with it.

    `dtype_bytes` is the widest element the site's own statements are declared to
    work on, and it is None when the source does not say. It exists because the
    single kernel-wide number it replaces was the MINIMUM width appearing
    anywhere in the file: an INT8 KV cache sets the currency to 1 byte for a
    softmax site whose every operand is the fp32 the matmul above it produced,
    and the 192 KB UB gate then passes tiles several times over the buffer.

    A directive governs the elementwise statements that follow it until the next
    `set_vec_tile_shapes`, and the walk restarts at every function boundary: a
    directive does not reach across a `def` in any form this module can read, and
    under-counting residency is the safe direction (see `_is_vector_op`).
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    roots = pypto_roots(tree)
    ids = _vec_site_ids(src)
    off = apply.line_starts(src)
    res = {}
    for scope in _kernel_scopes(tree):
        _vec_scope_walk(scope, ids, off, roots, res)
    return res


def vec_residency(src):
    """Peak vec-tile residency per site, `{site_id: n}`. See `vec_sites`."""
    return {sid: cell["live"] for sid, cell in vec_sites(src).items()}


def vec_dtype_bytes(src):
    """
    Element width per vec site, `{site_id: n}`, omitting the sites the source
    does not declare. See `vec_sites`.
    """
    return {
        sid: cell["dtype_bytes"]
        for sid, cell in vec_sites(src).items()
        if cell["dtype_bytes"]
    }


def _func_attr(call):
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


def vec_ladder_cap(domain, site_id, field, fields):
    """
    The largest value `field` may take before it overflows the UB even with
    every other dimension of this vec site at its smallest legal rung.

    The UB rule is joint -- the tiles one operation holds have to fit TOGETHER
    (`vec_op_groups`) -- so no single dimension has a fixed bound. But a rung is
    safe to DROP if it overflows when everything else is minimal: no feasible
    tile can contain it. For each operation this site takes part in, the budget
    left for this site is the UB less what that operation's OTHER entries spend
    at their minima, and this site's own other axes are 1. The cap is the
    smallest such bound over the operations, so it agrees with `verify` by
    construction.

    That agreement is the point, not a nicety. The cap trims the ladder and
    `verify` admits the seed; when the cap was the tighter of the two, a seed
    `verify` had passed was enqueued into a study whose categorical domain no
    longer contained it, and optuna raised out of `study.optimize` before the
    block's first measurement. Every other field's minimum is 1, the trailing one
    included: the 16-alignment rule does not forbid 1, and `space.vec_feasible`
    admits a trailing extent of 1 explicitly, because kernels carry
    `set_vec_tile_shapes(<rows>, 1)` reduction legs.

    Returns None when the footprint bound is off at this site, so the caller
    keeps the full ladder.
    """
    domain = domain or {}
    ub = domain.get("ub_bytes")
    groups = domain.get("vec_op_groups")
    if site_id in (domain.get("vec_unmodelled") or ()):
        return None
    if not (ub and groups):
        return None
    minimal = {field: 1 for field in fields}
    cap = None
    for group in groups:
        mine = [sh for sid, sh in group if sid == site_id]
        if not mine:
            continue
        # What the operation's other entries spend with this site's tile at its
        # floor -- they are tiles of OTHER sites, whose configs this call does
        # not carry, so they are charged at the one-element floor the same way.
        others = sum(
            group_bytes(domain, {sid: dict(minimal)}, sid, sh)
            for sid, sh in group
            if sid != site_id
        )
        room = ub - others
        if room <= 0:
            return 1
        # Every entry of THIS site in this operation scales with `field`, so they
        # are one coefficient, not a fixed cost. An operation holding four tiles
        # of this site charges four times what one unit of `field` costs. Pricing
        # the other three at their floor instead is what made the cap 4x too
        # loose, and a cap looser than `verify` is as broken as one tighter: the
        # ladder then offers rungs the gate refuses and the block spends its
        # startup draws on them.
        coef = sum(
            group_bytes(domain, {site_id: dict(minimal)}, site_id, sh) for sh in mine
        )
        if coef <= 0:
            continue
        bound = room // coef
        cap = bound if cap is None else min(cap, bound)
    return cap


def read_views(src):
    """View shapes as integers, at ANY rank, resolving named bindings.

    Two departures from `feasibility.extract`, and each has a reason.

    **Rank.** `feasibility` keeps `len(shape) >= 2`, which is right for the rule
    it feeds and wrong here: it makes a flattened kernel's view invisible, and a
    flatten is exactly the case this domain exists to constrain. Once a kernel
    has been flattened the only remaining knob IS the chunk size, cut from a 1-D
    view.

    **Symbols.** `feasibility` reads views literal-only, deliberately:

        resolving its symbols would feed the rejection rule a footprint the
        kernel never pays

    That reasoning is about FOOTPRINT, and this is not a footprint. Whether a
    tile dimension divides a view's extent does not depend on whether the view is
    resident. And literal-only is not a conservative choice here, it is an empty
    one: every level1 kernel writes `pypto.view(x, [TILE_B, D], ...)`, so a
    literal-only reader returns nothing on all six and the divisor rules never
    fire once.

    A name bound to two different values is dropped by `_int_env`, so the shape
    never depends on which binding happened to be walked last.

    Reading this here rather than widening `feasibility` keeps the baseline arms,
    which share that module, byte-identical.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    env = _int_env(tree)
    views = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name != "view":
            continue
        # The SHAPE is argument 1, positionally. Scanning args[1:] for "the first
        # list that resolves" reads the OFFSET as a shape the moment the real
        # shape is symbolic -- `pypto.view(x, [TILE_B, D], [0, 0])` yields a view
        # of [0, 0], which is worse than reading nothing: it is a fabricated
        # extent that suppresses the "no literal view" note and then constrains
        # the domain to divisors of zero.
        if len(node.args) < 2:
            continue
        shape = _int_list(node.args[1], env)
        if shape:
            views.append(shape)
    return views


def _int_bindings(target, value):
    """(name, int) pairs bound by one target, tuple unpacking included."""
    if isinstance(target, ast.Name) and _is_int_const(value):
        return [(target.id, value.value)]
    if not (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
    ):
        return []
    return [
        (t.id, v.value)
        for t, v in zip(target.elts, value.elts)
        if isinstance(t, ast.Name) and _is_int_const(v)
    ]


def _bind_ints(node, env, bad):
    """Fold one assignment's integer bindings into `env`, naming clashes in `bad`."""
    for t in node.targets:
        for name, value in _int_bindings(t, node.value):
            if name in env and env[name] != value:
                bad.add(name)
            env[name] = value


def _int_env(tree):
    """NAME -> int for unambiguous integer bindings anywhere in the module.

    Tuple unpacking included: `TILE_B, TILE_D = 16, 256` is how these kernels
    bind their tile constants, and `feasibility._int_env` -- which this used to
    delegate to -- reads simple assignments only, so both names came back
    unresolved and the view with them.

    A name bound to two different values is dropped, so a shape never depends on
    which binding happened to be walked last.
    """
    env, bad = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            _bind_ints(node, env, bad)
    return {k: v for k, v in env.items() if k not in bad}


def _is_int_const(n):
    return (
        isinstance(n, ast.Constant)
        and isinstance(n.value, int)
        and not isinstance(n.value, bool)
    )


def _int_list(node, env=None):
    if not isinstance(node, ast.List) or not node.elts:
        return None
    out = []
    for e in node.elts:
        if _is_int_const(e):
            out.append(e.value)
        elif isinstance(e, ast.Name) and (env or {}).get(e.id) is not None:
            out.append(env[e.id])
        else:
            return None
    return out


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def _running_trailing(src):
    """{site_id: the trailing tile extent the kernel is ALREADY running}.

    Read by `space.vec_feasible` and used only to admit. The alignment rule is
    written in elements against a constant that assumes a 2-byte dtype, so it
    can throw away the warm start on a kernel that is already running and
    golden-passing at that extent. Losing a bad draw costs one trial; losing
    the incumbent costs the block, because TPE then has no feasible observation
    to model and returns best_config: null.
    """
    try:
        running = apply.current_config(src)
    except SyntaxError:  # pragma: no cover
        return {}
    out = {}
    for sid, c in running.items():
        # The highest-numbered `d*`, the same reading `_effective` clamps
        # against. Positionally last and numerically last agree on a config
        # `apply` has just parsed out of the source, and disagree on one a block
        # winner wrote back, so naming the axis keeps the two sides of the clamp
        # talking about the same axis.
        dims = {
            int(k[1:]): v for k, v in c.items() if k.startswith("d") and k[1:].isdigit()
        }
        if dims:
            out[sid] = dims[max(dims)]
    return out


def _running_live_caps(src, report):
    """Per-site residency capped so the tile the kernel IS RUNNING fits the UB.

    `prod(dims) x dtype x residency <= 192 KB` rejects kernels' own delivered
    tile shapes. Two things could explain that: the buffer is larger than
    192 KB, or the residency walk over-counts. The source cannot say which, and
    the first version of this function bet on the buffer and raised the site's
    byte budget to the running footprint.

    The device settles it: a tile drawn inside a budget raised that way comes
    back `F40005 TENSOR_MEMORY_ALLOCATION ... exceeds MEM_UB 196608`. The
    buffer is exactly the 196608 the model says, so the residency is the wrong
    number.

    So the capacity stays at the model's and the RESIDENCY is capped instead, to
    the largest value consistent with both facts we have: the program runs, and
    the buffer is 192 KB. That is `ub // (prod(dims) x dtype)` for the running
    tile, and a draw the device would refuse for capacity is then refused for
    free.

    Only ever LOWERS a residency, and only at a site whose own running tile
    disproves it. A site the model already fits is untouched.

    Returns (caps, unmodelled). A site whose running tile exceeds the buffer on
    its OWN, at one live copy, is in `unmodelled`: no residency explains it, so
    something else about that site is wrong -- its width, or the assumption that
    `prod(dims)` is resident at all. `verify` then does not judge that site.
    That is not the raised budget again: a raised budget invents a number and
    licenses draws against it, while this records that the site is unreadable
    and leaves it to the device, where `_lower_ceiling` learns a real bound from
    real refusals. Every site we CAN model is still gated at 192 KB.

    Derived from the running tile, so it moves when the tile moves -- which is
    why both are kept out of `fingerprint`, for the same reason `per_site` is.
    """
    ub = report.get("ub_bytes")
    if not ub:
        return {}, []
    try:
        running = apply.current_config(src)
    except SyntaxError:  # pragma: no cover
        return {}, []
    caps, unmodelled = {}, []
    for sid, c in running.items():
        dims = [v for k, v in c.items() if k.startswith("d")]
        # Default 1, the same default `verify` charges a site the walk has no
        # entry for. Skipping those here leaves a site carrying a multi-million
        # element `prod(dims)` -- plainly not a UB-resident footprint --
        # unflagged and then rejected at live=1, which is the running program
        # excluded by a rule that never examined it.
        live = report["vec_residency"].get(sid, 1)
        if not dims:
            continue
        db = report["vec_dtype_bytes"].get(sid) or report["dtype_bytes"]
        per_tile = _prod(dims) * db
        if per_tile <= 0 or per_tile * live <= ub:
            continue
        if per_tile > ub:
            unmodelled.append(sid)
            continue
        caps[sid] = ub // per_tile
    return caps, sorted(unmodelled)


def live_for(domain, site_id):
    """
    The residency `verify` charges this site: the walk's count, capped by what
    the running tile proves (`_running_live_caps`).
    """
    res = (domain or {}).get("vec_residency") or {}
    live = res.get(site_id, 1)
    cap = ((domain or {}).get("vec_live_cap") or {}).get(site_id)
    return min(live, cap) if cap else live


def _note_residency_caps(report):
    """Record in the report where the residency walk capped its count."""
    residency = report.get("vec_residency")
    capped = report.get("vec_live_cap")
    if capped:
        report["notes"].append(
            "the residency walk over-counts at "
            + ", ".join(
                f"{sid} ({residency[sid]} -> {cap})"
                for sid, cap in sorted(capped.items())
            )
            + "; the running tile does not fit the UB at the walk's count, so "
            "the count is capped and the 192 KB budget stands"
        )
    if report.get("vec_unmodelled"):
        report["notes"].append(
            "the footprint bound is OFF at "
            + ", ".join(report.get("vec_unmodelled"))
            + ": one running tile there already exceeds the UB, so no residency "
            "explains it and this module cannot judge that site. The device "
            "does, through the learned ceiling."
        )


def _summarise_footprint(report):
    """Fold the per-site widths and residencies into the scalar summary."""
    kdb = report.get("dtype_bytes")
    per_site = report.get("vec_dtype_bytes")
    residency = report.get("vec_residency")
    # The worst site is the one that spends the most buffer per tile ELEMENT, which
    # is width x residency -- not residency alone. With one width for the whole
    # kernel those two orderings coincided; with a per-site width they do not.
    worst = (
        max(
            ((per_site.get(sid) or kdb) * live, live) for sid, live in residency.items()
        )
        if residency
        else None
    )
    report["live_tiles"] = worst[1] if worst else 0
    if worst and worst[0] > 0 and report.get("ub_bytes"):
        report["capacity_elems"] = report.get("ub_bytes") // worst[0]
    else:
        report["notes"].append(
            "no vec site found, so the footprint bound is off (alignment only)"
        )


def derive(op_file, live_tiles=None, ub_budget_kb=192):
    """What the program says about the tiles it can hold.

    Returns:
        {"capacity_elems": int or None,     the footprint bound, in elements
         "live_tiles": int, "dtype_bytes": int,
         "vec_residency": {site_id: int},   tiles the site holds at once
         "vec_dtype_bytes": {site_id: int}, element width, where the source says
         "ub_bytes": int,                   the per-site budget `verify` applies
         "vec_live_cap": {site_id: int},    residency capped by the running tile
         "vec_trailing": {site_id: int},    the trailing extent it is running
         "vec_unmodelled": [site_id],       sites the footprint bound is off at
         "views": [...],                    for change detection ONLY
         "per_site": {},                    always empty; see below
         "notes": [str]}

    `per_site` exists so callers stay stable, and it is always empty. An earlier
    version derived per-dimension domains from the view's extents -- the tile
    dimension had to divide it -- and that was simply wrong about PyPTO: a tile
    shape and a view shape are independent. The kernels say so plainly once you
    stop assuming otherwise: a generated kernel can set a leading tile dimension
    many times larger than the view it is cut from.

    The rule survived one review because it looked conservative and its failures
    looked like caution. It collapsed a leading axis to a single element and
    capped a trailing axis far below the extent the hand-optimised version runs
    at -- and a search inside those boxes would have converged
    cleanly, reported nothing unusual, and never once indicated that the optimum
    was outside. That is the failure mode this module's header warns about, built
    into the module itself.

    What is left is what was true all along: the tile is bounded by the hardware,
    not by the view. Alignment (in `space`) and footprint (here).
    """
    report = {
        "per_site": {},
        "capacity_elems": None,
        "views": [],
        "live_tiles": int(live_tiles or DEFAULT_LIVE_TILES),
        "dtype_bytes": 4,
        "notes": [],
    }
    if feasibility is None:
        report["unreadable"] = True
        report["notes"].append("feasibility unavailable; no footprint bound")
        return report
    try:
        info = feasibility.extract(op_file)
    except Exception as e:  # pragma: no cover
        report["unreadable"] = True
        report["notes"].append(f"extract failed ({e}); no footprint bound")
        return report
    if "parse_error" in info:
        report["unreadable"] = True
        report["notes"].append("kernel does not parse; no footprint bound")
        return report

    src = _read_text(op_file)
    # Kept for `changed()` only. A view is not a constraint on the tile, but a
    # view that moves is a program whose access pattern moved, which is a reason
    # to re-tune even though it is not a reason to reject anything.
    report["views"] = read_views(src)
    report["dtype_bytes"] = int(info.get("dtype_bytes") or 4)

    # Per-vec-site UB budget, in BYTES, derived from the source rather than
    # supplied by hand. Each vec directive may hold a different tile size and
    # governs a different number of simultaneously-resident tiles, so the bound
    # is per site: `prod(dims) * dtype * residency <= UB`. `verify` applies it.
    report["ub_bytes"] = int(ub_budget_kb) * 1024 if ub_budget_kb else 0
    sites = vec_sites(src)
    report["vec_residency"] = {sid: c["live"] for sid, c in sites.items()}
    # Per-site element width, for the sites whose statements declare one. Absent
    # sites fall back to the kernel-wide `dtype_bytes` in `verify`, so this map
    # only ever makes the gate more accurate where the source is explicit; it
    # never narrows on a guess.
    report["vec_dtype_bytes"] = {
        sid: c["dtype_bytes"] for sid, c in sites.items() if c["dtype_bytes"]
    }
    report["vec_trailing"] = _running_trailing(src)
    # What one operation holds in UB at once. This is what `verify` gates on;
    # `vec_residency` / `vec_live_cap` below are retained for the SEARCH SPACE
    # bounds in `space`, which still reason per site.
    report["vec_op_groups"] = vec_op_groups(src)
    report["vec_live_cap"], report["vec_unmodelled"] = _running_live_caps(src, report)
    _note_residency_caps(report)
    # `capacity_elems` and `live_tiles` are retained as a single-number summary
    # for `changed()` / logging only -- computed from the WORST site so the
    # fingerprint still moves when the footprint envelope moves. The real gate is
    # per-site in `verify`; nothing reads this scalar as a bound any more.
    _summarise_footprint(report)
    return report


def fingerprint(domain):
    """A comparable summary of the space, ignoring everything that is not it.

    Change 4 rests on this: when a structural action lands, the previously
    located tile optimum describes a program that no longer exists -- but only
    when the SPACE moved. `lever.structural_signature` is the obvious test and
    the wrong one, because it hashes the program with the tile literals erased,
    so a renamed variable or an added comment changes it and a block would fire
    after every non-tile action.

    What actually invalidates a tile optimum is a change to the inputs of the
    derivation: the view extents, the dtype, the residency. Those and nothing
    else are what this fingerprint carries -- the dtype both as the kernel-wide
    number and as the per-site map, since an edit that changes one site's
    precision moves that site's budget and nothing else would show it.

    Deliberately NOT including `per_site`. Since an axis is only constrained when
    the view genuinely exceeds the running tile, the per-dimension domains depend
    on the tile's current value -- so a block that improved the tile would change
    them, fire a retune, run another block, and loop.
    """
    return repr(
        (
            sorted(tuple(v) for v in (domain.get("views") or [])),
            domain.get("capacity_elems"),
            domain.get("dtype_bytes"),
            tuple(sorted((domain.get("vec_dtype_bytes") or {}).items())),
            # Residency per site, for the same reason as the widths, and it
            # was the one input to `verify` the fingerprint did not carry.
            # `capacity_elems` summarises the WORST site only, so a fusion
            # that takes a non-worst site from 2 live tiles to 4 moved that
            # site's budget while every field here stayed put: `changed()`
            # reported "unchanged", no retune fired, and `block.memory_key`
            # handed the next block a study whose observations were taken
            # when that site had twice the room. Unlike `vec_live_cap` this
            # does not depend on any tile VALUE, so including it cannot
            # loop.
            tuple(sorted((domain.get("vec_residency") or {}).items())),
        )
    )


def changed(before, after):
    """Did the tile space move? (bool, one-line reason).

    A program that could not be read is UNKNOWN, not changed. Reporting a move
    there would fire a block on every syntactically broken candidate -- and a
    broken candidate is about to fail its golden test anyway, so the block would
    be spent tuning a kernel that is on its way to being reverted.
    """
    if before.get("unreadable") or after.get("unreadable"):
        return False, "the program could not be read; no verdict"
    if fingerprint(before) == fingerprint(after):
        return False, "the derived tile domain is unchanged"
    bits = []
    if (before.get("views") or []) != (after.get("views") or []):
        bits.append(f"views {before.get('views')} -> {after.get('views')}")
    if before.get("capacity_elems") != after.get("capacity_elems"):
        bits.append(
            f"capacity {before.get('capacity_elems')} -> "
            f"{after.get('capacity_elems')} elements"
        )
    for sid in sorted(
        set(before.get("vec_dtype_bytes") or {})
        | set(after.get("vec_dtype_bytes") or {})
    ):
        b = (before.get("vec_dtype_bytes") or {}).get(sid)
        a = (after.get("vec_dtype_bytes") or {}).get(sid)
        if b != a:
            bits.append(f"{sid} width {b}B -> {a}B")
    return True, "; ".join(bits) or "the derived tile domain changed"


def _effective(site_config, trailing):
    """(elements, clamped?) for one site's tile, against what it is running on.

    The trailing dim is the highest-numbered `d*` key, not the last one the dict
    happens to hold: a block winner's config is written back with `d1` before
    `d0` on some sites, and reading position rather than name would clamp the
    wrong axis.
    """
    dims = {}
    for k, v in site_config.items():
        if k.startswith("d") and k[1:].isdigit():
            dims[int(k[1:])] = v
    if not dims:
        return 0, False
    clamped = False
    if trailing:
        last = max(dims)
        if dims.get(last, 0) > trailing:
            dims[last], clamped = trailing, True
    n = 1
    for v in dims.values():
        n *= v
    return n, clamped


# Calls whose first or second argument DECLARES the result's extent. The tile
# directive sets a granularity; what is allocated is also bounded by the tensor's
# own shape, and these are the places the source states it.
_SHAPE_ALLOC = frozenset(("full", "tensor", "empty", "zeros", "ones"))
_SHAPE_RESHAPE = frozenset(("view", "reshape"))
# Reductions: the reduced axis becomes 1 (or disappears without `keepdim`). A
# softmax's row statistic is ONE COLUMN, and charging it a whole tile makes
# most of this model's refusals wrong.
_REDUCE_CALLS = frozenset(
    (
        "sum",
        "amax",
        "amin",
        "mean",
        "prod",
        "any",
        "all",
        "argmax",
        "argmin",
        "rowmax",
        "rowsum",
    )
)


def int_literal(node):
    """A literal int, including `-1`, which the parser gives as a UnaryOp."""
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = int_literal(node.operand)
        return -inner if inner is not None else None
    return None


def _int_consts(tree):
    """`{name: int}` for the module's and kernel's plain integer assignments.

    `Q_TILE = 128`, `QT = 320`. These are what the shape arguments are written
    in, so without them every declared shape reads as unknown and the bound this
    module exists to apply is never applied.
    """
    out = {}
    for n in ast.walk(tree):
        if not isinstance(n, ast.Assign):
            continue
        v = int_literal(n.value)
        if v is None:
            continue
        for t in n.targets:
            if isinstance(t, ast.Name):
                out[t.id] = v
    return out


def literal_shape(node, consts):
    """`[int or None]` for a literal shape argument, or None when it is not one.

    A dimension the source does not state as an integer -- a kernel parameter
    like `head_dim`, a `SymbolicScalar` -- is None, which means "do not bound
    this axis", never zero and never a guess.
    """
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    out = []
    for e in node.elts:
        v = int_literal(e)
        if v is not None:
            out.append(v)
        elif isinstance(e, ast.Name) and e.id in consts:
            out.append(consts[e.id])
        else:
            out.append(None)
    return out


def _broadcast_shape(a, b):
    """The extent an elementwise result spans, right-aligned. Unknown stays
    unknown: `[320, 1]` with `[320, None]` is `[320, None]`, not `[320, 1]`."""
    if a is None:
        return b
    if b is None:
        return a
    n = max(len(a), len(b))
    a = [None] * (n - len(a)) + list(a)
    b = [None] * (n - len(b)) + list(b)
    return [None if (x is None or y is None) else max(x, y) for x, y in zip(a, b)]


def _covers(have, need):
    """Can a tensor of extent `have` hold a result of extent `need`?

    This decides whether the compiler may write an operation's result into an
    operand whose life ends at that operation -- the reuse `CheckOpBufferSize`
    sees as one memory id rather than two. An unknown extent is not a promise on
    either side, with one exception: identical extents, unknowns included. In an
    elementwise chain the result's shape IS the broadcast of its operands, so an
    operand carrying exactly the result's shape is the one that set it and holds
    exactly as much. That exception is the `sub(scores_scaled, mij)` step of a
    flash-attention softmax, where the subtrahend is one column and the minuend
    is the whole tile; without it the kernel's own running baseline comes out
    over the 192 KB buffer.
    """
    if have is None or need is None:
        return False
    if list(have) == list(need):
        return True
    n = max(len(have), len(need))
    a = [None] * (n - len(have)) + list(have)
    b = [None] * (n - len(need)) + list(need)
    for x, y in zip(a, b):
        if x is None or y is None or x < y:
            return False
    return True


def _kwarg(call, name):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _rhs_attr(rhs):
    """The method name a statement's right-hand side calls, or None."""
    if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Attribute):
        return rhs.func.attr
    return None


def _tile_names(rhs, tiles):
    """The names a right-hand side reads that already hold a tile."""
    return {n.id for n in ast.walk(rhs) if isinstance(n, ast.Name) and n.id in tiles}


def _last_reads(stmts):
    """The index of the last statement that reads each name."""
    last_read = {}
    for i, (_, _, kind, node) in enumerate(stmts):
        if kind != "assign":
            continue
        written = _assigned_names(node)
        for n in ast.walk(node.value):
            if isinstance(n, ast.Name) and n.id not in written:
                last_read[n.id] = i
    return last_read


def _reduced_shape(rhs, base, consts):
    """The shape a reduce call leaves of `base`, or None."""
    dim = _kwarg(rhs, "dim")
    if dim is None and len(rhs.args) > 1:
        dim = rhs.args[1]
    axis = int_literal(dim) if dim is not None else None
    if not base or axis is None:
        return None
    shape = list(base)
    i = axis if axis >= 0 else len(shape) + axis
    if not 0 <= i < len(shape):
        return shape
    keep = _kwarg(rhs, "keepdim")
    if isinstance(keep, ast.Constant) and keep.value is False:
        shape.pop(i)
    else:
        shape[i] = 1
    return shape


def _stmt_shape(rhs, attr, read, shapes, consts):
    """The declared extent of what one statement produces, or None."""
    if attr in _SHAPE_ALLOC and rhs.args:
        return literal_shape(rhs.args[0], consts)
    if attr in _SHAPE_RESHAPE and len(rhs.args) > 1:
        return literal_shape(rhs.args[1], consts)
    if attr in _REDUCE_CALLS:
        base = None
        for name in sorted(read):
            base = _broadcast_shape(base, shapes.get(name))
        return _reduced_shape(rhs, base, consts)
    shape = None
    for name in sorted(read):
        shape = _broadcast_shape(shape, shapes.get(name))
    return shape


def _group_entries(active, read, shapes, tiles):
    """The distinct tensors one vector statement holds, as [site, shape] entries."""
    group, seen = [], set()
    for name in sorted(read):
        if name in seen or shapes.get(name) is None and name not in tiles:
            continue
        seen.add(name)
        group.append([active, shapes.get(name)])
    return group


def _scope_op_groups(scope, ids, off, roots, consts):
    """The per-operation tile groups one kernel scope charges to the buffer."""
    stmts = _ordered_statements([scope])
    # The last statement that READS each name, for the liveness reuse rule.
    last_read = _last_reads(stmts)
    groups = []
    active = None
    tiles, shapes, env = set(), {}, {}
    for idx, (_, _, kind, node) in enumerate(stmts):
        if kind == "setter":
            active = ids.get(off(node.lineno, node.col_offset))
            continue
        rhs = node.value
        w = _dt_width(rhs, env)
        if w:
            for name in _assigned_names(node):
                env[name] = w
            continue
        written = _assigned_names(node)
        is_vec = active is not None and _is_vector_op(rhs, tiles, roots)
        attr = _rhs_attr(rhs)
        read = _tile_names(rhs, tiles)
        shape = _stmt_shape(rhs, attr, read, shapes, consts)
        if is_vec:
            group = _group_entries(active, read, shapes, tiles)
            reused = any(
                last_read.get(name) == idx and _covers(shapes.get(name), shape)
                for name in read
            )
            if not (written & read) and not reused:
                group.append([active, shape])
            if group:
                groups.append(group)
        if is_vec or _produces_a_tile(rhs):
            tiles |= written
            shapes.update(dict.fromkeys(written, shape))
    return groups


def vec_op_groups(src):
    """What ONE operation holds in UB at once: `[[[site_id, shape], ...], ...]`.

    This is the shape of the compiler's own check. `CheckOpBufferSize`
    (framework/src/passes/block_graph_pass/schedule_ooo/schedule_base.h) sums one
    operation's inputs AND outputs, deduplicated by `memoryrange.memId`, against
    the buffer -- not a site's tile against the buffer. No per-site rule can be
    right: the same tile at the same width and the same residency both RUNS on
    one kernel and is REFUSED for capacity on another, so the quantity that
    decides is which tiles share one operation.

    Each entry is `[site_id, shape]`, one per distinct tensor the operation
    holds. `site_id` is the site ACTIVE at that operation, not where the tensor
    was first allocated: `set_vec_tile_shapes` tiles the statements that FOLLOW
    it, so the directive in force governs every operand of those statements. An
    online-softmax update under `set_vec_tile_shapes(128, 128)` reads an
    accumulator allocated under `set_vec_tile_shapes(64, 512)`, and it reads it
    128 x 128 at a time. `shape` is the tensor's own declared extent, which
    bounds the allocation from the other side.

    Two tensors are the same memory, and counted once, when either
      * the statement writes into an operand (`oi[:] = ...`), or
      * an operand's life ends at this operation and `_covers` says the result
        fits in it. This is the compiler's liveness reuse. Without it every
        elementwise chain is charged one fresh tile per statement, and
        configurations these kernels are known to run come out over the buffer
        -- which would cost the warm start, and with it the block, since TPE
        then has no feasible observation and returns `best_config: null`.

    `tests/test_capacity_gate.py` holds the adjudication of this rule against
    the device's own verdicts.

    This is a BOUND that rejects the right candidates, and it is not a claim to
    reproduce the compiler's arithmetic. Two things the refusal messages show
    are not modelled here:

      * within a kernel the overflowing operation and its total do not move
        when the tile moves -- a total that is not a function of the tile, so
        what a draw decides is whether that operation is reached, not how big
        it is. A draw wider than the data does not shrink to the data the way
        `vec_trailing` assumes -- it appears to saturate;
      * some capacity refusals are not this rule at all. `ReplaceTensor ...
        can not copy to UB, tensor size [...] exceeds the UB size [196608]` is
        a single tensor the pass must copy whole, independent of the tile.
        Those are recorded as `incorrect_capacity` and a tile gate cannot
        answer them.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    roots = pypto_roots(tree)
    ids = _vec_site_ids(src)
    off = apply.line_starts(src)
    consts = _int_consts(tree)
    out = []
    for scope in _kernel_scopes(tree):
        out.extend(_scope_op_groups(scope, ids, off, roots, consts))
    return out


def group_bytes(domain, config, site_id, shape):
    """One entry's footprint: the site's tile, bounded by the tensor it holds.

    `min(tile_axis, tensor_axis)` per axis, because the directive sets a
    granularity and the tensor bounds it -- plus the running trailing extent,
    which bounds an axis the source does not state an integer for.
    """
    c = (config or {}).get(site_id)
    if not c:
        return 0
    dims = {
        int(k[1:]): v for k, v in c.items() if k.startswith("d") and k[1:].isdigit()
    }
    if not dims:
        return 0
    axes = [dims[k] for k in sorted(dims)]
    if shape:
        pad = (
            list(shape[-len(axes):])
            if len(shape) >= len(axes)
            else [None] * (len(axes) - len(shape)) + list(shape)
        )
        axes = [min(t, s) if s else t for t, s in zip(axes, pad)]
    trailing = (domain.get("vec_trailing") or {}).get(site_id)
    if trailing:
        axes[-1] = min(axes[-1], trailing)
    n = 1
    for v in axes:
        n *= v
    width = (
        (domain.get("vec_dtype_bytes") or {}).get(site_id)
        or domain.get("dtype_bytes")
        or 4
    )
    return n * width


def verify(domain, config, dtype_bytes=4):
    """Is `config` inside `domain`? (ok, reason).

    Called on every candidate, which is what makes a too-wide domain harmless.

    The UB footprint rule is per OPERATION, over `vec_op_groups`:

        sum over the tiles one operation holds of
            prod(min(tile_axis, tensor_axis, running trailing)) * dtype   <= UB

    `vec_op_groups` carries the whole derivation and the evidence for it; read it
    before changing anything here. In short: the buffer is shared, so the thing
    that has to fit is one operation's inputs and outputs together, and a rule
    written per site is wrong in both directions at once. The per-site rule this
    replaced both missed capacity refusals the device went on to make and
    free-rejected candidates the device would have run -- throwing away
    candidates that would have worked, which is why tile blocks collapsed: TPE
    had no feasible observations left to model.

    `dtype` is the width of the elements that directive works on, read from the
    source per site (`vec_sites`), because it is a property of the directive and
    not of the file. A single kernel-wide width was the currency here until it
    was noticed that it is the MINIMUM over every `DT_*` token in the file: an
    antiquant kernel's int8 cache sets it to 1 byte for an fp32 softmax site,
    and the gate then passes tiles several times over the buffer.

    A site the walk could not read is in `vec_unmodelled` and is not judged. With
    no groups at all -- an unparseable source, or a kernel with no vec directive
    -- the footprint bound is simply OFF, which is the existing rule for a domain
    that cannot state a bound. Cube sites carry no `d*` dims and are checked
    against L0A/B/C in `space` instead.
    """
    del dtype_bytes  # per-site now; see `vec_sites`
    ub = domain.get("ub_bytes")
    groups = domain.get("vec_op_groups") or ()
    if not ub or not groups:
        return True, ""
    skip = set(domain.get("vec_unmodelled") or ())
    for group in groups:
        total, parts, spend = 0, [], {}
        for sid, shape in group:
            if sid in skip:
                continue
            b = group_bytes(domain, config, sid, shape)
            if not b:
                continue
            total += b
            spend[sid] = spend.get(sid, 0) + b
            extent = (
                "x".join("?" if x is None else str(x) for x in shape)
                if shape
                else "unbounded"
            )
            parts.append(f"{sid}[{extent}]={b}B")
        if total > ub:
            # The reason opens with "<site_id>: ", which `space.repair` parses to
            # decide what to shrink, and carries " UB" so `_is_capacity` knows it
            # is a capacity refusal and not an alignment one. The site named is
            # the one spending the most in THIS operation -- shrinking any other
            # one costs a rung and does not clear the overflow.
            worst = max(spend, key=spend.get)
            return False, (
                f"{worst}: one operation holds "
                + " + ".join(parts)
                + f" = {total}B > {ub}B UB"
            )
    return True, ""
