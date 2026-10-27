# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Which per-case config-dict entries the deterministic core may tune.

`apply` rewrites a tile call only when its arguments are integer literals. A
kernel written as a FACTORY has none:

    _KERNEL_CONFIG = {
        10: dict(mm_l0=64, vec_rows=32, n_l1=256),
        13: dict(mm_l0=128, stitch=1024, max_ws_kb=524800),
    }

    def _make_static_kernel(M, K, E, N, rows, mm_l0=MM_L0, vec_rows=VEC_ROWS,
                            n_l1=N_L1, stitch=128, max_ws_kb=None, sched=0):
        @pypto.frontend.jit(runtime_options=_ro)
        def kernel(...):
            pypto.set_cube_tile_shapes([mm_l0, MM_L1], [K_L0, K_L1], [N_L0, n_l1])
            pypto.set_vec_tile_shapes(vec_rows, VEC_WIDE)

    for cid, M, K, E, N in _UNIFORM:
        _STATIC[(M, K, E, N)] = _make_static_kernel(M, K, E, N, M // E,
                                                    **_KERNEL_CONFIG.get(cid, {}))

Every tile argument here is an `ast.Name`, so `apply` reads the calls as
`cube_dyn` / `vec_dyn` -- discovered and left alone, because rewriting
`set_vec_tile_shapes(M, K)` with numbers would change what the kernel computes
rather than how it is tiled. That judgement is right in general and wrong here:
`mm_l0` and `vec_rows` are not shapes, they are tuning parameters with
defaults, and the per-case values live in `_KERNEL_CONFIG` as integer literals
already.

The cost is that the tile BO runs ZERO blocks on such a kernel: every case
logs "no tunable tile site", and the knobs that are in fact tunable are swept
by hand instead, one evaluation per value.

So this module admits the config entry instead of the call. Selection is
default-deny and three-staged, the same shape as `const.py`:

  1. Find a module-level dict whose values are per-case `dict(...)` entries of
     integer literals.
  2. Confirm it is `**`-expanded into a call, and resolve that call to a
     module-level function. THAT function's defaulted parameters are the only
     legal knob names -- which is also what separates a knob from a shape, since
     `M, K, E, N, rows` carry no default.
  3. Admit a knob only if its parameter reaches the kernel: read inside the
     nested `@jit` body, or reaching the decorator's `runtime_options` /
     `pass_options`. A parameter reaching a tensor annotation is excluded, for
     the reason stage 1 of `const.py` excludes one -- that is an extent, not a
     knob.

Nothing is admitted without a key: which case is being optimised is a fact the
dict cannot answer, because the whole point of it is that it holds several.
That fact is carried IN THE KERNEL, as a marker comment `normalize` writes on
the entry at INIT from `--config-key`:

    _KERNEL_CONFIG = {
        10: dict(mm_l0=64),   # panko:tune
        13: dict(mm_l0=128),
    }

A comment rather than a parameter threaded through `apply`, `block`, `lever`
and the evaluator, because the campaign's subject already lives in the kernel
file and travels with every snapshot of it: a candidate restored from
`nodes/<hash>.py` still says which case it is. No marker, no cfg sites.
"""
import ast
import re
from collections import namedtuple

MARKER = "panko:tune"
_MARKED = re.compile(r"#[^\n]*\b" + re.escape(MARKER) + r"\b")

# `slot` says where the knob lands, which is what gives it a LADDER and a
# footprint. ("cube", axis, level) with axis in m/k/n and level in L0/L1;
# ("vec", i) for the i-th argument of set_vec_tile_shapes; None for a knob that
# reaches the kernel by another route (a runtime option, an unroll list).
Knob = namedtuple("Knob", "name value start end slot present")

_CUBE_AXES = ("m", "k", "n")
_CUBE_LEVELS = ("L0", "L1")
_TILE_FUNCS = ("set_cube_tile_shapes", "set_vec_tile_shapes")


def _fname(func):
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _int_lit(n):
    return (isinstance(n, ast.Constant) and isinstance(n.value, int)
            and not isinstance(n.value, bool))


def _literal(n):
    """A value a config table can hold: a scalar, or a flat sequence of them.

    Not every entry is tunable -- `unroll_seq=(128, 64, 16, 4, 1)` is a tuple
    and the core tunes integers -- but one untunable entry must not disqualify
    the TABLE. Requiring int literals everywhere did: a single tuple value was
    enough to skip the whole table.
    """
    if isinstance(n, ast.Constant):
        return isinstance(n.value, (int, float, str, bool)) or n.value is None
    if isinstance(n, (ast.Tuple, ast.List)):
        return all(isinstance(e, ast.Constant) for e in n.elts)
    return False


def _tunable_default(fn, knob):
    """Is this parameter's default an INTEGER-shaped one?

    The core tunes integers. A parameter defaulting to a tuple or a list is a
    schedule, and offering to set it to a number would produce a kernel that
    does not compile. The tile knobs default to a module constant (a Name), so
    a Name is admitted and the value is read from the entry.
    """
    args = fn.args
    defaults = dict(zip([a.arg for a in args.args[len(args.args) - len(args.defaults):]],
                        args.defaults)) if args.defaults else {}
    defaults.update({a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults) if d})
    d = defaults.get(knob)
    if d is None:
        return False
    if isinstance(d, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        return False
    return True


def _offsets(src):
    """(line, col) -> absolute byte offset, for the spans `apply` rewrites."""
    starts, total = [], 0
    for line in src.splitlines(keepends=True):
        starts.append(total)
        total += len(line)
    starts.append(total)

    def off(lineno, col):
        return starts[lineno - 1] + col
    return off


def _is_jit(fn):
    for d in getattr(fn, "decorator_list", []):
        node = d.func if isinstance(d, ast.Call) else d
        if _fname(node) == "jit":
            return True
    return False


def _entry_pairs(value):
    """The keyword pairs of a `dict(...)` call or a `{...}` literal, or None.

    Both spellings occur and neither is more correct; what matters is that the
    entry is a flat mapping of names to integer literals.
    """
    if isinstance(value, ast.Call) and _fname(value.func) == "dict" and not value.args:
        return [(kw.arg, kw.value) for kw in value.keywords if kw.arg]
    if isinstance(value, ast.Dict):
        out = []
        for k, v in zip(value.keys, value.values):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                return None
            out.append((k.value, v))
        return out
    return None


def _is_dict_assign(node):
    """A module-level `NAME = {...}` statement with at least one key."""
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
        return False
    if not isinstance(node.targets[0], ast.Name):
        return False
    return isinstance(node.value, ast.Dict) and bool(node.value.keys)


def _entries_are_literal(value):
    """Whether every value of a dict is a flat mapping of integer literals."""
    for k, v in zip(value.keys, value.values):
        pairs = _entry_pairs(v)
        if k is None or not isinstance(k, ast.Constant):
            return False
        if pairs is None or not all(_literal(x) for _, x in pairs):
            return False
    return True


def _config_dicts(tree):
    """{name: ast.Dict} for every module-level per-case config dict.

    The shape is what identifies it: a mapping whose every value is itself a
    flat mapping of names to integer literals. A dict of anything else is not
    this idiom and is not looked at again.
    """
    out = {}
    for node in tree.body:
        if not _is_dict_assign(node):
            continue
        if _entries_are_literal(node.value):
            out[node.targets[0].id] = node.value
    return out


def _module_func(tree, name):
    """The module-level `def name` of this tree, or None."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _expanded_into(tree, dict_name):
    """The module-level function `dict_name` is `**`-expanded into, or None.

    `f(..., **CFG[k])` and `f(..., **CFG.get(k, {}))` are the two spellings the
    recorded kernels use. Anything else is not matched, and nothing is admitted
    -- a config dict that is never expanded into a function has no signature to
    take its legal knob names from.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg is not None:
                continue                     # a named keyword, not `**`
            names = {n.id for n in ast.walk(kw.value) if isinstance(n, ast.Name)}
            if dict_name not in names:
                continue
            fn = _module_func(tree, _fname(node.func))
            if fn is not None:
                return fn
    return None


def _defaulted_params(fn):
    """The parameter names that carry a default -- the legal knob names.

    This is also what separates a knob from a shape without having to reason
    about either: `M, K, E, N, rows` are passed positionally and carry no
    default, while every tuning parameter has one.
    """
    args = fn.args
    with_default = args.args[len(args.args) - len(args.defaults):] if args.defaults else []
    return [a.arg for a in with_default] + [a.arg for a in args.kwonlyargs]


def _jit_body(fn):
    """The nested `@jit` function inside the factory, or None."""
    for node in ast.walk(fn):
        if isinstance(node, ast.FunctionDef) and node is not fn and _is_jit(node):
            return node
    return None


def _bound_names(node):
    """Every name the target(s) of one assignment statement bind."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    out = set()
    for t in targets:
        for n in ast.walk(t):
            if isinstance(n, ast.Name):
                out.add(n.id)
    return out


def _taint(fn, names):
    """`names`, plus every local name an assignment derives from one of them.

    `stitch` reaches the kernel as `_ro = {...: stitch}` and then
    `@jit(runtime_options=_ro)`, so following one assignment is the difference
    between seeing three runtime knobs and seeing none. Propagated by name and
    ignoring scope, which over-taints; the admission below is what bounds that,
    and the golden is what bounds the admission.
    """
    tainted = set(names)
    for _ in range(4):                       # a fixpoint; the chains are short
        grew = False
        for node in ast.walk(fn):
            if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                continue
            rhs = node.value
            if rhs is None:
                continue
            used = {n.id for n in ast.walk(rhs) if isinstance(n, ast.Name)}
            if not (used & tainted):
                continue
            fresh = _bound_names(node) - tainted
            if fresh:
                tainted |= fresh
                grew = True
        if not grew:
            break
    return tainted


def _names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _reaches_kernel(fn, jit_fn, knob):
    """Does `knob` reach what the kernel compiles, and is it a knob not an extent?

    Excluded first, like `const.py`: a parameter reaching a tensor annotation in
    the jit signature is part of the declared interface, and rewriting it would
    change what the kernel computes rather than how it runs.
    """
    tainted = _taint(fn, {knob})
    for a in list(jit_fn.args.args) + list(jit_fn.args.kwonlyargs):
        if a.annotation is not None and (_names_in(a.annotation) & tainted):
            return False
    for d in jit_fn.decorator_list:           # runtime_options / pass_options
        if _names_in(d) & tainted:
            return True
    return bool(_names_in(jit_fn) & tainted)


def _cube_slot(node, knob):
    """Where `knob` sits in one cube tile call, or None."""
    for i, arg in enumerate(node.args[:3]):
        if not isinstance(arg, ast.List):
            continue
        for j, e in enumerate(arg.elts[:2]):
            if isinstance(e, ast.Name) and e.id == knob:
                return ("cube", _CUBE_AXES[i], _CUBE_LEVELS[j])
    return None


def _vec_slot(node, knob):
    """Where `knob` sits in one vec tile call, or None."""
    for i, arg in enumerate(node.args):
        if isinstance(arg, ast.Name) and arg.id == knob:
            return ("vec", i)
    return None


def _slot_of(jit_fn, knob):
    """Where `knob` sits in a tile call, or None.

    Only a DIRECT appearance counts. A knob one derivation removed from the
    call still reaches the kernel and is still tunable; what it does not get is
    a tile ladder and a footprint, because neither can be read off the call.
    """
    for node in ast.walk(jit_fn):
        if not (isinstance(node, ast.Call) and _fname(node.func) in _TILE_FUNCS):
            continue
        if _fname(node.func) == "set_cube_tile_shapes":
            slot = _cube_slot(node, knob)
        else:
            slot = _vec_slot(node, knob)
        if slot is not None:
            return slot
    return None


def marked_key(src):
    """The config key whose entry carries the `panko:tune` marker, or None.

    The marker is looked for on the entry's own line, which is where a comment
    after `10: dict(...),` lands. A file with two markers is a file whose
    subject is ambiguous, so it has none.
    """
    got = analyse(src)
    if got is None:
        return None
    cfg = got[4]
    lines = src.splitlines()
    found = None
    for k, v in zip(cfg.keys, cfg.values):
        if not isinstance(k, ast.Constant):
            continue
        line = lines[v.end_lineno - 1] if v.end_lineno <= len(lines) else ""
        if _MARKED.search(line):
            if found is not None:
                return None
            found = k.value
    return found


def _entry_node(cfg, key):
    """The value node for `key` in the config dict, or None."""
    for k, v in zip(cfg.keys, cfg.values):
        if isinstance(k, ast.Constant) and k.value == key:
            return v
    return None


def analyse(src):
    """(dict_name, factory, jit_fn, knobs, cfg) for the one config idiom, or None.

    `knobs` is {name: slot} over the admitted knob names -- the factory's
    defaulted parameters that reach the kernel. `cfg` is the dict node.
    """
    tree = ast.parse(src)
    for name, cfg in _config_dicts(tree).items():
        factory = _expanded_into(tree, name)
        if factory is None:
            continue
        jit_fn = _jit_body(factory)
        if jit_fn is None:
            continue
        knobs = {}
        for p in _defaulted_params(factory):
            if _tunable_default(factory, p) and _reaches_kernel(factory, jit_fn, p):
                knobs[p] = _slot_of(jit_fn, p)
        if knobs:
            return name, factory, jit_fn, knobs, cfg
    return None


def admitted(src, key=None):
    """{knob: Knob} for config entry `key`. Default-deny: {} without a match.

    A knob already in the entry carries its value and the byte span of that
    value. A legal knob the entry does not mention is returned too, with
    `present=False` and a zero-width span at the insertion point -- the same
    convention `apply` uses for an absent runtime option, so the kernel as
    written stays a reachable point and the core can also try setting it.

    An entry that does not exist admits nothing: inserting one is a structural
    edit and belongs to `normalize`, which runs once, not to `apply`, which
    runs on every trial.
    """
    got = analyse(src)
    if got is None:
        return {}
    if key is None:
        key = marked_key(src)
    if key is None:
        return {}
    _name, _factory, _jit_fn, knobs, cfg = got
    node = _entry_node(cfg, key)
    if node is None:
        return {}
    pairs = _entry_pairs(node) or []
    off = _offsets(src)
    out, present, frozen = {}, {}, set()
    for pname, v in pairs:
        if pname not in knobs:
            continue
        if _int_lit(v):
            present[pname] = v
        else:
            # Named by this case, but not as a number. It is neither tunable
            # (no ladder fits it) nor insertable (it is already set), so the
            # core does not see it at all.
            frozen.add(pname)
    # Just inside the entry's closing bracket, where a new keyword goes.
    insert = off(node.end_lineno, node.end_col_offset) - 1
    for pname, slot in sorted(knobs.items()):
        if pname in frozen:
            continue
        v = present.get(pname)
        if v is not None:
            out[pname] = Knob(pname, v.value, off(v.lineno, v.col_offset),
                              off(v.end_lineno, v.end_col_offset), slot, True)
        else:
            out[pname] = Knob(pname, None, insert, insert, slot, False)
    return out


def _strip_markers(src):
    """Every `panko:tune` comment removed, so marking is idempotent."""
    out = []
    for line in src.split("\n"):
        m = _MARKED.search(line)
        out.append(line[:m.start()].rstrip() if m else line)
    return "\n".join(out)


def mark(src, key):
    """(src, action) with `key`'s entry marked as the case being tuned.

    Three actions, because the three are worth telling apart in a log:
    `marked` (the entry was there), `inserted` (it was not, and an empty one was
    added), `unavailable` (this kernel is not written as a config factory).

    Inserting `<key>: dict(),` is semantically a no-op: the factory's own
    defaults then apply, exactly as they do for a case the dict does not
    mention. That is what makes this a normalisation rather than a change, and
    it is the whole reason it has to happen -- a case with no entry at all
    leaves nothing for the core to tune even once it can see the table.
    """
    got = analyse(src)
    if got is None:
        return src, "unavailable"
    src = _strip_markers(src)
    got = analyse(src)
    cfg = got[4]
    node = _entry_node(cfg, key)
    lines = src.split("\n")
    if node is not None:
        i = node.end_lineno - 1
        lines[i] = lines[i] + "  # " + MARKER
        return "\n".join(lines), "marked"
    last = cfg.values[-1]
    first_line = lines[last.lineno - 1]
    indent = first_line[:len(first_line) - len(first_line.lstrip())]
    off = _offsets(src)
    at = off(last.end_lineno, last.end_col_offset)
    tail = src[at:]
    if tail[:1] == ",":                      # keep the entry list well-formed
        at += 1
    key_src = repr(key) if isinstance(key, str) else str(key)
    return (src[:at] + f"\n{indent}{key_src}: dict(),  # {MARKER}" + src[at:],
            "inserted")


def entry_exists(src, key):
    """Whether the config dict already holds an entry for `key`.

    `normalize` asks before inserting one. Inserting `key: dict()` is
    semantically a no-op -- the factory's own defaults then apply, exactly as
    they do for a case the dict does not mention -- which is what makes it a
    normalisation rather than a change.
    """
    got = analyse(src)
    return bool(got) and _entry_node(got[4], key) is not None


def entry_insertion(src):
    """(offset, indent) where a new per-case entry goes, or None."""
    got = analyse(src)
    if got is None:
        return None
    cfg = got[4]
    off = _offsets(src)
    last = cfg.values[-1]
    line_start = off(last.end_lineno, 0)
    indent = len(src[line_start:].split("\n")[0]) - len(
        src[line_start:].split("\n")[0].lstrip())
    return off(last.end_lineno, last.end_col_offset), indent
