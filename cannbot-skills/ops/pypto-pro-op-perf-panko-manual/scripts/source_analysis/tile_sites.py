# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Read Pro tile declarations, source spans and ownership without executing the kernel."""

from __future__ import annotations

__all__ = [
    "ShapeEntry",
    "TileSite",
    "module_constants",
    "read_sites",
    "sibling_constants",
    "site_by_id",
]


import ast
import logging
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

from panko_core.pypto_ast import function_name, keyword_value
from source_analysis.case_context import active_scope_closure
from source_analysis.kernel_boundary import allocation_owners
from source_analysis.source_bindings import UNKNOWN, Binding, SourceBindings


logger = logging.getLogger(__name__)

_MAKE = "make_tile_group"
_SINGLE_TILE = "make_tile"
_TILETYPE = "TileType"
_SPACES = ("Vec", "Mat", "Left", "Right", "Scaling", "Acc", "Bias",
           "ScaleLeft", "ScaleRight")
# Report DDR as a known space and exclude it from on-chip high-water accounting.
OFF_CHIP = ("DDR",)
_DEFAULT_SPACE = "Vec"          # pl.TileType(target_memory=MemorySpace.Vec)


@dataclass(frozen=True)
class ShapeEntry:
    """One entry of a `shape=[...]`, as text, as a span, and as an int if known."""

    text: str
    start: int
    end: int
    value: Optional[int]


@dataclass(frozen=True)
class TileSite:
    """A tile allocation with its resolved fields and source spans."""

    site_id: str
    group_var: Optional[str]
    space: Optional[str]
    dtype_text: str
    shape: Tuple[ShapeEntry, ...]
    depth: Optional[int]
    depth_source: str
    depth_span: Optional[Tuple[int, int]]
    addrs_text: str
    addrs_span: Tuple[int, int]
    mutex_ids_text: Optional[str]
    call_span: Tuple[int, int]
    lineno: int
    resolvable: bool
    unresolved: Tuple[str, ...]
    # A scalar denotes the base. Lists preserve per-slot addresses. None marks unresolved entries.
    addrs_values: Tuple[Optional[int], ...] = ()
    # Empty when static resolution cannot determine the dtype.
    dtype_resolved: str = ""
    # Missing and -1 entries impose no lower bound.
    valid_shape: Tuple[Optional[int], ...] = ()
    # Layout and fractal box determine trailing-axis alignment.
    layout: str = ""
    # An explicit fractal excludes the tile from the RowMajor + NoneBox alignment rule.
    fractal: bool = False
    #: Stable lexical owner; sibling factories may use identical local names.
    scope: str = ""
    #: The actual shape keyword expression, including an alias when one is used.
    shape_span: Optional[Tuple[int, int]] = None
    shape_is_sequence: bool = True
    #: A resolved address alias can denote a list, including a one-slot list.
    addrs_is_sequence: bool = False
    function_lineno: int = 0
    shape_rank_known: bool = True
    allocation_scope: Tuple[str, ...] = ()


class Offsets:
    """Convert AST line and UTF-8 byte columns to character indices for slicing the original source."""

    def __init__(self, src: str) -> None:
        self._src = src
        # Match Python's physical lines without treating form feeds or Unicode
        # separators inside comments/strings as newlines, or changing source bytes.
        self._starts = [0] + [m.end() for m in re.finditer(r"\r\n?|\n", src)]

    def pos(self, lineno: int, col: int) -> int:
        if lineno < 1:
            return 0
        if lineno > len(self._starts):
            return len(self._src)
        start = self._starts[lineno - 1]
        end = self._starts[lineno] if lineno < len(self._starts) else len(self._src)
        line = self._src[start:end]
        if line.isascii():
            return start + min(max(0, col), len(line))
        prefix = line.encode("utf-8")[:max(0, col)]
        return start + len(prefix.decode("utf-8", errors="ignore"))

    def span(self, node: ast.AST) -> Tuple[int, int]:
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)
        if lineno is None or end_lineno is None:
            return (0, 0)
        start = self.pos(lineno, getattr(node, "col_offset", 0))
        end = self.pos(end_lineno, getattr(node, "end_col_offset", 0))
        return start, max(start, end)


class _ScopedValues(dict):
    """Integer-compatible view backed by captured lexical expression bindings."""

    def __init__(self, resolver, bindings):
        self.resolver = resolver
        self.bindings = bindings
        super().__init__((k, value) for k, binding in bindings.items()
                         if (value := resolver.value(binding)) is not UNKNOWN)

    def value(self, node):
        return self.resolver.value(Binding(node, self.bindings))

    def integer(self, node):
        value = self.value(node)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def resolve(self, node):
        return self.resolver.resolve(Binding(node, self.bindings))


def scoped_tiletype(binding, resolver, depth=0):
    """Follow lexical aliases and a transparent TileType factory, preserving spans."""
    if depth > 12:
        return None
    binding = resolver.resolve(binding)
    if binding is None:
        return None
    node = binding.node
    if is_named_call(node, _TILETYPE):
        return binding
    if not isinstance(node, ast.Call):
        return None
    fn = resolver.resolve(Binding(node.func, binding.env))
    if not fn or not isinstance(fn.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    body = [s for s in fn.node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    args = resolver.arguments(fn, node, binding.env)
    result = scoped_tiletype(Binding(body[0].value, args), resolver, depth + 1)
    if result is None:
        return result
    params = {p.arg for p in list(fn.node.args.posonlyargs) + list(fn.node.args.args) + list(fn.node.args.kwonlyargs)}
    bindings = {}
    for kw in result.node.keywords:
        bindings[kw.arg] = (args.get(kw.value.id) if isinstance(kw.value, ast.Name)
                            and kw.value.id in params else Binding(kw.value, result.env))
    copied = ast.Call(func=result.node.func, args=result.node.args,
                      keywords=result.node.keywords)
    copied.panko_keyword_bindings = bindings
    return Binding(copied, result.env)


def _scoped_dtype(node, values):
    def reduce(binding, depth=0):
        if binding is None or depth > 12:
            return ""
        resolved = values.resolver.resolve(binding)
        if resolved is None:
            return ""
        value = values.resolver.value(resolved)
        if isinstance(value, str) and value.startswith("DT_"):
            return value
        aliases = {name: reduce(resolved.env.get(name), depth + 1)
                   for name in _free_names(resolved.node)
                   if name in resolved.env}
        return _reduce_dtype(resolved.node, aliases)
    return reduce(Binding(node, values.bindings))


def _free_names(node: ast.AST) -> Tuple[str, ...]:
    """Every NAME read in this expression, in source order, deduplicated."""
    return tuple(dict.fromkeys(child.id for child in ast.walk(node) if isinstance(child, ast.Name)))


def module_constants(src: str,
                     seed: Optional[Mapping[str, int]] = None) -> Dict[str, int]:
    """Resolve module-level integers using arithmetic and source-visible pure scalar planners.

    Supply imported constants in seed before evaluation so derived values can use them.
    Local bindings take precedence; unknown runtime calls invalidate stale values.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    # An unresolved rebind invalidates earlier integer values.
    resolver = SourceBindings(tree, seed, override_extra=False)
    assigned = set()
    for stmt in tree.body:
        targets = (stmt.targets if isinstance(stmt, ast.Assign) else
                   [stmt.target] if isinstance(stmt, ast.AnnAssign) else [])
        for target in targets:
            assigned.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    constants = {}
    for name in assigned:
        value = resolver.value(resolver.module_env.get(name))
        if isinstance(value, int) and not isinstance(value, bool):
            constants[name] = value
    return constants


_DTYPE_PREFIX = "DT_"


def _reduce_dtype(node: Optional[ast.AST], aliases: Mapping[str, str],
                  depth: int = 0) -> str:
    """Resolve dtype aliases to a DT_ token or return an empty string."""
    if node is None or depth > 8:
        return ""
    if isinstance(node, ast.Attribute):
        return node.attr if node.attr.startswith(_DTYPE_PREFIX) else ""
    if isinstance(node, ast.Name):
        if node.id.startswith(_DTYPE_PREFIX):
            return node.id
        return aliases.get(node.id, "")
    if isinstance(node, ast.Call):
        if len(node.args) == 1 and not node.keywords:
            return _reduce_dtype(node.args[0], aliases, depth + 1)
        return ""
    if isinstance(node, ast.IfExp):
        a = _reduce_dtype(node.body, aliases, depth + 1)
        b = _reduce_dtype(node.orelse, aliases, depth + 1)
        return a if a and a == b else ""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value if node.value.startswith(_DTYPE_PREFIX) else ""
    return ""


def is_named_call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and function_name(node.func) == name


def space_of(target_memory: Optional[ast.AST]) -> Optional[str]:
    """"Vec" from `pl.MemorySpace.Vec`, or the DSL default when absent."""
    if target_memory is None:
        return _DEFAULT_SPACE
    known = _SPACES + OFF_CHIP
    if isinstance(target_memory, ast.Attribute) and target_memory.attr in known:
        return target_memory.attr
    if isinstance(target_memory, ast.Name) and target_memory.id in known:
        return target_memory.id
    if isinstance(target_memory, ast.Constant) and target_memory.value in known:
        return str(target_memory.value)
    return None


def _tiletype_arg(tiletype, name):
    node = keyword_value(tiletype, name)
    if node is None:
        index = ("shape", "dtype", "target_memory", "valid_shape", "layout", "fractal").index(name)
        if index < len(tiletype.args):
            return tiletype.args[index]
    return node


def _assigned_names(tree: ast.AST, offsets: Offsets) -> Dict[int, str]:
    """call object id -> the single NAME it is assigned to, when there is one."""
    out: Dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and is_named_call(node.value, _MAKE):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            if len(targets) == 1:
                out[id(node.value)] = targets[0].id
        elif isinstance(node, ast.AnnAssign) and is_named_call(node.value, _MAKE):
            if isinstance(node.target, ast.Name):
                out[id(node.value)] = node.target.id
    return out


def _shape_entries(
    shape: Optional[ast.AST], src: str, offsets: Offsets, env: _ScopedValues
):
    """Read each shape element with its source span and static value. Preserve the whole span when the
    expression is opaque."""
    if shape is None:
        return (), (), False
    resolved = env.resolve(shape)
    if resolved and isinstance(resolved.node, (ast.List, ast.Tuple)):
        shape = resolved.node
        env = _ScopedValues(env.resolver, resolved.env)
    elif isinstance(value := env.value(shape), (list, tuple)):
        start, end = offsets.span(shape)
        entries = tuple(ShapeEntry(src[start:end], start, end,
                        v if isinstance(v, int) and not isinstance(v, bool) else None)
                        for v in value)
        return entries, (), True
    nodes = shape.elts if isinstance(shape, (ast.List, ast.Tuple)) else [shape]
    entries = []
    unresolved: list = []
    for node in nodes:
        start, end = offsets.span(node)
        value = env.integer(node)
        entries.append(ShapeEntry(text=src[start:end], start=start, end=end, value=value))
        if value is None:
            unresolved.extend(name for name in _free_names(node) if name not in env and name not in unresolved)
    return tuple(entries), tuple(unresolved), isinstance(shape, (ast.Tuple, ast.List))


def _addrs_entries(addrs: Optional[ast.AST], env: _ScopedValues):
    """Resolve addresses through lexical bindings. The rewriter determines how to edit the original
    expression."""
    if addrs is None:
        return (), False
    value = env.value(addrs)
    if isinstance(value, (list, tuple)):
        return tuple(v if isinstance(v, int) and not isinstance(v, bool) else None
                     for v in value), True
    if isinstance(addrs, (ast.List, ast.Tuple)):
        return tuple(env.integer(e) for e in addrs.elts), False
    return (env.integer(addrs),), False


def _depth_of(call: ast.Call, src: str, offsets: Offsets, env: _ScopedValues):
    """(depth, depth_source, depth_span, mutex_ids_text), by pypto_pro's own rule."""
    depth_node = keyword_value(call, "depth")
    mutex_node = keyword_value(call, "mutex_ids")
    mutex_text = None
    if mutex_node is not None:
        m_start, m_end = offsets.span(mutex_node)
        mutex_text = src[m_start:m_end]

    if depth_node is not None:
        span = offsets.span(depth_node)
        return env.integer(depth_node), "depth=", span, mutex_text

    if isinstance(mutex_node, (ast.List, ast.Tuple)) and mutex_node.elts:
        return len(mutex_node.elts), "len(mutex_ids)", None, mutex_text
    if mutex_node is not None:
        value = env.value(mutex_node)
        if isinstance(value, (list, tuple)) and value:
            return len(value), "len(mutex_ids)", None, mutex_text

    if isinstance(mutex_node, ast.Constant) and mutex_node.value == "auto":
        return None, "auto", None, mutex_text

    # Depth cannot be inferred from an empty or absent mutex sequence.
    return None, "default", None, mutex_text


class TileProgram:
    """Resolve declarations against one source's lexical and allocation scopes."""

    def __init__(self, src, extra_consts, *, include_single_tiles=False):
        self.src = src
        tree = ast.parse(src)
        self.offsets = Offsets(src)
        self.bindings = SourceBindings(tree, extra_consts)
        self.active_scopes = active_scope_closure(tree, self.bindings, extra_consts)
        self.allocation_owners = allocation_owners(self.bindings.boundary)
        self.var_of = _assigned_names(tree, self.offsets)
        allocations = (_MAKE, _SINGLE_TILE) if include_single_tiles else (_MAKE,)
        self.calls = [node for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and function_name(node.func) in allocations]
        self.calls.sort(key=lambda node: self.offsets.span(node))
        # Known case branches may hide declarations, but unknown branches retain
        # both declarations and their budgets. Number all calls before filtering.
        self.active_spans = {self.offsets.span(call) for call in self.calls
                             if self.bindings.active_at.get(id(call), True)}
        self.per_space: Dict[str, int] = {}

        self.sites = tuple(self.read_call(call) for call in self.calls)
        if self.active_scopes is None:
            self.active_sites = self.sites
        else:
            self.active_sites = tuple(site for site in self.sites if
                                      site.scope in self.active_scopes and site.call_span in self.active_spans)

    @staticmethod
    def keyword_binding(tile_binding, name):
        """Retain the lexical environment carried by a constructed type keyword."""
        node = _tiletype_arg(tile_binding.node, name)
        if node is None:
            return None
        keywords = getattr(tile_binding.node, 'panko_keyword_bindings', {})
        return keywords.get(name) or Binding(node, tile_binding.env)

    def body_constraints(self):
        return _BodyConstraintReader(self.bindings, self.offsets, self.active_sites).read()

    def read_call(self, call):
        """Build one site without inheriting facts from a previous call."""
        call_span = self.offsets.span(call)
        here_env = _ScopedValues(self.bindings, self.bindings.environment(call))
        fields, shape_node, shape_env = self._type_fields(call, here_env)
        shape, unresolved, rank_known = _shape_entries(shape_node, self.src, self.offsets, shape_env)
        single = is_named_call(call, _SINGLE_TILE)
        addrs = keyword_value(call, "addr" if single else "addrs")
        addrs_span = self.offsets.span(addrs) if addrs is not None else (call_span[0], call_span[0])
        addresses, is_sequence = _addrs_entries(addrs, here_env)
        key = (fields["space"] or "unknown").lower()
        if single:
            depth, depth_source, depth_span, mutex_text = 1, "make_tile", None, None
            key += ".tile"
        else:
            depth, depth_source, depth_span, mutex_text = _depth_of(call, self.src, self.offsets, here_env)
        index = self.per_space.get(key, 0)
        self.per_space[key] = index + 1
        scope = self.bindings.scope(call)
        function_lineno = self.bindings.scope_lines.get(scope, 0)
        return TileSite(
            site_id="%s#%d" % (key, index), group_var=self.var_of.get(id(call)),
            **fields, shape=shape, shape_rank_known=rank_known, scope=scope,
            addrs_span=addrs_span, addrs_text=self.src[slice(*addrs_span)],
            addrs_values=addresses, addrs_is_sequence=is_sequence,
            function_lineno=function_lineno,
            allocation_scope=self.allocation_owners.get(function_lineno, ()),
            depth=depth, depth_source=depth_source, depth_span=depth_span,
            mutex_ids_text=mutex_text, call_span=call_span,
            lineno=getattr(call, "lineno", 0),
            resolvable=bool(shape) and all(e.value is not None for e in shape),
            unresolved=unresolved)

    def type_binding(self, call):
        if is_named_call(call, _SINGLE_TILE):
            node = call.args[0] if call.args else None
        else:
            node = keyword_value(call, "type")
        if node is not None:
            return scoped_tiletype(Binding(node, self.bindings.environment(call)), self.bindings)
        return None

    def _type_fields(self, call, here_env):
        """Read type facts with fresh defaults for every declaration."""
        fields = dict(space=None, dtype_text="", dtype_resolved="", valid_shape=(),
                      layout="", fractal=False, shape_span=None, shape_is_sequence=True)
        shape_node = None
        type_binding = self.type_binding(call)
        tiletype = type_binding.node if type_binding is not None else None
        if (tiletype is None or any(isinstance(arg, ast.Starred) for arg in tiletype.args)
                or any(kw.arg is None for kw in tiletype.keywords)):
            return fields, shape_node, here_env
        type_env = _ScopedValues(self.bindings, type_binding.env)
        fields["layout"] = _layout_of(_tiletype_arg(tiletype, "layout"), self.src, self.offsets)
        fields["fractal"] = _tiletype_arg(tiletype, "fractal") is not None
        fields["space"] = space_of(_tiletype_arg(tiletype, "target_memory"))
        dtype_binding = self.keyword_binding(type_binding, "dtype")
        dtype_node = dtype_binding.node if dtype_binding else None
        if dtype_node is not None:
            d_start, d_end = self.offsets.span(dtype_node)
            fields["dtype_text"] = self.src[d_start:d_end]
            fields["dtype_resolved"] = _scoped_dtype(
                dtype_node, _ScopedValues(self.bindings, dtype_binding.env))
        shape_binding = self.keyword_binding(type_binding, "shape")
        shape_node = shape_binding.node if shape_binding else None
        if shape_binding:
            fields["shape_span"] = self.offsets.span(shape_node)
            fields["shape_is_sequence"] = isinstance(shape_node, (ast.List, ast.Tuple))
            shape_env = _ScopedValues(self.bindings, shape_binding.env)
        else:
            shape_env = type_env
        valid_binding = self.keyword_binding(type_binding, "valid_shape")
        if valid_binding:
            fields["valid_shape"] = _shape_floors(
                valid_binding.node, _ScopedValues(self.bindings, valid_binding.env)) or ()
        return fields, shape_node, shape_env


def read_sites(
    src: str, extra_consts: Optional[Mapping[str, int]] = None, *,
    include_inactive: bool = False,
) -> Tuple[TileSite, ...]:
    """Return tiles in source order, restricting to the active case by default."""
    program = TileProgram(src, extra_consts)
    return program.sites if include_inactive else program.active_sites


def _layout_of(node, src, offsets):
    """Return the written layout name, including an unresolved alias."""
    if node is None:
        return ""
    try:
        a, b = offsets.span(node)
    except Exception:                                      # noqa: BLE001
        return ""
    text = (src[a:b] or "").strip()
    return text.rsplit(".", 1)[-1] if text else ""


def _shape_floors(node, env):
    """Read positive per-axis bounds. None denotes an unknown rank."""
    if node is None:
        return None
    resolved = env.resolve(node)
    if resolved:
        node = resolved.node
        env = _ScopedValues(env.resolver, resolved.env)
    if isinstance(node, (ast.List, ast.Tuple)):
        values = (env.integer(element) for element in node.elts)
        return tuple(value if value is not None and value > 0 else None for value in values)
    return None


# Track grp.current(), grp.next() and grp[i] back to their declaring site.
_SLOT_CALLS = ("current", "next")
_VALIDSHAPE = "set_validshape"
_MOVE = "move"


class _BodyConstraintReader:
    """Collect consumer windows and move relations in one lexical context."""

    def __init__(self, bindings, offsets, sites):
        self.tree = bindings.tree
        self.offsets = offsets
        self.bindings = bindings
        sites = tuple(sites)
        self.by_call = {site.call_span: site.site_id for site in sites}
        self.ranks = {site.site_id: len(site.shape) for site in sites}
        self.windows: Dict[str, list] = {}
        self.unresolved_windows = {}
        self.moves = []
        self.detail = []

    def read(self):
        """Return deterministic constraints from active calls only."""
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            if not self.bindings.active_at.get(id(node), True):
                continue
            env = _ScopedValues(self.bindings, self.bindings.environment(node))
            if is_named_call(node, _VALIDSHAPE) and len(node.args) >= 2:
                self._record_window(node, env)
            elif is_named_call(node, _MOVE) and len(node.args) >= 2:
                self._record_move(node, env)
        return {"windows": {k: tuple(v) for k, v in sorted(self.windows.items())},
                "unresolved_window_axes": {k: sorted(v) for (k, v) in sorted(self.unresolved_windows.items())},
                "moves": sorted(self.moves),
                "move_detail": sorted(self.detail, key=lambda d: d["pair"])}

    def _moved_site(self, expr, env, depth=0):
        """Follow lexical aliases, slots and indexing back to a tile declaration."""
        if depth > 16:
            return None
        bound = self.bindings.resolve(Binding(expr, env))
        if bound is None:
            return None
        expr, env = bound.node, bound.env
        if is_named_call(expr, _MAKE):
            return self.by_call.get(self.offsets.span(expr))
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            if expr.func.attr in _SLOT_CALLS:
                return self._moved_site(expr.func.value, env, depth + 1)
        if isinstance(expr, ast.Subscript):
            return self._moved_site(expr.value, env, depth + 1)
        return None

    def _record_window(self, node, env):
        """Merge each resolved consume window with the site's existing floors."""
        sid = self._moved_site(node.args[0], env.bindings)
        if not sid:
            return
        got = _shape_floors(node.args[1], env)
        if got is None:
            self.unresolved_windows.setdefault(sid, set()).update(range(self.ranks[sid]))
            return
        unknown = [i for i, value in enumerate(got) if value is None]
        if unknown:
            self.unresolved_windows.setdefault(sid, set()).update(unknown)
        cur = self.windows.setdefault(sid, [None] * len(got))
        while len(cur) < len(got):
            cur.append(None)
        for i, v in enumerate(got):
            if v is not None and (cur[i] is None or v > cur[i]):
                cur[i] = v

    def _record_move(self, node, env):
        """Keep offset-less pair identity together with direction and mode."""
        # An offset makes it a SUB-BLOCK copy, and then the shapes differ by
        # design. Both spellings: the third positional and `offset=`.
        if len(node.args) > 2 or keyword_value(node, "offset") is not None:
            return
        dst = self._moved_site(node.args[0], env.bindings)
        src_sid = self._moved_site(node.args[1], env.bindings)
        if dst and src_sid and dst != src_sid:
            pair = tuple(sorted((dst, src_sid)))
            if pair not in self.moves:
                self.moves.append(pair)
            # Keep move direction separately from the sorted pair. Source and
            # destination have different constraints for acc_to_vec_mode.
            mode = next((k.arg for k in node.keywords
                         if k.arg and k.arg.endswith("_mode")), None)
            if not any(d["pair"] == list(pair) for d in self.detail):
                self.detail.append({"pair": list(pair), "dst": dst,
                                    "src": src_sid, "mode": mode})


def unreferenced(src: str, sites: Iterable[TileSite]):
    """Find tile declarations with no AST load of their bound name; comments and strings do not count."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return ()
    reads: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            reads[node.id] = reads.get(node.id, 0) + 1
    out = []
    for site in sites:
        name = getattr(site, "group_var", None)
        if not name or not reads.get(name):
            out.append(site.site_id)
    return tuple(out)


def site_by_id(sites: Iterable[TileSite], site_id: str) -> TileSite:
    """The one site with this id, or KeyError -- a caller naming a site it tuned."""
    for site in sites:
        if site.site_id == site_id:
            return site
    raise KeyError(site_id)


def sibling_constants(src, op_file=None):
    """Read integer constants imported from sibling modules without executing the kernel or its imports."""
    if not op_file:
        return {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    base = os.path.dirname(os.path.abspath(op_file))
    out: Dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.level or not node.module:
            continue
        where = base
        for _ in range(node.level - 1):
            where = os.path.dirname(where)
        path = os.path.join(where, *node.module.split(".")) + ".py"
        if not os.path.isfile(path):
            logger.debug("relative import %r does not resolve to a file", node.module)
            continue
        try:
            consts = module_constants(Path(path).read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            logger.debug("cannot read %s: %s", path, exc)
            continue
        wanted = [a.name for a in node.names]
        if any(name == "*" for name in wanted):
            out.update(consts)
            continue
        for alias in node.names:
            value = consts.get(alias.name)
            if value is not None:
                out[alias.asname or alias.name] = value
    return out
