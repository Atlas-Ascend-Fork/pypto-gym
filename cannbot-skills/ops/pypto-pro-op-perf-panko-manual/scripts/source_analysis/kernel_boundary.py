# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Locate editable device functions and pure construction code while freezing unknown calls and host effects."""
import ast
import copy

from panko_core.pypto_ast import jit_decorators, keyword_value, parent_map
from memory.source_allocator import executable_body, recognize_allocator

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
_BUILTINS = frozenset({"abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list", "max", "min",
                       "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip"})
_EXCEPTIONS = frozenset({"ValueError", "TypeError", "RuntimeError", "AssertionError"})
_DSL = frozenset({"pl", "vf", "pypto", "pypto_pro"})


def decorated_jit(node):
    return any(jit_decorators(node))


def decorated_vector_function(node):
    """Recognize the DSL targets admitted by subscript launch resolution."""
    for dec in getattr(node, "decorator_list", ()):
        value = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(value, ast.Name) and value.id == "vector_function":
            return True
        if (isinstance(value, ast.Attribute) and value.attr == "vector_function" and root_name(value) in _DSL):
            return True
    return False


def _runtime_decorators(node):
    """Decorators outside the DSL can add runtime host work and stay frozen."""
    out = []
    for dec in node.decorator_list:
        value = dec.func if isinstance(dec, ast.Call) else dec
        if (root_name(value) in _DSL or isinstance(value, ast.Name) and value.id in {"jit", "vector_function"}):
            continue
        out.append(dec)
    return out


def root_name(node):
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def scope_nodes(roots, stop_at=(*_FUNCTION, ast.ClassDef)):
    """Visit each scope in source order, yielding nested definitions without descending into them."""
    pending = list(reversed(list(roots)))
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, stop_at):
            pending.extend(reversed(list(ast.iter_child_nodes(node))))


def loaded_names(node):
    return (child for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load))


def assignment_targets(node):
    return node.targets if isinstance(node, ast.Assign) else [node.target]


def _names(target):
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def argument_defaults(args):
    positional = args.posonlyargs + args.args
    defaults = {arg.arg: value for arg, value in zip(positional[len(positional) - len(args.defaults):], args.defaults)}
    defaults.update((arg.arg, value) for arg, value in zip(args.kwonlyargs, args.kw_defaults) if value is not None)
    return defaults


def definition_paths(tree):
    """Build lexical case names and persisted function-only shell names in one traversal."""
    lexical, shell = {tree: ""}, {tree: ""}
    lexical_counts, shell_counts = {}, {}

    def child_path(parent, name, counts):
        key = parent, name
        ordinal = counts[key] = counts.get(key, 0) + 1
        name = name if ordinal == 1 else f"{name}#{ordinal}"
        return f"{parent}/{name}" if parent else name

    def visit(node, case_owner, shell_owner):
        if isinstance(node, (*_FUNCTION, ast.ClassDef)):
            case_owner = lexical[node] = child_path(case_owner, node.name, lexical_counts)
            if isinstance(node, _FUNCTION):
                shell_owner = shell[node] = child_path(shell_owner, node.name, shell_counts)
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.expr):
                visit(child, case_owner, shell_owner)

    visit(tree, "", "")
    return lexical, shell


class Boundary:
    def __init__(self, tree, extra_consts=None, *, parents=None, paths=None):
        self.tree = tree
        self.node_parent = parent_map(tree) if parents is None else parents
        self.parent = {tree: None}
        self.scope = {}
        self.bindings = {tree: {}}
        self.functions = []
        self._index(tree, tree)
        self._index_consumers()
        self.scope_paths = definition_paths(tree)[1] if paths is None else paths
        self._index_case_parameters(extra_consts)
        self._index_allocators()
        self.entries = {f for f in self.functions if decorated_jit(f)}
        self.graph = {f: self.calls(f) for f in self.functions}
        self.runtime_metadata = {fn for fn in self.functions if self.cached_device_query(fn)}
        reachable = self.closure(self.entries)
        # Factory calls construct callables; they are not device call edges.
        host = self.closure(set(self.functions) - reachable, stop=self.entries)
        self.device = (reachable - host) | self.entries
        self.factories = set()
        for fn in self.device:
            scope = self.parent[fn]
            while isinstance(scope, _FUNCTION):
                self.factories.add(scope)
                scope = self.parent[scope]
        # Keep forwarding wrappers in the frozen host shell so tensor arguments remain checked.
        candidates = self.closure(self.factories) - self.device
        self._pure_cache = {}
        self.planners = {fn for fn in candidates if self.pure_function(fn)}
        self._retain_host_planners()
        # Retain factories using a frozen helper while eliding their device bodies.
        self.editable = self.device | self.planners

    @staticmethod
    def own_nodes(fn):
        return (node for node in scope_nodes(fn.body, _FUNCTION) if not isinstance(node, _FUNCTION))

    @staticmethod
    def _device_query_signature(fn):
        return not (fn.decorator_list or fn.args.posonlyargs or fn.args.args or fn.args.kwonlyargs
                    or fn.args.vararg or fn.args.kwarg)

    @staticmethod
    def _cache_zero_subscript(result):
        if not isinstance(result, ast.Subscript) or not isinstance(result.value, ast.Name):
            return False
        return isinstance(result.slice, ast.Constant) and result.slice.value == 0

    def tile_type_call(self, node):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in _DSL and node.func.attr == "TileType"):
            return False
        name = node.func.value.id
        owner, values = self.binding(name, self.scope.get(node, self.tree))
        if not values:
            return True  # standalone source fragments use the conventional DSL alias
        if owner is not self.tree or values != [None]:
            return False
        return any(
            isinstance(stmt, ast.Import) and any(
                alias.name in {"pypto_pro.language",
                               "pypto.language", "pypto_pro", "pypto"}
                and (alias.asname or alias.name.split(".")[0]) == name for alias in stmt.names)
            or isinstance(stmt, ast.ImportFrom) and stmt.module in {"pypto_pro", "pypto"}
            and any(alias.name == "language" and (alias.asname or alias.name) == name
                    for alias in stmt.names)
            for stmt in self.tree.body)

    def metadata_arguments(self, fn, call, scope, seen=frozenset()):
        """Check positional, keyword, and omitted/defaulted construction inputs."""
        positional = fn.args.posonlyargs + fn.args.args
        if not all(self.metadata(a, scope, seen) or i < len(positional)
                   and (fn, positional[i].arg) in self.case_parameters
                   and self.pure_expr(a, scope, frozenset()) for i, a in enumerate(call.args)):
            return False
        if not all(k.arg is not None and (self.metadata(k.value, scope, seen)
                   or (fn, k.arg) in self.case_parameters and isinstance(k.value, ast.Name)
                   and self.binding(k.value.id, scope)[1] == [None]) for k in call.keywords):
            return False
        supplied = {a.arg for a in positional[:len(call.args)]}
        supplied.update(k.arg for k in call.keywords)
        return all(name in supplied or self.metadata(value, self.parent[fn])
                   for name, value in argument_defaults(fn.args).items())

    def metadata_access(self, node):
        parent = self.node_parent.get(node)
        return (isinstance(parent, ast.Attribute) and parent.value is node
                and parent.attr in {"shape", "dtype", "device", "ndim", "dim", "numel", "size"})

    def binding(self, name, scope):
        while scope is not None:
            if name in self.bindings[scope]:
                return scope, self.bindings[scope][name]
            scope = self.parent.get(scope)
        return None, []

    def resolve(self, expr, scope, seen=frozenset()):
        if expr is None or (id(expr), id(scope)) in seen:
            return set()
        seen = seen | {(id(expr), id(scope))}
        if isinstance(expr, _FUNCTION):
            return {expr}
        if isinstance(expr, ast.Name):
            owner, values = self.binding(expr.id, scope)
            return set().union(*(self.resolve(v, owner, seen) for v in values))
        if isinstance(expr, ast.IfExp):
            return self.resolve(expr.body, scope, seen) | self.resolve(expr.orelse, scope, seen)
        if isinstance(expr, ast.Subscript):
            # A device launch is a call edge, not a Python dispatch-table lookup.
            launched = self.resolve(expr.value, scope, seen)
            if launched and all(decorated_jit(fn) or decorated_vector_function(fn) for fn in launched):
                return launched
            values = self.callable_table(expr.value, scope)
            resolved = [self.resolve(value, owner, seen) for value, owner in values]
            return set().union(*resolved) if resolved and all(resolved) else set()
        if isinstance(expr, ast.Call):
            return set().union(*(self.returned(fn, seen) for fn in self.resolve(expr.func, scope, seen)))
        return set()

    def callable_table(self, expr, scope, seen=frozenset()):
        """Bounded literal dispatch only; mutated or escaping tables stay opaque."""
        if isinstance(expr, ast.Dict) and all(isinstance(k, ast.Constant) for k in expr.keys):
            return [(value, scope) for value in expr.values]
        if isinstance(expr, (ast.List, ast.Tuple)):
            return [(value, scope) for value in expr.elts]
        if not isinstance(expr, ast.Name):
            return []
        owner, values = self.binding(expr.id, scope)
        key = (owner, expr.id)
        if key in seen or len(values) != 1:
            return []
        for node in self._loads.get(key, ()):
            parent = self.node_parent.get(node)
            if not (isinstance(parent, ast.Subscript) and parent.value is node and isinstance(parent.ctx, ast.Load)):
                return []
        return self.callable_table(values[0], owner, seen | {key})

    def returned(self, fn, seen=frozenset()):
        values = [self.resolve(node.value, fn, seen) for node in self.own_nodes(fn) if isinstance(node, ast.Return)]
        return set().union(*values)

    def calls(self, fn):
        return set().union(*(self.resolve(n.func, fn) for n in self.own_nodes(fn) if isinstance(n, ast.Call)))

    def closure(self, seeds, stop=frozenset()):
        seen, stack = set(), list(seeds)
        while stack:
            fn = stack.pop()
            if fn in seen:
                continue
            seen.add(fn)
            if fn not in stop:
                stack.extend(self.graph.get(fn, ()))
        return seen

    def allocator_call(self, call, fn):
        """Only isolated, recognized bump allocators are construction arithmetic."""
        if isinstance(call.func, ast.Name):
            return (call.func.id in self.allocators and not call.args
                    and not call.keywords and not self.binding(call.func.id, fn)[1])
        if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name):
            return False
        owner, values = self.binding(call.func.value.id, fn)
        return (owner is fn and bool(values) and all(
            isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and self.allocator_call(value, fn)
            and self.allocators[value.func.id].method == call.func.attr for value in values))

    def cached_device_query(self, fn):
        """Recognize a cached scalar scheduling query while keeping its state frozen."""
        if not self._device_query_signature(fn):
            return False
        body = executable_body(fn.body)
        if len(body) != 2 or not isinstance(body[-1], ast.Return):
            return False
        result = body[-1].value
        if not self._cache_zero_subscript(result):
            return False
        cache = result.value.id
        if not self._device_cache_is_private(cache, fn) or not self._canonical_device_query_imports(fn):
            return False
        fallback_name = self._device_query_fallback(fn)
        if fallback_name is None:
            return False
        expected = ast.parse('''if CACHE[0] is None:
    try:
        CACHE[0] = int(torch.npu.get_device_properties(0).vector_core_num)
    except Exception:
        CACHE[0] = DEFAULT
    if CACHE[0] <= 0:
        CACHE[0] = DEFAULT
return CACHE[0]
''')

        class Rename(ast.NodeTransformer):
            @staticmethod
            def visit_name(node):
                return ast.Name(id={cache: "CACHE", fallback_name: "DEFAULT"}.get(node.id, node.id), ctx=node.ctx)

            visit_Name = visit_name

        actual = Rename().visit(ast.Module(body=copy.deepcopy(body), type_ignores=[]))
        return ast.dump(actual) == ast.dump(expected)

    def math_call(self, call, scope):
        if not (isinstance(call.func, ast.Attribute) and call.func.attr == "sqrt"
                and isinstance(call.func.value, ast.Name)):
            return False
        name = call.func.value.id
        owner, values = self.binding(name, scope)
        return (owner is self.tree and values == [None] and any(
            isinstance(node, ast.Import) and any(alias.name == "math"
                and (alias.asname or alias.name) == name for alias in node.names)
            for node in self.tree.body))

    def pure_expr(self, node, fn, visiting):
        if node is None:
            return True
        if isinstance(node, (ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)):
            return False
        if isinstance(node, ast.Call):
            if self.tile_type_call(node) or self.allocator_call(node, fn) or self.math_call(node, fn):
                okay = True
            elif isinstance(node.func, ast.Name) and node.func.id in _BUILTINS | _EXCEPTIONS:
                okay = True
            elif (isinstance(node.func, ast.Attribute)
                  and node.func.attr in {"append", "extend", "insert", "sort", "reverse"}):
                okay = self._local_container(node.func.value, fn)
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                # Reading a scheduling environment variable or a local table is
                # data-free. Writing environment/runtime state is never allowed.
                okay = self._local_container(node.func.value, fn) or (
                    isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "environ"
                    and root_name(node.func.value) in {"os", "_os"})
            else:
                callees = self.resolve(node.func, fn)
                okay = bool(callees) and all(f in self.runtime_metadata
                                             or self.pure_function(f, visiting) for f in callees)
            return okay and all(self.pure_expr(a, fn, visiting) for a in node.args) and all(
                self.pure_expr(k.value, fn, visiting) for k in node.keywords)
        return all(self.pure_expr(child, fn, visiting) for child in ast.iter_child_nodes(node))

    def opaque_runtime_calls(self, fn):
        calls = (node for node in self.own_nodes(fn) if isinstance(node, ast.Call))
        return [call for call in calls if self.closure(self.resolve(call.func, fn)) & self.runtime_metadata]

    def unconditional_runtime_call(self, call, fn):
        """Only a direct, top-level statement has no hidden execution guard."""
        statement = self.node_parent.get(call)
        return (isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return))
                and statement.value is call and statement in fn.body)

    def pure_function(self, fn, visiting=frozenset()):
        if fn in self._pure_cache:
            return self._pure_cache[fn]
        if fn in visiting or fn in self.device:
            return False
        visiting = visiting | {fn}
        runtime_is_conditional = any(not self.unconditional_runtime_call(call, fn)
                                     for call in self.opaque_runtime_calls(fn)) if fn in self.factories else False
        if runtime_is_conditional:
            self._pure_cache[fn] = False
            return False
        result = all(self._planner_node_is_pure(node, fn, visiting) for node in self.own_nodes(fn))
        self._pure_cache[fn] = result
        return result

    def metadata(self, node, scope, seen=frozenset()):
        """Can a host call argument be established as data-free metadata?"""
        key = (id(node), id(scope))
        if node is None or key in seen:
            return False
        seen = seen | {key}
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            return self._metadata_name(node, scope, seen)
        if isinstance(node, ast.Attribute):
            return ((node.attr in {"shape", "dtype", "device", "ndim"}
                     and isinstance(node.value, ast.Name))
                    or root_name(node) in _DSL | {"torch"})
        if isinstance(node, ast.Subscript):
            return (self.metadata(node.value, scope, seen) and self.metadata(node.slice, scope, seen))
        if isinstance(node, ast.Compare) and all(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
            return all(isinstance(value, ast.Name) or self.metadata(value, scope, seen)
                       for value in [node.left] + node.comparators)
        if isinstance(node, ast.Call):
            return self._metadata_call(node, scope, seen)
        return all(self.metadata(n, scope, seen) for n in ast.iter_child_nodes(node)
                   if not isinstance(n, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context)))

    def definition_runtime(self, fn):
        """Non-DSL decorators and effectful defaults execute outside the kernel."""
        expressions = list(_runtime_decorators(fn))
        defaults = list(fn.args.defaults) +\
            [d for d in fn.args.kw_defaults if d is not None]

        scope = self.parent[fn]
        expressions.extend(value for value in defaults
                           if not (self.metadata(value, scope) and self.pure_expr(value, scope, frozenset())))
        expressions.extend(self.opaque_runtime_calls(fn))
        return expressions

    def shell(self, owned_bindings=()):
        """Freeze host work while projecting editable device and construction code."""
        return _ShellProjection(self).project(owned_bindings)

    def _device_cache_is_private(self, cache, fn):
        owner, values = self.binding(cache, fn)
        if owner is not self.tree or len(values) != 1:
            return False
        value = values[0]
        if not isinstance(value, ast.List) or len(value.elts) != 1:
            return False
        if not isinstance(value.elts[0], ast.Constant) or value.elts[0].value is not None:
            return False
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Name) or node.id != cache:
                continue
            scope = self.scope.get(node, self.tree)
            if scope is fn or self.binding(cache, scope)[0] is not self.tree:
                continue
            if not isinstance(node.ctx, ast.Store) or self.node_parent.get(node) is not self.node_parent.get(value):
                return False
        return True

    def _device_query_fallback(self, fn):
        writes = [node for node in ast.walk(fn) if isinstance(node, ast.Assign)]
        fallback = [node.value.id for node in writes if isinstance(node.value, ast.Name)]
        if len(fallback) != 2 or fallback[0] != fallback[1]:
            return None
        owner, values = self.binding(fallback[0], fn)
        if owner is not self.tree or not values:
            return None
        if not all(isinstance(value, ast.Constant) and type(value.value) is int for value in values):
            return None
        return fallback[0]

    def _planner_node_is_pure(self, node, fn, visiting):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            owner, values = self.binding(node.id, fn)
            if owner is self.tree and not self.metadata_access(node):
                if any(value is not None and not isinstance(value, _FUNCTION)
                       and not self.metadata(value, owner) for value in values):
                    return False
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith,
                             ast.Import, ast.ImportFrom, ast.Delete)):
            return False
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = assignment_targets(node)
            if any(not isinstance(target, (ast.Name, ast.Tuple, ast.List)) for target in targets):
                return False
        return not isinstance(node, ast.Call) or self.pure_expr(node, fn, visiting)

    def _index_consumers(self):
        # Index immutable table loads once for callable lookup.
        self._loads = {}
        self._call_nodes = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                self._call_nodes.append(node)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                owner, _ = self.binding(node.id, self.scope.get(node, self.tree))
                self._loads.setdefault((owner, node.id), []).append(node)

    def _index_case_parameters(self, extra_consts):
        self.case_parameters = {}
        scoped = (extra_consts or {}).get("__panko_scoped_constants__", {})
        for fn in self.functions:
            values = scoped.get(self.scope_paths[fn]) if isinstance(scoped, dict) else None
            if not isinstance(values, dict):
                continue
            for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
                value = values.get(arg.arg)
                if arg.arg in values and (value is None or type(value) in (int, float, bool, str)):
                    self.case_parameters[fn, arg.arg] = value

    def _index_allocators(self):
        self.allocators = {}
        for node in self.tree.body:
            specification = recognize_allocator(node)
            if specification is not None:
                references = [n for n in ast.walk(self.tree) if isinstance(n, ast.Name) and n.id == node.name]
                if all(isinstance(self.node_parent.get(n), ast.Call)
                       and self.node_parent[n].func is n for n in references):
                    self.allocators[node.name] = specification

    def _retain_host_planners(self):
        """Keep helpers whose host consumers require unknown or tensor arguments."""
        while True:
            unsafe = set()
            for fn in self.planners:
                if self._planner_uses_host_value(fn) or self._planner_has_unknown_call(fn):
                    unsafe.add(fn)
            if not unsafe:
                return
            self.planners -= unsafe

    def _planner_uses_host_value(self, fn):
        for node in self.own_nodes(fn):
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
                continue
            owner, _ = self.binding(node.id, fn)
            if not isinstance(owner, _FUNCTION) or owner is fn or owner in self.planners:
                continue
            if not self.metadata_access(node) and not self.resolve(node, fn) and not self.metadata(node, fn):
                return True
        return False

    def _planner_has_unknown_call(self, fn):
        for call in self._call_nodes:
            scope = self.scope.get(call, self.tree)
            if scope in self.device or scope in self.planners:
                continue
            if fn in self.resolve(call.func, scope) and not self.metadata_arguments(fn, call, scope):
                return True
        return False

    def _bind(self, scope, name, value):
        self.bindings[scope].setdefault(name, []).append(value)

    def _index(self, node, scope):
        self.scope[node] = scope
        if isinstance(node, _FUNCTION):
            self._bind(scope, node.name, node)
            self.parent[node] = scope
            self.bindings[node] = {}
            self.functions.append(node)
            for arg in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs):
                self._bind(node, arg.arg, None)
            for arg in (node.args.vararg, node.args.kwarg):
                if arg:
                    self._bind(node, arg.arg, None)
            for statement in node.body:
                self._index(statement, node)
            for dec in node.decorator_list:
                self._index(dec, scope)
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in assignment_targets(node):
                for name in _names(target):
                    self._bind(scope, name, node.value)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in node.names:
                self._bind(scope, name.asname or name.name.split(".")[0], None)
        for child in ast.iter_child_nodes(node):
            self._index(child, scope)

    def _local_container(self, node, fn):
        owner, values = self.binding(root_name(node), fn)
        return owner is fn and any(isinstance(v, (ast.List, ast.Dict, ast.Set)) for v in values)

    def _canonical_device_query_imports(self, fn):
        torch_owner, torch_values = self.binding("torch", fn)
        if torch_owner is not self.tree or torch_values != [None]:
            return False
        canonical_import = any(isinstance(node, ast.Import) and any(
            alias.name == "torch" and alias.asname in (None, "torch") for alias in node.names)
            for node in self.tree.body)
        return canonical_import and not self.binding("int", fn)[1] and not self.binding("Exception", fn)[1]

    def _metadata_name(self, node, scope, seen):
        owner, values = self.binding(node.id, scope)
        if (owner, node.id) in self.case_parameters and values == [None]:
            return True
        if isinstance(owner, _FUNCTION) and values == [None]:
            arguments = []
            parameters = [a.arg for a in owner.args.posonlyargs + owner.args.args]
            for call in self._call_nodes:
                caller = self.scope.get(call, self.tree)
                if owner not in self.resolve(call.func, caller):
                    continue
                value = keyword_value(call, node.id)
                if value is None and node.id in parameters and parameters.index(node.id) < len(call.args):
                    value = call.args[parameters.index(node.id)]
                arguments.append(value is not None and self.metadata(value, caller, seen))
            return bool(arguments) and all(arguments)
        return bool(values) and all(self.metadata(v, owner, seen) for v in values)

    def _metadata_call(self, node, scope, seen):
        if self.tile_type_call(node):
            return self._metadata_call_inputs(node, scope, seen)
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"dim", "numel", "size"}:
            return isinstance(node.func.value, ast.Name) and self._metadata_call_inputs(node, scope, seen)
        if isinstance(node.func, ast.Name) and node.func.id == "len":
            return (len(node.args) == 1 and not node.keywords
                    and (isinstance(node.args[0], ast.Name)
                         or self.metadata(node.args[0], scope, seen)))
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"split", "rsplit", "lower", "upper"}:
            return self.metadata(node.func.value, scope, seen)
        if (_environment_lookup(node.func)):
            return True
        allowed = isinstance(node.func, ast.Name) and node.func.id in _BUILTINS
        callees = self.resolve(node.func, scope)
        allowed |= bool(callees) and callees <= (getattr(self, "planners", set()) | self.runtime_metadata)
        return allowed and self._metadata_call_inputs(node, scope, seen)

    def _metadata_call_inputs(self, call, scope, seen):
        return (all(self.metadata(arg, scope, seen) for arg in call.args)
                and all(kw.arg is not None and self.metadata(kw.value, scope, seen) for kw in call.keywords))


class _ShellProjection:
    """Track editable consumers and preserve their host dependencies."""

    def __init__(self, boundary):
        self.boundary = boundary
        self.tunable, self.retained = set(), set()
        self.assignments, self.candidates, self.grids = {}, set(), set()
        self.construction_inputs = set()
        self.frozen_planners = set()
        self.movable = set()

    @staticmethod
    def _cached_factory_shape(body):
        """Recognize assignment, conditional cache fill, and return statements."""
        if len(body) != 3:
            return False
        assignment, condition, result = body
        if not isinstance(assignment, ast.Assign) or len(assignment.targets) != 1:
            return False
        if not isinstance(assignment.targets[0], ast.Name) or not isinstance(assignment.value, ast.Tuple):
            return False
        if not isinstance(condition, ast.If) or condition.orelse or not isinstance(result, ast.Return):
            return False
        if len(condition.body) != 1 or not isinstance(condition.body[0], ast.Assign):
            return False
        return len(condition.body[0].targets) == 1

    @staticmethod
    def _cached_factory_lookup(key, condition, result, store):
        """Require the lookup, fill, and result to refer to the identical cache key."""
        test = condition.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.NotIn):
            return False
        if not isinstance(test.left, ast.Name) or test.left.id != key.id:
            return False
        target = store.targets[0]
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
            return False
        if not isinstance(target.slice, ast.Name) or target.slice.id != key.id:
            return False
        if not isinstance(result.value, ast.Subscript) or not isinstance(store.value, ast.Call):
            return False
        return (ast.dump(test.comparators[0]) == ast.dump(target.value)
                and ast.dump(result.value.value) == ast.dump(target.value)
                and ast.dump(result.value.slice) == ast.dump(target.slice))

    def project(self, owned_bindings):
        self.collect_construction_inputs()
        self.collect_cached_inputs()
        self.collect_assignments()
        self.reads(self.boundary.tree, self.boundary.tree)
        self.preserve_owned_bindings(owned_bindings)
        self.propagate_dependencies()
        self.boundary.discover_owned_bindings = sorted(
            (self.boundary.scope_paths.get(scope, ""), name)
            for statement in self.movable
            for scope, name in self.assignments[statement][0])
        return self.project_tree()

    def symbol(self, name, scope):
        owner, values = self.boundary.binding(name, scope)
        return (owner, name) if values else (scope, name)

    def names(self, node, scope):
        return {self.symbol(n.id, self.boundary.scope.get(n, scope)) for n in loaded_names(node)}

    def dependencies(self, node, scope):
        if node is None or node in self.construction_inputs:
            return set()
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            return {self.symbol(node.id, self.boundary.scope.get(node, scope))}
        return set().union(*(self.dependencies(child, scope) for child in ast.iter_child_nodes(node)))

    def reads(self, node, scope):
        if isinstance(node, _FUNCTION):
            if node in self.boundary.editable:
                self.tunable.update(self.names(node, node))
                for value in self.boundary.definition_runtime(node):
                    self.reads(value, self.boundary.parent[node])
                return
            scope = node
        if node in self.grids or node in self.construction_inputs:
            self.tunable.update(self.names(node, scope))
            return
        if node in self.candidates:
            return
        if node in self.assignments:
            self.retained.update(self.assignments[node][0])
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            self.retained.add(self.symbol(node.id, self.boundary.scope.get(node, scope)))
        for child in ast.iter_child_nodes(node):
            self.reads(child, scope)

    def collect_construction_inputs(self):
        for node in ast.walk(self.boundary.tree):
            if not isinstance(node, ast.Call):
                continue
            scope = self.boundary.scope.get(node, self.boundary.tree)
            callees = self.boundary.resolve(node.func, scope)
            if callees and callees <= self.boundary.planners & self.boundary.factories:
                arguments = [*node.args, *(kw.value for kw in node.keywords)]
                self.construction_inputs.update(arg for arg in arguments if self.boundary.metadata(arg, scope))

    def collect_cached_inputs(self):
        """Retain the cache protocol while allowing shared factory inputs to vary."""
        for fn in self.boundary.functions:
            body = executable_body(fn.body)
            if not self._cached_factory_shape(body):
                continue
            assignment, condition, result = body
            store = condition.body[0]
            if not self._cached_factory_lookup(assignment.targets[0], condition, result, store):
                continue
            callees = self.boundary.resolve(store.value.func, fn)
            if not callees or not callees <= self.boundary.planners & self.boundary.factories:
                continue
            arguments = list(store.value.args) + [k.value for k in store.value.keywords]
            dependency_keys = set().union(*(self.names(a, fn) for a in arguments))
            self.construction_inputs.update(
                value for value in assignment.value.elts
                if self.boundary.metadata(value, fn) and self.names(value, fn) <= dependency_keys)

    def collect_assignments(self):
        for node in ast.walk(self.boundary.tree):
            scope = self.boundary.scope.get(node, self.boundary.tree)
            if scope in self.boundary.editable:
                continue
            self._collect_grid(node, scope)
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = assignment_targets(node)
            if not all(isinstance(t, (ast.Name, ast.Tuple, ast.List)) for t in targets):
                continue
            keys = {(scope, name) for target in targets for name in _names(target)}
            if not keys:
                continue
            self.assignments[node] = (keys, self.dependencies(node.value, scope))
            if (self.boundary.metadata(node.value, scope) and self.boundary.pure_expr(node.value, scope, frozenset())):
                self.candidates.add(node)

    def preserve_owned_bindings(self, owned_bindings):
        # Preserve ownership established by the authenticated baseline when a
        # deterministic rewrite inlines the final device use of a constant.
        # This does not thaw unrelated unused globals or any new host consumer.
        original_owned = {tuple(key) for key in owned_bindings}
        for statement, (keys, _) in self.assignments.items():
            if statement in self.candidates:
                self.tunable.update(key for key in keys
                                    if (self.boundary.scope_paths.get(key[0]), key[1]) in original_owned)

        local_keys = set()
        for stmt in self.candidates:
            if stmt in self.assignments and self.boundary.scope.get(stmt) is not self.boundary.tree:
                local_keys.update(self.assignments[stmt][0])
        self.tunable.update(local_keys - self.retained)

        self.frozen_planners = set()
        for fn in self.boundary.planners:
            calls = self.boundary.opaque_runtime_calls(fn)
            if any(not self.boundary.unconditional_runtime_call(call, fn) for call in calls):
                self.frozen_planners.add(fn)
        for fn in self.frozen_planners:
            self.retained.update(self.names(fn, fn))

    def propagate_dependencies(self):
        for _ in range(len(self.assignments) + len(self.boundary.planners) + 1):
            old = (set(self.tunable), set(self.retained))
            for statement, (keys, deps) in self.assignments.items():
                if keys & self.retained or statement not in self.candidates:
                    self.retained.update(deps)
                if keys & self.tunable:
                    self.tunable.update(deps)
            # A pure helper can also feed retained tensor work. Keep its entire
            # definition and its scalar dependencies, without freezing factories
            # that merely call it for their own construction arithmetic.
            for fn in self.boundary.planners:
                if (fn not in self.boundary.factories
                        and self.symbol(fn.name, self.boundary.parent[fn]) in self.retained):
                    self.frozen_planners.add(fn)
                    self.retained.update(self.names(fn, fn))
            if old == (self.tunable, self.retained):
                break
        self.movable = {statement for statement, (keys, _) in self.assignments.items()
                        if statement in self.candidates and keys <= self.tunable and not keys & self.retained}

    def project_tree(self):
        boundary = self.boundary
        movable = self.movable
        frozen_planners = self.frozen_planners
        grids = self.grids

        class Project(ast.NodeTransformer):

            def visit_function_def(self, node):
                runtime = boundary.definition_runtime(node)
                if node in boundary.device:
                    marker = "<device code>"
                    keep = node in boundary.entries
                elif node in boundary.planners and node not in frozen_planners:
                    marker = "<kernel construction>"
                    keep = node in boundary.factories
                else:
                    return self.generic_visit(node)
                if not keep and not runtime:
                    return None
                value = ast.Tuple(elts=[ast.Constant(marker), *copy.deepcopy(runtime)], ctx=ast.Load())
                return ast.copy_location(ast.Expr(value), node)

            visit_FunctionDef = visit_function_def

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_assign(self, node):
                return None if node in movable else self.generic_visit(node)

            visit_Assign = visit_assign

            visit_AnnAssign = visit_Assign

            def visit_subscript(self, node):
                if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
                    if node.slice.elts[1] in grids:
                        node.slice.elts[1] = ast.Constant("<launch grid>")
                return self.generic_visit(node)

            visit_Subscript = visit_subscript

        return Project().visit(boundary.tree)

    def _collect_grid(self, node, scope):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            parts = node.func.slice
            if (isinstance(parts, ast.Tuple) and len(parts.elts) == 2
                    and self.boundary.resolve(node.func.value, scope) & self.boundary.entries):
                grid = parts.elts[1]
                if (self.boundary.metadata(grid, scope) and self.boundary.pure_expr(grid, scope, frozenset())):
                    self.grids.add(grid)


def device_functions(tree):
    """Function nodes, never unqualified names: sibling factories remain distinct."""
    return Boundary(tree).device


def shell_dump(tree, extra_consts=None, owned_bindings=()):
    return ast.dump(Boundary(tree, extra_consts).shell(owned_bindings))


def discover_owned_bindings(src, extra_consts=None):
    boundary = Boundary(ast.parse(src), extra_consts)
    boundary.shell()
    return boundary.discover_owned_bindings


def measurement_dump(tree):
    """Measurement flags and their lexical dependencies remain fixed."""
    boundary = Boundary(tree)
    flags = []
    for fn in sorted(boundary.entries, key=lambda n: n.lineno):
        for value in _measurement_options(fn):
            deps = _measurement_dependencies(boundary, value, boundary.parent[fn])
            flags.append((fn.name, ast.dump(value), deps) if deps else (fn.name, ast.dump(value)))
    return repr(sorted(flags, key=repr))


def _measurement_options(fn):
    """Yield debug configuration expressions attached to an entry point."""
    for decorator in fn.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "debug_options":
                yield keyword.value


def _measurement_dependencies(boundary, expr, scope, seen=frozenset()):
    """Canonicalize recursively referenced values without revisiting cycles."""
    out = []
    names = {n.id for n in loaded_names(expr)}
    for name in sorted(names):
        owner, values = boundary.binding(name, scope)
        key = (id(owner), name)
        if key in seen:
            continue
        for value in values:
            if value is not None:
                deps = _measurement_dependencies(boundary, value, owner, seen | {key})
                out.append((name, ast.dump(value), deps))
    return out


def allocation_owners(boundary):
    owners = {fn: set() for fn in boundary.functions}
    for entry in boundary.entries:
        # Allocation receipts number every component from zero, including the first.
        parts = (part.partition("#") for part in boundary.scope_paths[entry].split("/"))
        identity = "/".join(f"{name}#{int(ordinal or 1) - 1}" for name, _, ordinal in parts)
        for fn in boundary.closure({entry}, stop=boundary.entries - {entry}):
            owners[fn].add(identity)
    return {fn.lineno: tuple(sorted(entries)) for fn, entries in owners.items()}


def _environment_lookup(function):
    if not isinstance(function, ast.Attribute) or function.attr != "get":
        return False
    value = function.value
    return isinstance(value, ast.Attribute) and value.attr == "environ" and root_name(value) in {"os", "_os"}
