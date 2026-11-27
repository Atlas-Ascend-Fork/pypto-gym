# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Resolve lexical bindings against their assignment-time environments without executing source."""
import ast
from functools import cached_property
import operator
from dataclasses import dataclass

from panko_core.pypto_ast import parent_map
from memory.source_allocator import AllocatorCursor, recognize_allocator
from source_analysis.case_context import (DEFAULT_ARGUMENTS, active_scope_closure, derived_fact_bindings,
                                          split_constants)
from source_analysis.kernel_boundary import (Boundary, argument_defaults, assignment_targets, definition_paths,
                                             loaded_names, root_name, scope_nodes)

UNKNOWN = object()


@dataclass
class Binding:
    node: ast.AST
    env: dict

    @classmethod
    def literal(cls, value):
        return cls(ast.Constant(value), {})


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv,
           ast.Mod: operator.mod, ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor,
           ast.LShift: operator.lshift, ast.RShift: operator.rshift}
_COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
            ast.GtE: operator.ge, ast.Is: operator.is_, ast.IsNot: operator.is_not, ast.In: lambda a, b: a in b,
            ast.NotIn: lambda a, b: a not in b}


_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: operator.invert, ast.Not: operator.not_}
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
# Statement kinds that share the _suite_<kind> handler of another kind.
_SUITE_KINDS = {ast.AsyncFunctionDef: ast.FunctionDef, ast.AnnAssign: ast.Assign, ast.AsyncWith: ast.With,
                ast.AsyncFor: ast.For, ast.While: ast.For, ast.ImportFrom: ast.Import}
# += can mutate a caller-owned list.
_EFFECTFUL = (ast.Global, ast.Nonlocal, ast.Delete, ast.Yield, ast.YieldFrom, ast.Await, ast.With, ast.AsyncWith,
              ast.Try, ast.Raise, ast.Import, ast.ImportFrom, *_DEFINITIONS, ast.AugAssign)


def augmented_value(stmt):
    return ast.BinOp(left=ast.Name(id=stmt.target.id, ctx=ast.Load()), op=stmt.op, right=stmt.value)


def _bound_name(node):
    """Name introduced by a definition, a store or delete, or an import alias."""
    if isinstance(node, _DEFINITIONS):
        return node.name
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        return node.id
    if isinstance(node, ast.alias):
        return node.asname or node.name.split(".")[0]
    return None


def _constructs_or_aliases(parent, node):
    """The load is the callee of a constructor call, or the value of a plain-name assignment."""
    if isinstance(parent, ast.Call):
        return parent.func is node
    return (isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is node
            and all(isinstance(target, ast.Name) for target in assignment_targets(parent)))


def _assigned_names(stmt):
    if not isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)) or stmt.value is None:
        return
    for target in assignment_targets(stmt):
        for node in ast.walk(target):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                yield node.id


class SourceBindings:
    def __init__(self, tree, extra=None, override_extra=True):
        self.tree = tree
        self.parents = parent_map(tree)
        self.context = extra
        self.extra, self.scoped_extra = split_constants(extra)
        self.derived_extra, self.derived_guard_error = derived_fact_bindings(tree, extra)
        self.default_arguments = (extra or {}).get(DEFAULT_ARGUMENTS, {})
        self.override_extra = override_extra
        self.specialized = {}
        self.paths, self.shell_paths = definition_paths(tree)
        self.scopes = {id(node): path for node, path in self.paths.items() if node is not tree}
        self.scope_lines = {path: getattr(node, "lineno", 0) for node, path in self.paths.items()}
        self._device_scopes = ({self.scopes[id(fn)] for fn in self.boundary.device}
                               if (extra or {}).get("__panko_active_scopes__") else set())
        self._allocator_specs = self._allocator_specifications(tree, self.parents)
        self._late_rebindings = self._rebound_values(self.paths)
        # Revisit after finding actual factory invocations. A default alone is
        # not evidence: make(width=128) can still be called with a dynamic width.
        for _ in range(4):
            self.stale_derived_used = set()
            self._value_cache = {}
            self.env_at = {}
            self.scope_at = {}
            self.active_at = {}
            self.calls = []
            seed = {k: Binding.literal(v) for k, v in self.extra.items()}
            self.module_env = self._suite(tree.body, seed, "")
            specialized = self._specializations()
            if self._fingerprint(specialized) == self._fingerprint(self.specialized):
                break
            self.specialized = specialized
        if self.stale_derived_used:
            raise ValueError(self.derived_guard_error + "; unresolved bindings: " +
                             ", ".join(sorted(self.stale_derived_used)))

    @staticmethod
    def resolve(binding):
        while binding is not None and isinstance(binding.node, ast.Name):
            binding = binding.env.get(binding.node.id)
        return binding

    @staticmethod
    def arguments(fn, call, caller_env):
        node = fn.node
        env = dict(fn.env)
        params = node.args.posonlyargs + node.args.args
        env.update((p.arg, None) for p in params + node.args.kwonlyargs)
        if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
            return env
        env.update((name, Binding(default, fn.env)) for name, default in argument_defaults(node.args).items())
        env.update((p.arg, Binding(arg, caller_env)) for p, arg in zip(params, call.args))
        env.update((kw.arg, Binding(kw.value, caller_env)) for kw in call.keywords if kw.arg in env)
        return env

    @staticmethod
    def _rebound_values(owners):
        """Track late rebindings separately for each previously indexed lexical scope."""
        writes = {}
        for owner in owners:
            for stmt in scope_nodes(ast.iter_child_nodes(owner)):
                for name in _assigned_names(stmt):
                    writes.setdefault((owner, name), []).append(stmt.value)
        unsafe = {}
        for (_, name), values in writes.items():
            if len(values) > 1:
                last_write = max(value.lineno for value in values)
                unsafe.setdefault(name, {}).update((id(n), last_write) for value in values for n in ast.walk(value))
        return unsafe

    @staticmethod
    def _same_value(left, right):
        """Compare types and container contents, including dict order, before joining facts across branches."""
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            left, right = left.items(), right.items()
        elif not isinstance(left, (list, tuple)):
            return left == right
        return len(left) == len(right) and all(SourceBindings._same_value(a, b) for a, b in zip(left, right))

    @staticmethod
    def _simple_target(target):
        return (isinstance(target, ast.Name) or
                isinstance(target, (ast.Tuple, ast.List)) and
                all(SourceBindings._simple_target(item) for item in target.elts))

    @staticmethod
    def _target_accepts(target, value):
        """Check unpacking arity for a target already accepted by _simple_target."""
        return isinstance(target, ast.Name) or (
            isinstance(value, (list, tuple, range)) and len(target.elts) == len(value)
            and all(SourceBindings._target_accepts(child, item) for child, item in zip(target.elts, value)))

    @staticmethod
    def _valid_call(function, call):
        parameters = list(function.args.posonlyargs) + list(function.args.args)
        if function.args.vararg or function.args.kwarg or len(call.args) > len(parameters):
            return False
        if any(isinstance(arg, ast.Starred) for arg in call.args):
            return False
        assigned = {parameter.arg for parameter in parameters[:len(call.args)]}
        keyword_names = {parameter.arg for parameter in function.args.args + function.args.kwonlyargs}
        for keyword in call.keywords:
            if keyword.arg not in keyword_names or keyword.arg in assigned:
                return False
            assigned.add(keyword.arg)
        required = {p.arg for p in parameters + function.args.kwonlyargs} - argument_defaults(function.args).keys()
        return required <= assigned

    @staticmethod
    def _locals(fn):
        return {name for name in map(_bound_name, scope_nodes(ast.iter_child_nodes(fn))) if name}

    @staticmethod
    def _contains_mutable(value):
        return (isinstance(value, (list, dict, set)) or
                isinstance(value, tuple) and any(SourceBindings._contains_mutable(item) for item in value))

    @staticmethod
    def _contains_reference(value, target, seen=None):
        if value is target:
            return True
        seen = set() if seen is None else seen
        if id(value) in seen or not isinstance(value, (list, tuple, dict)):
            return False
        seen.add(id(value))
        items = [item for pair in value.items() for item in pair] if isinstance(value, dict) else value
        return any(SourceBindings._contains_reference(item, target, seen) for item in items)

    @staticmethod
    def _allocator_specifications(tree, parents):
        """Reject escaped allocator classes and generators with shadowed builtins."""
        nodes = list(ast.walk(tree))
        builtins_shadowed = any(SourceBindings._shadows_allocator_builtin(n) for n in nodes)
        specifications = {}
        for cls in nodes:
            specification = recognize_allocator(cls)
            if specification is None or specification.returns_ids and builtins_shadowed:
                continue
            aliases = SourceBindings._allocator_aliases(nodes, cls.name)
            if SourceBindings._allocator_uses_are_safe(nodes, parents, aliases):
                specifications[id(cls)] = specification
        return specifications

    @staticmethod
    def _shadows_allocator_builtin(node):
        name = node.arg if isinstance(node, ast.arg) else _bound_name(node)
        return name in {"list", "range"} or isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names)

    @staticmethod
    def _allocator_aliases(nodes, class_name):
        aliases = {class_name}
        while True:
            previous = set(aliases)
            for node in nodes:
                if (isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Name)
                        and node.value.id in aliases):
                    aliases.update(t.id for t in assignment_targets(node) if isinstance(t, ast.Name))
            if aliases == previous:
                return aliases

    @staticmethod
    def _allocator_uses_are_safe(nodes, parents, aliases):
        """Every load of the class is a constructor call or a plain name alias."""
        loads = (node for node in nodes
                 if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in aliases)
        return all(_constructs_or_aliases(parents.get(node), node) for node in loads)

    @staticmethod
    def _repeatable_sequence(sequence, count):
        return (isinstance(sequence, (list, tuple)) and isinstance(count, int)
                and all(isinstance(item, int) for item in sequence))

    @staticmethod
    def _bounded_power(base, exponent):
        if not isinstance(base, int) or not isinstance(exponent, int):
            return UNKNOWN
        if not 0 <= exponent <= 64 or abs(base).bit_length() * exponent > 4096:
            return UNKNOWN
        return base ** exponent  # |base| < 2**bits, so the result has at most bits * exponent <= 4096 bits

    @staticmethod
    def _value_builtin(name, args):
        if name in ("min", "max"):
            return (min if name == "min" else max)(*args)
        if name == "len" and len(args) == 1:
            return len(args[0])
        if name == "int" and len(args) == 1 and isinstance(args[0], (str, int)):
            return int(args[0])
        if name in ("list", "tuple") and len(args) <= 1:
            return SourceBindings._sequence_builtin(name, args)
        if name == "range" and 1 <= len(args) <= 3:
            if all(isinstance(arg, int) for arg in args):
                result = range(*args)
                return result if len(result) <= 1024 else UNKNOWN
        return UNKNOWN

    @staticmethod
    def _sequence_builtin(name, args):
        sequence = args[0] if args else ()
        if not isinstance(sequence, (list, tuple, range, str, bytes, dict)) or len(sequence) > 1024:
            return UNKNOWN
        return (list if name == "list" else tuple)(sequence)

    @staticmethod
    def _value_atom(node, child):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Attribute) and node.attr.startswith("DT_"):
            return node.attr
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [child(element) for element in node.elts]
            if any(value is UNKNOWN for value in values):
                return UNKNOWN
            return values if isinstance(node, ast.List) else tuple(values)
        if isinstance(node, ast.Subscript):
            base, index = child(node.value), child(node.slice)
            return UNKNOWN if base is UNKNOWN or index is UNKNOWN else base[index]
        if isinstance(node, ast.UnaryOp):
            value = child(node.operand)
            return UNKNOWN if value is UNKNOWN else _UNARY[type(node.op)](value)
        return UNKNOWN

    @staticmethod
    def _value_dict(n, child):
        pairs = [(child(k), child(v)) for k, v in zip(n.keys, n.values) if k is not None]
        if len(pairs) != len(n.keys) or any(k is UNKNOWN or v is UNKNOWN for k, v in pairs):
            return UNKNOWN
        return dict(pairs)

    @staticmethod
    def _value_compare(n, child):
        a = child(n.left)
        for op, rhs in zip(n.ops, n.comparators):
            b = child(rhs)
            fn = _COMPARE.get(type(op))
            if a is UNKNOWN or b is UNKNOWN or fn is None:
                return UNKNOWN
            # Constant interning is a compiler/runtime detail. Object
            # identity is statically unambiguous only against singletons.
            if isinstance(op, (ast.Is, ast.IsNot)) and not any(v is None or isinstance(v, bool) for v in (a, b)):
                return UNKNOWN
            if not fn(a, b):
                return False
            a = b
        return True

    @staticmethod
    def _value_boolean(n, child):
        decisive = isinstance(n.op, ast.Or)  # `or` stops at the first truthy operand, `and` at the first falsy one
        for expression in n.values:
            value = child(expression)
            if value is UNKNOWN or bool(value) is decisive:
                return value
        return value

    @cached_property
    def boundary(self):
        return Boundary(self.tree, parents=self.parents, paths=self.shell_paths)

    def environment(self, node):
        return self.env_at.get(id(node), {})

    def scope(self, node):
        return self.scope_at.get(id(node), "")

    def value(self, binding, depth=0):
        if binding is None or depth > 24:
            return UNKNOWN
        cached = self._value_cache.get(id(binding))
        if cached is not None and cached[0] is binding:
            return cached[1]
        result = self._value(binding, depth)
        self._value_cache[id(binding)] = (binding, result)
        return result

    def _value(self, binding, depth=0):
        node, env = binding.node, binding.env

        def child(expression):
            return self.value(Binding(expression, env), depth + 1)

        handlers = {ast.Dict: self._value_dict, ast.BinOp: self._value_binary, ast.Compare: self._value_compare,
                    ast.BoolOp: self._value_boolean, ast.IfExp: self._value_conditional}
        try:
            if isinstance(node, ast.Name):
                return self.value(env.get(node.id), depth + 1)
            if isinstance(node, ast.ListComp):
                return self._value_comprehension(node, env, depth)
            if isinstance(node, ast.Call):
                return self._value_call(node, env, depth)
            return handlers.get(type(node), self._value_atom)(node, child)
        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, OverflowError):
            return UNKNOWN

    def _value_conditional(self, node, child):
        predicate = child(node.test)
        if predicate is not UNKNOWN:
            return child(node.body if predicate else node.orelse)
        before, after = child(node.body), child(node.orelse)
        return before if before is not UNKNOWN and self._same_value(before, after) else UNKNOWN

    def _value_comprehension(self, n, env, depth):
        results = []
        remaining = [2048]

        def expand(index, local):
            if len(results) > 1024:
                raise ValueError("comprehension limit")
            if index == len(n.generators):
                item = self.value(Binding(n.elt, local), depth + 1)
                if item is UNKNOWN:
                    raise ValueError("unknown comprehension element")
                results.append(item)
                return
            gen = n.generators[index]
            if gen.is_async or not self._simple_target(gen.target):
                raise ValueError("unsupported comprehension")
            sequence = self.value(Binding(gen.iter, local), depth + 1)
            if not isinstance(sequence, (tuple, list, range)) or len(sequence) > 1024:
                raise ValueError("unknown comprehension sequence")
            for item in sequence:
                remaining[0] -= 1
                if remaining[0] < 0:
                    raise ValueError("comprehension work limit")
                if not self._target_accepts(gen.target, item):
                    raise ValueError("invalid comprehension unpacking")
                inner = dict(local)
                self._bind(gen.target, Binding.literal(item), inner, {})
                predicates = [self.value(Binding(test, inner), depth + 1) for test in gen.ifs]
                if any(test is UNKNOWN for test in predicates):
                    raise ValueError("unknown comprehension predicate")
                if all(predicates):
                    expand(index + 1, inner)
        expand(0, dict(env))
        return results

    def _value_binary(self, n, child):
        a, b = child(n.left), child(n.right)
        if a is UNKNOWN or b is UNKNOWN:
            return UNKNOWN
        if isinstance(n.op, ast.Mult):
            sequence, count = (a, b) if isinstance(a, (list, tuple)) else (b, a)
            if self._repeatable_sequence(sequence, count):
                return sequence * count if len(sequence) * max(count, 0) <= 1024 else UNKNOWN
        if isinstance(n.op, ast.Pow):
            return self._bounded_power(a, b)
        if isinstance(n.op, (ast.LShift, ast.RShift)) and not (isinstance(b, int) and 0 <= b <= 64):
            return UNKNOWN
        operation = _BINARY.get(type(n.op))
        if not operation or not isinstance(a, int) or not isinstance(b, int):
            return UNKNOWN
        result = operation(a, b)
        return result if result.bit_length() <= 4096 else UNKNOWN

    def _value_call(self, n, env, depth):
        args = [self.value(Binding(arg, env), depth + 1) for arg in n.args]
        if any(arg is UNKNOWN for arg in args):
            return UNKNOWN
        if isinstance(n.func, ast.Name) and n.func.id not in env:
            return UNKNOWN if n.keywords else self._value_builtin(n.func.id, args)
        function = self.resolve(Binding(n.func, env))
        if not function or not isinstance(function.node, ast.FunctionDef):
            return UNKNOWN
        if function.node.decorator_list or not self._valid_call(function.node, n):
            return UNKNOWN
        if any(self.value(Binding(keyword.value, env), depth + 1) is UNKNOWN for keyword in n.keywords):
            return UNKNOWN
        # Interpret only bounded, pure integer/sequence planner statements.
        call_env = self.arguments(function, n, env)
        action, result = self._run_planner_suite(function.node.body, call_env, depth + 1, [2048])
        return result if action == "return" else UNKNOWN

    def _run_planner_suite(self, suite, env, depth, remaining):
        for stmt in suite:
            remaining[0] -= 1
            if remaining[0] < 0:
                return "unknown", UNKNOWN
            action, value = self._run_planner_statement(stmt, env, depth, remaining)
            if action != "next":
                return action, value
        return "next", None

    def _run_planner_statement(self, stmt, env, depth, remaining):
        if isinstance(stmt, ast.Return):
            return "return", self.value(Binding(stmt.value, dict(env)), depth + 1)
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and stmt.value is not None:
            return self._planner_assign(stmt, env, depth)
        if isinstance(stmt, ast.If):
            predicate = self.value(Binding(stmt.test, env), depth + 1)
            if predicate is UNKNOWN:
                return "unknown", UNKNOWN
            return self._run_planner_suite(stmt.body if predicate else stmt.orelse, env, depth, remaining)
        if isinstance(stmt, ast.For) and self._simple_target(stmt.target):
            return self._run_planner_for(stmt, env, depth, remaining)
        if isinstance(stmt, (ast.Break, ast.Continue)):
            return ("break" if isinstance(stmt, ast.Break) else "continue"), None
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            return "next", None
        return "unknown", UNKNOWN

    def _planner_assign(self, stmt, env, depth):
        targets = assignment_targets(stmt)
        if not all(self._simple_target(target) for target in targets):
            return "unknown", UNKNOWN
        expression = augmented_value(stmt) if isinstance(stmt, ast.AugAssign) else stmt.value
        value = self.value(Binding(expression, dict(env)), depth + 1)
        if value is UNKNOWN or not all(self._target_accepts(target, value) for target in targets):
            return "unknown", UNKNOWN
        for target in targets:
            self._bind(target, Binding.literal(value), env, {})
        return "next", None

    def _run_planner_for(self, stmt, env, depth, remaining):
        sequence = self.value(Binding(stmt.iter, env), depth + 1)
        if not isinstance(sequence, (tuple, list, range)) or len(sequence) > 1024:
            return "unknown", UNKNOWN
        for item in sequence:
            if not self._target_accepts(stmt.target, item):
                return "unknown", UNKNOWN
            self._bind(stmt.target, Binding.literal(item), env, {})
            action, value = self._run_planner_suite(stmt.body, env, depth, remaining)
            if action == "break":
                return "next", None
            if action not in ("next", "continue"):
                return action, value
        return self._run_planner_suite(stmt.orelse, env, depth, remaining)

    def _record(self, node, env, scope, active=True):
        self.env_at[id(node)] = dict(env)
        self.scope_at[id(node)] = scope
        self.active_at[id(node)] = active
        if active and isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = self.resolve(Binding(node.func, env))
            if fn and isinstance(fn.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.calls.append((fn, node, dict(env)))
        for child in ast.iter_child_nodes(node):
            self._record(child, env, scope, active)

    def _bind(self, target, value, env, constants=None, fallback=None):
        constants = self.extra if constants is None else constants
        if isinstance(target, ast.Name):
            env[target.id] = value
            if fallback and target.id in fallback and self.value(value) is UNKNOWN:
                if self.derived_guard_error:
                    self.stale_derived_used.add(target.id)
                else:
                    env[target.id] = Binding.literal(fallback[target.id])
            if self.override_extra and target.id in constants:
                env[target.id] = Binding.literal(constants[target.id])
        elif isinstance(target, (ast.Tuple, ast.List)):
            resolved = self.resolve(value)
            if (resolved and isinstance(resolved.node, (ast.Tuple, ast.List))
                    and len(resolved.node.elts) == len(target.elts)):
                values = [Binding(x, resolved.env) for x in resolved.node.elts]
            else:
                got = self.value(value)
                values = ([Binding.literal(x) for x in got]
                          if isinstance(got, (tuple, list, range)) and len(got) == len(target.elts)
                          else [None] * len(target.elts))
            for t, v in zip(target.elts, values):
                self._bind(t, v, env, constants, fallback)
        elif isinstance(target, (ast.Subscript, ast.Attribute)):
            self._invalidate(target.value, env)

    def _binding_depends_on(self, binding, node, seen=None):
        seen = set() if seen is None else seen
        if binding is None or id(binding) in seen:
            return False
        seen.add(id(binding))
        return binding.node is node or any(self._binding_depends_on(binding.env.get(name.id), node, seen)
                                           for name in loaded_names(binding.node))

    def _invalidate(self, expression, env):
        root = root_name(expression)
        if root is None:
            return
        original = self.resolve(env.get(root))
        original_value = self.value(env.get(root))
        if isinstance(original_value, AllocatorCursor):
            self._replace_allocator(original_value, None, env)
        if original is not None and isinstance(original.node, ast.ClassDef):
            for binding in list(env.values()):
                cursor = self.value(binding)
                if isinstance(cursor, AllocatorCursor) and cursor.specification.class_node is original.node:
                    self._replace_allocator(cursor, None, env)
        for name, binding in list(env.items()):
            resolved = self.resolve(binding)
            value = self.value(binding)
            dependent_mutable = (
                self._contains_mutable(original_value) and self._contains_reference(
                    value, original_value)
                or original is not None and value is UNKNOWN
                and self._binding_depends_on(binding, original.node))
            if name == root or dependent_mutable or _same_binding_node(original, resolved):
                env[name] = None

    def _join(self, a, b):
        out = {}
        for name in a.keys() | b.keys():
            x, y = a.get(name), b.get(name)
            if x is not y:
                vx, vy = self.value(x), self.value(y)
                x = x if vx is not UNKNOWN and self._same_value(vx, vy) else None
            if self.override_extra and x is None and name in self.extra:
                x = Binding.literal(self.extra[name])
            out[name] = x
        return out

    def _readonly_call(self, call, env, seen=frozenset()):
        """Recognize the tiny pure helper/declaration subset used by inspection."""
        if isinstance(call.func, ast.Attribute):
            return (isinstance(call.func.value, ast.Name) and call.func.value.id in {"pl", "pypto", "pypto_pro"}
                    and call.func.attr in {"TileType", "make_tile", "make_tile_group", "set_validshape"})
        if not isinstance(call.func, ast.Name):
            return False
        if call.func.id not in env:
            return call.func.id in {"min", "max", "len", "int", "list", "tuple", "range"}
        binding = self.resolve(Binding(call.func, env))
        if binding is None or not isinstance(binding.node, ast.FunctionDef) or binding.node.decorator_list:
            return False
        function = binding.node
        if id(function) in seen or len(seen) >= 16:
            return False
        seen = seen | {id(function)}
        parameters = function.args.posonlyargs + function.args.args + function.args.kwonlyargs
        local = {**binding.env, **dict.fromkeys(self._locals(function) | {p.arg for p in parameters})}
        return all(self._readonly_node(node, local, seen) for stmt in function.body for node in ast.walk(stmt))

    def _readonly_node(self, node, local, seen):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            return all(self._simple_target(target) for target in assignment_targets(node))
        if isinstance(node, ast.Call):
            return self._readonly_call(node, local, seen)
        return not isinstance(node, _EFFECTFUL)

    def _container_escapes(self, expression, env):
        if isinstance(expression, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Defining a function does not execute its body. Decorator/default
            # calls do execute, and may consume an enclosing mutable object.
            expressions = (list(expression.decorator_list) + list(expression.args.defaults)
                           + [value for value in expression.args.kw_defaults if value is not None])
        else:
            expressions = [expression]
        for expression in expressions:
            for call in (node for node in ast.walk(expression) if isinstance(node, ast.Call)):

                self._invalidate_escaping_call(call, env)

    def _invalidate_escaping_call(self, call, env):
        names = [name for name in loaded_names(call) if self._contains_mutable(self.value(env.get(name.id)))]
        if not names or self._readonly_call(call, env):
            return
        for name in names:
            self._invalidate(name, env)

    def _allocator_escapes(self, statement, env):
        self._container_escapes(statement, env)
        allowed = None
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            allowed = statement.value  # an alias is tracked by instance
            if isinstance(allowed, ast.Call) and isinstance(allowed.func, ast.Attribute):
                allowed = allowed.func.value
        for node in loaded_names(statement):
            if node is not allowed:
                cursor = self.value(env.get(node.id))
                if isinstance(cursor, AllocatorCursor):
                    self._replace_allocator(cursor, None, env)

    def _replace_allocator(self, previous, replacement, env):
        for name, binding in list(env.items()):
            value = self.value(binding)
            if isinstance(value, AllocatorCursor) and value.instance == previous.instance:
                env[name] = None if replacement is None else Binding.literal(replacement)

    def _allocator_assignment(self, stmt, env):
        call = stmt.value
        if not isinstance(call, ast.Call):
            return None
        if isinstance(call.func, ast.Name):
            constructor = self.resolve(Binding(call.func, env))
            specification = self._allocator_specs.get(id(constructor.node)) if constructor else None
            if specification and not call.args and not call.keywords:
                return Binding.literal(AllocatorCursor(specification, id(call), 0))
        if not isinstance(call.func, ast.Attribute):
            return None
        cursor = self.value(Binding(call.func.value, env))
        if not isinstance(cursor, AllocatorCursor):
            return None
        specification = cursor.specification
        values, parameters, alignment, valid = self._allocator_arguments(call, specification, env)
        result = cursor.allocate(values.get(parameters[0]), alignment) if valid else None
        if result is None:
            self._replace_allocator(cursor, None, env)
            return Binding(ast.Name(id="__unknown_allocator_result", ctx=ast.Load()), {})
        address, advanced = result
        self._replace_allocator(cursor, advanced, env)
        return Binding.literal(address)

    def _allocator_arguments(self, call, specification, env):
        method = next(fn for fn in specification.class_node.body
                      if isinstance(fn, ast.FunctionDef) and fn.name == specification.method)
        parameters = [argument.arg for argument in method.args.args[1:]]
        supplied = list(zip(parameters, call.args)) + [(keyword.arg, keyword.value) for keyword in call.keywords]
        names = [name for name, _ in supplied]
        valid = (call.func.attr == specification.method and len(call.args) <= len(parameters)
                 and len(set(names)) == len(names) and set(names) <= set(parameters))
        values = {parameters[1]: specification.default_alignment} if len(parameters) == 2 else {}
        values.update((name, self.value(Binding(expression, env))) for name, expression in supplied)
        alignment = values.get(parameters[1]) if len(parameters) == 2 else 1
        return values, parameters, alignment, valid

    def _suite(self, statements, env, scope, active=True):
        env = dict(env)
        for stmt in statements:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr,
                                 ast.Return, ast.FunctionDef, ast.AsyncFunctionDef)):
                self._allocator_escapes(stmt, env)
            kind = _SUITE_KINDS.get(type(stmt), type(stmt)).__name__.lower()
            env = getattr(self, "_suite_" + kind, self._suite_other)(stmt, env, scope, active)
        return env

    def _suite_functiondef(self, stmt, env, scope, active):
        self.env_at[id(stmt)] = dict(env)
        self.scope_at[id(stmt)] = scope
        captured = dict(env)
        # Use the same closure environment for function bodies and pure-helper evaluation.
        for name, definitions in self._late_rebindings.items():
            bound = captured.get(name)
            if bound is not None and definitions.get(id(bound.node), 0) > stmt.lineno:
                captured[name] = None
        fn = Binding(stmt, captured)
        env[stmt.name] = fn
        fn.env[stmt.name] = fn
        local = {**captured, **dict.fromkeys(self._locals(stmt))}
        fn_scope = self.scopes[id(stmt)]
        # Closure capture propagates outer facts. Only this exact owner may
        # supply a local/parameter; nested shadowing must stay lexical.
        scoped = self.scoped_extra.get(fn_scope, {})
        defaults = argument_defaults(stmt.args)
        omitted = self.default_arguments.get(fn_scope, ())
        missing_defaults = set(omitted) - defaults.keys()
        if missing_defaults:
            raise ValueError("recorded omitted arguments no longer have source defaults in %s: %s; "
                             "refresh the actual factory call context" % (
                                 fn_scope, ", ".join(sorted(missing_defaults))))
        for p in list(stmt.args.posonlyargs) + list(stmt.args.args) + list(stmt.args.kwonlyargs):
            local[p.arg] = self.specialized.get(id(stmt), {}).get(p.arg)
            if p.arg in scoped and self.value(local[p.arg]) is UNKNOWN:
                local[p.arg] = Binding.literal(scoped[p.arg])
            elif (p.arg in omitted and id(stmt) not in self.specialized and self.value(local[p.arg]) is UNKNOWN):
                local[p.arg] = Binding(defaults[p.arg], captured)
            elif self.override_extra and local[p.arg] is None and p.arg in self.extra:
                local[p.arg] = Binding.literal(self.extra[p.arg])
        self._suite(stmt.body, local, fn_scope, active)
        return env

    def _suite_classdef(self, stmt, env, scope, active):
        self._allocator_escapes(stmt, env)
        self._record(stmt, env, scope, active)
        env[stmt.name] = Binding(stmt, dict(env)) if id(stmt) in self._allocator_specs else None
        return env

    def _suite_assign(self, stmt, env, scope, active):
        self._record(stmt, env, scope, active)
        if stmt.value is not None:
            value = self._allocator_assignment(stmt, env) or Binding(stmt.value, dict(env))
            for target in assignment_targets(stmt):
                self._bind(target, value, env, fallback=self.derived_extra.get(scope))
        return env

    def _suite_augassign(self, stmt, env, scope, active):
        self._record(stmt, env, scope, active)
        if isinstance(stmt.target, ast.Name):
            expr = augmented_value(stmt)
            previous_env = dict(env)
            if self._contains_mutable(self.value(env.get(stmt.target.id))):
                # Sequence += and *= mutate aliases in Python. Their
                # evaluation is unsupported; do not retain stale aliases.
                self._invalidate(stmt.target, env)
            self._bind(stmt.target, Binding(expr, previous_env), env, fallback=self.derived_extra.get(scope))
        else:
            self._invalidate(stmt.target, env)
        return env

    def _suite_if(self, stmt, env, scope, active):
        self._allocator_escapes(stmt.test, env)
        pred = self.value(Binding(stmt.test, env))
        self._record(stmt.test, env, scope, active)
        yes = self._suite(stmt.body, env, scope, active and (pred is UNKNOWN or bool(pred)))
        no = self._suite(stmt.orelse, env, scope, active and (pred is UNKNOWN or not pred))
        if pred is UNKNOWN:
            return self._join(yes, no)
        return yes if pred else no

    def _suite_with(self, stmt, env, scope, active):
        for item in stmt.items:
            self._allocator_escapes(item.context_expr, env)
            self._record(item.context_expr, env, scope, active)
        return self._suite(stmt.body, env, scope, active)

    def _suite_for(self, stmt, env, scope, active):
        condition = stmt.test if isinstance(stmt, ast.While) else stmt.iter
        self._allocator_escapes(condition, env)
        self._record(condition, env, scope, active)
        if isinstance(stmt, ast.While):
            pred = self.value(Binding(stmt.test, env))
            if pred is not UNKNOWN and not pred:
                # Index the unreachable body without evaluating its calls or mutating bindings.
                for child in stmt.body:
                    self._record(child, env, scope, False)
                return self._suite(stmt.orelse, env, scope, active)
        loop = dict(env)
        loop.update((n.id, None) for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
        env = self._join(env, self._suite(stmt.body, loop, scope, active))
        return self._suite(stmt.orelse, env, scope, active)

    def _suite_try(self, stmt, env, scope, active):
        for handler in stmt.handlers:
            if handler.type is not None:
                self._allocator_escapes(handler.type, env)
        branches = [self._suite(stmt.body + stmt.orelse, env, scope, active)]
        branches.extend(self._suite(h.body, env, scope, active) for h in stmt.handlers)
        for branch in branches:
            env = self._join(env, branch)
        return self._suite(stmt.finalbody, env, scope, active)

    def _suite_delete(self, stmt, env, scope, active):
        self._record(stmt, env, scope, active)
        for target in stmt.targets:
            self._invalidate(target, env)
        return env

    def _suite_import(self, stmt, env, scope, active):
        self._record(stmt, env, scope, active)
        for alias in stmt.names:
            name = _bound_name(alias)
            env[name] = Binding.literal(self.extra[name]) if name in self.extra else None
        return env

    def _suite_other(self, stmt, env, scope, active):
        self._allocator_escapes(stmt, env)
        self._record(stmt, env, scope, active)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Attribute) and func.attr in {"append", "extend", "insert", "pop", "remove", "clear",
                                                                 "update", "setdefault", "reverse", "sort"}:
                self._invalidate(func.value, env)
        return env

    def _specializations(self):
        selected = active_scope_closure(self.tree, self, self.context)
        calls = {}
        for fn, call, env in self.calls:
            scope = self.scope_at.get(id(call), "")
            if selected is not None and scope in self._device_scopes and scope not in selected:
                continue
            calls.setdefault(id(fn.node), []).append((fn, self.arguments(fn, call, env)))
        escaped = self._escaped_functions(selected)
        return {key: self._invocation_facts(invocations) for key, invocations in calls.items() if key not in escaped}

    def _escaped_functions(self, selected):
        escaped = set()
        for node in loaded_names(self.tree):
            scope = self.scope(node)
            if selected is not None and scope in self._device_scopes and scope not in selected:
                continue
            bound = self.resolve(Binding(node, self.environment(node)))
            if bound is None or not isinstance(bound.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            parent = self.parents.get(node)
            if isinstance(parent, ast.Call) and parent.func is node:
                continue
            if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is node:
                continue  # direct aliases are followed at their call sites
            escaped.add(id(bound.node))
        return escaped

    def _same_handles(self, bindings):
        handles = [self._handle(binding) for binding in bindings]
        return handles[0] is not None and all(handle == handles[0] for handle in handles)

    def _invocation_facts(self, invocations):
        fn = invocations[0][0].node
        params = list(fn.args.posonlyargs) +\
            list(fn.args.args) + list(fn.args.kwonlyargs)
        facts = {}
        for param in params:
            bindings = [env.get(param.arg) for _, env in invocations]
            values = [self.value(b) for b in bindings]
            if (values[0] is not UNKNOWN and all(self._same_value(v, values[0]) for v in values)
                    or self._same_handles(bindings)):
                facts[param.arg] = bindings[0]
        return facts

    def _fingerprint(self, specializations):
        return {k: {name: (repr(self.value(b)), self._handle(b)) for name, b in vals.items()}
                for k, vals in specializations.items()}

    def _handle(self, binding, depth=0):
        """Identity of a statically passed tile/group, without executing the DSL."""
        if depth > 16:
            return None
        binding = self.resolve(binding)
        if binding is None:
            return None
        n = binding.node
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr == "make_tile_group":
                return id(n)
            if n.func.attr in ("next", "current"):
                return self._handle(Binding(n.func.value, binding.env), depth + 1)
        if isinstance(n, ast.Subscript):
            return self._handle(Binding(n.value, binding.env), depth + 1)
        return None


def _same_binding_node(original, resolved):
    return original is not None and resolved is not None and resolved.node is original.node
