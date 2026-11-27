# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Resolve active factory scopes and arguments from case metadata without executing kernel code."""
from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping

from source_analysis.kernel_boundary import Boundary, loaded_names

ACTIVE_SCOPES = "__panko_active_scopes__"
SCOPED_CONSTANTS = "__panko_scoped_constants__"
CASE_ID = "__panko_case_id__"
DERIVED_FACTS = "__panko_derived_facts__"
DEFAULT_ARGUMENTS = "__panko_default_arguments__"
_RESERVED = frozenset({ACTIVE_SCOPES, SCOPED_CONSTANTS, CASE_ID, DERIVED_FACTS, DEFAULT_ARGUMENTS})


def validate_context(extra):
    """Validate the context envelope without evaluating its source or metadata."""
    if extra is None:
        return
    if not isinstance(extra, Mapping) or not all(isinstance(key, str) for key in extra):
        raise ValueError("case constants must be an object with string keys")
    _validate_active_scopes(extra.get(ACTIVE_SCOPES))
    scoped = extra.get(SCOPED_CONSTANTS, {})
    _validate_scoped_constants(scoped)
    if CASE_ID in extra and not isinstance(extra[CASE_ID], str):
        raise ValueError("case id must be a string")
    _validate_default_arguments(extra.get(DEFAULT_ARGUMENTS, {}), scoped)
    _validate_derived_facts(extra.get(DERIVED_FACTS))


def _lexical_path(value):
    return isinstance(value, str) and bool(value) and not value.startswith("/") and not value.endswith("/")


def _validate_active_scopes(active):
    if active is None:
        return
    valid = isinstance(active, (list, tuple)) and bool(active) and all(_lexical_path(scope) for scope in active)
    if not valid or len(set(active)) != len(active):
        raise ValueError("active scopes must be a nonempty list of distinct exact lexical paths")


def _validate_scoped_constants(scoped):
    if not isinstance(scoped, Mapping):
        raise ValueError("scoped constants must map lexical paths to binding objects")
    for scope, values in scoped.items():
        valid_values = isinstance(values, Mapping) and all(isinstance(k, str) and k.isidentifier() for k in values)
        if not _lexical_path(scope) or not valid_values:
            raise ValueError("invalid scoped constants for lexical path %r" % (scope,))


def _validate_default_arguments(defaults, scoped):
    if not isinstance(defaults, Mapping):
        raise ValueError("default arguments must map exact lexical scopes to omitted argument names")
    for scope, names in defaults.items():
        valid_scope = isinstance(scope, str) and bool(scope)
        valid_names = isinstance(names, (list, tuple)) and all(isinstance(n, str) and n.isidentifier() for n in names)
        if not valid_scope or not valid_names or len(set(names)) != len(names):
            raise ValueError("invalid default arguments for lexical path %r" % scope)
        if set(names) & set(scoped.get(scope, {})):
            raise ValueError("an argument cannot be both explicitly bound and omitted")


def _validate_derived_facts(derived):
    if derived is None:
        return
    if not isinstance(derived, Mapping) or set(derived) != {"construction_sha256", "inputs_sha256", "values"}:
        raise ValueError("derived facts require construction_sha256, inputs_sha256, and values")
    for key in ("construction_sha256", "inputs_sha256"):
        digest = derived[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("derived facts require a lowercase SHA-256 for %s" % key)
    validate_context({SCOPED_CONSTANTS: derived["values"]})


def split_constants(extra):
    """Return ordinary constants and exact-scope bindings, preserving values."""
    validate_context(extra)
    extra = extra or {}
    return ({k: v for k, v in extra.items() if k not in _RESERVED},
            {scope: dict(values) for (scope, values) in extra.get(SCOPED_CONSTANTS, {}).items()})


def construction_fingerprint(tree):
    """
    Bind recorded planner outputs to all non-device source dependencies.

    Keep the complete module AST, including globals, imports, host construction,
    helper bodies and device signatures/decorators. Only proven device bodies
    are elided. This deliberately invalidates on unrelated host changes too;
    missing dependency edges must never leave a stale planner output valid.
    """
    boundary = Boundary(tree)

    def project(node):
        if isinstance(node, list):
            return [project(value) for value in node]
        if not isinstance(node, ast.AST):
            return node
        elided = {"body": [ast.Pass()]} if node in boundary.device else {}
        return type(node)(**{field: elided.get(field) or project(value) for field, value in ast.iter_fields(node)})

    payload = ast.dump(project(tree), include_attributes=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _inputs_fingerprint(context):
    payload = {key: value for key, value in context.items() if key != DERIVED_FACTS}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def derived_fact_bindings(tree, context):
    """Return recorded values and an actionable stale-evidence reason, if any."""
    context = context or {}
    guard = context.get(DERIVED_FACTS)
    if guard is None:
        return {}, None
    reasons = []
    if guard["inputs_sha256"] != _inputs_fingerprint(context):
        reasons.append("selected case inputs changed")
    if guard["construction_sha256"] != construction_fingerprint(tree):
        reasons.append("kernel construction or planner dependencies changed")
    message = ("recorded derived facts are stale (%s); refresh them from the actual case/environment "
               "or remove them so the current source can resolve its own values" %
               "; ".join(reasons)) if reasons else None
    return {scope: dict(values) for scope, values in guard["values"].items()}, message


def active_scope_closure(tree, bindings, extra):
    """
    Resolve selected roots and their conservative local-function closure.

    A reference to a function also keeps its declarations: this covers helpers
    returned or passed as callables, not only direct calls. Unknown declarations
    in a reached helper are retained and will fail the ordinary layout gate.
    Source-order ids are assigned before this selection, so a retained site's
    identity never shifts merely because another factory is inactive.
    """
    extra = extra or {}
    roots = extra.get(ACTIVE_SCOPES)
    supplied_scopes = (set(extra.get(SCOPED_CONSTANTS, {})) | set(extra.get(DEFAULT_ARGUMENTS, {}))
                       | set(extra.get(DERIVED_FACTS, {}).get("values", {})))
    functions = {path: n for n, path in bindings.paths.items()
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for scope in supplied_scopes:
        if scope not in functions:
            raise ValueError("case constants name missing function scope %r" % scope)
    if roots is None:
        return None
    missing = set(roots) - functions.keys()
    if missing:
        raise ValueError("case selects missing function scope(s): %s" % ", ".join(sorted(missing)))
    pending = list(roots)
    selected = set()
    seen_bindings = set()
    boundary = bindings.boundary

    def follow(name, env, depth):
        binding = env.get(name.id)
        if binding is not None and depth <= 32 and id(binding) not in seen_bindings:
            seen_bindings.add(id(binding))
            if isinstance(binding.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                pending.append(bindings.scopes[id(binding.node)])
            else:
                for child in loaded_names(binding.node):
                    follow(child, binding.env, depth + 1)
        # Global names may refer to helpers defined later in the source.
        pending.extend(bindings.scopes[id(target)] for target in boundary.resolve(name, boundary.scope.get(name, tree)))

    while pending:
        root = pending.pop()
        for scope, function in functions.items():
            if scope in selected or (scope != root and not scope.startswith(root + "/")):
                continue
            selected.add(scope)
            active_names = (n for n in loaded_names(function) if bindings.active_at.get(id(n), True))
            for name in active_names:
                follow(name, bindings.environment(name), 0)
    return frozenset(selected | {""})
