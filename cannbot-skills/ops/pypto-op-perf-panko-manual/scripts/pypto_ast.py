# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Which names in a source file mean `pypto`.

`<op>_impl.py` carries the kernel and the host wrapper together, and an alias is
legal, so every reader that has to tell a pypto call from a torch one needs this
same answer. It lived in two copies -- `predicates` and `bayesian_optimization.domain`
-- which is one copy too many for a test that decides whether a candidate's
statements count as kernel work.

Nothing here imports anything but `ast`, so both readers can have it without
taking on each other's dependencies.
"""

import ast


def pypto_roots(tree, *, packages=("pypto",), seeds=("pypto",), include_from=False):
    """Names bound by DSL imports, including any aliases conventional for this dialect."""
    roots = set(seeds)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = (alias for alias in node.names if alias.name.split(".")[0] in packages)
        elif include_from and isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in packages:
            names = node.names
        else:
            continue
        roots.update((alias.asname or alias.name).split(".")[0] for alias in names)
    return frozenset(roots)


def parent_map(tree):
    """Index an AST by child node, including nodes inside nested definitions."""
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def function_name(node):
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def jit_decorators(fn):
    for decorator in getattr(fn, "decorator_list", ()):
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if function_name(target) == "jit":
            yield decorator


def keyword_value(call, name):
    return next((kw.value for kw in call.keywords if kw.arg == name), None)
