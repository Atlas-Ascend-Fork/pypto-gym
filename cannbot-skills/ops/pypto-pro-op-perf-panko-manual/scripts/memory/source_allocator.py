# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Recognize supported address and mutex allocators statically."""
from __future__ import annotations

import ast
from dataclasses import dataclass


def executable_body(statements):
    """Ignore standalone string literals when matching a function body."""
    def is_string(stmt):
        return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)

    return [stmt for stmt in statements if not is_string(stmt)]


@dataclass(frozen=True)
class BumpAllocator:
    class_node: ast.ClassDef
    method: str
    default_alignment: int
    returns_ids: bool = False


@dataclass(frozen=True)
class AllocatorCursor:
    specification: BumpAllocator
    instance: int
    position: int

    def allocate(self, size, alignment):
        # bool is intentionally excluded: it is not an explicit byte extent.
        if type(size) is not int or type(alignment) is not int or size < 0:
            return None
        if alignment <= 0 or max(size.bit_length(), alignment.bit_length()) > 63:
            return None
        address = (self.position + alignment - 1) // alignment * alignment
        if (address + size).bit_length() > 63:
            return None
        if self.specification.returns_ids and size > 1024:
            return None
        value = list(range(address, address + size)) if self.specification.returns_ids else address
        return value, AllocatorCursor(self.specification, self.instance, address + size)


def _allocator_methods(node):
    if not isinstance(node, ast.ClassDef):
        return None
    if node.bases or node.keywords or node.decorator_list:
        return None
    methods = executable_body(node.body)
    if len(methods) != 2 or not all(isinstance(x, ast.FunctionDef) for x in methods):
        return None
    init = next((x for x in methods if x.name == '__init__'), None)
    alloc = next((x for x in methods if x is not init), None)
    if init is None or alloc is None or alloc.name.startswith("__"):
        return None
    return (init, alloc) if all(map(_simple_allocator_method, methods)) else None


def _simple_allocator_method(fn):
    args = fn.args
    extras = (fn.decorator_list, fn.returns, args.posonlyargs, args.kwonlyargs, args.vararg, args.kwarg)
    return not any(extras) and not any(argument.annotation for argument in args.args)


def recognize_allocator(node):
    methods = _allocator_methods(node)
    if methods is None:
        return None
    init, alloc = methods
    returns_ids = len(alloc.args.args) == 2 and not alloc.args.defaults
    if (len(init.args.args) != 1 or init.args.defaults or not _allocation_arity_matches(alloc, returns_ids)):
        return None
    default = ast.Constant(1) if returns_ids else alloc.args.defaults[0]
    if not isinstance(default, ast.Constant) or type(default.value) is not int or default.value <= 0:
        return None
    cursor = _allocator_cursor(init, alloc, returns_ids)
    if cursor is None or not _allocator_shape_matches(alloc, cursor, returns_ids):
        return None
    return BumpAllocator(node, alloc.name, default.value, returns_ids)


def _allocator_cursor(init, alloc, returns_ids):
    ibody, abody = executable_body(init.body), executable_body(alloc.body)
    if len(ibody) != 1 or len(abody) != (3 if returns_ids else 4):
        return None
    first = ibody[0]
    if not isinstance(first, ast.Assign) or len(first.targets) != 1:
        return None
    target = first.targets[0]
    if not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name):
        return None
    if target.value.id != init.args.args[0].arg or not isinstance(first.value, ast.Constant):
        return None
    if type(first.value.value) is not int or first.value.value != 0:
        return None
    cursor = target.attr
    # Special attributes invoke Python object protocols; a field equal to the
    # allocator method would replace that bound method during construction.
    if cursor.startswith("__") or cursor == alloc.name:
        return None
    return cursor


def _allocator_shape_matches(alloc, cursor, returns_ids):
    abody = executable_body(alloc.body)
    save = abody[0 if returns_ids else 1]
    if not isinstance(save, ast.Assign) or len(save.targets) != 1 or not isinstance(save.targets[0], ast.Name):
        return False
    names = {arg.arg: canonical for (arg, canonical) in zip(alloc.args.args, ('self', 'nbytes', 'align'))}
    names[save.targets[0].id] = 'result'

    class Canonical(ast.NodeTransformer):
        @staticmethod
        def visit_name(expr):
            return ast.Name(id=names.get(expr.id, expr.id), ctx=expr.ctx)

        visit_Name = visit_name

        def visit_attribute(self, expr):
            return ast.Attribute(value=self.visit(expr.value),
                                 attr='cursor' if expr.attr == cursor else expr.attr, ctx=expr.ctx)

        visit_Attribute = visit_attribute

    # Parse a copy: never mutate AST nodes owned by the caller or source reader.
    actual = ast.parse('\n'.join(ast.unparse(stmt) for stmt in abody))
    actual = Canonical().visit(actual)
    expected = ast.parse('''result = self.cursor
self.cursor += nbytes
return list(range(result, result + nbytes))
''' if returns_ids else '''self.cursor = (self.cursor + align - 1) // align * align
result = self.cursor
self.cursor += nbytes
return result
''')
    return ast.dump(actual, include_attributes=False) == ast.dump(expected, include_attributes=False)


def _allocation_arity_matches(alloc, returns_ids):
    return returns_ids or (len(alloc.args.args) == 3 and len(alloc.args.defaults) == 1)
