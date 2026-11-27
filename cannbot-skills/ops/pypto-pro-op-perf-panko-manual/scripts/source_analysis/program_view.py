# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Map edits in a namespaced view to the original bytes. Rebuild the view after structural changes."""
import ast
import base64
import json
from functools import lru_cache

from source_analysis.source_bindings import scope_nodes
from source_analysis.tile_sites import Offsets

HEADER = '# PANKO-PROGRAM-'
DSL_IMPORTS = {'pl', 'vf', 'pypto', 'pypto_pro', 'torch', 'math', 'os', '_os'}


def _stores(body):
    names = set()

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda)):
            # Comprehension targets and lambda arguments have their own scope.
            for child in ast.walk(node):
                if isinstance(child, ast.NamedExpr):
                    visit(child.target)
            return
        if isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or (a.name.split('.')[0] if isinstance(node, ast.Import) else a.name)
                         for a in node.names)
        for child in ast.iter_child_nodes(node):
            visit(child)
    for statement in body:
        visit(statement)
    return names


def _runtime_import_name(node, module, symbol):
    if not isinstance(node, ast.ImportFrom) or not node.level:
        return module
    parts = [node.module] if node.module else []
    if symbol is None:
        parts.append(module.rsplit('.', 1)[-1])
    return '.' * node.level + '.'.join(parts)


def _prior_package_exports(body, node):
    pending = {alias.name for alias in node.names}
    exports = set()
    for stmt in reversed(body[:body.index(node)]):
        if isinstance(stmt, ast.AnnAssign) and stmt.value is None:
            continue
        bound = pending.intersection(_stores([stmt]))
        if not bound:
            continue
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Assign, ast.AnnAssign, ast.AugAssign)):
            exports.update(bound)
        elif not isinstance(stmt, ast.Delete):
            raise ValueError('conditional package binding needs explicit initialization: %s' % sorted(bound))
        pending.difference_update(bound)
        if not pending:
            break
    return exports


def _absolute_import_names(source):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and not node.level:
            yield node.module
            for alias in node.names:
                yield node.module + '.' + alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


class _Rename(ast.NodeTransformer):
    def __init__(self, view, filename):
        self.view = view
        self.filename = filename
        self.mapping = view.names[filename]
        self.aliases = view.module_aliases[filename]
        self.scopes = []

    def global_name(self, name):
        for local, declared_global in reversed(self.scopes):
            if name in declared_global:
                return True
            if name in local:
                return False
        return True


    def visit_name(self, node):
        if self.global_name(node.id):
            if isinstance(node.ctx, ast.Load) and node.id in self.aliases:
                raise ValueError('module alias escapes static attribute access: %s:%d' % (
                    self.filename, node.lineno))
            node.id = self.mapping.get(node.id, node.id)
        return node

    visit_Name = visit_name

    def visit_attribute(self, node):
        module = self._module_of(node.value)
        if module is not None:
            target = self.view.by_module[module]
            if node.attr not in self.view.names[target]:
                raise ValueError(
                    'unresolved local module attribute: %s.%s' % (target, node.attr))
            replacement = ast.Name(
                id=self.view.names[target][node.attr], ctx=node.ctx)
            replacement.origin = getattr(node, 'origin', None)
            return ast.copy_location(replacement, node)
        return self.generic_visit(node)

    visit_Attribute = visit_attribute

    def visit_function_def(self, node):
        original_name = node.name
        if self.global_name(original_name):
            node.name = self.mapping.get(original_name, original_name)
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        args, parameters = self._visit_parameters(node.args)
        for arg in args:
            if arg.annotation:
                arg.annotation = self.visit(arg.annotation)
        if node.returns:
            node.returns = self.visit(node.returns)
        local = _stores(node.body) | parameters

        globals_ = set()
        for stmt in scope_nodes(node.body):
            if isinstance(stmt, ast.Global):
                globals_.update(stmt.names)
        self.scopes.append((local, globals_))
        enclosing_aliases = self.aliases
        self.aliases = self.view.module_aliases[self.filename]
        node.body = [self.visit(n) for n in node.body]
        self.aliases = enclosing_aliases
        self.scopes.pop()
        return node

    visit_FunctionDef = visit_function_def

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_lambda(self, node):
        _, parameters = self._visit_parameters(node.args)
        self.scopes.append((parameters, ()))
        enclosing_aliases = self.aliases
        self.aliases = self.view.module_aliases[self.filename]
        node.body = self.visit(node.body)
        self.aliases = enclosing_aliases
        self.scopes.pop()
        return node

    visit_Lambda = visit_lambda

    def visit_list_comp(self, node):
        local = set()
        for index, generator in enumerate(node.generators):
            generator.iter = self.visit(generator.iter)
            if index == 0:
                self.scopes.append((local, ()))
            local.update(n.id for n in ast.walk(generator.target) if isinstance(n, ast.Name))
            generator.ifs = [self.visit(n) for n in generator.ifs]
        for field in ('elt', 'key', 'value'):
            if hasattr(node, field):
                setattr(node, field, self.visit(getattr(node, field)))
        self.scopes.pop()
        return node

    visit_ListComp = visit_list_comp

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_class_def(self, node):
        if self.global_name(node.name):
            node.name = self.mapping.get(node.name, node.name)
        node.bases = [self.visit(n) for n in node.bases]
        node.keywords = [self.visit(n) for n in node.keywords]
        node.decorator_list = [self.visit(n) for n in node.decorator_list]
        # Class attributes are not lexical closure variables in methods.
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                saved = child.name
                self.visit(child)
                child.name = saved
            elif not isinstance(child, (ast.Expr, ast.Pass)):
                raise ValueError(
                    'class-level planning needs explicit initialization: %s' % self.filename)
        return node

    visit_ClassDef = visit_class_def

    def visit_global(self, node):
        node.names = [self.mapping.get(n, n) for n in node.names]
        return node

    visit_Global = visit_global

    def visit_import(self, node):
        return self.import_node(node)

    visit_Import = visit_import

    def visit_import_from(self, node):
        if node.module == '__future__':
            return ast.Pass()
        return self.import_node(node)

    visit_ImportFrom = visit_import_from

    def import_node(self, node):
        local = self.view.local_imports.get(id(node))
        if local is not None:
            statements = []
            for alias, module, symbol in local:
                if symbol is not None:
                    statements.append(ast.Assign(targets=[ast.Name(id=self.mapping[alias], ctx=ast.Store())],
                                                 value=ast.Name(
                                                     id=self.view.names[self.view.by_module[module]][symbol],
                                                     ctx=ast.Load())))
            return statements or [ast.Pass()]
        for alias in node.names:
            bound = alias.asname or (alias.name.split(
                '.')[0] if isinstance(node, ast.Import) else alias.name)
            if self.global_name(bound):
                alias.asname = self.mapping.get(bound, bound)
        return node

    def _module_of(self, expr):
        if isinstance(expr, ast.Name) and self.global_name(expr.id):
            return self.aliases.get(expr.id)
        if isinstance(expr, ast.Attribute):
            dotted = ast.unparse(expr)
            if dotted in self.aliases and self.global_name(dotted.split('.', 1)[0]):
                return self.aliases[dotted]
            parent = self._module_of(expr.value)
            if parent is not None:
                return self.view.module_aliases[self.view.by_module[parent]].get(expr.attr)
        return None

    def _visit_parameters(self, arguments):
        # Defaults resolve in the enclosing scope, before parameters shadow it.
        arguments.defaults = [self.visit(value) for value in arguments.defaults]
        arguments.kw_defaults = [self.visit(value) if value else None for value in arguments.kw_defaults]
        args = arguments.posonlyargs + arguments.args + arguments.kwonlyargs
        names = {arg.arg for arg in (*args, arguments.vararg, arguments.kwarg) if arg}
        return args, names


class ProgramView:
    def __init__(self, files, package=None):
        if package and "." in package:
            raise ValueError(f"Package root {package!r} omits enclosing package initializers. "
                             f"Prepare the outermost package {package.split('.')[0]!r} as --root.")
        self.files = dict(files)  # relative name -> original UTF-8 text
        self.package = package
        self.trees = {name: ast.parse(src, filename=name)
                      for name, src in files.items()}
        self.modules = {name: (name[:-3].replace('/', '.').removesuffix('.__init__')
                               if name != '__init__.py' else '') for name in files}
        self.by_module = {module: name for (name, module) in self.modules.items()}
        if package:
            for name, module in self.modules.items():
                qualified = package + '.' + module if module else package
                if self.by_module.setdefault(qualified, name) != name:
                    raise ValueError('ambiguous staged module: %s' % qualified)
        self.names = {name: {symbol: '__panko_m%d_%s' % (index, symbol)
                             for symbol in _stores(self.trees[name].body)}
                      for index, name in enumerate(sorted(files))}
        self.local_imports, self.module_aliases = {}, {}
        self.edges = {name: set() for name in files}
        self.runtime_imports = {name: {} for name in files}
        self._scan_imports()
        self._order_modules()
        self._build()

    @staticmethod
    def _check_import_alias(statement, protected):
        for node in scope_nodes((statement,)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in protected:
                    raise ValueError('canonical import alias is rebound: %s' % node.name)
                continue
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                if node.id in protected:
                    raise ValueError('canonical import alias is rebound: %s' % node.id)

    @staticmethod
    def _unique_statements(statements, imports):
        for statement in statements:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                key = ast.dump(statement)
                if key in imports:
                    continue
                imports.add(key)
            yield statement

    @classmethod
    def from_source(cls, source):
        """Rebuild the source map from the first line of a nonempty analysis view.

        The V1 or V2 provenance header contains the original module sources.
        A missing prefix or unknown version raises ValueError. Payload decoding
        and module parsing errors propagate. materialize() checks edits to the body.
        """
        first = source.splitlines()[0]
        if not first.startswith(HEADER):
            raise ValueError('not a PANKO program view')
        return _cached_view(first)

    def acquisition_imports(self, original_package):
        """Recover qualified dependencies for legacy views prepared without package metadata."""
        if self.package:
            return self.runtime_imports
        if original_package:
            return ProgramView(self.files, original_package).runtime_imports
        suffixes = tuple('.' + module for module in self.modules.values() if module)
        for source in self.files.values():
            for name in _absolute_import_names(source):
                if name not in self.by_module and name.endswith(suffixes):
                    raise ValueError('package namespace unavailable; prepare the view at the original package root')
        return self.runtime_imports

    def materialize(self, candidate):
        """Map an edited view with the original header back to {relative filename: source text}.

        Mapped expressions may be replaced with literals, lists, tuples or supported
        arithmetic on literals. Only those source spans are rewritten, preserving the
        original UTF-8 bytes elsewhere, including comments, spacing and line endings.
        A changed header, unsupported structural edits, or conflicting or overlapping
        replacements raise ValueError. Invalid Python syntax raises SyntaxError.
        """
        if not candidate.startswith(self.header):
            raise ValueError(
                'analysis-view provenance changed; rebuild from the original files')
        changed = ast.parse(candidate[len(self.header):])
        edits = {name: {} for name in self.files}
        allowed = (ast.Expression, ast.Constant, ast.List, ast.Tuple, ast.Load, ast.UnaryOp, ast.USub, ast.UAdd,
                   ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)

        def compare(old, new):
            if isinstance(old, ast.AST) and isinstance(new, ast.AST):
                if ast.dump(old) == ast.dump(new):
                    return
                origin = self.origins.get(id(old))
                if origin and isinstance(new, ast.expr) and all(isinstance(n, allowed) for n in ast.walk(new)):
                    filename, start, end = origin
                    replacement = ast.unparse(new)
                    previous = edits[filename].get((start, end))
                    if previous is not None and previous != replacement:
                        raise ValueError(
                            'conflicting edits to shared declaration')
                    edits[filename][start, end] = replacement
                    return
                if type(old) is not type(new):
                    raise ValueError('TPE changed a nonliteral program node')
                for field, value in ast.iter_fields(old):
                    compare(value, getattr(new, field))
            elif isinstance(old, list) and isinstance(new, list):
                if len(old) != len(new):
                    raise ValueError('TPE changed program structure')
                for a, b in zip(old, new):
                    compare(a, b)
            elif old != new:
                raise ValueError('TPE changed non-shape syntax')
        compare(self.parsed, changed)
        output = dict(self.files)
        for name, changes in edits.items():
            previous = len(output[name]) + 1
            for (start, end), value in sorted(changes.items(), reverse=True):
                if end > previous:
                    raise ValueError('overlapping source-map edits')
                output[name] = output[name][:start] + value + output[name][end:]
                previous = start
            if changes:
                ast.parse(output[name], filename=name)
        return output

    def _scan_imports(self):
        kept_imports = {}
        for name, tree in self.trees.items():
            self.module_aliases[name] = {}
            if self.package:
                tree.body = list(self._split_mixed_imports(tree.body))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._scan_import(name, tree, node, kept_imports)

    def _split_mixed_imports(self, statements):
        """Keep external imports in the view when a statement also imports staged modules."""
        for stmt in statements:
            if isinstance(stmt, ast.Import) and len({a.name in self.by_module for a in stmt.names}) > 1:
                for alias in stmt.names:
                    yield ast.copy_location(ast.Import(names=[alias]), stmt)
            else:
                yield stmt

    def _scan_import(self, name, tree, node, kept_imports):
        target = self._local_import(name, node)
        if target is not None:
            # Function-local imports require execution-order reasoning; fail closed.
            if node not in tree.body:
                raise ValueError('local import inside a scope needs initialization repair: %s:%d' % (name, node.lineno))
            self.local_imports[id(node)] = target
            for _alias, module, symbol in target:
                dependency = self.by_module[module]
                if dependency != name or symbol is None or name.rsplit('/', 1)[-1] != '__init__.py':
                    self.edges[name].add(dependency)
                self._bind_runtime_import(name, _runtime_import_name(node, module, symbol), dependency)
            return
        if isinstance(node, ast.ImportFrom) and node.level:
            raise ValueError('relative import leaves the program manifest: %s:%d' % (name, node.lineno))
        for alias in node.names:
            if alias.name == '*':
                raise ValueError('star imports need explicit bindings: %s:%d' % (name, node.lineno))
            bound = alias.asname or (alias.name.split('.')[0] if isinstance(node, ast.Import) else alias.name)
            if node in tree.body and bound in DSL_IMPORTS:
                self._keep_runtime_import(name, node, alias, bound, kept_imports)

    def _bind_runtime_import(self, name, lookup, dependency):
        imports = self.runtime_imports[name]
        imports[lookup] = dependency
        if self.package and not lookup.startswith('.'):
            parent = lookup.rpartition('.')[0]
            while parent:
                initializer = self.by_module.get(parent)
                if initializer and initializer.rsplit('/', 1)[-1] == '__init__.py':
                    imports[parent] = initializer
                parent = parent.rpartition('.')[0]

    def _keep_runtime_import(self, name, node, alias, bound, kept_imports):
        identity = (type(node).__name__, getattr(node, 'module', None), alias.name)
        if bound in kept_imports and kept_imports[bound] != identity:
            raise ValueError('conflicting DSL/runtime import alias %s' % bound)
        kept_imports[bound] = identity
        self.names[name][bound] = bound

    def _resolve_module_aliases(self, name, tree):
        """Resolve re-exports in execution order, retaining each statement's visible aliases."""
        protected = {symbol for symbol, mapped in self.names[name].items() if symbol == mapped}
        aliases = self.module_aliases[name]
        before = []
        for node in tree.body:
            self._check_import_alias(node, protected)
            before.append(aliases.copy())
            imports = self.local_imports.get(id(node))
            if imports is None:
                for bound in _stores([node]):
                    aliases.pop(bound, None)
                continue
            self._apply_import_aliases(imports, aliases)
        return before

    def _apply_import_aliases(self, imports, aliases):
        for i, (alias, module, symbol) in enumerate(imports):
            if symbol is not None:
                source_aliases = self.module_aliases[self.by_module[module]]
                if symbol not in source_aliases:
                    aliases.pop(alias, None)
                    continue
                module = source_aliases[symbol]
                imports[i] = (alias, module, None)
            aliases[alias] = module

    def _order_modules(self):
        self.order = []
        visiting, done = set(), set()

        def visit(name):
            if name in visiting:
                raise ValueError(
                    'cyclic local imports need explicit initialization: %s' % name)
            if name in done:
                return
            visiting.add(name)
            for dependency in sorted(self.edges[name]):
                visit(dependency)
            visiting.remove(name)
            done.add(name)
            self.order.append(name)
        for name in sorted(self.files):
            visit(name)

    def _local_import(self, name, node):
        package = self.modules[name].split(
            '.')[:-1] if not name.endswith('__init__.py') else self.modules[name].split('.')
        package = [p for p in package if p]
        if isinstance(node, ast.ImportFrom):
            base = list(package)
            if node.level:
                if node.level - 1 > len(base):
                    return None
                base = base[:len(base) - node.level + 1]
                module = '.'.join(
                    base + ([node.module] if node.module else []))
            else:
                module = node.module or ''
            targets = self._from_targets(name, node, module)
            if targets is not None:
                return targets
            if node.level:
                raise ValueError('unresolved local import: %s:%d' %
                                 (name, node.lineno))
            if self.package and module in self.by_module:
                raise ValueError('unresolved staged export from %s' % module)
        else:
            if all(a.name in self.by_module and (self.package or a.asname or '.' not in a.name) for a in node.names):
                return [(a.asname or a.name, a.name, None) for a in node.names]
            if self.package and any(a.name in self.by_module for a in node.names):
                raise ValueError('local import inside a scope needs initialization repair: %s:%d' % (name, node.lineno))
        return None

    def _from_targets(self, name, node, module):
        owner = self.by_module.get(module)
        if node.level and owner and self.modules[owner] != module:
            owner = None
        exports = self.names.get(owner, {})
        if owner == name and name.rsplit('/', 1)[-1] == '__init__.py':
            body = self.trees[name].body
            if node in body:
                exports = _prior_package_exports(body, node)
        targets = []
        for alias in node.names:
            bound = alias.asname or alias.name
            child = '.'.join(filter(None, (module, alias.name)))
            if alias.name in exports:
                targets.append((bound, module, alias.name))
            elif child in self.by_module and (not node.level or self.modules[self.by_module[child]] == child):
                targets.append((bound, child, None))
            else:
                return None
        return targets

    def _build(self):
        body = []
        imports = set()
        for filename in self.order:
            tree = self.trees[filename]
            offsets = Offsets(self.files[filename])
            for node in ast.walk(tree):
                if isinstance(node, ast.expr) and hasattr(node, 'end_lineno'):
                    node.origin = (
                        filename, *offsets.span(node))
            # Transform each node once to preserve its original span.
            rename = _Rename(self, filename)
            aliases_before = self._resolve_module_aliases(filename, tree)
            for statement, aliases in zip(tree.body, aliases_before):
                rename.aliases = aliases
                transformed = rename.visit(statement)
                statements = transformed if isinstance(transformed, list) else [transformed]
                body.extend(self._unique_statements(statements, imports))
        self.template = ast.fix_missing_locations(
            ast.Module(body=body, type_ignores=[]))
        self.code = ast.unparse(self.template) + '\n'
        self.parsed = ast.parse(self.code)
        self.origins = {}

        self._pair_origins(self.template, self.parsed)
        payload = {name: base64.b64encode(src.encode(
            'utf-8')).decode('ascii') for name, src in sorted(self.files.items())}
        version = 'V1 '
        if self.package:
            version = 'V2 '
            payload = {'package': self.package, 'files': payload}
        self.header = HEADER + version +\
            base64.b64encode(json.dumps(
                payload, sort_keys=True).encode()).decode('ascii') + '\n'
        self.source = self.header + self.code

    def _pair_origins(self, left, right):
        if not isinstance(left, ast.AST):
            return
        origin = getattr(left, 'origin', None)
        if origin:
            self.origins[id(right)] = origin
        for field, value in ast.iter_fields(left):
            actual = getattr(right, field)
            if isinstance(value, ast.AST):
                self._pair_origins(value, actual)
            elif isinstance(value, list):
                self._pair_sequence(value, actual)

    def _pair_sequence(self, source, parsed):
        for left, right in zip(source, parsed):
            self._pair_origins(left, right)


@lru_cache(maxsize=4)
def _cached_view(header):
    version, _, encoded = header[len(HEADER):].partition(' ')
    payload = json.loads(base64.b64decode(encoded, validate=True))
    package = None
    if version == 'V2':
        package, payload = payload['package'], payload['files']
    elif version != 'V1':
        raise ValueError('unknown program view version: %s' % version)
    return ProgramView({name: base64.b64decode(data, validate=True).decode('utf-8') for name, data in payload.items()},
                       package)
