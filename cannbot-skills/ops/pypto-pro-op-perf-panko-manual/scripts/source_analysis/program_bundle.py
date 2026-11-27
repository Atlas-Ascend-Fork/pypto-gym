# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Checkpoint and restore a local Python package as one optimization program."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import ast
import base64
import json
import os

from bootstrap import activate
activate()
import jsonio
from source_analysis.runtime_files import cached_bytecode, check_files, digest, file_digest, stage_bytes
from source_analysis.kernel_boundary import Boundary


def inventory(root):
    files = []
    for path in root.rglob('*.py'):
        relative = path.relative_to(root)
        if relative.parts[0] == 'optimization' or {'__pycache__', '.git'} & set(relative.parts):
            continue
        files.append(str(relative))
    return sorted(files)


def member(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or rel.suffix != '.py':
        raise ValueError('program member must be a relative Python file: %s' % relative)
    path = root / rel
    if any(p.is_symlink() for p in [path, *list(path.parents)[:len(rel.parts) - 1]]):
        raise ValueError('program members cannot traverse symlinks: %s' % relative)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('program member escapes root: %s' % relative)
    return path


def plan(root):
    """Scan an existing package directory for Python files and JIT entry points.

    The returned manifest includes relative paths, SHA-256 hashes and target scopes.
    Files with syntax or decoding errors remain in the inventory with a diagnostic
    and contribute no targets. Unsafe member paths raise ValueError, and errors
    resolving the root or reading a file propagate to the caller.
    """
    root = Path(root).resolve(strict=True)
    files = inventory(root)
    targets, diagnostics = [], []
    for relative in files:
        path = member(root, relative)
        try:
            tree = ast.parse(path.read_bytes(), filename=relative)
        except (SyntaxError, UnicodeError) as exc:
            diagnostics.append({'file': relative, 'reason': str(exc)})
            continue
        boundary = Boundary(tree)
        for entry in sorted(boundary.entries, key=lambda node: node.lineno):
            scope = boundary.scope_paths[entry]
            targets.append({'file': relative, 'entry': scope,
                            'case_context': {'__panko_active_scopes__': [scope]}})
    return {'version': 1, 'root': str(root), 'files': files,
            'sha256': {rel: file_digest(member(root, rel)) for rel in files},
            'targets': targets, 'diagnostics': diagnostics}


def _view_path(root, path):
    path = Path(path).absolute()
    if path.is_symlink():
        raise ValueError('analysis view cannot be a symlink')
    path = path.resolve()
    if path.is_relative_to(root) and path.relative_to(root).parts[:1] != ('optimization',):
        raise ValueError('place the analysis view outside the package or under optimization/')
    return path


def package_namespace(root):
    """Return the import prefix at the original root, including enclosing Python packages."""
    if not root.name.isidentifier():
        return None
    parts = [root.name]
    for parent in root.parents:
        if not parent.name.isidentifier() or not (parent / '__init__.py').is_file():
            break
        parts.append(parent.name)
    return '.'.join(reversed(parts))


def prepare(root, view_path, required=None):
    """Build a joint view without changing a byte in any runtime source file."""
    from source_analysis.program_view import ProgramView
    manifest = plan(root)
    if manifest['diagnostics']:
        raise ValueError('program contains unparseable sources: %s' % manifest['diagnostics'])
    root = Path(manifest['root'])
    path = _view_path(root, view_path)
    if path.exists():
        raise ValueError('analysis view already exists; use refresh for an existing campaign')
    package = package_namespace(root)
    view = ProgramView({rel: member(root, rel).read_bytes().decode('utf-8') for rel in manifest['files']}, package)
    for target in manifest['targets']:
        head, *tail = target['entry'].split('/')
        target['view_entry'] = '/'.join([view.names[target['file']][head], *tail])
        target['case_context'] = {'__panko_active_scopes__': [target['view_entry']]}
    required = sorted(set(required or [t['file'] for t in manifest['targets']]))
    if not required or not set(required) <= set(manifest['files']):
        raise ValueError('required runtime entry files must be present in the program')
    manifest.update(version=2, view=str(path), package=package, required=required, symbol_map=view.names,
                    view_sha256=digest(view.source.encode('utf-8')))
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_runtime(_view_writes(manifest, view.source, view.files))
    return manifest


def _materialized(program, source):
    from source_analysis.program_view import ProgramView
    files = ProgramView.from_source(source).materialize(source)
    if sorted(files) != sorted(program['files']):
        raise ValueError('analysis-view inventory differs from campaign')
    return files


def _view_writes(program, source, files):
    root = Path(program['root'])
    spec = {'root': str(root.resolve()), 'view_sha256': digest(source.encode('utf-8')),
            'files': {str(member(root, rel).resolve()): digest(src.encode('utf-8')) for rel, src in files.items()},
            'required': [str(member(root, rel).resolve()) for rel in program['required']]}
    return [(Path(program['view']), source.encode('utf-8')),
            (Path(str(program['view']) + '.runtime.json'), json.dumps(spec, sort_keys=True).encode('utf-8'))]


def joint_drift(program):
    try:
        root = Path(program['root'])
        if inventory(root) != sorted(program['files']):
            return 'program Python file inventory changed'
        source = Path(program['view']).read_text(encoding='utf-8')
        files = _materialized(program, source)
        for rel, src in files.items():
            if member(root, rel).read_bytes() != src.encode('utf-8'):
                return 'program view is stale for %s; refresh after structural edits' % rel
        runtime_spec(program['view'])
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, IndexError) as exc:
        return 'program view is invalid: %s' % exc
    return ''


def _rollback_files(committed, retain, failure):
    errors = []
    for path, old in reversed(committed):
        try:
            if old is None:
                path.unlink(missing_ok=True)
            else:
                os.replace(old, path)
        except OSError as exc:
            if old:
                retain.add(old)
            errors.append(f'{path}: {exc}; backup={old}')
    if errors:
        raise OSError('program rollback failed; ' + '; '.join(errors)) from failure


def _cleanup_staged_files(staged, retain):
    for _path, new, old in staged:
        for temporary in (new, old):
            if temporary is not None and temporary not in retain:
                temporary.unlink(missing_ok=True)


def _replace_files(writes):
    """Roll back failed replacements. Runtime hashes detect mixed files after a crash between renames."""
    staged, committed = [], []
    retain = set()
    try:
        for path, data in writes:
            new = stage_bytes(path, data)
            staged.append((path, new, None))
            old = stage_bytes(path, path.read_bytes()) if path.exists() else None
            staged[-1] = (path, new, old)
        for path, new, old in staged:
            os.replace(new, path)
            committed.append((path, old))
    except BaseException as failure:
        _rollback_files(committed, retain, failure)
        raise
    finally:
        _cleanup_staged_files(staged, retain)


def _write_runtime(writes):
    changed = [(path, data) for path, data in writes if not path.exists() or path.read_bytes() != data]
    cached = [item for path, _ in changed for item in cached_bytecode(path)]
    for item in cached:
        item.unlink(missing_ok=True)
    _replace_files(changed)


def write_candidate(program, source, *, exporting=False):
    """Validate all mappings before writing a TPE candidate to runtime modules."""
    root = Path(program['root'])
    existing = set(inventory(root))
    expected = set(program['files'])
    if existing - expected or (not exporting and existing != expected):
        raise ValueError('program inventory changed')
    files = _materialized(program, source)
    writes = [(member(root, rel), src.encode('utf-8')) for rel, src in files.items()]
    _write_runtime(writes + _view_writes(program, source, files))


def refresh(manifest_path):
    """Rebuild analysis after coordinated structural edits to the runtime files."""
    return refresh_program(json.loads(Path(manifest_path).read_text(encoding='utf-8')))


def refresh_program(program, *, preserve_current=False):
    from source_analysis.program_view import ProgramView
    if program.get('version') != 2:
        raise ValueError('refresh requires a joint program manifest')
    root = Path(program['root'])
    if inventory(root) != sorted(program['files']):
        raise ValueError('program inventory changed')
    files = {rel: member(root, rel).read_bytes().decode('utf-8') for rel in program['files']}
    if preserve_current:
        source = Path(program['view']).read_text(encoding='utf-8')
        if _materialized(program, source) == files:
            return {'view': program['view'], 'code_hash': digest(source.encode('utf-8'))}
    view = ProgramView(files, program['package'] if 'package' in program else package_namespace(root))
    _write_runtime(_view_writes(program, view.source, view.files))
    return {'view': program['view'], 'code_hash': digest(view.source.encode('utf-8'))}


def runtime_spec(view_path):
    """Validate staged runtime identity; a missing/stale manifest is a refusal."""
    path = Path(view_path)
    source = path.read_text(encoding='utf-8')
    from source_analysis.program_view import HEADER, ProgramView
    if not source.startswith(HEADER):
        return None
    spec = json.loads(Path(str(path) + '.runtime.json').read_text(encoding='utf-8'))
    if spec['view_sha256'] != digest(source.encode('utf-8')):
        raise ValueError('runtime identity is stale')
    if not spec['required'] or not set(spec['required']) <= set(spec['files']):
        raise ValueError('invalid runtime entry identity')
    # Also bind the sidecar to the source-map payload, not just a self-reported hash.
    materialized = ProgramView.from_source(source).materialize(source)
    expected_files = {str(member(Path(spec['root']), rel).resolve()): digest(
        src.encode('utf-8')) for rel, src in materialized.items()}
    if spec['files'] != expected_files:
        raise ValueError('runtime identities differ from analysis view')
    check_files(spec['files'], 'runtime source changed')
    return spec


def attached(view_path):
    """Recover the joint staging contract for direct evaluator API callers."""
    spec = runtime_spec(view_path)
    if not spec:
        return None
    from source_analysis.program_view import ProgramView
    view = ProgramView.from_source(Path(view_path).read_text(encoding='utf-8'))
    root = Path(spec['root'])
    return {'version': 2, 'root': str(root), 'view': str(Path(view_path).resolve()),
            'package': view.package or package_namespace(root), 'files': sorted(view.files),
            'required': [str(Path(p).relative_to(root)) for p in spec['required']]}


def require_current(program):
    reason = joint_drift(program)
    if reason:
        raise ValueError(reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'prepare', 'refresh'])
    parser.add_argument('--root')
    parser.add_argument('--view')
    parser.add_argument('--manifest')
    parser.add_argument('--entry-file', action='append')
    args = parser.parse_args()
    if args.action == 'refresh':
        if not args.manifest:
            parser.error('refresh requires --manifest')
        result = refresh(args.manifest)
    elif args.action == 'prepare':
        if not args.root or not args.view or not args.manifest:
            parser.error('prepare requires --root, --view and --manifest')
        result = prepare(args.root, args.view, args.entry_file)
        Path(args.manifest).write_text(json.dumps(result, indent=2), encoding='utf-8')
    else:
        if not args.root:
            parser.error('plan requires --root')
        result = plan(args.root)
    jsonio.emit(result, indent=2)


if __name__ == '__main__':
    main()
