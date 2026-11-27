# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Hash frozen inputs and replace runtime sources with fresh bytecode."""
import hashlib
import os
import stat
import tempfile
from pathlib import Path


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_files(manifest, reason):
    for path, expected in manifest.items():
        if file_digest(path) != expected:
            raise ValueError(f"{reason}: {path}")


def cached_bytecode(path):
    cache = path.parent / '__pycache__'
    if cache.is_symlink():
        raise ValueError('program bytecode cache cannot be a symlink: %s' % cache)
    entries = cache.iterdir() if cache.is_dir() else ()
    cached = [p for p in entries if p.name.startswith(path.stem + '.') and p.suffix == '.pyc']
    legacy = path.with_suffix('.pyc')
    if legacy.exists() or legacy.is_symlink():
        cached.append(legacy)
    for item in cached:
        if not item.is_file() and not item.is_symlink():
            raise ValueError('invalid program bytecode cache: %s' % item)
    return cached


def stage_bytes(path, data):
    if path.is_symlink():
        raise ValueError(f"program output cannot be a symlink: {path}")
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.panko-',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644)
        return temporary
    except BaseException:
        if temporary:
            temporary.unlink(missing_ok=True)
        raise


def write_runtime_file(path, data):
    """Replace one source atomically. Invalidate bytecode even for same-size, same-tick edits."""
    if path.exists() and path.read_bytes() == data:
        return
    cached = cached_bytecode(path)
    temporary = stage_bytes(path, data)
    try:
        for item in cached:
            item.unlink(missing_ok=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
