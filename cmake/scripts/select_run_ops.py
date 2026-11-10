# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resolve the .run operator scope shared by CMake and system tests."""
import argparse
import itertools
import logging
from pathlib import Path, PurePosixPath
import re
import sys


_PATH_PART = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


def _available_operators(root):
    available = {}
    for paradigm, depth in itertools.product(("pro", "tensor"), (1, 2, 3)):
        pattern = "/".join(["*"] * depth + ["CMakeLists.txt"])
        for cmake in (root / "ops" / paradigm).glob(pattern):
            op = cmake.parent
            if (op / "op_host").is_dir():
                available.setdefault(op.name, []).append(op)
    return available


def _load_manifest(manifest):
    try:
        import yaml
    except ImportError as error:
        raise ValueError("reading build_ops.yaml requires PyYAML; install PyYAML>=6.0") from error
    try:
        with manifest.open(encoding="utf-8") as stream:
            entries = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"{manifest}: {error}") from error
    if entries is None:
        entries = []
    if not isinstance(entries, list) or any(not isinstance(entry, str) or not entry for entry in entries):
        raise ValueError(f"{manifest}: expected a YAML list of nonempty operator path strings")
    if not entries:
        raise ValueError(f"{manifest}: no operators listed; add at least one uncommented path")
    return entries


def _select_from_manifest(root, available, manifest):
    selected = []
    for entry in _load_manifest(manifest):
        relative = PurePosixPath(entry)
        if (relative.is_absolute() or entry != relative.as_posix() or
                any(part in (".", "..") or not _PATH_PART.fullmatch(part) for part in relative.parts)):
            raise ValueError(f"{manifest}: invalid path relative to ops/pro: {entry!r}")
        op = root / "ops/pro" / relative
        if op not in available.get(op.name, []):
            raise ValueError(f"{manifest}: {entry!r} has no binary build directory "
                             "(requires operator CMakeLists.txt and op_host/)")
        selected.append(op)
    return selected


def _select_from_names(available, requested):
    names = [name.strip() for name in requested.replace(";", ",").split(",")]
    if not all(names):
        raise ValueError("--ops must be ALL or a nonempty comma-separated operator list")
    selected = []
    for name in names:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid operator directory name: {name!r}")
        candidates = available.get(name, [])
        if not candidates:
            raise ValueError(f"operator {name!r} has no binary build directory "
                             "(requires operator CMakeLists.txt and op_host/)")
        if len(candidates) > 1:
            raise ValueError(f"ambiguous operator {name!r}: {candidates}")
        selected.append(candidates[0])
    return selected


def select_run_op_dirs(source_root, requested="ALL"):
    root = Path(source_root).resolve()
    requested = requested.strip()
    available = _available_operators(root)
    if requested.upper() == "ALL":
        selected = _select_from_manifest(root, available, root / "ops/pro/build_ops.yaml")
    else:
        selected = _select_from_names(available, requested)
    selected = sorted(set(selected))
    names = [op.name for op in selected]
    if len(names) != len(set(names)):
        raise ValueError("selected operator directories must have unique operator names")
    return selected


def select_run_ops(source_root, requested="ALL"):
    return [op.name for op in select_run_op_dirs(source_root, requested)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--ops", default="ALL")
    parser.add_argument("--format", choices=("names", "paths"), default="names")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    try:
        selected = select_run_op_dirs(args.source_root, args.ops)
    except ValueError as error:
        parser.error(str(error))
    if args.format == "paths":
        root = args.source_root.resolve()
        logging.info("%s", "\n".join(op.relative_to(root).as_posix() for op in selected))
    else:
        logging.info("%s", "\n".join(op.name for op in selected))


if __name__ == "__main__":
    main()
