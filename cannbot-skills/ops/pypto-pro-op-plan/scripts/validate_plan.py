#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate PyPTO-Pro specifications, material evidence and current kernel references."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import logging
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Optional
from urllib.parse import unquote


OPS_ROOT = Path(__file__).resolve().parents[2]
MATERIAL_ROOT = OPS_ROOT / "pypto-pro-material-explore"
KERNEL_ROOT = OPS_ROOT / "pypto-pro-kernel"
KERNEL_PREFIX = "pypto-pro-kernel/"
PLACEHOLDER_RE = re.compile(r"\{[^{}\r\n]+\}")
LOGGER = logging.getLogger(__name__)


class PlanConfigurationError(RuntimeError):
    pass


def _read(path: Path, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not text.strip():
        raise ValueError(f"{label} is empty")
    return text


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path) if path.is_file() else None
    if spec is None or spec.loader is None:
        raise PlanConfigurationError(f"missing checker dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise PlanConfigurationError(f"cannot load {path}: {error}") from error
    return module


def _check_spec(op_dir: Path, validator: Any):
    try:
        path = op_dir / "SPEC.md"
        text = _read(path, "SPEC.md")
        op_name = validator.load_spec_contract(path)["op_name"]
        errors = [] if op_name == op_dir.name else [
            f"op_name={op_name!r}, directory={op_dir.name!r}"
        ]
        matches = list(re.finditer(r"(?m)^## kernel 契约补充\s*$", text))
        if len(matches) != 1 or not text[matches[0].end():].strip():
            errors.append("kernel contract section must occur once with content")
        return op_name, errors
    except (AttributeError, KeyError, OSError, ValueError) as error:
        return op_dir.name, [str(error)]


def _check_index(op_dir: Path, devkit: Path, material: Any):
    try:
        expected, _ = material.build_index(devkit, material.manifest_path())
    except Exception as error:
        raise PlanConfigurationError(f"cannot build canonical material index: {error}") from error
    try:
        actual = _read(op_dir / "PRO_MATERIAL_INDEX.md", "PRO_MATERIAL_INDEX.md")
    except ValueError as error:
        return expected, [str(error)]
    return expected, [] if actual == expected else ["PRO_MATERIAL_INDEX.md is stale"]


def _frontmatter(text: str) -> dict[str, str]:
    """Read the template's flat single-line scalars, not general YAML."""
    match = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)", text, re.DOTALL)
    if match is None:
        raise ValueError("EXPLORE_REPORT.md has no leading frontmatter")
    result = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        key = key.strip()
        if not separator or not key or key in result:
            raise ValueError(f"invalid frontmatter line: {line!r}")
        scalar = re.fullmatch(
            r'''("(?:[^"\\]|\\.)*"|'(?:[^']|'')*'|[^'"#].*?)(?:[ \t]+#.*)?''',
            value.strip(),
        )
        if scalar is None:
            raise ValueError(f"invalid frontmatter scalar: {line!r}")
        value = scalar.group(1)
        if value.startswith('"'):
            value = json.loads(value)
        elif value.startswith("'"):
            value = value[1:-1].replace("''", "'")
        if not value.strip():
            raise ValueError(f"empty frontmatter value: {key}")
        result[key] = value
    return result


def _file_paths(text: str) -> set[str]:
    """Extract backtick file paths, retaining suffixes and possible anchors."""
    return {
        path for path in re.findall(r"`([^`\r\n]+)`", text)
        if re.match(r"^\S*[/\\]", path) and re.search(r"[^/\\]\.[^/\\#]+(?:#.*)?$", path)
    }


def _path_error(label: str, paths: list[str], reason: str) -> str:
    shown = ", ".join(paths[:8])
    remainder = f" (+{len(paths) - 8} more)" if len(paths) > 8 else ""
    return f"{label} {reason} ({len(paths)}): {shown}{remainder}"


def _check_citations(label: str, prefix: str, cited: set[str], allowed: set[str]) -> list[str]:
    errors = []
    if label != "API":
        missing = []
        for source in allowed - cited:
            if source.startswith(prefix) and not any(path.startswith(source + "#") for path in cited):
                missing.append(source)
        if missing:
            errors.append(_path_error(label, sorted(missing), "paths not covered in report"))
    # Match anchors against complete indexed names; filenames may contain '#'.
    extra = []
    for path in cited - allowed:
        if not any(path.startswith(source + "#") for source in allowed):
            extra.append(path)
    if extra:
        errors.append(_path_error(label, sorted(extra), "referenced paths absent from material index"))
    return errors


def check_report(op_dir: Path, op_name: str, index: str) -> list[str]:
    """Check an explore report against the template and canonical material index."""
    try:
        report = _read(op_dir / "EXPLORE_REPORT.md", "EXPLORE_REPORT.md")
        template = _read(MATERIAL_ROOT / "templates/explore_report.md", "report template")
        actual_meta, template_meta = _frontmatter(report), _frontmatter(template)
    except ValueError as error:
        return [str(error)]
    errors = []
    if actual_meta.get("schema_version") != template_meta.get("schema_version"):
        errors.append("frontmatter schema_version does not match template")
    if actual_meta.get("op_name") != op_name:
        errors.append(f"frontmatter op_name={actual_meta.get('op_name')!r}, expected {op_name!r}")
    if not actual_meta.get("feasibility") or "{" in actual_meta["feasibility"]:
        errors.append("frontmatter feasibility is missing")
    elif actual_meta["feasibility"].strip().lower() in {"unsupported", "不可行", "不支持"}:
        errors.append("frontmatter feasibility has an unresolved unsupported blocker")
    expected = [line for line in template.splitlines() if line.startswith("## ")]
    actual = [line for line in report.splitlines() if line.startswith("## ")]
    if actual != expected:
        errors.append("level-2 headings must exactly match template order")
    remaining = sorted(set(PLACEHOLDER_RE.findall(template)) & set(PLACEHOLDER_RE.findall(report)))
    if remaining:
        errors.append("unresolved template placeholders: " + ", ".join(remaining[:8]))
    paths = _file_paths(report)
    categories = (("sample", "pro_ops/"), ("guide", "docs/guide/"), ("API", "docs/pypto_pro/api/"))
    rooted = r"(?:/|\.\.?/|[A-Za-z]:/|\$(?:\w+|\{[^}]+\})/)"
    allowed = _file_paths(index)
    # Check devkit citations and required sample/guide coverage across the report.
    # Reading and applicability follow the Material Skill and semantic review.
    # Directories and commands are not evidence. Kernel guidance is checked
    # separately against its own installed root, not the devkit material index.
    for label, prefix in categories:
        cited = set()
        for path in paths:
            before, found, _ = path.replace("\\", "/").partition(prefix)
            # Preserve spaces in paths, but exclude command prefixes and
            # separate rooted arguments such as `/usr/bin/cat /cache/...`.
            if not found:
                continue
            if before and (not before.endswith("/") or not re.match(rooted, before)):
                continue
            if re.search(r"\s+" + rooted, before):
                continue
            cited.add(path)
        errors.extend(_check_citations(label, prefix, cited, allowed))
    return errors


def _kernel_link_checker(kernel_root: Path):
    """Reuse the current kernel's Markdown reader and anchor rules."""
    package_root = kernel_root / "scripts/_checks"
    name = "_plan_kernel_" + hashlib.sha256(str(kernel_root).encode()).hexdigest()[:16]
    if name not in sys.modules:
        _module(name, package_root / "__init__.py")
    try:
        return importlib.import_module(name + ".links")
    except Exception as error:
        raise PlanConfigurationError(f"cannot load kernel link checker: {error}") from error


def _collect_kernel_references(report: str, links: Any) -> tuple[set[str], list[str]]:
    """Separate portable owner citations from shell excerpts and invalid prefixes."""
    references = set()
    errors = []
    for _, line in links.prose_lines(report):
        candidates = {target: False for target in _file_paths(line)}
        candidates.update((target, True) for target in links.inline_targets(line))
        for target, explicit_link in candidates.items():
            if KERNEL_PREFIX not in target or "://" in target:
                continue
            before = target.partition(KERNEL_PREFIX)[0]
            # A shell excerpt is not a citation. Rooted or relative paths to
            # the skill must use the portable named-root form documented here.
            if before and not explicit_link and any(char.isspace() for char in before):
                continue
            if before:
                errors.append(f"kernel reference must start with {KERNEL_PREFIX}: {target}")
            else:
                references.add(target)
    return references, errors


def _check_kernel_fragment(reference: str, target: Path, fragment: str, links: Any) -> list[str]:
    if target.suffix.lower() != ".md":
        return [f"kernel fragment requires a Markdown owner: {reference}"]
    try:
        if not fragment or fragment not in links.anchors(target.read_text(encoding="utf-8")):
            return [f"kernel referenced anchor does not exist: {reference}"]
    except (OSError, UnicodeError) as error:
        return [f"cannot read kernel reference {reference}: {error}"]
    return []


def _check_kernel_reference(reference: str, kernel_root: Path, links: Any) -> list[str]:
    """Validate one owner path and its optional Markdown fragment."""
    relative, separator, fragment = reference[len(KERNEL_PREFIX):].partition("#")
    relative, fragment = unquote(relative), unquote(fragment)
    pure = PurePosixPath(relative)
    unsafe_path = pure.is_absolute() or ".." in pure.parts or "\\" in relative
    if not relative or unsafe_path:
        return [f"kernel reference escapes its root: {reference}"]
    try:
        target = (kernel_root / relative).resolve()
    except (OSError, ValueError) as error:
        return [f"invalid kernel reference {reference}: {error}"]
    if not target.is_relative_to(kernel_root.resolve()):
        return [f"kernel reference resolves outside its root: {reference}"]
    if not target.is_file():
        return [f"kernel referenced file does not exist: {reference}"]
    if separator:
        return _check_kernel_fragment(reference, target, fragment, links)
    return []


def check_kernel_references(op_dir: Path, kernel_root: Path) -> list[str]:
    """Check direct owner citations without maintaining a second knowledge manifest."""
    try:
        report = _read(op_dir / "EXPLORE_REPORT.md", "EXPLORE_REPORT.md")
    except ValueError as error:
        return [str(error)]
    links = _kernel_link_checker(kernel_root)
    references, errors = _collect_kernel_references(report, links)
    if not references:
        return errors + ["report must cite an applicable pypto-pro-kernel owner"]
    for reference in sorted(references):
        errors.extend(_check_kernel_reference(reference, kernel_root, links))
    return errors


def validate(op_dir: Path, devkit: Path, kernel_root: Path = KERNEL_ROOT):
    for path, label in ((op_dir, "operator"), (devkit, "devkit"), (kernel_root, "kernel root")):
        if not path.is_dir():
            raise PlanConfigurationError(f"{label} directory does not exist: {path}")
    intent = _module(
        "_plan_spec", OPS_ROOT / "pypto-pro-intent-understand/scripts/validate_spec.py",
    )
    material = _module(
        "_plan_material", MATERIAL_ROOT / "scripts/build_material_index.py",
    )
    op_name, spec_errors = _check_spec(op_dir, intent)
    index, index_errors = _check_index(op_dir, devkit, material)
    return [
        ("SPEC", spec_errors),
        ("INDEX", index_errors),
        ("REPORT", check_report(op_dir, op_name, index)),
        ("REFERENCES", check_kernel_references(op_dir, kernel_root)),
    ]


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op-dir", required=True)
    parser.add_argument("--devkit", required=True)
    parser.add_argument("--kernel-root", default=str(KERNEL_ROOT),
                        help="current pypto-pro-kernel skill root (defaults to its sibling installation)")
    args = parser.parse_args(argv)
    try:
        results = validate(*(
            Path(value).resolve() for value in (args.op_dir, args.devkit, args.kernel_root)
        ))
    except PlanConfigurationError as error:
        LOGGER.error("ERROR: %s", error)
        return 2
    failed = False
    for name, errors in results:
        failed = failed or bool(errors)
        for error in errors:
            LOGGER.info("[FAIL] %s: %s", name, error)
        if not errors:
            LOGGER.info("[PASS] %s", name)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
