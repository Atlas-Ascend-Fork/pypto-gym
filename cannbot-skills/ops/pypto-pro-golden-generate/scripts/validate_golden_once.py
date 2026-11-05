#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run each current Stage-2 Golden once and verify its hash-bound receipt."""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional


RECEIPT_NAME = "GOLDEN_VALIDATION.json"
SCHEMA_VERSION = 2
ROLES = ("cpu", "npu")
LOGGER = logging.getLogger(__name__)
VALIDATION_DRIVER = r"""
import ast
import runpy
import sys
from pathlib import Path

path, *required = sys.argv[1:]
required = set(required)
called = set()
tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
locations = {
    node.name: (
        min([node.lineno, *(item.lineno for item in node.decorator_list)]),
        node.lineno,
    )
    for node in tree.body if isinstance(node, ast.FunctionDef)
}

def is_top_level(frame, name):
    if frame.f_code.co_filename != path or frame.f_code.co_name != name:
        return False
    start, stop = locations[name]
    return start <= frame.f_code.co_firstlineno <= stop

def trace(frame, event, _arg):
    name = frame.f_code.co_name
    if event != "call" or name not in required or not is_top_level(frame, name):
        return None
    caller = frame.f_back
    while caller and not is_top_level(caller, "_validate"):
        caller = caller.f_back
    if name == "_validate" or caller:
        called.add(name)
    return None

code = 0
sys.path[0] = str(Path(path).parent)
sys.argv = [path]
sys.settrace(trace)
try:
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit as error:
        if error.code is not None:
            if isinstance(error.code, int):
                code = error.code
            else:
                sys.stderr.write(str(error.code) + "\n")
                code = 1
finally:
    sys.settrace(None)
missing = sorted(required - called)
if missing:
    sys.stderr.write("validation did not execute: " + ", ".join(missing) + "\n")
    code = code or 1
raise SystemExit(code)
"""


class ReceiptError(ValueError):
    """Validation inputs or receipt data are not trustworthy."""


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReceiptError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    except (json.JSONDecodeError, OSError, UnicodeError) as error:
        raise ReceiptError(f"invalid {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ReceiptError(f"{path.name} root must be an object")
    return value


def _op_name(spec_path: Path) -> str:
    validator = Path(__file__).resolve().parents[2] / (
        "pypto-pro-kernel/scripts/validate_spec.py"
    )
    if not validator.is_file():
        raise ReceiptError(f"SPEC validation requires pypto-pro-kernel: {validator}")
    spec = importlib.util.spec_from_file_location("_pypto_spec_validator", validator)
    if spec is None or spec.loader is None:
        raise ReceiptError(f"cannot load canonical SPEC validator: {validator}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module.load_spec_contract(spec_path)["op_name"]
    except Exception as error:
        raise ReceiptError(f"invalid SPEC.md: {error}") from error


def _framework_imports(tree: ast.Module) -> list[str]:
    """Find direct framework imports, including literal dynamic imports."""
    names: set[str] = set()
    importlib_aliases = {"importlib"}
    loader_aliases = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.add(module)
            names.update(
                f"{module}.{alias.name}" if module else alias.name
                for alias in node.names
            )
            if module == "importlib":
                loader_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names if alias.name == "import_module"
                )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        is_loader = isinstance(function, ast.Name) and function.id in loader_aliases
        is_module_loader = (
            isinstance(function, ast.Attribute)
            and function.attr == "import_module"
            and isinstance(function.value, ast.Name)
            and function.value.id in importlib_aliases
        )
        argument = node.args[0]
        if (is_loader or is_module_loader) and isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            names.add(argument.value)
    return sorted(name for name in names if name.split(".", 1)[0] in {"pypto", "pypto_pro"})


def _golden_contract(path: Path, op_name: str, role: str) -> None:
    """Check direct framework independence and required validation functions."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as error:
        raise ReceiptError(f"invalid {path.name}: {error}") from error
    forbidden = _framework_imports(tree)
    if forbidden:
        raise ReceiptError(f"{path.name} must be independent of PyPTO: {', '.join(forbidden)}")
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    target = f"{op_name}_golden_cpu" if role == "cpu" else f"{op_name}_golden"
    required = {target, "_validate"} | ({"_make_inputs"} if role == "npu" else set())
    missing = sorted(required - set(functions))
    if missing:
        raise ReceiptError(f"{path.name} missing functions: {', '.join(missing)}")


def _snapshot(op_dir: Path, spec_path: Optional[Path] = None):
    if not op_dir.is_dir():
        raise ReceiptError(f"operator directory does not exist: {op_dir}")
    spec_path = spec_path.resolve() if spec_path is not None else op_dir / "SPEC.md"
    op_name = _op_name(spec_path)
    paths = {
        "spec": spec_path,
        "cpu": op_dir / f"{op_name}_golden_cpu.py",
        "npu": op_dir / f"{op_name}_golden.py",
        "receipt": op_dir / RECEIPT_NAME,
    }
    for role in ("spec",) + ROLES:
        path = paths.get(role)
        if not path.is_file():
            raise ReceiptError(f"missing {path.name}")
    for role in ROLES:
        _golden_contract(paths.get(role), op_name, role)
    return op_name, paths, {role: _sha(paths.get(role)) for role in ("spec",) + ROLES}


def _record_valid(record: Any) -> bool:
    if not isinstance(record, dict) or set(record) != {
        "sha256", "exit_code", "duration_ms", "stable",
    }:
        return False
    duration = record.get("duration_ms")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        return False
    code = record.get("exit_code")
    if not isinstance(record.get("sha256"), str) or not isinstance(code, int) or isinstance(code, bool):
        return False
    return duration >= 0 and isinstance(record.get("stable"), bool)


def _record_passes(record: Any, digest: str) -> bool:
    if not _record_valid(record) or record["sha256"] != digest:
        return False
    return record["exit_code"] == 0 and record["stable"]


def _base(op_name: str, spec_hash: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "op_name": op_name, "spec_sha256": spec_hash}


def _load(path: Path, op_name: str, spec_hash: str) -> dict[str, Any]:
    if not path.exists():
        return _base(op_name, spec_hash)
    receipt = _read_json(path)
    if receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("op_name") != op_name:
        raise ReceiptError("receipt schema or operator is invalid")
    if receipt.get("spec_sha256") != spec_hash:
        return _base(op_name, spec_hash)
    if set(receipt) - {"schema_version", "op_name", "spec_sha256", *ROLES}:
        raise ReceiptError("receipt has unknown fields")
    for role in ROLES:
        if role in receipt and not _record_valid(receipt[role]):
            raise ReceiptError(f"invalid {role} validation record")
    return receipt


def _write(path: Path, receipt: dict[str, Any]) -> None:
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def check_receipt(
    op_dir: Path, expected_op: Optional[str] = None, spec_path: Optional[Path] = None,
) -> tuple[bool, str]:
    """Check hashes and exit codes without importing or executing a Golden."""
    try:
        op_name, paths, hashes = _snapshot(op_dir.resolve(), spec_path)
        if expected_op is not None and op_name != expected_op:
            raise ReceiptError(f"SPEC op_name={op_name!r}, expected {expected_op!r}")
        receipt = _load(paths["receipt"], op_name, hashes["spec"])
        if set(receipt) != {"schema_version", "op_name", "spec_sha256", *ROLES}:
            raise ReceiptError("receipt does not contain both validations")
        for role in ROLES:
            if not _record_passes(receipt[role], hashes[role]):
                raise ReceiptError(f"{role} validation is missing, failed, or stale")
        return True, "current SPEC and both Golden hashes have successful validations"
    except (ReceiptError, OSError, UnicodeError) as error:
        return False, str(error)


def run_once(
    op_dir: Path, retry_failed: Optional[str] = None,
    spec_path: Optional[Path] = None, reuse_from: Optional[Path] = None,
) -> int:
    op_dir = op_dir.resolve()
    lock_name = hashlib.sha256(str(op_dir).encode("utf-8")).hexdigest() + ".lock"
    lock_path = Path(tempfile.gettempdir()) / f"pypto-golden-{lock_name}"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _run_locked(op_dir, retry_failed, spec_path, reuse_from)


def _run_locked(
    op_dir: Path, retry_failed: Optional[str],
    spec_path: Optional[Path] = None, reuse_from: Optional[Path] = None,
) -> int:
    op_name, paths, hashes = _snapshot(op_dir, spec_path)
    if reuse_from is not None:
        receipt = _load(reuse_from / RECEIPT_NAME, op_name, hashes["spec"])
        if any(not _record_passes(receipt.get(role), hashes[role]) for role in ROLES):
            raise ReceiptError("candidate receipt does not validate current SPEC and both Golden files")
        _write(paths["receipt"], receipt)
    else:
        try:
            receipt = _load(paths["receipt"], op_name, hashes["spec"])
        except ReceiptError as error:
            LOGGER.info("[RESET] invalid receipt: %s", error)
            receipt = _base(op_name, hashes["spec"])
    for role in ROLES:
        record = receipt.get(role)
        if _record_passes(record, hashes[role]):
            LOGGER.info("[CACHED] %s: %s", role, paths[role].name)
            continue
        if (
            _record_valid(record)
            and record["sha256"] == hashes[role]
            and retry_failed not in (role, "all")
        ):
            LOGGER.error("[FAIL] %s: unchanged failure; use --retry-failed %s", role, role)
            return 1
        start = time.perf_counter()
        target = f"{op_name}_golden_cpu" if role == "cpu" else f"{op_name}_golden"
        required = ["_validate", target] + (["_make_inputs"] if role == "npu" else [])
        result = subprocess.run(
            [sys.executable, "-c", VALIDATION_DRIVER, str(paths[role]), *required],
            check=False,
        )
        stable = _sha(paths[role]) == hashes[role]
        receipt[role] = {
            "sha256": hashes[role],
            "exit_code": result.returncode,
            "duration_ms": round((time.perf_counter() - start) * 1000, 3),
            "stable": stable,
        }
        _write(paths["receipt"], receipt)
        if result.returncode != 0 or not stable:
            LOGGER.error("[FAIL] %s: exit_code=%s, stable=%s", role, result.returncode, stable)
            return 1
        LOGGER.info("[PASS] %s: exit_code=0, duration_ms=%s", role, receipt[role]["duration_ms"])
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op-dir", required=True)
    parser.add_argument("--spec", type=Path, help="SPEC path; defaults to <op-dir>/SPEC.md")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--retry-failed", choices=("cpu", "npu", "all"))
    mode.add_argument("--reuse-from", type=Path, help="reuse a matching candidate directory's receipt")
    args = parser.parse_args(argv)
    try:
        if args.check:
            valid, message = check_receipt(Path(args.op_dir), spec_path=args.spec)
            LOGGER.info("%s%s", "[PASS] " if valid else "[FAIL] ", message)
            return 0 if valid else 1
        return run_once(Path(args.op_dir), args.retry_failed, args.spec, args.reuse_from)
    except (ReceiptError, OSError, UnicodeError) as error:
        LOGGER.error("[ERROR] %s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
