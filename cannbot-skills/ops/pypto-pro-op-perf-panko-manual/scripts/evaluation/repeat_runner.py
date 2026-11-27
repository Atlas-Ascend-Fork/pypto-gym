# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Repeat a Python test in one profiler process. Preserve its arguments, outputs and exit status."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import builtins
import hashlib
import json
import logging
import os
import runpy
import time
import shutil
import traceback
from contextlib import contextmanager
from importlib.machinery import SourceFileLoader

logger = logging.getLogger(__name__)

# Emit after the first launch returns so timeout classification can distinguish
# compilation from a later stall.
FIRST_CALL_MARKER = "[panko-progress] first_call_done"

LAUNCH_MARKER = "[panko-progress] launch %d/%d done"

# Bind the measurement to the loaded module path and hash.
IMPL_ENV = "PANKO_IMPL_PATH"
PROGRAM_IMPL_ENV = "PANKO_PROGRAM_IMPL_PATHS"
EXPORT_PATHS_ENV = "PANKO_EXPORT_IMPORT_PATHS"

IMPL_MARKER = "[panko-progress] impl %s %s"
MODULE_MARKER = "[panko-progress] program module "

# A frozen test can read its launch grid from this environment variable.
BLOCK_DIM_ENV = "PANKO_BLOCK_DIM"

#: Entry points that own the process and cannot be re-entered in it.
_PYTEST_ENTRIES = ("pytest", "py.test")

_REFUSE_PYTEST = "%r is a pytest entry point. Use a Python script that launches the kernel once per call."


def _basename(token: str) -> str:
    base = os.path.basename(str(token))
    return base[:-4] if base.endswith(".exe") else base


def _is_python(token: str) -> bool:
    return _basename(token).startswith("python")


def interpreter_of(argv):
    """Resolve the named Python interpreter. A bare script has none."""
    parts = [str(a) for a in (argv or [])]
    if not parts or parts[0].endswith(".py") or not _is_python(parts[0]):
        return None
    found = shutil.which(parts[0])
    if found:
        return os.path.abspath(found)
    if os.path.isabs(parts[0]):
        return parts[0]
    # A missing named interpreter must not fall back to the default interpreter.
    raise ValueError(f"The requested interpreter {parts[0]!r} is missing from PATH. Give an absolute path.")


def _is_pytest(token: str) -> bool:
    base = _basename(token)
    return base in _PYTEST_ENTRIES or base.startswith("pytest")


_SCRIPT_FLAGS = {"-b", "-bb", "-B", "-E", "-O", "-OO", "-q", "-s", "-S", "-u"}


def split_script_command(argv):
    """Return interpreter options, script path and script arguments for a repeatable command."""
    parts = [str(a) for a in (argv or [])]
    if not parts:
        raise ValueError("the test command is empty, so there is nothing to launch")
    if _is_pytest(parts[0]):
        raise ValueError(_REFUSE_PYTEST % parts[0])
    if parts[0].endswith(".py"):
        return [], parts[0], parts[1:]
    if not _is_python(parts[0]):
        raise ValueError(f"Expected a Python interpreter or .py script, got {parts[0]!r}")
    i = 1
    while i < len(parts) and parts[i].startswith("-"):
        flag = parts[i]
        i += 1
        if flag == "--":
            break
        if flag in ("-W", "-X"):
            if i >= len(parts):
                raise ValueError(f"Python option {flag} requires an argument")
            i += 1
        elif flag.startswith(("-W", "-X")) or flag in _SCRIPT_FLAGS:
            continue
        elif flag.startswith("-m") and _is_pytest(flag[2:] or (parts[i] if i < len(parts) else "")):
            raise ValueError(_REFUSE_PYTEST % flag)
        else:
            raise ValueError(f"Unsupported Python interpreter option {flag!r} for a repeated script")
    if i == len(parts) or not parts[i].endswith(".py"):
        raise ValueError("the profiled command requires a .py script path")
    return parts[1:i], parts[i], parts[i + 1:]


def loopable(argv, launches=None):
    """Return the script and its arguments, or (None, reason) for an unsupported command."""
    try:
        _, script, args = split_script_command(argv)
        return script, args
    except ValueError as exc:
        return None, str(exc)


def _positive_int(name, value):
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if number < 1:
        raise ValueError(f"{name} must be at least 1, got {number}")
    return number


def looped_argv(argv, launches, python=None, block_dim=None, block_dim_flag=None):
    """Return (runner argv, None) or (None, refusal), preserving the original interpreter and arguments."""
    try:
        n = _positive_int("launches", launches)
        if block_dim is not None:
            block_dim = _positive_int("block_dim", block_dim)
        elif block_dim_flag:
            raise ValueError(f"block_dim_flag={block_dim_flag!r} requires block_dim")
        flags, script, rest = split_script_command(argv)
        if python is None:
            python = interpreter_of(argv)
    except ValueError as exc:
        return None, str(exc)
    target = [script, *[str(a) for a in rest]]
    if block_dim is not None and block_dim_flag:
        target += [str(block_dim_flag), str(block_dim)]
    own = ["--launches", str(n)]
    if block_dim is not None:
        own += ["--block-dim", str(block_dim)]
    return ([str(python or sys.executable), *flags, os.path.abspath(__file__), *own, "--", *target], None)


def run(script, args, launches, block_dim=None, timing=None):
    """Apply launch overrides and repeat the script in one interpreter with its own argv and import path."""
    out, err = sys.stdout, sys.stderr
    script = os.path.abspath(script)
    here = os.path.dirname(script)
    if here and here not in sys.path:
        sys.path.insert(0, here)
    # Frozen tests can import either the qualified package or its top-level modules.
    sys.path[:0] = json.loads(os.environ.pop(EXPORT_PATHS_ENV, "[]"))
    if block_dim is not None:
        os.environ[BLOCK_DIM_ENV] = str(int(block_dim))
    started = time.perf_counter() if timing is not None else None
    for i in range(launches):
        sys.argv = [script, *[str(a) for a in args]]
        try:
            with capture_impl_imports(i == 0) as imported_paths:
                runpy.run_path(script, run_name="__main__")
        except builtins.SystemExit as exc:
            # A script exit is a result, including a successful exit(None/0).
            code = exc.code
            if code not in (None, 0):
                err.write("panko-repeat-runner: launch %d/%d exited %r\n" % (i + 1, launches, code))
                err.flush()
                return code if isinstance(code, int) else 1
        except BaseException:                      # noqa: BLE001 -- reported, not hidden
            err.write("panko-repeat-runner: launch %d/%d raised\n" % (i + 1, launches))
            traceback.print_exc(file=err)
            err.flush()
            return 1
        if timing is not None:
            timing["completed"] = i + 1
            if i == 0:
                timing["first_test_call"] = time.perf_counter() - started
        if i == 0:
            err.write(FIRST_CALL_MARKER + "\n")
            err.flush()
            report_impl_modules(out, imported_paths)
        out.write((LAUNCH_MARKER % (i + 1, launches)) + "\n")
        out.flush()
    return 0


@contextmanager
def capture_impl_imports(enabled):
    """Observe source loads on the first launch, including modules removed from the import cache."""
    paths = set()
    if not enabled or not os.environ.get(PROGRAM_IMPL_ENV):
        yield paths
        return
    original = SourceFileLoader.exec_module

    def observed(loader, module):
        result = original(loader, module)
        paths.add((loader.name, module.__package__, loader.path))
        return result

    SourceFileLoader.exec_module = observed
    try:
        yield paths
    finally:
        SourceFileLoader.exec_module = original


def report_impl_modules(out, imported_paths=()):
    """Report loaded implementation paths and hashes after launch. Diagnostic failures are nonfatal."""
    staged = os.environ.get(IMPL_ENV)
    if not staged:
        return
    try:
        program_paths = json.loads(os.environ.get(PROGRAM_IMPL_ENV, "[]"))
        want = {os.path.basename(p) for p in program_paths} if program_paths else {os.path.basename(staged)}
        seen = set()
        modules = [(name, getattr(module, "__package__", None), getattr(module, "__file__", None))
                   for name, module in list(sys.modules.items())]
        reported = set()
        for name, package, path in [*modules, *imported_paths]:
            if not isinstance(path, str) or os.path.basename(path) not in want:
                continue
            real = os.path.realpath(path)
            if program_paths and (name, real) not in reported:
                out.write(MODULE_MARKER + json.dumps([name, package, real]) + "\n")
                reported.add((name, real))
            if real in seen:
                continue
            seen.add(real)
            digest = hashlib.sha256(Path(real).read_bytes()).hexdigest()
            out.write((IMPL_MARKER % (digest if program_paths else digest[:8], real)) + "\n")
        out.flush()
    except Exception:                       # noqa: BLE001 -- a diagnostic only
        pass


_USAGE = ("usage: repeat_runner.py --launches N [--block-dim M] -- <python3> "
          "<script.py> [args...]\n")


def main(argv=None):
    """Run the CLI and translate script exits to a status."""
    raw = [str(a) for a in (sys.argv[1:] if argv is None else argv)]
    if len(raw) < 4 or "--" not in raw:
        logger.error("%s", _USAGE.rstrip())
        return 2
    try:
        separator = raw.index("--")
        options = raw[:separator]
        if len(options) % 2 or options[::2] not in (["--launches"], ["--launches", "--block-dim"]):
            raise ValueError(_USAGE.rstrip())
        launches = _positive_int("launches", options[1])
        block_dim = _positive_int("block_dim", options[3]) if len(options) == 4 else None
    except ValueError as exc:
        logger.error("panko-repeat-runner: %s", exc)
        return 2
    try:
        flags, script, target = split_script_command(raw[separator + 1:])
        if flags:
            raise ValueError("pass interpreter options before repeat_runner.py")
    except ValueError as exc:
        logger.error("panko-repeat-runner: refused: %s", exc)
        return 3
    from evaluation.acquisition_receipt import capture
    return capture(run, script, target, launches, block_dim)


if __name__ == "__main__":
    sys.exit(main())
