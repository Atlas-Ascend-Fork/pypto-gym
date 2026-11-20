# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""JIT compiler implementation: versioned cache + flock + cpp_extension.load.

Cache key dimensions (structural correctness, pypto _compute_cache_dir style):
    gym_ver     distribution version; source trees without dist-info fall back
                to a sha1 of the _bindings/csrc payload so edits invalidate the cache
    build_ts    X-Build-Timestamp from the wheel METADATA (build identity):
                same-version rebuilds get a fresh key, so a stale cache can
                never shadow changed binding sources; omitted for source trees
                (their dev-<sha1> gym_ver is content-addressed already)
    torch_ver   torch.__version__ (local part kept: +cpu vs +npu differ)
    tonpu_ver   torch_npu.__version__
    ascend_hash md5 of the CANN install path, ascend_<hash> (acl/aclnn headers
                affect the compile; same name/resolution as pypto); omitted
                when unknown
    arch        platform.machine()
No Python dimension: form-B artifacts carry zero Python symbols (verified),
one .so serves every supported interpreter version.
"""

import fcntl
import hashlib
import logging
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import sysconfig
import time

_log = logging.getLogger(__name__)

_LOCK_TIMEOUT = 1800.0     # seconds to wait for a concurrent compiler
_POLL_INTERVAL = 1.0


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(text))


def _csrc_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "_bindings" / "csrc"


def _gym_version() -> str:
    from importlib import metadata
    try:
        return metadata.version("pypto-gym")
    except Exception:
        # Source/editable tree without dist-info: fingerprint the JIT inputs so
        # source edits invalidate the cache (replaces the missing version key).
        digest = hashlib.sha1()
        csrc = _csrc_dir()
        if csrc.is_dir():
            for path in sorted(csrc.rglob("*")):
                if path.is_file():
                    digest.update(str(path.relative_to(csrc)).encode())
                    digest.update(path.read_bytes())
        return "dev-" + digest.hexdigest()[:12]


def _build_timestamp() -> str:
    """Build-identity segment from the wheel METADATA (X-Build-Timestamp).

    Written by setup.py's CustomEggInfo at build time (pypto-style); build_ci
    injects one timestamp per build round shared by the full and shell wheels.
    Empty for source trees (no dist-info); 'nots' with a warning when an
    installed artifact predates the field.
    """
    from importlib import metadata
    try:
        meta = metadata.metadata("pypto-gym")
    except Exception:
        return ""
    ts = (meta["X-Build-Timestamp"] or "").strip() if meta is not None else ""
    if not ts:
        _log.warning("pypto-gym METADATA has no X-Build-Timestamp; using the 'nots' cache segment")
        return "nots"
    return ts


def _resolve_cann_path() -> str:
    """CANN install path resolution, pypto _get_ascend_path parity:
    CANN_PATH > ASCEND_HOME_PATH > dirname(ASCEND_OPP_PATH)."""
    for env_key in ("CANN_PATH", "ASCEND_HOME_PATH"):
        val = os.environ.get(env_key, "")
        if val:
            return val
    opp = os.environ.get("ASCEND_OPP_PATH", "")
    return str(Path(opp).parent) if opp else ""


def _cann_hash() -> str:
    """CANN install identity segment (same name and resolution order as pypto's
    ascend_<hash>). The CANN headers/libs are compile inputs; switching or
    upgrading CANN must not reuse bindings compiled against another install.
    """
    cann = _resolve_cann_path()
    if not cann:
        return ""
    return "ascend_" + hashlib.md5(cann.encode()).hexdigest()[:8]


def cache_dir() -> Path:
    """Versioned cache directory (created on demand by ensure_compiled)."""
    import torch
    import torch_npu
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    root = Path(xdg) if xdg and Path(xdg).is_absolute() else Path.home() / ".cache"
    path = root / "cann" / "pypto_gym"
    for segment in (
        _sanitize(_gym_version()),
        _sanitize(_build_timestamp()),
        _sanitize(str(torch.__version__)),
        _sanitize(str(getattr(torch_npu, "__version__", "0"))),
        _cann_hash(),
        platform.machine(),
    ):
        if segment:
            path = path / segment
    return path


def _cann_path() -> Path:
    cann = _resolve_cann_path()
    if not cann or not (Path(cann) / "lib64" / "libascendcl.so").is_file():
        raise ImportError(
            "CANN not found for JIT binding compilation: set ASCEND_HOME_PATH "
            "(or CANN_PATH / ASCEND_OPP_PATH) to a CANN installation providing "
            "lib64/libascendcl.so")
    return Path(cann)


def _ensure_ninja_on_path() -> None:
    """Make the pip-installed ninja visible when its scripts dir is off PATH."""
    if shutil.which("ninja"):
        return
    candidates = [str(Path(sys.executable).parent)]
    scripts = sysconfig.get_path("scripts")
    if scripts:
        candidates.append(scripts)
    try:
        import ninja
        bin_dir = getattr(ninja, "BIN_DIR", "")
        if bin_dir:
            candidates.append(str(bin_dir))
        module_file = getattr(ninja, "__file__", "")
        if module_file:
            for parent in Path(module_file).resolve().parent.parents:
                candidates.append(
                    str(parent / ("Scripts" if os.name == "nt" else "bin")))
    except ImportError:
        pass
    for prefix in (sys.prefix, sys.base_prefix):
        candidates.append(
            str(Path(prefix) / ("Scripts" if os.name == "nt" else "bin")))
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if shutil.which("ninja", path=candidate):
            current = os.environ.get("PATH", "")
            os.environ["PATH"] = candidate + (os.pathsep + current if current else "")
            return
    raise RuntimeError(
        "ninja executable not found; install it with 'pip install ninja' "
        "or add the Python environment's scripts directory to PATH")


def _try_lock(fd) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _poll_lock(fd, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _try_lock(fd):
            return True
        time.sleep(_POLL_INTERVAL)
    return False


def ensure_compiled(op_name: str) -> Path:
    """Return the compiled binding .so for op_name, compiling on first use."""
    build_dir = cache_dir()
    so_path = build_dir / f"pypto_gym_{op_name}.so"
    if so_path.is_file():
        return so_path
    build_dir.mkdir(parents=True, exist_ok=True)
    lock_path = build_dir / f".{op_name}.compile.lock"
    lock_fd = open(lock_path, "w")
    acquired = False
    try:
        acquired = _try_lock(lock_fd) or _poll_lock(lock_fd, _LOCK_TIMEOUT)
        if not acquired:
            _log.warning("timeout waiting for the compilation lock; compiling independently")
        if so_path.is_file():          # a peer finished while we waited
            return so_path
        _compile(op_name, build_dir)
        if not so_path.is_file():
            raise RuntimeError(
                f"JIT compilation reported success but produced no {so_path}")
        return so_path
    finally:
        if acquired:
            # unlink while still holding: after unlock+unlink a peer could lock
            # the old inode while a newcomer locks a fresh one (pypto pattern).
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
        lock_fd.close()


def _compile(op_name: str, build_dir: Path) -> None:
    _ensure_ninja_on_path()
    import torch_npu  # noqa: F401  (headers/libs for the binding; torch pulled in by cpp_extension)
    from torch.utils import cpp_extension

    csrc = _csrc_dir()
    source = csrc / f"{op_name}.cpp"
    if not source.is_file():
        raise ImportError(
            f"binding source not found: {source}; install a pypto-gym artifact "
            f"that ships _bindings/csrc (wheel or run-registered shell)")
    cann = _cann_path()
    tonpu = Path(torch_npu.__file__).parent
    verbose = os.environ.get("GYM_JIT_VERBOSE", "") == "1"
    _log.info("JIT compiling pypto_gym_%s into %s", op_name, build_dir)
    try:
        cpp_extension.load(
            name=f"pypto_gym_{op_name}",
            sources=[str(source)],
            extra_include_paths=[
                str(csrc),
                str(tonpu / "include"),
                str(cann / "include"),
                str(cann / "include" / "aclnnop"),
            ],
            extra_ldflags=[
                f"-L{cann / 'lib64'}", "-lascendcl",
                f"-L{tonpu / 'lib'}", "-ltorch_npu",
            ],
            build_directory=str(build_dir),
            is_python_module=False,
            verbose=verbose,
        )
    except Exception as error:
        raise RuntimeError(
            f"JIT compilation of pypto_gym_{op_name} failed: {error}\n"
            f"Requirements: g++ (GCC 9.4.0+), ninja on PATH (pip install ninja, or apt install ninja-build), torch/torch_npu "
            f"headers (from the installed pip packages), CANN headers/libs "
            f"(ASCEND_HOME_PATH). Set GYM_JIT_VERBOSE=1 for the full compile log."
        ) from error


def prewarm(ops=None):
    """Compile bindings into the versioned cache without executing any op.

    Intended for image builds / post-install warm-up so the production runtime
    stays compile-free. Imports torch. Returns the compiled op names.
    """
    if ops is None:
        from . import binding_names
        names = list(binding_names())
    elif isinstance(ops, str):
        names = [ops]
    else:
        names = list(ops)
    for name in names:
        ensure_compiled(name)
    return names
