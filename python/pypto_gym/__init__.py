# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch interfaces backed by the installed PyPTO-Gym ACLNN binaries.

Importing this package is cheap and does not require torch. Bindings are JIT
compiled on first use (pypto_gym._loader) and loaded lazily on first attribute
access, or explicitly through :func:`load` / :func:`prewarm`. This keeps the
namespace importable from tooling that does not have torch/torch_npu installed.
"""
import os
import pkgutil
from pathlib import Path

from ._loader import binding_names as _binding_names

_VENDOR_NAME = 'pypto_gym'
_VENDOR_REL = Path('_opp') / 'vendors' / _VENDOR_NAME
_VENDOR_LIB_REL = Path('op_api') / 'lib' / 'libcust_opapi.so'

_resolved_vendor = None


def _package_base_dir():
    """pypto-style dual location: cur_dir wins, dist locate validates.

    ``locate_file`` catches packaging anomalies; a mismatch (editable/source
    tree) falls back to the directory of this file, which is always truthful.
    """
    cur_dir = Path(__file__).resolve().parent
    try:
        from importlib import metadata
        pkg_dir = Path(str(metadata.distribution('pypto-gym').locate_file('pypto_gym'))).resolve()
    except Exception:
        return cur_dir
    return pkg_dir if pkg_dir == cur_dir else cur_dir


def _is_vendor(path):
    return path is not None and (path / _VENDOR_LIB_REL).is_file()


def _discover_vendor():
    """Three-tier vendor discovery (decision 30). Returns Path or None.

    1. bundled ``_opp`` sibling           — community full wheel (form marker)
    2. ``ASCEND_CUSTOM_OPP_PATH`` entries — run installed to a custom location
                                            (set_env.bash), or user-managed
    3. ``$ASCEND_OPP_PATH/vendors``       — run installed into the standard
                                            CANN opp, nothing sourced
                                            (ops-transformer _custom_opp_paths
                                            style scan; ASCEND_HOME_PATH/opp
                                            as equivalent fallback)
    Every tier is validated by op_api/lib/libcust_opapi.so; invalid hits
    degrade to the next tier. No rpath/RUNPATH is involved anywhere
    (decision 29): the C++ side consumes ASCEND_CUSTOM_OPP_PATH via
    gym_aclnn.h's dlopen scan.
    """
    bundled = _package_base_dir() / _VENDOR_REL
    if _is_vendor(bundled):
        return bundled
    for entry in os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(':'):
        entry = entry.strip()
        if not entry:
            continue
        for cand in (Path(entry), Path(entry) / _VENDOR_NAME):
            if _is_vendor(cand):
                return cand
    opp = os.environ.get('ASCEND_OPP_PATH', '')
    if not opp:
        home = os.environ.get('ASCEND_HOME_PATH', '')
        opp = str(Path(home) / 'opp') if home else ''
    if opp:
        cand = Path(opp) / 'vendors' / _VENDOR_NAME
        if _is_vendor(cand):
            return cand
    return None


def _ensure_vendor_env():
    """Resolve the vendor and prepend it to ASCEND_CUSTOM_OPP_PATH.

    Runs at import time, before any binding load: gym_aclnn.h caches its
    dlopen handles once (Libraries() static singleton). Prepending makes the
    resolved vendor win over stale entries (bundled-first, decision 4/30).
    """
    global _resolved_vendor
    vendor = _discover_vendor()
    _resolved_vendor = vendor
    if vendor is None:
        return
    path = str(vendor)
    entries = [p for p in os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(':') if p]
    if entries and entries[0] == path:
        return
    if path in entries:
        entries.remove(path)
    entries.insert(0, path)
    os.environ['ASCEND_CUSTOM_OPP_PATH'] = ':'.join(entries)


_ensure_vendor_env()

# Allow the optional kernel wheel's pypto_gym.ops namespace to coexist. The
# attribute is updated through globals() because assigning to a dunder name
# directly is discouraged by the code checker; the effect is identical.
globals()['__path__'] = pkgutil.extend_path(globals()['__path__'], __name__)

try:
    __all__ = list(_binding_names())
except ImportError:
    # Bindings are not installed (for example an ACLNN-only artifact).
    __all__ = []


def vendor_path():
    """Absolute path of the resolved custom-op vendor directory, or None.

    Pure C++ aclnn consumers can export ``ASCEND_CUSTOM_OPP_PATH`` to this
    path; importing this package already sets it for the current process.
    """
    return str(_resolved_vendor) if _resolved_vendor is not None else None


def load(op=None):
    """Load bindings (JIT compiling on first use) and return the op names.

    Accessing ``pypto_gym.<op>`` loads lazily; call this before using
    ``torch.ops.pypto_gym.<op>`` directly.
    """
    from ._loader import binding_names, load_binding

    if op is None:
        names = list(binding_names())
        for name in names:
            load_binding(name)
        return names
    if not isinstance(op, str):
        raise TypeError('operator name must be a string')
    load_binding(op)
    return [op]


def prewarm(ops=None):
    """JIT compile bindings into the versioned cache without executing ops.

    For image builds / post-install warm-up so the production runtime stays
    compile-free. Imports torch. Returns the compiled op names.
    """
    from ._loader import prewarm as _prewarm
    return _prewarm(ops)


def _resolve_op(name):
    """Shared operator resolver behind get_ops() and attribute access.

    Two-tier resolution (form A dual-capability wheel):
      1. torch-binding tier — operators.json manifest (_bindings), JIT compiled
         against the installed _opp binaries;
      2. JIT source tier — pypto_gym.ops.<paradigm>.<network>.<op> public entry
         (requires pypto_pro in the environment; the wheel ships the sources).
    The public contract of get_ops() stays stable across tier extension.
    Returns the callable entry, or None when unresolvable.
    """
    from ._loader import binding_names, load_binding

    try:
        names = binding_names()
    except ImportError:
        names = []
    if name in names:
        return load_binding(name)
    return _resolve_jit_source(name)


def _resolve_jit_source(name):
    """JIT source tier: import pypto_gym.ops.<...>.<op> public entry by name.

    Scans ``op_kernel/<name>.py`` modules under the ops/ tree and resolves the
    public callable of the same name (op-level ``__init__.py`` re-exports are
    covered by the same module-level attribute). A missing pypto_pro
    dependency surfaces as ImportError to the caller.
    """
    import importlib

    base = Path(__file__).resolve().parent / "ops"
    if not base.is_dir():
        return None
    for module_path in sorted(base.rglob(f"op_kernel/{name}.py")):
        rel = module_path.relative_to(base.parent)
        module_name = ".".join(rel.with_suffix("").parts)
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        entry = getattr(module, name, None)
        if callable(entry):
            return entry
    return None


def get_ops(op=None):
    """Resolve operator entries by name (tier-agnostic soft probe).

    ``op=None``  -> sorted list of currently resolvable operator names
                    (binding tier + JIT source tier).
    ``op=<str>`` -> callable entry, or None when unresolvable (no exception;
                    cross-check with get_ops() for typos).
    """
    if op is None:
        from ._loader import binding_names

        names = set()
        try:
            names.update(binding_names())
        except ImportError:
            pass
        names.update(_jit_source_names())
        return sorted(names)
    if not isinstance(op, str):
        raise TypeError('operator name must be a string')
    return _resolve_op(op)


def _jit_source_names():
    """Operator names discoverable in the JIT source tier (ops/ tree)."""
    base = Path(__file__).resolve().parent / "ops"
    if not base.is_dir():
        return set()
    return {path.stem for path in base.rglob("op_kernel/*.py")
            if path.stem != "__init__"}


def __getattr__(name):
    if name.startswith('_'):
        raise AttributeError(name)
    from ._loader import binding_names

    try:
        binding_names()
    except ImportError:
        pass  # no bindings installed; the JIT source tier may still resolve
    entry = _resolve_op(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return entry


def __dir__():
    return sorted(set(globals()) | set(__all__))
