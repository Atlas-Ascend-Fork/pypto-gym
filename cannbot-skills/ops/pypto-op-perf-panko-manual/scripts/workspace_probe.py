#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Read a PyPTO kernel's workspace size without rebuilding PyPTO.

The runtime asks the Python side for the workspace buffer. Verified on the
installed build (`pypto/frontend/parser/entry.py:356`, and
`python/src/bindings/runtime.cpp` `KernelLauncher::DoLaunch`):

    int64_t wsSize = kmodule->GetWorkspaceSize(kbinary, tensors);
    if (wsSize) {
        auto pyalloc = py::getattr(module, "alloc");
        wsAddr = (int64_t*)pyalloc(wsSize).cast<int64_t>();
    }

    class JitCallableWrapper:
        @staticmethod
        def alloc(size):
            return torch.empty(size, dtype=torch.int8, device='npu').data_ptr()

So `size` IS `memBudget.Total()` in bytes, and it is the only place Python sees
it: `GetWorkspaceSize` is not exposed to Python at all. Wrapping `alloc` on the
wrapper CLASS therefore sees every size the runtime requests, for every kernel,
with no change to PyPTO.

Two things the real signature forces. `alloc` is a STATICMETHOD, so the hook
must be installed as one and must accept the call with or without an instance
in front -- a hook installed as an ordinary method turns `Class.alloc(size)`
into `self = size`. And it returns a POINTER (`data_ptr()`), not a buffer: an
int return is never read as a size, or the probe would report an address.

Three states, never conflated:

  measured     the hook was installed and `alloc` was called: the size is read
  zero         the hook saw the kernel run and `alloc` was never called. The
               launcher skips the call exactly when wsSize == 0, so this IS a
               reading of 0
  unavailable  no JitCallableWrapper with an `alloc` was found, no size could be
               read from a call, or the hook saw nothing run. NOT zero: a probe
               that read nothing must never look like a kernel that needs no
               workspace.

`__call__` is counted for ONE reason, and it is not about the kernel: a wrapper
always invokes its kernel exactly once, so `launches == 0` means the hook is on
the wrong class object -- two pypto installs, a reloaded module, a version whose
wrapper lives elsewhere. In that state `alloc` is not seen either, and without
the counter "alloc was never called" would be recorded as a workspace of 0 that
nothing measured. With it, the same state reads `unavailable`.

The size is taken from the call itself when it carries one (the first positive
int argument, or a size-like keyword), else from what `alloc` returned (a
tensor's `numel() * element_size()`, or `nbytes`). Which of the two was used is
reported, so a PyPTO version whose `alloc` has another shape shows up as a
changed `source` rather than as a wrong number.
"""

import inspect

SIZE_KWARGS = ("size", "nbytes", "ws_size", "workspace_size", "num_bytes", "n")


def find_jit_classes(module):
    """
    Classes of the module-level jit kernels that have an `alloc` (duck-typed,
    as `pypto-op-develop/scripts/smoke_check_impl.py` finds them), else any class
    named JitCallableWrapper already imported -- a kernel defined in a helper
    module and called through it is not a name in the impl module.
    """
    import sys
    out = []
    for name in dir(module):
        if name.startswith("__"):
            continue
        obj = getattr(module, name, None)
        if (hasattr(obj, "compile") and hasattr(obj, "_cached_signature")
                and hasattr(type(obj), "alloc")):
            cls = type(obj)
            if cls not in out:
                out.append(cls)
    if not out:
        for mod in list(sys.modules.values()):
            cls = getattr(mod, "JitCallableWrapper", None)
            if isinstance(cls, type) and hasattr(cls, "alloc") and cls not in out:
                out.append(cls)
    return out


def _size_from_call(args, kwargs):
    for a in args:
        if isinstance(a, bool):
            continue
        if isinstance(a, int) and a > 0:
            return a
    for k in SIZE_KWARGS:
        v = kwargs.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return None


def _size_from_result(res):
    # NOT an int: the installed `alloc` returns `data_ptr()`, and an address
    # read as a size is a plausible-looking wrong number.
    if isinstance(res, int):
        return None
    try:
        if hasattr(res, "numel") and hasattr(res, "element_size"):
            return int(res.numel()) * int(res.element_size())
        if hasattr(res, "nbytes"):
            return int(res.nbytes)
    except Exception:          # a result we cannot size is "unread", not an error
        return None
    return None


def _alloc_hook(orig, cls, static, probe):
    """`alloc` wrapped so every call's byte count lands on `probe.calls`.

    A factory rather than a closure written inside the install loop: the four
    values are bound as parameters, so each patched class gets its own.
    """
    def hooked(*args, **kwargs):
        # The launcher calls `alloc(wsSize)` through a getattr, which may or
        # may not bind an instance; a staticmethod takes none.
        call = list(args)
        if call and isinstance(call[0], cls):
            call = call[1:]
        res = orig(*call, **kwargs) if static else orig(*args, **kwargs)
        try:
            n = _size_from_call(call, kwargs)
            src = "call"
            if n is None:
                n, src = _size_from_result(res), "result"
            probe.calls.append({"bytes": n, "source": src if n is not None else None})
        except Exception as e:   # noqa: BLE001
            probe.calls.append({"bytes": None, "source": None,
                                "error": f"{type(e).__name__}: {e}"})
        return res
    return hooked


class WorkspaceProbe:
    """Install on a loaded impl module; read after the calls you care about."""

    def __init__(self):
        self.installed = []          # (class, original alloc attribute)
        self.wrapped_calls = []      # (class, original __call__)
        self.calls = []              # one entry per alloc call: {"bytes", "source"}
        self.launches = 0            # jit wrapper invocations seen
        self.error = None

    def install(self, module):
        try:
            classes = find_jit_classes(module)
        except Exception as e:     # noqa: BLE001 -- a probe must never break the measurement
            self.error = f"scan failed: {type(e).__name__}: {e}"
            return self
        for cls in classes:
            # BEFORE the patch, or uninstall restores the hook over itself.
            was = inspect.getattr_static(cls, "alloc", None)
            static = isinstance(was, staticmethod)
            orig = cls.alloc                       # resolved callable either way
            hooked = _alloc_hook(orig, cls, static, self)
            try:
                cls.alloc = staticmethod(hooked) if static else hooked
                self.installed.append((cls, was))
            except (AttributeError, TypeError) as e:
                self.error = f"cannot patch {cls.__name__}.alloc: {e}"
                continue
            self._count_launches(cls)
        if not self.installed and self.error is None:
            self.error = "no JitCallableWrapper with an alloc method in the module"
        return self

    def uninstall(self):
        for cls, orig in self.wrapped_calls:
            try:
                cls.__call__ = orig
            except (AttributeError, TypeError):
                pass
        self.wrapped_calls = []
        for cls, orig in self.installed:
            try:
                # The STATIC attribute, so a staticmethod is restored as one.
                cls.alloc = orig
            except (AttributeError, TypeError):
                pass
        self.installed = []

    def reading(self, wrapper_calls=None):
        """
        The result dict fields. `wrapper_calls` = how many times the wrapper
        ran, so a wrapper that launches several kernels reports their sum too.
        """
        calls = self.calls
        if not self.installed:
            return {"workspace_b": None, "workspace_state": "unavailable",
                    "workspace_error": self.error}
        if not calls:
            if not self.launches:
                return {"workspace_b": None, "workspace_state": "unavailable",
                        "workspace_calls": 0, "workspace_launches": 0,
                        "workspace_error": "the hook saw no kernel run (a wrapper "
                                           "always invokes one), so it is not on the "
                                           "class this program uses"}
            return {"workspace_b": 0, "workspace_state": "zero", "workspace_calls": 0,
                    "workspace_launches": self.launches}
        sizes = [c["bytes"] for c in calls if c.get("bytes") is not None]
        if not sizes:
            return {"workspace_b": None, "workspace_state": "unavailable",
                    "workspace_calls": len(calls),
                    "workspace_error": "alloc was called but no size could be read"}
        return {"workspace_b": max(sizes), "workspace_state": "measured",
                "workspace_calls": len(calls), "workspace_launches": self.launches,
                # One kernel launch per call is what was observed on one recorded kernel;
                # a wrapper that launches several kernels, or a shape that moves
                # between calls, shows up here rather than being averaged away.
                "workspace_distinct": sorted(set(sizes)),
                "workspace_sum_per_call_b": (sum(sizes) // wrapper_calls
                                             if wrapper_calls else None),
                "workspace_source": sorted({c["source"] for c in calls if c.get("source")})}

    def _count_launches(self, cls):
        """Count kernel invocations: `launches == 0` is the hook on the wrong
        class, and must not be read as a workspace of 0 (see the module
        docstring). A wrapper always invokes its kernel once."""
        call = getattr(cls, "__call__", None)
        if call is None:
            return
        probe = self

        def called(self_, *args, **kwargs):
            probe.launches += 1
            return call(self_, *args, **kwargs)
        try:
            cls.__call__ = called
            self.wrapped_calls.append((cls, call))
        except (AttributeError, TypeError):
            # Not fatal: without it `zero` cannot be distinguished from "no
            # kernel ran", so `reading` reports `unavailable` in that case.
            pass
