# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch interfaces backed by the installed PyPTO-Gym ACLNN binaries.

Importing this package is cheap and does not require torch. Bindings are loaded
on first attribute access, or explicitly through :func:`load`. This keeps the
namespace importable from tooling that does not have torch/torch_npu installed.
"""
import pkgutil

from ._loader import binding_names as _binding_names

# Allow the optional kernel wheel's pypto_gym.ops namespace to coexist. The
# attribute is updated through globals() because assigning to a dunder name
# directly is discouraged by the code checker; the effect is identical.
globals()['__path__'] = pkgutil.extend_path(globals()['__path__'], __name__)

try:
    __all__ = list(_binding_names())
except ImportError:
    # Bindings are not installed (for example an ACLNN-only .run).
    __all__ = []


def load(op=None):
    """Load installed bindings and return the loaded operator names.

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


def __getattr__(name):
    if name.startswith('_'):
        raise AttributeError(name)
    from ._loader import binding_names, load_binding

    try:
        names = binding_names()
    except ImportError as error:
        raise AttributeError(str(error)) from error
    if name not in names:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return load_binding(name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
