# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Load shipped libraries on demand; no runtime compilation or kernel-source imports."""
import importlib
import json
from pathlib import Path

_loaded = {}
_names = None


def binding_names():
    """Return the installed operator names without importing torch."""
    global _names
    if _names is None:
        package = Path(__file__).resolve().parent
        manifest = package / 'operators.json'
        if not manifest.is_file():
            raise ImportError('Torch bindings are not installed; build the .run with --torch-bindings '
                              'and install it into the active CANN environment')
        try:
            names = json.loads(manifest.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise ImportError(f'Invalid PyPTO-Gym binding manifest {manifest}: {error}') from error
        if not isinstance(names, list):
            raise ImportError('Invalid PyPTO-Gym binding manifest: expected a list of operator names')
        for name in names:
            if not isinstance(name, str) or not name.isidentifier():
                raise ImportError('Invalid operator name in PyPTO-Gym binding manifest')
        _names = tuple(names)
    return _names


def load_binding(name):
    """Import torch/torch_npu and load one operator extension on first use."""
    if name in _loaded:
        return _loaded[name]
    if name not in binding_names():
        raise KeyError(name)
    import torch
    import torch_npu  # Register PrivateUse1 and load libtorch_npu before extensions.

    package = Path(__file__).resolve().parent
    library = package / '_native' / f'libpypto_gym_{name}.so'
    try:
        torch.ops.load_library(str(library))
    except OSError as error:
        raise ImportError(f'Cannot load {library}; use the torch/torch_npu ABI '
                          f'used to build this .run: {error}') from error
    module = importlib.import_module(f'._frontends.{name}', __package__)
    entry = getattr(module, name)
    _loaded[name] = entry
    return entry


def load_bindings():
    """Load every installed binding; prefer load_binding() for lazy access."""
    return {name: load_binding(name) for name in binding_names()}
