#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""算子调用入口适配(仓级测试公共层).

按 pytest --mode 选择 jit / aclnn / torch 路径:
    jit   -> import <op_kernel>/<op>.py 的宿主入口(默认与算子同名)
    aclnn -> import 预编译桥 aclnn_bridge_<op> 的 <op> 接口(签名与 jit 入口一致)
    torch -> 已安装 .run 中 pypto_gym.<op> 的正式 Torch 绑定

路径由根 conftest.py / pytest.ini 统一注入, 本模块不做 sys.path 操作.
算子入口名不合约定时, 用 jit_module/jit_attr 显式指定。
pytest --op-entry OP=FILE:CALLABLE 仅替换指定算子的 JIT 公开入口，保留原始测试与 golden。
"""

import importlib
import hashlib
import json
import os
from pathlib import Path
import sys
import types


_ENTRY_OVERRIDES = {}
_LOADED_ENTRIES = {}
_MODULE_SOURCES = {}


def configure_op_entries(entries, *, mode="jit", base_dir=None):
    """Validate explicit per-operator candidates without importing device code."""
    if entries and mode != "jit":
        raise ValueError("--op-entry requires --mode jit")
    selected = {}
    base = Path(base_dir or Path.cwd())
    for entry in entries:
        op, separator, target = entry.partition("=")
        filename, colon, attribute = target.rpartition(":")
        if (
            not separator
            or not colon
            or not op.isidentifier()
            or not attribute.isidentifier()
        ):
            raise ValueError("--op-entry expects OP=FILE.py:CALLABLE")
        if op in selected:
            raise ValueError(f"duplicate --op-entry for {op!r}")
        path = Path(filename).expanduser()
        path = (base / path).resolve() if not path.is_absolute() else path.resolve()
        if path.suffix != ".py" or not path.is_file():
            raise ValueError(f"candidate Python source does not exist: {path}")
        selected[op] = (path, attribute)
    _ENTRY_OVERRIDES.clear()
    _ENTRY_OVERRIDES.update(selected)
    _LOADED_ENTRIES.clear()


def loaded_op_entries():
    """Return loaded public-entry identities; this is not execution evidence."""
    return {op: dict(identity) for op, identity in _LOADED_ENTRIES.items()}


def unused_op_entries():
    """Identify requested candidates that no test selected through the public loader."""
    return sorted(_ENTRY_OVERRIDES.keys() - _LOADED_ENTRIES.keys())


def merge_op_entry_identities(identities):
    """Combine subprocess reports' entry identities without importing their kernels."""
    for op, identity in identities.items():
        path, attribute = _ENTRY_OVERRIDES[op]
        expected = {
            "path": str(path),
            "callable": attribute,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if identity != expected or (
            op in _LOADED_ENTRIES and _LOADED_ENTRIES[op] != identity
        ):
            raise RuntimeError(f"candidate identity differs between workers: {op}")
        _LOADED_ENTRIES[op] = dict(identity)


def _file_caller(op, path, attribute):
    """Load candidates in separate package namespaces, preserving relative imports."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    prior = _LOADED_ENTRIES.get(op)
    if prior and prior["sha256"] != digest:
        raise RuntimeError(f"candidate source changed during this process: {path}")
    root = path.parent
    while (root / "__init__.py").is_file():
        root = root.parent
    prefix = "_gym_candidate_" + hashlib.sha256(str(root).encode()).hexdigest()[:20]
    if prefix not in sys.modules:
        package = types.ModuleType(prefix)
        package.__path__ = [str(root)]
        package.__package__ = prefix
        sys.modules[prefix] = package
    parts = path.relative_to(root).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    module_name = ".".join((prefix, *parts))
    if module_name in _MODULE_SOURCES and _MODULE_SOURCES[module_name] != digest:
        raise RuntimeError(f"candidate source changed; use a fresh process: {path}")
    _MODULE_SOURCES[module_name] = digest
    module = importlib.import_module(module_name)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"candidate source changed while importing: {path}")
    if Path(module.__file__).resolve() != path:
        raise RuntimeError(
            f"candidate import resolved to a different file: {module.__file__}"
        )
    caller = getattr(module, attribute)
    if not callable(caller):
        raise TypeError(f"candidate entry is not callable: {path}:{attribute}")
    _LOADED_ENTRIES[op] = {"path": str(path), "callable": attribute, "sha256": digest}
    return caller


def get_mode() -> str:
    """返回当前测试模式(jit/aclnn/torch), 由根 conftest 的 --mode 写入 GYM_TEST_MODE."""
    return os.environ.get("GYM_TEST_MODE", "jit")


def get_op_caller(
    op: str,
    *,
    jit_module: str = None,
    jit_attr: str = None,
    aclnn_module: str = None,
    aclnn_attr: str = None,
):
    """返回算子在当前模式下的调用入口(torch.Tensor 接口)."""
    mode = get_mode()
    if mode == "jit":
        if op in _ENTRY_OVERRIDES:
            return _file_caller(op, *_ENTRY_OVERRIDES[op])
        module_name = jit_module
        if not module_name:
            registry = os.environ.get("GYM_OPS_REGISTRY")
            if registry:
                with open(registry, "r", encoding="utf-8") as fh:
                    mapping = json.load(fh)
                module_name = mapping.get(op)
                if not module_name:
                    raise RuntimeError(
                        f"op {op!r} not found in GYM_OPS_REGISTRY={registry}"
                    )
            else:
                module_name = op
        module = importlib.import_module(module_name)
        caller = getattr(module, jit_attr or op)
        if not callable(caller):
            raise TypeError(
                f"operator entry is not callable: {module_name}:{jit_attr or op}"
            )
        return caller
    if mode == "aclnn":
        module = importlib.import_module(aclnn_module or f"aclnn_bridge_{op}")
        return getattr(module, aclnn_attr or op)
    if mode == "torch":
        return getattr(importlib.import_module("pypto_gym"), op)
    raise ValueError(f"unsupported test mode: {mode} (expected jit/aclnn/torch)")
