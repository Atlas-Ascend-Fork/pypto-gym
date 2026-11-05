# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public-entry selection is testable without importing torch or a device runtime."""

import hashlib
import json
import sys
import types

import pytest

from op_caller import (
    configure_op_entries,
    get_op_caller,
    loaded_op_entries,
    merge_op_entry_identities,
    unused_op_entries,
)


@pytest.fixture(autouse=True)
def entry_selection(monkeypatch):
    monkeypatch.setenv("GYM_TEST_MODE", "jit")
    monkeypatch.delenv("GYM_OPS_REGISTRY", raising=False)
    configure_op_entries([])
    yield
    configure_op_entries([])


def source(path, text="def entry():\n    return 7\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_candidate_preserves_package_initializers_relative_imports_and_sys_path(
    tmp_path,
):
    root = tmp_path / "candidate with spaces"
    source(root / "__init__.py", "from .helper import VALUE\n")
    source(root / "helper.py", "VALUE = 19\n")
    path = source(
        root / "kernel.py", "from . import VALUE\ndef entry():\n    return VALUE\n"
    )
    before = list(sys.path)
    configure_op_entries([f"engram_gate={path}:entry"])
    assert loaded_op_entries() == {}
    assert unused_op_entries() == ["engram_gate"]
    assert get_op_caller("engram_gate")() == 19
    assert unused_op_entries() == []
    assert sys.path == before
    identity = loaded_op_entries()["engram_gate"]
    assert identity == {
        "path": str(path),
        "callable": "entry",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_two_kernel_files_and_an_existing_module_do_not_collide(tmp_path, monkeypatch):
    def existing_entry():
        return -1

    existing = types.ModuleType("kernel")
    existing.entry = existing_entry
    monkeypatch.setitem(sys.modules, "kernel", existing)
    first = source(tmp_path / "first/kernel.py")
    second = source(tmp_path / "second/kernel.py", "def entry():\n    return 23\n")
    configure_op_entries([f"first={first}:entry", f"second={second}:entry"])
    assert get_op_caller("first")() == 7
    assert get_op_caller("second")() == 23
    assert sys.modules["kernel"] is existing


def test_relative_candidate_resolves_from_invocation_directory(tmp_path):
    source(tmp_path / "nested/kernel.py")
    configure_op_entries(["op=nested/kernel.py:entry"], base_dir=tmp_path)
    assert get_op_caller("op")() == 7


@pytest.mark.parametrize("reconfigure", [False, True])
def test_changed_source_cannot_be_claimed_as_the_cached_callable(tmp_path, reconfigure):
    path = source(tmp_path / "kernel.py")
    selection = [f"op={path}:entry"]
    configure_op_entries(selection)
    assert get_op_caller("op")() == 7
    path.write_text("def entry():\n    return 89\n")
    if reconfigure:
        configure_op_entries(selection)
    with pytest.raises(RuntimeError, match="source changed"):
        get_op_caller("op")


@pytest.mark.parametrize(
    "body,error",
    [
        ("raise RuntimeError('candidate import failed')\n", RuntimeError),
        ("other = 1\n", AttributeError),
        ("entry = 1\n", TypeError),
    ],
)
def test_invalid_candidate_never_falls_back_to_default(
    tmp_path, body, error, monkeypatch
):
    def default_op():
        return "fallback"

    module = types.ModuleType("default_operator")
    module.op = default_op
    monkeypatch.setitem(sys.modules, module.__name__, module)
    path = source(tmp_path / "kernel.py", body)
    configure_op_entries([f"op={path}:entry"])
    with pytest.raises(error):
        get_op_caller("op", jit_module=module.__name__)
    assert loaded_op_entries() == {}


@pytest.mark.parametrize(
    "selection",
    [
        "op",
        "op=file.py",
        "=file.py:entry",
        "op=file.py:",
        "op=missing.py:entry",
        "bad-name=file.py:entry",
    ],
)
def test_invalid_selection_fails_before_device_import(tmp_path, selection):
    with pytest.raises(ValueError):
        configure_op_entries([selection], base_dir=tmp_path)


def test_duplicates_and_aclnn_overrides_are_rejected(tmp_path):
    path = source(tmp_path / "kernel.py")
    selection = f"op={path}:entry"
    with pytest.raises(ValueError, match="duplicate"):
        configure_op_entries([selection, selection])
    with pytest.raises(ValueError, match="requires --mode jit"):
        configure_op_entries([selection], mode="aclnn")


def test_default_registry_and_aclnn_routes_are_preserved(tmp_path, monkeypatch):
    def registered_entry():
        return 11

    module = types.ModuleType("registered_operator")
    module.entry = registered_entry
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"op": module.__name__}))
    monkeypatch.setenv("GYM_OPS_REGISTRY", str(registry))
    assert get_op_caller("op", jit_attr="entry")() == 11
    assert loaded_op_entries() == {}
    monkeypatch.setenv("GYM_TEST_MODE", "aclnn")
    assert get_op_caller("op", aclnn_module=module.__name__, aclnn_attr="entry")() == 11


def test_worker_identities_are_merged_without_loading_and_reject_changed_bytes(
    tmp_path,
):
    path = source(tmp_path / "kernel.py", "raise RuntimeError('must not import')\n")
    configure_op_entries([f"op={path}:entry"])
    identity = {
        "path": str(path),
        "callable": "entry",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    merge_op_entry_identities({"op": identity})
    assert unused_op_entries() == []
    assert loaded_op_entries() == {"op": identity}
    path.write_text("changed = True\n")
    with pytest.raises(RuntimeError, match="differs between workers"):
        merge_op_entry_identities({"op": identity})
