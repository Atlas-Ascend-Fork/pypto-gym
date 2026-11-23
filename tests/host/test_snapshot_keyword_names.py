# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Use real YAML parsing and the actual snapshot Python probe generator."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "cannbot-skills/ops/pypto-op-verify/scripts"
sys.path.insert(0, str(SCRIPTS))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCHEMA = load_module(
    "snapshot_manifest_schema", SCRIPTS / "snapshot_manifest_schema.py"
)
GENERATOR = load_module("snapshot_keyword_generator", SCRIPTS / "snapshot_generator.py")


def manifest_path(tmp_path, name):
    path = tmp_path / "snapshot_manifest.yaml"
    document = {
        "op": "example_op",
        "module": "M2",
        "case": "test-case",
        "intermediates": [
            {"name": name, "shape": [4], "probe_point": "before_nt_loop"}
        ],
    }
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


@pytest.mark.parametrize("name", ["class", "for", "return", "async"])
def test_reserved_names_are_rejected_by_manifest_loader(tmp_path, name):
    # The real generator embeds the unprefixed local name in pypto.reshape.
    probe = getattr(GENERATOR, "_gen_impl_probe_block")
    lines = probe({"name": name, "shape": [4], "probe_point": "before_nt_loop"}, "    ")
    with pytest.raises(SyntaxError):
        compile(
            "def generated_probe():\n" + "\n".join(lines), "<snapshot-probe>", "exec"
        )
    with pytest.raises(SCHEMA.SnapshotManifestError, match="valid Python identifier"):
        SCHEMA.load_manifest(manifest_path(tmp_path, name))


@pytest.mark.parametrize("name", ["buffer", "class_", "match"])
def test_valid_local_names_keep_generating_parseable_probes(tmp_path, name):
    manifest = SCHEMA.load_manifest(manifest_path(tmp_path, name))
    entry = manifest["intermediates"][0]
    probe = getattr(GENERATOR, "_gen_impl_probe_block")
    lines = probe(entry, "    ")
    compile("def generated_probe():\n" + "\n".join(lines), "<snapshot-probe>", "exec")
    assert entry["name"] == name


def test_existing_non_identifier_rejection_is_retained(tmp_path):
    with pytest.raises(SCHEMA.SnapshotManifestError, match="valid Python identifier"):
        SCHEMA.load_manifest(manifest_path(tmp_path, "bad name"))
