# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host regressions for tolerances in the snapshot manifest contract."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from snapshot_manifest_schema import SnapshotManifestError, load_manifest


@pytest.fixture
def manifest_path(tmp_path):
    manifest = {
        "op": "probe",
        "module": "M1",
        "case": "host",
        "intermediates": [
            {"name": "value", "shape": [1], "probe_point": "before_nt_loop"}
        ],
    }
    path = tmp_path / "snapshot_manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def set_tolerance(path, name, value):
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest[name] = value
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")


@pytest.mark.parametrize("name", ["atol", "rtol"])
@pytest.mark.parametrize(
    "value", [float("inf"), float("-inf"), float("nan"), -1.0, None, "invalid", True]
)
def test_invalid_tolerance_is_a_manifest_error(manifest_path, name, value):
    set_tolerance(manifest_path, name, value)
    with pytest.raises(SnapshotManifestError, match=name):
        load_manifest(manifest_path)


@pytest.mark.parametrize("name", ["atol", "rtol"])
@pytest.mark.parametrize("value", [0, 1, 1e-3, "1.0e-3"])
def test_finite_nonnegative_tolerance_is_normalized(manifest_path, name, value):
    set_tolerance(manifest_path, name, value)
    assert load_manifest(manifest_path)[name] == float(value)


def test_missing_tolerances_keep_defaults(manifest_path):
    manifest = load_manifest(manifest_path)
    assert manifest["atol"] == manifest["rtol"] == 1e-3


def run_manifest_cli(manifest_path):
    script = Path(__file__).with_name("snapshot_manifest_schema.py").resolve()
    return subprocess.run(
        [
            str(Path(sys.executable).absolute()),
            str(script),
            str(manifest_path),
            "--print",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("value", [float("inf"), None, "invalid"])
def test_cli_rejects_invalid_tolerance_without_traceback(manifest_path, value):
    set_tolerance(manifest_path, "atol", value)
    result = run_manifest_cli(manifest_path)
    assert result.returncode == 1
    assert "[INVALID]" in result.stderr
    assert "atol" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_valid_manifest_has_json_output(manifest_path):
    result = run_manifest_cli(manifest_path)
    assert result.returncode == 0
    assert "[OK]" in result.stderr
    json_start = result.stderr.index("{")
    payload = result.stderr[json_start:]
    assert json.loads(payload)["atol"] == 1e-3
