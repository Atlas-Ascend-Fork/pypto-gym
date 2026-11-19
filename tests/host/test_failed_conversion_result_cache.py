# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
CONVERT = ROOT / "cannbot-skills/model/pypto-convert-model/scripts"
FUSED = ROOT / "cannbot-skills/model/pypto-fused-op-integration/scripts"
PERF = ROOT / "cannbot-skills/ops/pypto-pro-op-perf-tune/scripts"


def module(path):
    """Load the complete production module and its actual sibling dependencies."""
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("host_" + path.stem, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def definitions(path, names, **namespace):
    """Execute real definitions while isolating unrelated network/model imports."""
    nodes = [
        n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) in names
    ]
    assert {n.name for n in nodes} == set(names)
    namespace["__name__"] = __name__
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_failed_model_load_replaces_prior_success_evidence(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"pt": {"status": "PASS"}}))

    def fail_load(*args):
        raise ValueError("controlled missing local model")

    from contextlib import nullcontext

    function = definitions(
        CONVERT / "run.py",
        ["process_model"],
        OUT=tmp_path / "outputs",
        RESULTS=tmp_path,
        CACHE=tmp_path,
        TARGET_FORMATS=["pt", "onnx"],
        load=fail_load,
        master_logger=lambda: logging.getLogger(__name__),
        timed=lambda *args: nullcontext(),
        json=json,
    )["process_model"]
    result = function("model")
    assert json.loads(path.read_text()) == result
    assert all(r["status"] == "LOAD_FAIL" for r in result.values())
