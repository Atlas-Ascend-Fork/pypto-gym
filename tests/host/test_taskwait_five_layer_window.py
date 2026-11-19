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


@pytest.mark.parametrize("count", [5, 6, 7, 8, 12])
def test_five_available_anchors_produce_four_nonempty_segments(count):
    loaded = module(FUSED / "analyze_msprof_taskwait.py")
    rows = []
    for _ in range(count):
        rows.extend(
            [
                {"OP Type": "FlashAttentionScore"},
                {"OP Type": "Add", "Task Wait Time(us)": "2", "Task Duration(us)": "3"},
            ]
        )
    segments = getattr(loaded, "_analyse_single_layer_delay")(rows, "")
    assert len(segments) == 4
    assert all(s["tasks"] == 1 and s["dur"] == 3 for s in segments)


def test_fewer_than_five_anchors_still_skip():
    loaded = module(FUSED / "analyze_msprof_taskwait.py")
    assert (
        getattr(loaded, "_analyse_single_layer_delay")(
            [{"OP Type": "FlashAttentionScore"}] * 4, ""
        )
        == []
    )
