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


@pytest.mark.parametrize("filename", ["compare.py", "port.py"])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
    ],
)
def test_float8_real_cpu_outputs_export_as_float32(filename, dtype):
    tensor = torch.tensor([0.0, 1.0, -2.0]).to(dtype)
    function = definitions(CONVERT / filename, ["_to_numpy_safe"], torch=torch)[
        "_to_numpy_safe"
    ]
    output = function(tensor)
    assert output.dtype == np.float32
    np.testing.assert_array_equal(output, [0, 1, -2])


def test_integer_output_dtype_is_preserved():
    function = definitions(CONVERT / "compare.py", ["_to_numpy_safe"], torch=torch)[
        "_to_numpy_safe"
    ]
    assert function(torch.tensor([3], dtype=torch.int64)).dtype == np.int64
