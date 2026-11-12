# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read the single kernel's integer contract without importing PyPTO or NPU runtime."""

import ast
import operator
from pathlib import Path

_CONTRACT_NAMES = [
    "HEADS",
    "HEAD_DIM",
    "GROUPS",
    "GROUP_N",
    "GEMM1_K",
    "OUT_FEATURES",
    "GEMM2_K",
    "PAD_MULTIPLE",
    "AQ_BYTES_PER_ROW",
    "ASC_BYTES_PER_ROW",
    "YQ_BYTES_PER_ROW",
    "YS_BYTES_PER_ROW",
    "ASC_OFFSET_PER_ROW",
    "YQ_OFFSET_PER_ROW",
    "YS_OFFSET_PER_ROW",
    "WORKSPACE_BYTES_PER_ROW",
]
_INTEGER_OPERATORS = {
    ast.Add: operator.add,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
}


def _integer_value(node, values):
    """Evaluate only the contract's integer literals, references and arithmetic."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _INTEGER_OPERATORS:
        return _INTEGER_OPERATORS[type(node.op)](
            _integer_value(node.left, values), _integer_value(node.right, values)
        )
    raise ValueError("Unsupported expression in the kernel integer contract")


def _read_contract():
    """Extract the required assignments; never execute or import the kernel."""
    path = Path(__file__).resolve().parents[2] / "op_kernel/atten_epilogue.py"
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in _CONTRACT_NAMES:
                values[target.id] = _integer_value(node.value, values)
    if set(values) != set(_CONTRACT_NAMES):
        raise ValueError("Incomplete kernel integer contract")
    return values


_CONTRACT = _read_contract()
globals().update(_CONTRACT)
WIDE_CUBE_CORES = 32
PROFILE_COUNT = 19


def padded_tokens(t):
    """Round token rows up to the kernel's required granularity."""
    multiple = _CONTRACT["PAD_MULTIPLE"]
    return (t + multiple - 1) // multiple * multiple


def workspace_bytes(t):
    """Return the flat workspace size reported by host tiling, in bytes."""
    return _CONTRACT["WORKSPACE_BYTES_PER_ROW"] * padded_tokens(t)
