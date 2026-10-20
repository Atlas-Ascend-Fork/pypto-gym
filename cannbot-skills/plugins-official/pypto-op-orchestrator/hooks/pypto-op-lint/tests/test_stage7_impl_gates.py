# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Stage-7 enforcement of impl-structure lint rules.

Stage 7 (K-Search) rewrites `<op>_impl.py`, so the impl kernel-contract rules
must hold there too — otherwise the optimizer can ship structural violations
(e.g. a host-side `for ... in range(...)` that launches the JIT kernel N times to
game the per-launch latency measurement). We added stage 7 to the S0+S1 impl
rules, EXCEPT OL56 (multi-value `unroll_list` is a legitimate Stage-7 tuning).
"""

import json
from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule, write_file

_JIT = """import pypto
_N = pypto.DYNAMIC
_M = pypto.STATIC
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([_N, _M], pypto.DT_BF16), y: pypto.Tensor([_N, _M], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x
"""

_HOST_LOOP = (
    _JIT
    + """
def demo_wrapper(x, y):
    n_slices = 24
    for s in range(n_slices):
        demo_kernel(x, y)
    return y
"""
)

_SINGLE_LAUNCH = (
    _JIT
    + """
def demo_wrapper(x, y):
    demo_kernel(x, y)
    return y
"""
)

_S0S1 = [
    "OL01",
    "OL07",
    "OL45",
    "OL48",
    "OL55",
    "OL57",
    "OL58",
    "OL02",
    "OL03",
    "OL04",
    "OL05",
    "OL06",
    "OL08",
    "OL16",
    "OL25",
    "OL26",
    "OL49",
    "OL50",
    "OL52",
]


def test_ol45_host_loop_now_fails_at_stage7(tmp_path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    write_file(op_dir / "demo_impl.py", _HOST_LOOP)
    assert (
        run_rule(mod, op_dir, "OL45", stage=7).status == "FAIL"
    )  # newly enforced at Stage 7
    assert (
        run_rule(mod, op_dir, "OL45", stage=5).status == "FAIL"
    )  # unchanged at authoring stage


def test_ol45_single_launch_passes_at_stage7(tmp_path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    write_file(op_dir / "demo_impl.py", _SINGLE_LAUNCH)
    assert run_rule(mod, op_dir, "OL45", stage=7).status == "PASS"


def test_ol56_multi_unroll_exempt_at_stage7(tmp_path):
    # OL56 (single-value unroll_list before Stage 6) must NOT gate Stage 7 —
    # multi-value unroll tuning is a legitimate optimization action there.
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    assert (
        run_rule(mod, op_dir, "OL56", stage=7).status == "SKIP"
    )  # not applicable at Stage 7
    assert (
        run_rule(mod, op_dir, "OL56", stage=5).status != "SKIP"
    )  # still active while authoring


def test_rules_json_stage7_membership():
    rules_path = Path(__file__).resolve().parents[1] / "rules.json"
    with open(rules_path, encoding="utf-8") as fh:
        rules = json.load(fh)["rules"]
    byid = {r["id"]: r for r in rules}
    for rid in _S0S1:
        assert 7 in byid[rid]["stages"], f"{rid} must be enforced at Stage 7"
    assert 7 not in byid["OL56"]["stages"], "OL56 must stay exempt at Stage 7"
