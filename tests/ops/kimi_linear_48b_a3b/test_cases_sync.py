# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contract test: ``test_cases.json`` documents EXACTLY the op-test run grid.

``test_cases.json`` is a descriptive file — ``test_kda_chunk.py`` drives its grid
directly from ``_kda_test_common.chunk_grid`` and does **not** read the JSON. That
left the JSON free to drift from what actually runs (it did: the no-initial-state
cases were mis-documented as having an ``initial_state``). This test closes that
gap by asserting the JSON documents exactly the same set of cases and the same
shape/dtype/tolerance/scale envelope. It needs no NPU, so it runs everywhere.
"""

import json
import logging
from pathlib import Path

from _kda_test_common import chunk_grid, HEAD_DIM, RTOL, ATOL

logging.basicConfig(level=logging.INFO, format="%(message)s")

_JSON = Path(__file__).resolve().parent / "test_cases.json"


def _case_key(case):
    """(B, H, T, gate, with_state) identity of a run-grid KdaCase."""
    return (case.batch, case.num_heads, case.seq_len, case.gate, case.with_state)


def _json_key(c):
    """Same identity, read back from a test_cases.json entry (q is [B,T,H,K])."""
    q = c["input"]["q"]["shape"]
    has_state = c["input"].get("initial_state") is not None
    return (q[0], q[2], q[1], c["gate"], has_state)


def test_cases_json_matches_grid():
    data = json.loads(_JSON.read_text())
    tc = data["test_cases"]
    by_key = {_case_key(c): c for c in chunk_grid()}

    for label, grid, cases in (
            ("chunk", chunk_grid(), [c for c in tc if c["id"].startswith("chunk")]),):
        want = {_case_key(c) for c in grid}
        got = {_json_key(c) for c in cases}
        assert len(grid) == len(cases), (
            f"{label}: run grid has {len(grid)} cases, test_cases.json has {len(cases)}")
        assert want == got, (
            f"{label}: test_cases.json case set != run grid.\n"
            f"  only in grid: {sorted(want - got)}\n"
            f"  only in json: {sorted(got - want)}")

    for c in tc:
        inp = c["input"]
        cid = c["id"]
        for t in ("q", "k", "v"):
            assert inp[t]["dtype"] == "bfloat16", f"{cid}: {t} must be bf16 (real model)"
        assert inp["g"]["dtype"] == "float32", f"{cid}: g must be fp32 (fused_kda_gate)"
        assert inp["beta"]["dtype"] == "float32", f"{cid}: beta must be fp32 (.float().sigmoid())"
        assert inp["q"]["shape"][-1] == HEAD_DIM and inp["v"]["shape"][-1] == HEAD_DIM, \
            f"{cid}: head dim must be {HEAD_DIM}"
        assert abs(c["rtol"] - RTOL) < 1e-12 and abs(c["atol"] - ATOL) < 1e-12, \
            f"{cid}: tolerance must be rtol=atol={RTOL}"
        assert abs(inp["scale"]["value"] - HEAD_DIM ** -0.5) < 1e-12, \
            f"{cid}: scale must be HEAD_DIM**-0.5"
        case = by_key[_json_key(c)]
        b, t, h = case.batch, case.seq_len, case.num_heads
        assert inp["q"]["shape"] == [b, t, h, HEAD_DIM], f"{cid}: q shape"
        assert inp["beta"]["shape"] == [b, t, h], f"{cid}: beta shape"
        if case.with_state:
            assert inp.get("initial_state") is not None, \
                f"{cid}: with_state case must document initial_state"
            assert inp["initial_state"]["shape"] == [b, h, HEAD_DIM, HEAD_DIM], \
                f"{cid}: initial_state shape"
        else:
            assert inp.get("initial_state") is None, \
                f"{cid}: no-state case must have initial_state null/absent"

    logging.info(f"[sync] test_cases.json documents exactly {len(tc)} cases, "
                 f"matching the chunk_grid() run set")


def main():
    test_cases_json_matches_grid()
    logging.info("[PRECISION_PASS]")
    logging.info("All tests passed!")


if __name__ == "__main__":
    main()
