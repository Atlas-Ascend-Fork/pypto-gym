# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Front acceptance: the ratchet keeps what extends the front (B1).

`_apply_ratchet` used to keep a candidate iff `j > bar` -- a scalar on the
primary objective, whatever `config.objectives` declared, so a multi-objective
run searched exactly like a single-objective one. It now keeps a candidate iff
no archive point dominates it on the medians and it is no slower than the
preopt (the reference point, primary axis only).

What these pin down:
  * one objective: the rule is `j > bar`, verdict for verdict;
  * a kept trade-off is where the action CONTINUES from (`base`), but the
    fastest point it accepted (`best`) is what `close` promotes;
  * a block winner already in the archive is not compared with itself;
  * the outcome store keeps its latency meaning: a kept trade-off is
    `accepted`, not `kept` with a negative gain.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import CHIP_ARGS, DEMO_KERNEL as KERNEL, file_hash, harness, new_op_dir, read_state, run_cli  # noqa: E402


class _Run(unittest.TestCase):
    OBJECTIVES = None

    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        extra = []
        if self.OBJECTIVES:
            extra = ["--config", json.dumps({"objectives": self.OBJECTIVES})]
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", self.pre,
                "--preopt-workspace-b", str(1000 << 20),
                "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS, *extra)

    def _record(self, tile, p, ws=1000 << 20):
        self.f.write_text(KERNEL.replace("64, 128", f"{tile}, 128"), encoding="utf-8")
        h = file_hash(self.f)
        res, _ = run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                         "--code-hash", h, "--s", "1", "--p", str(p),
                         "--workspace-b", str(ws))
        return h, res

    def _outcomes(self):
        path = self.op_dir / "optimization" / "action_outcomes.jsonl"
        return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


class OneObjectiveTest(_Run):
    """With one objective, front acceptance is the scalar ratchet."""
    OBJECTIVES = ["aicore_e2e"]

    def test_the_verdicts_are_the_scalar_ratchet_s(self):
        seq = [(128, 90.0, "kept"), (256, 90.0, "tied"), (32, 95.0, "reverted"),
               (16, 85.0, "kept"), (8, 85.5, "reverted")]
        bar = 100.0
        for tile, p, want in seq:
            h, res = self._record(tile, p)
            self.assertEqual(read_state(self.op_dir)["eval_cache"][h]["verdict"], want, (tile, p))
            self.assertEqual(res["improved"], p < bar, (tile, p))
            self.assertEqual(res["faster"], p < bar)
            if p < bar:
                bar = p


class TradeOffTest(_Run):
    def test_a_trade_off_is_kept_and_the_action_continues_from_it(self):
        fast, _ = self._record(128, 90.0, 1600 << 20)
        small, res = self._record(256, 95.0, 380 << 20)
        self.assertTrue(res["improved"])
        self.assertFalse(res["faster"])
        self.assertTrue(res["acceptance"]["on_front"])
        self.assertTrue(res["acceptance"]["within_reference"])
        r = read_state(self.op_dir)["progress"]["refine"]
        self.assertEqual(r["base"]["code_hash"], small)
        self.assertEqual(r["best"]["code_hash"], fast, "best stays the fastest")
        # A dominated candidate is rolled back onto the BASE, not onto best.
        self._record(32, 99.0, 1700 << 20)
        self.assertEqual(file_hash(self.f), small)

    def test_close_promotes_the_fastest_not_the_base(self):
        fast, _ = self._record(128, 90.0, 1600 << 20)
        self._record(256, 95.0, 380 << 20)
        # The base is now the trade-off; exhaust the action from there.
        for i, tile in enumerate((8, 16, 24, 32, 40, 48, 56)):
            self._record(tile, 99.0 + i, 1700 << 20)
        res, rc = run_cli("close", "--op-dir", str(self.op_dir), "--action", "u1")
        self.assertEqual(rc, 0, res)
        self.assertEqual(res["best_aicore_e2e_us"], 90.0)
        self.assertEqual(file_hash(self.f), fast)

    def test_slower_than_the_preopt_is_not_accepted(self):
        _, res = self._record(256, 105.0, 380 << 20)
        self.assertFalse(res["improved"])
        self.assertTrue(res["acceptance"]["on_front"])
        self.assertFalse(res["acceptance"]["within_reference"])
        self.assertEqual(res["acceptance"]["reference"], {"aicore_e2e": 100.0})
        self.assertEqual(res["archive"], "inserted", "the front still keeps it")
        self.assertEqual(file_hash(self.f), self.pre)

    def test_a_dominated_point_is_reverted(self):
        self._record(128, 90.0, 1600 << 20)
        _, res = self._record(256, 95.0, 1700 << 20)
        self.assertFalse(res["improved"])
        self.assertFalse(res["acceptance"]["on_front"])
        self.assertLess(res["acceptance"]["front_margin"], 0.0)

    def test_the_outcome_store_keeps_its_latency_meaning(self):
        self._record(128, 90.0, 1600 << 20)
        self._record(256, 95.0, 380 << 20)
        rows = self._outcomes()
        self.assertEqual([(r["kept"], r["accepted"]) for r in rows],
                         [(True, True), (False, True)])
        self.assertEqual(rows[1]["log_gain"], 0.0)


class SelfExclusionTest(unittest.TestCase):
    """A block winner is archived by the block before `record` promotes it."""

    def test_a_point_is_not_compared_with_its_own_row(self):
        st = {"config": {"objectives": ["aicore_e2e", "workspace"]},
              # `_failure_clarity` builds the candidate's archive row, which
              # carries `J` and the evaluation number.
              "target": {"P_ref": 50.0}, "progress": {"evals_used": 0},
              "nodes": {"closed": [{"id": "root", "code_hash": "pre"}]},
              "eval_cache": {"pre": {"s": 1, "p": 100.0,
                                     "m": {"workspace_b": 1000,
                                           "workspace_state": "measured"}}},
              "archive": [{"code_hash": "pre",
                           "values": {"aicore_e2e": 100.0, "workspace": 1000.0}},
                          {"code_hash": "win",
                           "values": {"aicore_e2e": 90.0, "workspace": 1000.0}}]}
        m = {"workspace_b": 1000, "workspace_state": "measured"}
        ok, detail = harness.front_acceptance(st, "win", 90.0, m)
        self.assertTrue(ok)
        self.assertEqual(detail["compared_with"], 1)
        # And the same vector under another hash ties with it.
        ok, _ = harness.front_acceptance(st, "other", 90.0, m)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
