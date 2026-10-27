# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""The run delivers the best program it measured, closed or not.

`best_J` moves at `close`, and the budget does not wait for one. A campaign
measured a candidate better than its preopt, recorded it as `kept`,
and then ran out of evaluations with the action still open -- so `stop`
delivered the preopt and the verification read `no_change`.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import DEMO_KERNEL as KERNEL, file_hash, INIT_ARGS, new_op_dir, read_state, run_cli  # noqa: E402


class PromoteAtStopTest(unittest.TestCase):

    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", self.pre,
                "--op-file", str(self.f), "--allow-unversioned", *INIT_ARGS,
                "--config", json.dumps({"eval_budget": 2}))

    def test_an_open_actions_winner_is_delivered_when_the_budget_ends_the_run(self):
        better, res = self._record(128, 90.0)
        self.assertTrue(res["improved"], res)
        st = read_state(self.op_dir)
        self.assertEqual(st["progress"]["best_aicore_e2e_us"], 100.0,
                         "the ratchet still waits for close")
        stop, _ = run_cli("stop", "--op-dir", str(self.op_dir))
        self.assertTrue(stop["stop"])
        self.assertEqual(stop["reason"], "eval_budget")
        self.assertEqual(stop["best_aicore_e2e_us"], 90.0)
        st = read_state(self.op_dir)
        self.assertEqual(st["progress"]["promoted_at_stop"]["p"], 90.0)
        self.assertEqual(file_hash(self.f), better, "and it is what is on disk")
        node = next(c for c in st["nodes"]["closed"] if c["id"] == st["best_node"])
        self.assertEqual(node["code_hash"], better)
        self.assertEqual(node["closed_at"], "stop")
        self.assertIsNone(st["progress"]["refine"], "the action is consumed")
        log = next((self.op_dir / "optimization").glob("*.md")).read_text()
        self.assertIn("- promote-at-stop[u1]", log)

    def test_a_candidate_that_did_not_beat_the_best_is_not_promoted(self):
        _, res = self._record(128, 120.0)
        self.assertFalse(res["improved"])
        stop, _ = run_cli("stop", "--op-dir", str(self.op_dir))
        self.assertTrue(stop["stop"])
        self.assertEqual(stop["best_aicore_e2e_us"], 100.0)
        st = read_state(self.op_dir)
        self.assertNotIn("promoted_at_stop", st["progress"])
        self.assertEqual(file_hash(self.f), self.pre)

    def test_stop_is_idempotent(self):
        """
        `stop` is sticky and may be called again; the promotion is not a
        second closed node.
        """
        better, _ = self._record(128, 90.0)
        run_cli("stop", "--op-dir", str(self.op_dir))
        first = read_state(self.op_dir)
        run_cli("stop", "--op-dir", str(self.op_dir))
        st = read_state(self.op_dir)
        self.assertEqual(len(st["nodes"]["closed"]), len(first["nodes"]["closed"]))
        self.assertEqual(st["best_node"], first["best_node"])
        self.assertEqual(file_hash(self.f), better)

    def _record(self, tile, p):
        self.f.write_text(KERNEL.replace("64, 128", f"{tile}, 128"), encoding="utf-8")
        h = file_hash(self.f)
        res, _ = run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                         "--code-hash", h, "--s", "1", "--p", str(p))
        return h, res


if __name__ == "__main__":
    unittest.main()
