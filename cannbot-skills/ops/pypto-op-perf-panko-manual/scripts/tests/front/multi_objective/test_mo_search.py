# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""One search on the front, with no phase split.

The phase split is gone (C2). Acceptance is non-domination within the
reference point and the next action starts from the largest hypervolume
contribution, from the first evaluation -- so there is nothing left for a
second phase to do, and `mo_phases` / `greedy_stagnation_limit` /
`front_stagnation_limit` / `progress.phase` with them.

These pin the base rule and what replaced the phase-2 bookkeeping.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import (CHIP_ARGS, DEMO_KERNEL as KERNEL, harness, new_op_dir, read_state,  # noqa: E402
                      record_measurement, run_cli)


def _pt(h, p, w, tried=0):
    return {"code_hash": h, "latency_us": p, "workspace_b": w, "tried_from": tried,
            "values": {"mix_aic": p, "workspace": w}}


class BaseSelectionTest(unittest.TestCase):
    """
    The base is the point with the largest exclusive hypervolume
    contribution; actions tried from it only break ties.
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.op_dir = Path(self.tmp_dir.name)
        nodes = self.op_dir / "optimization" / "nodes"
        nodes.mkdir(parents=True)
        for h in ("a" * 64, "b" * 64, "c" * 64):
            (nodes / f"{h}.py").write_text(KERNEL, encoding="utf-8")
        self.st = {"config": {"objectives": ["mix_aic", "workspace"]},
                   "chip_envelope": {"memory_mib": 65536},
                   "nodes": {"closed": [{"id": "root", "code_hash": "p" * 64}]},
                   "eval_cache": {"p" * 64: {"s": 1, "p": 100.0, "m": {
                       "mix_aic": 100.0, "workspace_b": 1000 << 20,
                       "workspace_state": "measured"}}},
                   "archive": [_pt("a" * 64, 90.0, 1600 << 20),
                               _pt("b" * 64, 95.0, 380 << 20),
                               _pt("c" * 64, 99.0, 200 << 20)]}

    def test_the_largest_contribution_is_the_base(self):
        """
        The reference is the preopt on latency and the chip's memory on the
        byte axis, so the FASTEST point holds the widest slab: contributions
        18.56 / 5.75 / 0.64. That lean toward the extreme is the intended
        effect of a far reference, not an accident of it.
        """
        detail = {}
        base = harness.front_base(self.st, str(self.op_dir), detail)
        self.assertEqual(base["code_hash"], "a" * 64)
        self.assertIn("candidates", detail)
        self.assertEqual(detail.get("candidates"), 3)

    def test_tried_from_only_breaks_a_tie(self):
        """
        Every contribution zero -- here because no reference can be formed
        without the chip's memory -- falls through to fewest-tried, then the
        primary objective.
        """
        self.st = dict(self.st, chip_envelope={})
        self.st["archive"] = [_pt("a" * 64, 1.0, 3.0), _pt("c" * 64, 3.0, 1.0)]
        base = harness.front_base(self.st, str(self.op_dir))
        self.assertEqual(base["code_hash"], "a" * 64, "the tie goes to the faster")
        self.st["archive"][0]["tried_from"] = 1
        base = harness.front_base(self.st, str(self.op_dir))
        self.assertEqual(base["code_hash"], "c" * 64, "then to the one tried less")

    def test_a_tried_point_keeps_the_base_if_it_contributes_most(self):
        self.st["archive"][0]["tried_from"] = 5
        base = harness.front_base(self.st, str(self.op_dir))
        self.assertEqual(base["code_hash"], "a" * 64)

    def test_a_point_whose_source_is_gone_is_skipped(self):
        (self.op_dir / "optimization" / "nodes" / f"{'a' * 64}.py").unlink()
        base = harness.front_base(self.st, str(self.op_dir))
        self.assertEqual(base["code_hash"], "b" * 64,
                         "a delta has to be applied to something")

    def test_no_placeable_point_is_no_base(self):
        for h in ("a", "b", "c"):
            (self.op_dir / "optimization" / "nodes" / f"{h * 64}.py").unlink()
        self.assertIsNone(harness.front_base(self.st, str(self.op_dir)))


class OneSearchTest(unittest.TestCase):
    """
    Through the CLI: no phase anywhere, and the front counter is a
    diagnostic rather than a halt.
    """

    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        res, _ = run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo",
                         "--p-ref", "50", "--preopt-p", "100", "--preopt-hash", self.pre,
                         "--preopt-workspace-b", str(1000 << 20),
                         "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS)
        self.assertTrue(res["initialized"], res)

    def test_no_command_reports_a_phase(self):
        self.assertNotIn("phase", read_state(self.op_dir)["progress"])
        self.assertNotIn("phase", run_cli("select", "--op-dir", str(self.op_dir))[0])
        self._record(128, 90.0, 1600 << 20)
        res, _ = run_cli("stop", "--op-dir", str(self.op_dir))
        for k in ("phase", "phase_switched", "note"):
            self.assertNotIn(k, res)

    def test_the_front_counter_measures_every_candidate(self):
        """
        It only counted inside phase 2 before, so it stayed 0 in every run.
        """
        self._record(128, 90.0, 1600 << 20)
        self.assertEqual(read_state(self.op_dir)["progress"]["front_stagnation"], 0)
        self._record(32, 99.0, 1700 << 20)                  # dominated
        self.assertEqual(read_state(self.op_dir)["progress"]["front_stagnation"], 1)
        self._record(16, 98.0, 1700 << 20)                  # dominated again
        self.assertEqual(read_state(self.op_dir)["progress"]["front_stagnation"], 2)
        self._record(256, 95.0, 380 << 20)                  # a new trade-off
        self.assertEqual(read_state(self.op_dir)["progress"]["front_stagnation"], 0)

    def test_the_front_counter_is_not_a_halt(self):
        self._record(128, 90.0, 1600 << 20)
        for i, tile in enumerate((8, 16, 24, 32, 40, 48, 56)):
            self._record(tile, 110.0 + i, 1700 << 20)
        self.assertGreaterEqual(read_state(self.op_dir)["progress"]["front_stagnation"], 7)
        res, _ = run_cli("stop", "--op-dir", str(self.op_dir))
        self.assertFalse(res["stop"], res)

    def test_the_outcome_row_carries_one_label(self):
        self._record(128, 90.0, 1600 << 20)
        self._record(256, 95.0, 380 << 20)
        outcomes = self.op_dir / "optimization" / "action_outcomes.jsonl"
        rows = []
        for ln in outcomes.read_text().splitlines():
            if ln.strip():
                rows.append(json.loads(ln))
        self.assertEqual({r["phase"] for r in rows}, {"search"})
        self.assertEqual([r["front_insert"] for r in rows], [True, True])

    def test_every_measurement_is_evidence_about_the_arm(self):
        """
        `_observe_action` skipped phase 2 entirely. With one search it
        observes every candidate, on the LATENCY question.
        """
        self._record(128, 90.0, 1600 << 20)
        trade = self._record(256, 95.0, 380 << 20)[1]
        self.assertTrue(trade["improved"], "accepted: it extends the front")
        self.assertFalse(trade["faster"])
        stats = read_state(self.op_dir)["action_stats"]["F-1"]
        self.assertEqual((stats["n"], stats["kept"]), (2, 1),
                         "both measured, only the faster one kept")

    def _record(self, tile, p, ws):
        code_hash, reply = record_measurement(
            self, KERNEL.replace("64, 128", f"{tile}, 128"), p, ws)
        return code_hash, reply


if __name__ == "__main__":
    unittest.main()
