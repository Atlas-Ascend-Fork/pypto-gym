# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""What the report says at the end of a run.

A run is reviewed from its trajectory file, and its real story -- a budget
eaten by one action's tile blocks, a measured improvement that was never
delivered -- used to be visible only to someone who opened the state JSON. So
the accounting is written out, and the front is presented with the picks a
person would otherwise have to work out by hand.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import (CHIP_ARGS, DEMO_KERNEL as KERNEL, file_hash, harness, new_op_dir,  # noqa: E402
                      record_measurement, run_cli)

OBJS = harness.parse_objectives({"objectives": ["mix_aic", "workspace"]})


def _row(h, lat, ws):
    return {"code_hash": h, "values": {"mix_aic": lat, "workspace": float(ws)},
            "objectives_text": f"mix_aic={lat} workspace={ws / 2 ** 20:.1f}MiB",
            "source": f"optimization/frontier/{h[:8]}.py", "verified": None}


class RecommendTest(unittest.TestCase):
    """The extremes are facts about the front; `balanced` is a convention."""

    def test_each_objective_gets_its_best_point(self):
        rows = [_row("a" * 64, 80.0, 1600 << 20),
                _row("b" * 64, 90.0, 800 << 20),
                _row("c" * 64, 99.0, 300 << 20)]
        rec = harness.recommend(OBJS, rows)
        self.assertEqual(rec["best"]["mix_aic"]["code_hash"], "a" * 64)
        self.assertEqual(rec["best"]["workspace"]["code_hash"], "c" * 64)

    def test_the_knee_is_the_point_nearest_the_ideal_corner(self):
        rows = [_row("a" * 64, 80.0, 1600 << 20),
                _row("b" * 64, 82.0, 400 << 20),      # nearly as fast, a quarter
                _row("c" * 64, 99.0, 300 << 20)]
        self.assertEqual(harness.recommend(OBJS, rows)["balanced"]["code_hash"],
                         "b" * 64)

    def test_two_points_have_no_knee(self):
        """Each one IS an extreme, so choosing between them is the preference."""
        rows = [_row("a" * 64, 80.0, 1600 << 20), _row("b" * 64, 99.0, 300 << 20)]
        rec = harness.recommend(OBJS, rows)
        self.assertNotIn("balanced", rec)
        self.assertEqual(set(rec["best"]), {"mix_aic", "workspace"})

    def test_one_point_is_nothing_to_choose_between(self):
        self.assertEqual(harness.recommend(OBJS, [_row("a" * 64, 80.0, 1 << 20)]), {})


class ReportBlockTest(unittest.TestCase):

    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "100",
                # Not 100: a preopt that already meets the target is refused
                # (test_target.py). The 80 us candidate below meets it, which
                # is what releases the close.
                "--preopt-p", "120", "--preopt-hash", file_hash(self.f),
                "--preopt-workspace-b", str(1000 << 20),
                "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS,
                "--config", json.dumps({"objectives": ["mix_aic", "workspace"],
                                        "eval_budget": 4}))
        self.fast = self._record("128, 128", 80.0, 1600 << 20)
        self.mid = self._record("96, 128", 82.0, 400 << 20)
        self.small = self._record("32, 128", 99.0, 300 << 20)
        run_cli("close", "--op-dir", str(self.op_dir), "--action", "u1")
        run_cli("stop", "--op-dir", str(self.op_dir))
        run_cli("verify", "--op-dir", str(self.op_dir), "--skip", "not under test")
        self.res = run_cli("report", "--op-dir", str(self.op_dir))[0]
        self.md = next((self.op_dir / "optimization").glob("*.md")).read_text()

    def test_the_accounting_says_where_the_budget_went(self):
        block = self.md.split("## Accounting")[1]
        self.assertIn("evaluations: 4 charged | 0 in tile blocks", block)
        self.assertIn("share 0.5", block)
        self.assertIn("since the archive last accepted a point", block)
        self.assertIn("hv contribution", block)
        self.assertIn("kept though slower than the fastest known", block)
        self.assertIn("sigma_log", block)

    def test_the_picks_are_named_with_their_sources(self):
        block = self.md.split("## Recommended")[1]
        self.assertIn(f"balanced               {self.mid[:8]}", block)
        self.assertIn(f"best mix_aic           {self.fast[:8]}", block)
        self.assertIn(f"best workspace         {self.small[:8]}", block)
        self.assertIn("optimization/frontier/", block)
        self.assertIn("A convention, not a preference", block)
        self.assertIn("impl.py is unchanged by this section", block)

    def test_the_payload_carries_them_too(self):
        rec = self.res["recommended"]
        self.assertEqual(rec["balanced"], self.mid)
        self.assertEqual(rec["best"]["mix_aic"], self.fast)
        self.assertEqual(rec["best"]["workspace"], self.small)

    def test_impl_py_is_still_the_fastest_point(self):
        self.assertEqual(file_hash(self.f), self.fast)

    def _record(self, tile, p, ws):
        return record_measurement(self, KERNEL.replace("64, 128", tile), p, ws)[0]


if __name__ == "__main__":
    unittest.main()
