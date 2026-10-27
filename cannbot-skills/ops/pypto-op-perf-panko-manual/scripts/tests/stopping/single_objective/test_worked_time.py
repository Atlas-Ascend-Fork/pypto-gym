# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""`wall_clock_s` is the time the campaign was WORKED, not the time it existed.

One field, one meaning: `wall_clock_limit_s` bounds `wall_clock_s` and that is
the whole story. It used to be `now - _start_ts`, and a PANKO campaign is driven
one harness invocation at a time, so its elapsed time includes every hour nobody
was calling it -- an agent session that ended, a night, a device lent out.

A recorded run can spend most of its elapsed time idle -- a single gap between
two consecutive successful outcomes on the same action is enough -- and so
carry an elapsed time past its limit while having worked a fraction of it. A
`stop` reading elapsed time would halt such a run with most of its time and
most of its evaluations unspent.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import harness_source, SCRIPTS_DIR  # noqa: E402,F401

from _support import harness as H  # noqa: E402

GRACE = H.DEFAULT_CONFIG["idle_grace_s"]


class AccumulateTest(unittest.TestCase):

    def test_a_short_interval_is_charged_in_full(self):
        pr = self._pr(300)
        H.accumulate_wall_clock(pr, {})
        self.assertAlmostEqual(pr["wall_clock_s"], 300, delta=2)

    def test_intervals_add_up(self):
        pr = self._pr(300, worked=1000)
        H.accumulate_wall_clock(pr, {})
        self.assertAlmostEqual(pr["wall_clock_s"], 1300, delta=2)

    def test_an_idle_gap_is_capped_at_the_grace(self):
        """The idle gap sat between two successful outcomes."""
        pr = self._pr(46.5 * 3600)
        H.accumulate_wall_clock(pr, {})
        self.assertEqual(pr["wall_clock_s"], GRACE)

    def test_the_longest_real_cycle_is_still_charged_in_full(self):
        """A tile block is the longest cycle a campaign has."""
        pr = self._pr(3360)
        H.accumulate_wall_clock(pr, {})
        self.assertAlmostEqual(pr["wall_clock_s"], 3360, delta=2)
        self.assertLess(3360, GRACE)

    def test_the_grace_is_configurable(self):
        pr = self._pr(1000)
        H.accumulate_wall_clock(pr, {"idle_grace_s": 60})
        self.assertEqual(pr["wall_clock_s"], 60)

    def test_the_first_interval_runs_from_init(self):
        now = time.time()
        pr = {"_start_ts": now - 120}
        H.accumulate_wall_clock(pr, {})
        self.assertIn("wall_clock_s", pr)
        self.assertAlmostEqual(pr.get("wall_clock_s"), 120, delta=2)

    def test_a_clock_that_went_backwards_charges_nothing(self):
        pr = {"_start_ts": time.time() + 500, "last_seen_ts": time.time() + 500}
        H.accumulate_wall_clock(pr, {})
        self.assertIn("wall_clock_s", pr)
        self.assertEqual(pr.get("wall_clock_s"), 0)

    def test_two_saves_in_one_cycle_do_not_double_charge(self):
        pr = self._pr(300)
        H.accumulate_wall_clock(pr, {})
        first = pr["wall_clock_s"]
        H.accumulate_wall_clock(pr, {})
        self.assertAlmostEqual(pr["wall_clock_s"], first, delta=2)

    def _pr(self, last_ago, worked=0):
        now = time.time()
        return {"_start_ts": now - last_ago, "last_seen_ts": now - last_ago,
                "wall_clock_s": worked}


class ResumeTest(unittest.TestCase):
    """A state written when `wall_clock_s` meant elapsed."""

    def test_the_old_elapsed_value_becomes_the_starting_point(self):
        """
        It overstates the work done, so the resumed run keeps a budget no
        larger than the one it was launched under.
        """
        now = time.time()
        pr = {"_start_ts": now - 300, "wall_clock_s": 228126}
        H.accumulate_wall_clock(pr, {})
        self.assertGreaterEqual(pr["wall_clock_s"], 228126)
        self.assertLess(pr["wall_clock_s"], 228126 + GRACE + 2)

    def test_elapsed_time_is_not_stored(self):
        src = harness_source()
        self.assertNotIn('pr["wall_clock_s"] = int(time.time() - pr["_start_ts"])',
                         src)
        self.assertIn('"elapsed_s": int(time.time() - pr["_start_ts"])', src)


class StopConditionTest(unittest.TestCase):
    """The shape at issue: elapsed past the limit, worked well inside it."""

    class _A:
        op_dir = "."

    def test_an_idle_run_is_not_halted(self):
        st = self._st(worked=int(17.4 * 3600), elapsed=int(63.4 * 3600))
        self.assertNotEqual(H.stop_condition(st, self._A()), "wall_clock")

    def test_a_worked_run_past_the_limit_is_halted(self):
        st = self._st(worked=48 * 3600, elapsed=48 * 3600)
        self.assertEqual(H.stop_condition(st, self._A()), "wall_clock")

    def test_the_evaluation_budget_still_halts(self):
        st = self._st(worked=60, elapsed=60)
        st["progress"]["evals_used"] = 300
        self.assertEqual(H.stop_condition(st, self._A()), "eval_budget")

    def _st(self, worked, elapsed):
        return {"progress": {"wall_clock_s": worked,
                             "_start_ts": time.time() - elapsed,
                             "evals_used": 117, "global_stagnation": 0},
                "config": {"wall_clock_limit_s": 48 * 3600, "eval_budget": 300,
                           "success_stop": False},
                # `frontier_exhausted` is checked after the time and budget
                # conditions and walks the open set, so the fixture carries one.
                "nodes": {"open": [{"id": "x1"}], "closed": [], "pruned": [],
                          "held": []},
                "archive": []}


if __name__ == "__main__":
    unittest.main()
