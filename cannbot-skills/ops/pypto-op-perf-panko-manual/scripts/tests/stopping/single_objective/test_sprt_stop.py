# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""What closes an action, and nothing else.

There were ten OR'd conditions. Six of them asked the same question -- the
action proposed something that produced no measurement -- with five different
bounds set at five different times, and two of those bounds had never fired at
all. One, the SPRT, tested the mean of the margins while improvements live in
the tail: replayed over recorded campaigns it saved almost nothing, it could
not close an action whose candidates were disastrously slow (its MAD grew with
the disasters), and removing its MAD guard cut the action that won its
campaign.

Three questions remain, and they are the whole rule:

  * `n >= stagnation_K` -- measured, and the readings put it off the front;
  * `n_undetermined >= undetermined_K` -- measured, and they could not tell;
  * `n_unmeasured >= unmeasured_K` -- no measurement was produced at all.

Plus one escape that is not a judgement: an action that produced nothing
because the coder found the delta inexpressible, which would otherwise
deadlock.

`refuse_early_close` calls `disproved_on_structure`, so what `close` honours cannot drift
from what `record` reports -- the drift was a real deadlock, and the per-kind
escapes existed to patch it one kind at a time.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import DEMO_KERNEL as KERNEL, file_hash, harness, INIT_ARGS, new_op_dir, run_cli  # noqa: E402


class RuleTest(unittest.TestCase):
    """`disproved_on_structure` is the rule; this is its whole truth table."""

    def setUp(self):
        self.st = {"config": dict(harness.DEFAULT_CONFIG)}
        self.k = harness.DEFAULT_CONFIG["stagnation_K"]

    def test_nothing_yet_is_not_a_close(self):
        self.assertFalse(self._ask({"n": 0})[0])
        self.assertFalse(self._ask({"n": self.k - 1})[0])

    def test_the_three_bounds(self):
        for rec in ({"n": self.k},
                    {"n_undetermined": harness.DEFAULT_CONFIG["undetermined_K"]},
                    {"n_dup": harness.DEFAULT_CONFIG["unmeasured_K"]}):
            ok, why = self._ask(rec)
            self.assertTrue(ok, rec)
            self.assertTrue(why, rec)

    def test_the_unmeasured_kinds_share_one_bound(self):
        """Six conditions became one. A reader still gets the breakdown."""
        unm = harness.DEFAULT_CONFIG["unmeasured_K"]
        self.assertFalse(self._ask({"n_static": unm - 1})[0])
        ok, why = self._ask({"n_static": unm - 1, "n_dup": 1})
        self.assertTrue(ok)
        self.assertIn("static", why)
        self.assertIn("dup", why)

    def test_the_counters_do_not_borrow_from_each_other(self):
        """One short on each of the three is not a close."""
        self.assertFalse(self._ask({
            "n": self.k - 1,
            "n_undetermined": harness.DEFAULT_CONFIG["undetermined_K"] - 1,
            "n_static": harness.DEFAULT_CONFIG["unmeasured_K"] - 1})[0])

    def test_what_is_gone(self):
        for name in ("sprt_decision", "_sprt_update", "margin_sigma"):
            self.assertFalse(hasattr(harness, name), name)
        for key in ("sprt", "sprt_alpha", "sprt_min_n", "infeasible_K", "timeout_K"):
            self.assertNotIn(key, harness.DEFAULT_CONFIG, key)

    def test_a_configured_bound_is_honoured(self):
        self.st["config"]["unmeasured_K"] = 2
        self.assertTrue(self._ask({"n_dup": 2})[0])
        self.st["config"]["unmeasured_K"] = None
        self.assertFalse(self._ask({"n_dup": 99})[0],
                         "None turns the bound off, as for the others")

    def _ask(self, rec):
        return harness.disproved_on_structure(self.st, rec, self.k)


class CloseAgreesTest(unittest.TestCase):
    """`record` must never report a close that `close` then refuses."""

    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", file_hash(self.f),
                "--op-file", str(self.f), "--allow-unversioned", *INIT_ARGS)

    def test_each_bound_releases_the_close(self):
        cfg = harness.DEFAULT_CONFIG
        for counters in ({"n": cfg["stagnation_K"]},
                         {"n_undetermined": cfg["undetermined_K"]},
                         {"n_dup": cfg["unmeasured_K"]},
                         {"n_static": 4, "n_noop": 3, "n_timeout": 3}):
            self._refine(**counters)
            st = json.loads(self._path().read_text())
            self.assertIsNone(harness.refuse_early_close(st, "u1"), counters)

    def test_below_every_bound_the_close_is_refused_and_says_all_three(self):
        self._refine(n=1, n_undetermined=1, n_static=1)
        st = json.loads(self._path().read_text())
        why = harness.refuse_early_close(st, "u1")
        self.assertIsNotNone(why)
        for part in ("n=1 of stagnation_K=7",
                     "n_undetermined=1 of 7",
                     "n_unmeasured=1 of 10"):
            self.assertIn(part, why)

    def test_an_action_that_produced_nothing_can_still_be_closed(self):
        """The inexpressible-delta escape: nothing measured, nothing counted."""
        st = json.loads(self._path().read_text())
        st["progress"]["refine"] = None
        self._path().write_text(json.dumps(st), encoding="utf-8")
        st = json.loads(self._path().read_text())
        self.assertIsNone(harness.refuse_early_close(st, "u1"))

    def test_reaching_the_target_releases_the_close_without_retiring(self):
        """
        `success` is closeable but not spent -- it drives delivery, and an
        action that just hit the target is the last thing to retire.
        """
        self._refine(n=0, best={"code_hash": "x" * 64, "J": 125.0, "p": 80.0})
        st = json.loads(self._path().read_text())
        self.assertIsNone(harness.refuse_early_close(st, "u1"))
        self.assertFalse(harness.disproved_on_structure(st, st["progress"]["refine"],
                                                        st["config"]["stagnation_K"])[0])

    def _path(self):
        return self.op_dir / "optimization" / "search_state.json"

    def _refine(self, **counters):
        st = json.loads(self._path().read_text())
        st["progress"]["refine"] = dict(
            {"action": "u1", "best": None, "n": 0, "n_undetermined": 0,
             "n_static": 0, "n_noop": 0, "n_dup": 0, "n_timeout": 0},
            **counters)
        self._path().write_text(json.dumps(st), encoding="utf-8")
        return st


if __name__ == "__main__":
    unittest.main()
