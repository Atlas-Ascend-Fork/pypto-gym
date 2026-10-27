# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""A device context that was never opened is not a verdict on a tile.

A recorded campaign met `ctx is NULL! ... the context is a null pointer` and
scored every occurrence as "this candidate is incorrect": the blocks hit
`failure_limit` and aborted, the evaluations were charged, and the tile regions
went to the sampler as infeasible -- on the strength of a device id that was
never set.

These tests pin the three things that follow from calling it a fault instead:
the evaluator raises rather than returning a verdict, it does not wait (there
is nothing to wait for), and the halt names what to fix.
"""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import harness  # noqa: E402

from bayesian_optimization import evaluator as ev_mod  # noqa: E402

CTX_NULL = ("[ERROR] ctx is NULL!\n"
            "[ERROR] rtGetDevMsg execution failed, the context is a null pointer\n")


class _Evaluator(ev_mod.RealEvaluator):
    """A RealEvaluator whose command prints `output` and exits `rc`.

    A subclass, so the method under test is the real one and the stubbing
    happens inside the class that owns those names.
    """

    @classmethod
    def over(cls, output, rc=1):
        """One instance whose command prints `output` and exits `rc`.

        Built without the real constructor: that one reads the kernel off disk
        and binds its own command runner over the one stubbed below.
        """
        stub = cls.__new__(cls)
        stub.test_command, stub.eval_timeout_s = "true", 1
        stub.evaluation_mode = "subprocess"
        stub.fault_backoff_s = (30, 90)
        stub.last_detail, stub.last_retries = "", 0
        stub._last_output = output
        stub.slept = []
        stub._sleep = stub.slept.append
        stub._rc = rc
        return stub

    def run_test(self):
        """`_run_test` under its own name, so a caller need not reach in."""
        return self._run_test()

    def _run_cmd(self, cmd, timeout, env=None):
        return self._rc


class MarkerTest(unittest.TestCase):

    def test_the_context_markers_are_invocation_faults(self):
        for m in ev_mod.DEVICE_CONTEXT_MARKERS:
            self.assertIn(m, ev_mod.INVOCATION_FAULT_MARKERS)
            self.assertEqual(ev_mod.fault_kind(m), "invocation")

    def test_a_null_context_raises_instead_of_failing_the_candidate(self):
        ev = _Evaluator.over(CTX_NULL)
        with self.assertRaises(ev_mod.EnvironmentFault) as c:
            ev.run_test()
        self.assertEqual(c.exception.kind, "invocation")
        self.assertIn("ctx is NULL", c.exception.marker)

    def test_it_does_not_wait_because_there_is_nothing_to_wait_for(self):
        ev = _Evaluator.over(CTX_NULL)
        with self.assertRaises(ev_mod.EnvironmentFault):
            ev.run_test()
        self.assertEqual(ev.slept, [])
        self.assertEqual(ev.last_retries, 0)

    def test_an_ordinary_wrong_answer_is_still_a_verdict(self):
        ev = _Evaluator.over("golden mismatch: max abs err 0.31\n")
        self.assertEqual(ev.run_test(), 1)


class HintTest(unittest.TestCase):

    def test_the_halt_names_the_device_id_to_check(self):
        hint = harness.invocation_hint(["ctx is NULL"])
        self.assertIn("TILE_FWK_DEVICE_ID", hint)
        self.assertIn("never obtained a device context", hint)

    def test_a_path_fault_keeps_the_old_advice(self):
        hint = harness.invocation_hint(["No module named"])
        self.assertIn("--lint-root", hint)
        self.assertNotIn("TILE_FWK_DEVICE_ID", hint)

    def test_the_verdict_records_the_markers_for_the_next_session(self):
        st = {"progress": {},
              "blocks": [{"action": "I-1",
                          "aborted": "3 consecutive environment faults (invocation)",
                          "fault_kind": "invocation",
                          "fault_markers": ["ctx is NULL"],
                          "env_faults": 3}]}
        reason, probe = harness.device_verdict(st, types.SimpleNamespace())
        self.assertEqual(reason, "harness_fault")
        self.assertIsNone(probe)
        rec = st["progress"]["harness_fault"]
        self.assertEqual(rec["markers"], ["ctx is NULL"])
        self.assertIn("TILE_FWK_DEVICE_ID", rec["hint"])
        self.assertEqual(rec["block"], "I-1")


class BudgetReasonTest(unittest.TestCase):
    """`no evaluation budget remaining` covered three different situations."""

    def test_the_tile_share_is_named_as_the_share_not_the_run_budget(self):
        why = harness.budget_reason({"evals_used": 71, "block_evals": 60},
                                    {"eval_budget": 120, "block_budget_share": 0.5},
                                    self._args())
        self.assertIn("TILE share", why)
        self.assertIn("49 still unspent", why)
        self.assertNotIn("campaign is over", why)

    def test_a_spent_run_budget_says_the_campaign_is_over(self):
        why = harness.budget_reason({"evals_used": 120, "block_evals": 60},
                                    {"eval_budget": 120, "block_budget_share": 0.5},
                                    self._args())
        self.assertIn("RUN budget", why)
        self.assertIn("call `stop`", why)

    def test_the_per_block_cap_is_named_when_nothing_else_bound(self):
        """
        With the share off, the absolute per-block cap is the only limit
        left, and it has to say so rather than blame a budget with 110
        evaluations still in it.
        """
        why = harness.budget_reason({"evals_used": 10, "block_evals": 0},
                                    {"eval_budget": 120, "block_budget_share": None},
                                    self._args(max_trials=0))
        self.assertIn("block_max_trials", why)
        self.assertNotIn("RUN budget", why)

    def test_the_share_binds_without_a_run_budget(self):
        """
        `eval_budget: null` used to remove the guard entirely, because a
        share of an unstated total is undefined. It is a running ratio now.
        """
        why = harness.budget_reason({"evals_used": 60, "block_evals": 30},
                                    {"eval_budget": None, "block_budget_share": 0.5},
                                    self._args())
        self.assertIn("TILE share", why)
        self.assertNotIn("still unspent", why, "there is no budget to report")

    def _args(self, max_trials=None):
        return types.SimpleNamespace(max_trials=max_trials)


if __name__ == "__main__":
    unittest.main()
