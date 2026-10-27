# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A candidate that was never measured is neither charged nor labelled.

`references/evaluator.md` called a timed-out candidate `s = 0` and discarded it.
That is right for a compile FAILURE and wrong for a compile that ran past the
budget: nothing about the candidate was established, so it must not be charged
to `eval_budget` and must not be recorded as incorrect.

The structural path runs the measurement under `timeout <eval_timeout_s>` itself,
so only the caller can say so -- which is why these drive the real CLI: the
defect is in which command owns the accounting.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import DEMO_KERNEL as KERNEL, file_hash, harness, INIT_ARGS, run_cli  # noqa: E402


def _run(*args):
    return run_cli(*args)[0]


class DeriveEvalTimeoutTest(unittest.TestCase):
    """The right timeout is a property of the operator, not of the method."""

    def test_it_is_derived_from_the_preopt_compile(self):
        v, src = harness.derive_eval_timeout(self._cfg(), 200.0)
        self.assertEqual(v, 800)
        self.assertEqual(src, "derived:preopt")

    def test_the_floor_holds_for_a_fast_preopt(self):
        """
        A small kernel must not get a timeout so tight that a slightly larger
        candidate cannot compile at all.
        """
        v, src = harness.derive_eval_timeout(self._cfg(), 10.0)
        self.assertEqual(v, 300)
        self.assertEqual(src, "derived:preopt")

    def test_the_cap_holds_for_a_slow_preopt(self):
        """
        Wall clock is the real budget: a 24 h run cannot afford an unbounded
        per-candidate timeout.
        """
        v, _ = harness.derive_eval_timeout(self._cfg(), 900.0)
        self.assertEqual(v, 1800)

    def test_a_missing_reading_keeps_the_configured_value_and_says_so(self):
        for bad in (None, "", "n/a", 0, -1):
            v, src = harness.derive_eval_timeout(self._cfg(), bad)
            self.assertEqual(v, 300)
            self.assertEqual(src, "default",
                             "a silent derivation is the thing to avoid")

    def test_an_explicit_config_wins_over_the_derivation(self):
        v, src = harness.derive_eval_timeout(
            self._cfg(eval_timeout_s=900, eval_timeout_explicit=True), 200.0)
        self.assertEqual(v, 900)
        self.assertEqual(src, "explicit:config")

    def test_the_recorded_compile_times_land_in_a_usable_range(self):
        """
        Compile times measured on 910B3, at a small and at a large workspace.
        Both must derive a timeout above the compile they came from.
        """
        for compile_s in (125.0, 350.0):
            v, _ = harness.derive_eval_timeout(self._cfg(), compile_s)
            self.assertGreater(v, compile_s)

    def _cfg(self, **kw):
        c = {"eval_timeout_s": 300, "eval_timeout_k": 4,
             "eval_timeout_floor_s": 300, "eval_timeout_cap_s": 1800}
        c.update(kw)
        return c


class UnevaluatedAccountingTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.op_dir = Path(self._tmp.name) / "demo"
        self.op_dir.mkdir(parents=True)
        self.op_file = self.op_dir / "demo_impl.py"
        self.addCleanup(self._tmp.cleanup)
        self.op_file.write_text(KERNEL, encoding="utf-8")
        res = _run("init", "--op-dir", str(self.op_dir), "--op", "demo",
                   "--p-ref", "100", "--preopt-p", "100",
                   # p_ref == preopt_p: no target, see test_target.py.
                   "--config", json.dumps({"success_stop": False}),
                   "--preopt-compile-s", "200",
                   "--preopt-hash", file_hash(self.op_file),
                   "--op-file", str(self.op_file),
                   "--allow-unversioned", *INIT_ARGS)
        self.assertTrue(res["initialized"], res)

    def test_init_derives_the_timeout_and_records_the_source(self):
        cfg = self._state()["config"]
        self.assertEqual(cfg["eval_timeout_s"], 800)
        self.assertEqual(cfg["eval_timeout_source"], "derived:preopt")

    def test_it_is_not_charged_to_eval_budget(self):
        before = self._state()["progress"]["evals_used"]
        self._unevaluated()
        self.assertEqual(self._state()["progress"]["evals_used"], before,
                         "the budget counts numbers, and a timeout produced none")

    def test_it_is_not_recorded_as_a_measurement(self):
        self._unevaluated()
        st = self._state()
        self.assertEqual(st.get("eval_cache", {}).get(f"{0:064x}"), None,
                         "memoizing it would answer for a program nobody measured")
        self.assertEqual(st["progress"]["refine"]["n"], 0,
                         "`n` counts measurements that failed to improve")

    def test_it_is_counted_under_its_own_name(self):
        self._unevaluated()
        pr = self._state()["progress"]
        self.assertEqual(pr["unevaluated_rejected"], 1)
        self.assertEqual(pr["unevaluated_reasons"]["compile_timeout"], 1)
        self.assertEqual(pr["static_rejected"], 0,
                         "a timeout is not a static rejection")

    def test_the_reply_says_it_was_not_a_verdict(self):
        res = self._unevaluated()
        self.assertFalse(res["recorded"])
        self.assertEqual(res["unevaluated"], "compile_timeout")
        self.assertEqual(res["eval_timeout_s"], 800)
        self.assertEqual(res["eval_timeout_source"], "derived:preopt")

    def test_unmeasured_k_bounds_an_action_that_only_compiles_slowly(self):
        """
        `n` counts measurements and there are none, and `eval_budget` is not
        being charged, so only the unmeasured bound can end it.
        """
        k = self._state()["config"]["unmeasured_K"]
        seen = [self._unevaluated(i) for i in range(k)]
        self.assertEqual([r["n_timeout"] for r in seen], list(range(1, k + 1)))
        self.assertEqual([r["n_unmeasured"] for r in seen], list(range(1, k + 1)))
        self.assertEqual([r["stop_refine"] for r in seen],
                         [False] * (k - 1) + [True])
        self.assertEqual(seen[-1]["stop_cause"], "unmeasured_K")
        self.assertIsNotNone(seen[-1]["hint"])

    def test_reaching_the_bound_also_releases_the_close(self):
        """
        Otherwise the harness reports stop_refine and then refuses the close,
        which is a deadlock: `n` cannot advance on proposals that were not
        measurements.
        """
        k = self._state()["config"]["unmeasured_K"]
        for i in range(k):
            self._unevaluated(i)
        st = self._state()
        self.assertIsNone(harness.refuse_early_close(st, "u1"))

    def test_the_counters_stay_separate(self):
        self._unevaluated(0)
        pr = self._state()["progress"]
        r = pr["refine"]
        self.assertEqual(r["n_timeout"], 1)
        self.assertEqual(r.get("n_static", 0), 0)
        self.assertEqual(r.get("n_noop", 0), 0)
        self.assertEqual(r["n"], 0)

    def test_other_unevaluated_reasons_are_kept_apart(self):
        self._unevaluated(0, "compile_timeout")
        self._unevaluated(1, "oom")
        reasons = self._state()["progress"]["unevaluated_reasons"]
        self.assertEqual(reasons, {"compile_timeout": 1, "oom": 1})

    def test_a_normal_record_is_untouched(self):
        res = _run("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                   "--code-hash", f"{99:064x}", "--s", "1", "--p", "50")
        self.assertTrue(res["improved"])
        pr = self._state()["progress"]
        self.assertEqual(pr["evals_used"], 2)
        self.assertEqual(pr["unevaluated_rejected"], 0)

    def _state(self):
        return json.loads((self.op_dir / "optimization" / "search_state.json")
                          .read_text(encoding="utf-8"))

    def _unevaluated(self, i=0, reason="compile_timeout"):
        return _run("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                    "--code-hash", f"{i:064x}", "--s", "0",
                    "--unevaluated", reason)


if __name__ == "__main__":
    unittest.main()
