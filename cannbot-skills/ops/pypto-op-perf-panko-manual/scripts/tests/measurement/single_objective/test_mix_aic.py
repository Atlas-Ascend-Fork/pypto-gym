# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""`p` is msprof's MIX_AIC.

The reading comes from op_summary after msprof exits, one MIX_AIC row per
launch. These tests pin the parts that do not need a device: which rows are
read, how they map to calls, when the reading is refused, and that a refused
reading ends a tile block instead of scoring the candidate.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
import _support  # noqa: E402,F401  (puts scripts/ on sys.path)

import measure_latency as ml  # noqa: E402
from bayesian_optimization import evaluator as ev_mod  # noqa: E402

HEADER = ["Device_id", "Op Name", "Task Type", "Task Start Time(us)",
          "Task Duration(us)", "aicore_time(us)", "aiv_time(us)"]


def _csv(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(HEADER) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


class ReadingTest(unittest.TestCase):

    def test_the_diagnostic_columns_come_from_the_same_rows(self):
        rows = [{"Task Type": "MIX_AIC", "aicore_time(us)": v, "aiv_time(us)": v + 1,
                 "aic_mac_ratio": 0.03, "cube_utilization(%)": 99.9}
                for v in [500.0, 10.0, 12.0, 11.0]]
        m = ml.msprof_metrics(rows, total=4, warmup=1)
        self.assertEqual(m["aiv_time"], 12.0)
        self.assertEqual(m["aic_mac_ratio"], 0.03)
        self.assertEqual(m["cube_utilization"], 99.9)
        self.assertNotIn("vector_utilization", m, "a column the CSV lacks is absent")

    def test_only_mix_aic_rows_in_launch_order(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "op_summary_0.csv")
            _csv(p, [[0, "PYPTO__k", "MIX_AIC", 300, 1, 12.0, 11.0],
                     [0, "aicpu", "AI_CPU", 150, 1, 99.0, 0],
                     [0, "PYPTO__k", "MIX_AIC", 100, 1, 10.0, 9.0],
                     [0, "PYPTO__k", "MIX_AIC", 200, 1, 11.0, 10.0]])
            rows, task_types, raw = ml.mix_aic_rows(p)
        self.assertEqual([v for _, v in rows], [10.0, 11.0, 12.0])
        self.assertEqual(task_types, ["AI_CPU", "MIX_AIC"])
        self.assertEqual(len(raw), 3, "the raw rows the diagnostics are read from")

    def test_the_warmup_is_dropped_and_p_is_the_median(self):
        r = ml.mix_aic_reading([("k", v) for v in [500.0, 10.0, 12.0, 11.0]], total=4, warmup=1)
        self.assertEqual(r["samples"], [10.0, 12.0, 11.0])
        self.assertEqual(r["p"], 11.0)
        self.assertEqual(r["mix_aic_per_call"], 1)
        self.assertIsNotNone(r["sigma_log"])

    def test_k_launches_per_call_are_summed_per_call(self):
        r = ml.mix_aic_reading([("a", 1.0), ("b", 2.0)] * 3, total=3, warmup=1)
        self.assertEqual(r["mix_aic_per_call"], 2)
        self.assertEqual(r["samples"], [3.0, 3.0])

    def test_rows_that_do_not_divide_into_calls_are_refused(self):
        r = ml.mix_aic_reading([("k", 1.0)] * 7, total=10, warmup=1)
        self.assertIsNone(r["p"])
        self.assertIn("cannot assign", r["p_error"])

    def test_no_rows_is_no_reading_and_names_what_was_there(self):
        r = ml.mix_aic_reading([], total=10, warmup=1, task_types=["AI_VECTOR_CORE"])
        self.assertIsNone(r["p"])
        self.assertIn("AI_VECTOR_CORE", r["p_error"])

    def test_a_missing_msprof_is_no_reading(self):
        r = ml.measure_under_msprof([], 1, 9, "median",
                                    ml.ProfOptions(msprof="no-such-msprof-xyz"))
        self.assertIsNone(r["p"])
        self.assertIn("not found", r["p_error"])


class _Evaluator(ev_mod.RealEvaluator):
    """A RealEvaluator whose gates pass and whose measurement returns `m`.

    Only the branch after the measurement is under test, so everything that
    would touch a file, a compiler or the device is overridden here. A subclass
    rather than attributes set from outside: the names being replaced belong to
    this class.
    """

    @classmethod
    def over(cls, m):
        """
        One instance returning `m` from its measurement, built without the
        real constructor, which reads the kernel off disk.
        """
        stub = cls.__new__(cls)
        stub.base_src, stub._last_output, stub.last_detail = "", "", ""
        stub._m = m
        return stub

    def _write(self, src):
        return None

    def _run_test(self):
        return 0

    def _lint_ok(self):
        return True

    def _env_fault(self):
        return None

    def _measure(self):
        return self._m


class EvaluatorTest(unittest.TestCase):

    def setUp(self):
        def _applied(src, cfg):
            return "x", 1

        saved = ev_mod.apply.apply
        ev_mod.apply.apply = _applied
        self.addCleanup(setattr, ev_mod.apply, "apply", saved)

    def test_the_block_is_ranked_on_p_the_end_to_end_time(self):
        """
        `p` IS the profiler's AICore End-to-End Time, so the block and the
        ratchet climb one reading. MIX_AIC rides along as its own objective.
        """
        self.assertIn("mix_aic", ev_mod.CARRIED_METRICS)
        self.assertEqual(ev_mod.J_OBJECTIVE, "aicore_e2e")
        ev = _Evaluator.over({"p": 10.5, "samples": [10.5], "mix_aic": 11.0})
        self.assertEqual(ev({}), (1, 10.5, "ok"))
        self.assertEqual(ev.last_metrics["mix_aic"], 11.0)

    def test_a_missing_end_to_end_time_ends_the_block(self):
        """
        No `p` means the reading the search climbs was not obtained, so
        nothing was learned about this tile and it would repeat every trial.
        """
        ev = _Evaluator.over({"p": None, "mix_aic": 11.0})
        with self.assertRaises(ev_mod.EnvironmentFault) as c:
            ev({})
        self.assertEqual(c.exception.marker, "objective_unmeasured")
        self.assertIn("runtime_debug_mode", str(c.exception.detail))

    def test_an_unreadable_mix_aic_ends_the_block_and_is_not_a_verdict(self):
        ev = _Evaluator.over({"p": None, "p_error": "msprof not found on PATH",
                              "aicore_e2e": 10.5})
        with self.assertRaises(ev_mod.EnvironmentFault) as c:
            ev({})
        self.assertEqual(c.exception.marker, "objective_unmeasured")
        self.assertEqual(c.exception.kind, "invocation")
        self.assertIn("msprof not found", ev.last_detail)


if __name__ == "__main__":
    unittest.main()
