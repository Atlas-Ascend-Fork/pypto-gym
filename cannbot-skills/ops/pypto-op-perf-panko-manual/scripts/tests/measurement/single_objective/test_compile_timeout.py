# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A compile killed by `eval_timeout_s` is not a verdict about the candidate.

`references/evaluator.md` lumped "timed-out / crashed / OOM / compile-failed"
together and called all four `s = 0`. Three of them are statements about the
candidate. A compile that ran past the budget is not: nothing was measured.

On this hardware the distinction decides whole campaigns. The compile IS the
measurement -- minutes against a second or two for a warm call -- so a run can
lose most of its evaluations to the cap. Compile time scales with program size,
so the candidates the old rule discarded were systematically the large ones,
and it charged `eval_budget` for each.

`measure_latency.capture_readings` writes a marker to the REAL stderr once the first
call returns, so the split is exact: absent means the compile never finished.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import SCRIPTS_DIR, harness  # noqa: E402,F401

sys.path.insert(0, str(SCRIPTS_DIR))
import measure_latency  # noqa: E402
from bayesian_optimization import driver, evaluator  # noqa: E402

MARKER = measure_latency.FIRST_CALL_MARKER


class ClassifyTimeoutTest(unittest.TestCase):

    def test_no_marker_means_the_compile_never_finished(self):
        v = evaluator.classify_timeout("Traceback: killed mid-compile\n")
        self.assertEqual(v["unevaluated"], "compile_timeout")
        self.assertEqual(v["timeout"], "compile")
        self.assertIsNone(v.get("p"))

    def test_marker_present_means_it_compiled_then_ran_long(self):
        v = evaluator.classify_timeout(f"noise\n{MARKER} compile_s=348.0\nmore\n")
        self.assertEqual(v["timeout"], "exec")
        self.assertNotIn("unevaluated", v,
                         "a hang after a successful compile IS a verdict")
        self.assertEqual(v["compile_s"], 348.0)

    def test_empty_and_none_output_are_compile_timeouts(self):
        for out in ("", None):
            self.assertEqual(evaluator.classify_timeout(out)["unevaluated"],
                             "compile_timeout")

    def test_a_malformed_marker_still_splits_the_case(self):
        """The split must not depend on parsing the number."""
        v = evaluator.classify_timeout(f"{MARKER} compile_s=not-a-number\n")
        self.assertEqual(v["timeout"], "exec")
        self.assertIsNone(v["compile_s"])

    def test_marker_without_a_value_still_splits_the_case(self):
        v = evaluator.classify_timeout(f"{MARKER}\n")
        self.assertEqual(v["timeout"], "exec")
        self.assertIsNone(v["compile_s"])


class CaptureMarkerTest(unittest.TestCase):
    """The marker has to reach the PARENT, which means the saved stderr.

    Written into the redirected fd it would land in the temp file `capture_readings` is
    about to read and parse, and the parent -- which only has
    `TimeoutExpired.output` -- would never see it.
    """

    def test_the_marker_goes_to_the_saved_stderr_not_the_capture(self):
        text, _, err, n = self._capture_with_real_stderr_to_file(4)
        self.assertEqual(n, 4)
        self.assertIn(MARKER, err, "the parent must be able to see it")
        self.assertNotIn(MARKER, text, "it must not pollute the profiler capture")

    def test_it_is_written_once_for_the_first_call_only(self):
        _, _, err, _ = self._capture_with_real_stderr_to_file(4)
        self.assertEqual(err.count(MARKER), 1)

    def test_capture_returns_the_first_call_seconds(self):
        _, first, err, _ = self._capture_with_real_stderr_to_file(2)
        self.assertIsNotNone(first)
        self.assertGreaterEqual(first, 0.0)
        self.assertEqual(evaluator.marker_compile_s(err), first,
                         "the marker must carry the same number measure() reports")

    def test_a_single_call_still_reports_and_marks(self):
        _, first, err, n = self._capture_with_real_stderr_to_file(1)
        self.assertEqual(n, 1)
        self.assertIsNotNone(first)
        self.assertIn(MARKER, err)

    def _capture_with_real_stderr_to_file(self, total):
        calls = []

        def run_once():
            calls.append(1)
            os.write(1, b"AICore End-to-End Time: 100.0\n")

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".err",
                                         delete=False) as tf:
            name = tf.name
        saved = os.dup(2)
        try:
            with open(name, "w", encoding="utf-8") as fh:
                os.dup2(fh.fileno(), 2)
            text, first = measure_latency.capture_readings(run_once, total)
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        with open(name, encoding="utf-8") as fh:
            err = fh.read()
        os.unlink(name)
        return text, first, err, len(calls)


class MeasureReportsCompileSecondsTest(unittest.TestCase):

    def test_compile_s_is_in_the_measurement(self):
        def run_once():
            os.write(1, b"AICore End-to-End Time: 100.0\n")

        # fd 2 is the test runner's stderr and the marker is written to it by
        # design; park it so the suite's output stays clean.
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".err") as tf:
            saved = os.dup(2)
            try:
                os.dup2(tf.fileno(), 2)
                m = measure_latency.measure(run_once, warmup=1, runs=2, agg="min")
            finally:
                os.dup2(saved, 2)
                os.close(saved)
        self.assertIn("compile_s", m)
        self.assertIsNotNone(m["compile_s"])
        self.assertIsNotNone(m["p"], "the measurement itself must be unaffected")


class _Trial:
    def __init__(self):
        self.attrs = {}

    def set_user_attr(self, k, v):
        self.attrs[k] = v


class _Opts:
    on_trial = None


def _ctx():
    return {"state": {"ceiling": 4096.0, "floor": 0, "faults": 0,
                      "failures": 0, "fault_kind": None, "unevaluated": 0,
                      "seen": {}},
            "history": [], "new_refusals": [], "opts": _Opts()}


class RecordUnevaluatedTest(unittest.TestCase):
    """None of the four lasting effects `_record_refusal` has may happen here."""

    def test_it_does_not_memoize_the_configuration_as_measured(self):
        ctx, trial = _ctx(), _Trial()
        driver.record_unevaluated(ctx, {"m": 128}, "compile", trial)
        self.assertEqual(ctx["state"]["seen"], {},
                         "an unevaluated point must stay drawable")

    def test_it_does_not_lower_the_capacity_ceiling(self):
        ctx, trial = _ctx(), _Trial()
        driver.record_unevaluated(ctx, {"m": 128}, "compile", trial)
        self.assertEqual(ctx["state"]["ceiling"], 4096.0,
                         "no size limit was demonstrated")

    def test_it_files_no_cross_structure_refusal(self):
        ctx, trial = _ctx(), _Trial()
        driver.record_unevaluated(ctx, {"m": 128}, "compile", trial)
        self.assertEqual(ctx["new_refusals"], [],
                         "a timeout is not a statement about the tile")

    def test_it_is_recorded_under_its_own_status(self):
        ctx, trial = _ctx(), _Trial()
        driver.record_unevaluated(ctx, {"m": 128}, "why", trial)
        self.assertEqual(trial.attrs["status"], "unevaluated")
        self.assertEqual(ctx["history"][-1]["status"], "unevaluated")
        self.assertIsNone(ctx["history"][-1]["objective_us"])

    def test_consecutive_unevaluated_trials_are_counted(self):
        ctx = _ctx()
        for _ in range(3):
            driver.record_unevaluated(ctx, {"m": 128}, "compile", _Trial())
        self.assertEqual(ctx["state"]["unevaluated"], 3)

    def test_the_limit_exists_and_is_small(self):
        """
        A compile that will not finish rarely finishes for the next draw
        either, so the block must have a brake.
        """
        self.assertGreaterEqual(driver.UNEVALUATED_LIMIT, 1)
        self.assertLessEqual(driver.UNEVALUATED_LIMIT, 8)


if __name__ == "__main__":
    unittest.main()
