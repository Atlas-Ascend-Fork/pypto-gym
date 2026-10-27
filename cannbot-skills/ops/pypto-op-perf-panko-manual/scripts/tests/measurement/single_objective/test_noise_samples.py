# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""The run carries its own measurement noise, read from the readings themselves.

The measured cv varies by orders of magnitude across operators and moves with
the program, so a transcribed constant is wrong for almost every operator.
Every rule that asks "is this difference real" -- the margin-based stop,
end-of-run verification -- needs the operator's own figure, and it can only
come from the individual readings behind each median.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import DEMO_KERNEL as KERNEL, file_hash, harness, INIT_ARGS, new_op_dir, run_cli  # noqa: E402


def _run(*args):
    return run_cli(*args)[0]


class ParseSamplesTest(unittest.TestCase):
    def test_a_plain_list(self):
        self.assertEqual(harness.parse_samples("12.3,12.5, 12.4"), [12.3, 12.5, 12.4])

    def test_malformed_entries_are_dropped_not_fatal(self):
        """The measurement happened; a bad list must not fail the record."""
        self.assertEqual(harness.parse_samples("12.3,,abc,-1,0,12.5"), [12.3, 12.5])

    def test_nothing_is_nothing(self):
        for v in (None, "", " , "):
            self.assertEqual(harness.parse_samples(v), [])


class SigmaLogTest(unittest.TestCase):
    def test_below_two_readings_there_is_no_estimate(self):
        self.assertIsNone(harness.sigma_log([]))
        self.assertIsNone(harness.sigma_log([10.0]))

    def test_it_is_scale_free(self):
        """
        On the log scale a 10 us and a 10 ms kernel with the same relative
        spread read the same sigma -- which is what makes one threshold usable
        across operators.
        """
        a = harness.sigma_log([10.0, 11.0, 9.0, 10.5])
        b = harness.sigma_log([10000.0, 11000.0, 9000.0, 10500.0])
        self.assertAlmostEqual(a, b, places=6)

    def test_identical_readings_have_zero_spread(self):
        self.assertEqual(harness.sigma_log([5.0, 5.0, 5.0]), 0.0)


class ObserveNoiseTest(unittest.TestCase):
    def test_the_window_is_bounded(self):
        st = {}
        for i in range(harness.NOISE_WINDOW + 10):
            harness.observe_noise(st, [10.0, 10.0 + 0.1 * (i + 1)])
        self.assertIn("noise", st)
        self.assertEqual(len(st.get("noise")["recent"]), harness.NOISE_WINDOW)
        self.assertEqual(st.get("noise")["measurements"], harness.NOISE_WINDOW + 10)

    def test_one_pathological_measurement_does_not_set_the_estimate(self):
        st = {}
        for _ in range(8):
            harness.observe_noise(st, [10.0, 10.2, 9.8])
        self.assertIn("noise", st)
        quiet = st.get("noise")["sigma_log"]
        harness.observe_noise(st, [10.0, 40.0, 2.0])
        self.assertAlmostEqual(st.get("noise")["sigma_log"], quiet, places=6,
                               msg="the aggregate is a median")
        self.assertGreater(st.get("noise")["latest_sigma_log"], 10 * quiet)

    def test_no_samples_leaves_the_estimate_alone(self):
        st = {}
        self.assertIsNone(harness.observe_noise(st, []))
        self.assertNotIn("noise", st)


class NoiseCliTest(unittest.TestCase):
    """Driven through the real CLI: the defect would be in the plumbing."""

    def test_without_preopt_samples_the_noise_is_explicitly_unmeasured(self):
        self._init()
        noise = self._state()["noise"]
        self.assertEqual(noise["source"], "unmeasured")
        self.assertIsNone(noise["sigma_log"])

    def test_the_preopt_seeds_the_estimate(self):
        self._init("--preopt-samples", "100,102,98,101,99,100,103,97,100")
        st = self._state()
        self.assertEqual(st["noise"]["source"], "measured")
        self.assertEqual(st["noise"]["measurements"], 1)
        self.assertEqual(st["noise"]["protocol"], harness.SEARCH_PROTOCOL)
        self.assertGreater(st["noise"]["sigma_log"], 0)
        self.assertEqual(len(st["eval_cache"][self.preopt_hash]["samples"]), 9)
        self.assertEqual(len(st["nodes"]["closed"][0]["eval"]["samples"]), 9)

    def test_record_stores_samples_and_updates_the_estimate(self):
        self._init("--preopt-samples", "100,102,98")
        self._record(1, 95.0, "95,96,94,95.5")
        st = self._state()
        self.assertEqual(st["eval_cache"][f"{1:064x}"]["samples"], [95, 96, 94, 95.5])
        self.assertEqual(st["noise"]["measurements"], 2)
        expected = harness.sigma_log([95, 96, 94, 95.5])
        self.assertTrue(math.isclose(st["noise"]["latest_sigma_log"], expected))

    def test_a_record_without_samples_behaves_as_before(self):
        self._init()
        res = self._record(1, 95.0)
        self.assertIn("J", res)
        st = self._state()
        self.assertNotIn("samples", st["eval_cache"][f"{1:064x}"])
        self.assertEqual(st["noise"]["measurements"], 0)
        self.assertEqual(st["progress"]["evals_used"], 2)

    def test_a_duplicate_does_not_count_twice(self):
        """A byte-identical resubmission is not a measurement of anything."""
        self._init()
        self._record(1, 95.0, "95,96,94")
        self._record(1, 95.0, "95,96,94")
        self.assertEqual(self._state()["noise"]["measurements"], 1)

    def _init(self, *extra):
        self.op_file = new_op_dir(self, KERNEL)
        self.preopt_hash = file_hash(self.op_file)
        res = _run("init", "--op-dir", str(self.op_dir), "--op", "demo",
                   "--p-ref", "100", "--preopt-p", "100",
                   # p_ref == preopt_p, so the target is met before the run
                   # starts; `init` refuses that unless the caller says there
                   # is no target. These fixtures are not about the target.
                   "--config", json.dumps({"success_stop": False}),
                   "--preopt-hash", self.preopt_hash,
                   "--op-file", str(self.op_file),
                   "--allow-unversioned", *INIT_ARGS, *extra)
        self.assertTrue(res["initialized"], res)

    def _state(self):
        return json.loads((self.op_dir / "optimization" / "search_state.json")
                          .read_text(encoding="utf-8"))

    def _record(self, i, p, samples=None):
        args = ["record", "--op-dir", str(self.op_dir), "--parent", "u1",
                "--code-hash", f"{i:064x}", "--s", "1", "--p", str(p)]
        if samples is not None:
            args += ["--samples", samples]
        return _run(*args)


if __name__ == "__main__":
    unittest.main()
