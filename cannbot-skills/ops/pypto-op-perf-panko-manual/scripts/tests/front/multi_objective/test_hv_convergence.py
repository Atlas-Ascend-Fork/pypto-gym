# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""The run's convergence test, on the hypervolume (C3).

`front_stagnation` answers "did the front move" but not "did it move enough":
a point 0.1% better than its neighbour resets it exactly like one that doubles
the front. `progress.hv_stagnation` counts measurements since the front last
grew by more than `hv_min_gain` AND more than the readings' own noise, and
`hv_stagnation_limit` halts the run on it.

The noise part matters because a large share of candidates cannot be ordered
against the bar at all (see close_rule_noise_20261001.md). A front that grows
only by what the measurement can produce on its own is not growth.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import CHIP_ARGS, DEMO_KERNEL as KERNEL, file_hash, harness, new_op_dir, read_state, run_cli  # noqa: E402


def _row(h, p, w, samples=None):
    row = {"code_hash": h, "values": {"aicore_e2e": p, "workspace": float(w)}}
    if samples:
        row["samples_by_objective"] = {"aicore_e2e": list(samples)}
    return row


class GainTest(unittest.TestCase):
    """`hv_gain` on states built by hand, so the readings are the variable."""

    def test_both_fronts_are_measured_in_one_box(self):
        """
        One reference, fixed for the run, so the two volumes are directly
        comparable -- and so is the whole series.
        """
        st = self._st()
        before = [_row("a" * 64, 90.0, 1000.0)]
        after = before + [_row("b" * 64, 95.0, 500.0)]
        d = harness.hv_gain(st, before, after)
        self.assertGreater(d["gain"], 0.0)
        self.assertEqual(d["hv_before"], harness.hv_gain(st, before, before)["hv"])

    def test_a_refused_candidate_is_a_gain_of_exactly_zero(self):
        st = self._st()
        front = [_row("a" * 64, 90.0, 1000.0, [90.0] * 9)]
        d = harness.hv_gain(st, front, front)
        self.assertEqual((d["gain"], d["ci_low"], d["ci_high"]), (0.0, 0.0, 0.0))

    def test_the_readings_decide_whether_the_same_gain_clears_the_floor(self):
        """The identical point estimate, twice, on different readings.

        An insert that only ADDS points can never lose volume, so noise shows
        up as an interval reaching down toward zero rather than below it. What
        `hv_observe` asks is whether the LOWER end clears `hv_min_gain`.
        """
        st = self._st()

        def tight(p):
            return [p + i * 0.05 for i in range(9)]

        before = [_row("a" * 64, 90.0, 1600.0, tight(90.0)),
                  _row("b" * 64, 95.0, 380.0, tight(95.0))]

        def add(s):
            return harness.hv_gain(
                st, before, before + [_row("c" * 64, 94.0, 370.0, s)])

        clean = add(tight(94.0))
        noisy = add([94.0 * (1 + 0.04 * ((i % 3) - 1)) for i in range(9)])
        self.assertEqual(clean["gain"], noisy["gain"], "the medians are identical")
        # A floor between the two lower ends: the same point estimate counts
        # on one set of readings and not on the other.
        floor = (clean["ci_low"] + noisy["ci_low"]) / 2
        self.assertGreater(clean["ci_low"], floor)
        self.assertLess(noisy["ci_low"], floor)
        self.assertGreater(noisy["ci_high"], clean["ci_high"],
                           "and the noisy interval is the wider one")

    def test_a_program_is_resampled_once_per_draw(self):
        """
        `before` and `after` share programs. Redrawing them independently
        would measure the bootstrap instead of the front: the gain of a front
        against ITSELF would stop being zero.
        """
        st = self._st()
        noisy = [90.0 * (1 + 0.1 * ((i % 4) - 2)) for i in range(9)]
        front = [_row("a" * 64, 90.0, 1000.0, noisy),
                 _row("b" * 64, 95.0, 500.0, noisy)]
        d = harness.hv_gain(st, front, front)
        self.assertEqual((d["ci_low"], d["ci_high"]), (0.0, 0.0))

    def test_the_workspace_axis_contributes_no_uncertainty(self):
        """
        It is a probe, not a sample. Said out loud because the interval
        would otherwise look like it covered both axes.
        """
        st = self._st()
        row = _row("a" * 64, 90.0, 1000.0, [90.0] * 9)
        self.assertEqual(harness.row_samples(
            harness.parse_objectives(st["config"])[1], row), [])

    def _st(self, objectives=("aicore_e2e", "workspace"), preopt=(100.0, 1000.0)):
        return {"config": {"objectives": list(objectives)},
                # The reference on a byte axis. Without it `hv_reference_point`
                # returns None and there is no volume to compare.
                "chip_envelope": {"memory_mib": 65536},
                "nodes": {"closed": [{"id": "root", "code_hash": "p" * 64}]},
                "eval_cache": {"p" * 64: {"s": 1, "p": preopt[0], "m": {
                    "workspace_b": preopt[1], "workspace_state": "measured"}}}}


class CliTest(unittest.TestCase):
    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", self.pre,
                "--preopt-workspace-b", str(1000 << 20),
                "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS)

    def test_the_preopt_sets_the_baseline_without_counting(self):
        pr = read_state(self.op_dir)["progress"]
        self.assertEqual((pr["hv"], pr["hv_stagnation"]), (0.0, 0))
        self.assertIsNone(pr["hv_last"]["counted_as_gain"],
                          "the search has not paid for a measurement yet")

    def test_a_real_gain_resets_the_count_and_a_refusal_advances_it(self):
        self._record(128, 90.0, 1600, self._tight(90.0))
        self.assertEqual(read_state(self.op_dir)["progress"]["hv_stagnation"], 0)
        self._record(32, 99.0, 1700, self._tight(99.0))        # dominated
        self.assertEqual(read_state(self.op_dir)["progress"]["hv_stagnation"], 1)
        self._record(256, 95.0, 380, self._tight(95.0))        # a trade-off
        pr = read_state(self.op_dir)["progress"]
        self.assertEqual(pr["hv_stagnation"], 0)
        self.assertTrue(pr["hv_last"]["counted_as_gain"])
        self.assertGreater(pr["hv_last"]["ci_low"], 0.0)

    def test_a_gain_inside_the_noise_is_kept_on_the_front_but_not_counted(self):
        self._record(128, 90.0, 1600, self._tight(90.0))
        self._record(256, 95.0, 380, self._tight(95.0))
        noisy = [94.0 * (1 + 0.04 * ((i % 3) - 1)) for i in range(9)]
        h = self._record(16, 94.0, 370, noisy)
        st = read_state(self.op_dir)
        pr, arc = st["progress"], st["archive"]
        self.assertIn(h, [e["code_hash"] for e in arc], "the front took it")
        self.assertEqual(st["eval_cache"][h]["verdict"], "kept")
        self.assertFalse(pr["hv_last"]["counted_as_gain"])
        self.assertEqual(pr["hv_stagnation"], 1)
        self.assertLess(pr["hv_last"]["ci_low"], pr["hv_last"]["gain"])

    def test_the_limit_halts_the_run_and_the_report_says_why(self):
        self._record(128, 90.0, 1600, self._tight(90.0))
        st = read_state(self.op_dir)
        st["config"]["hv_stagnation_limit"] = 3
        (self.op_dir / "optimization" / "search_state.json").write_text(json.dumps(st))
        for i, tile in enumerate((8, 16, 24)):
            self._record(tile, 110.0 + i, 1700, self._tight(110.0 + i))
        self.assertEqual(read_state(self.op_dir)["progress"]["hv_stagnation"], 3)
        res, _ = run_cli("stop", "--op-dir", str(self.op_dir))
        self.assertEqual((res["stop"], res["reason"]), (True, "hv_stagnation"))
        self.assertEqual(res["hv_stagnation"], 3)
        self.assertIsNotNone(res["hv"])
        run_cli("report", "--op-dir", str(self.op_dir))
        log = next((self.op_dir / "optimization").glob("*.md")).read_text()
        self.assertIn("hypervolume:", log)
        self.assertIn("since it last grew by more than 0.01", log)

    def test_the_halt_is_off_when_the_limit_is_none(self):
        self._record(128, 90.0, 1600, self._tight(90.0))
        st = read_state(self.op_dir)
        st["config"]["hv_stagnation_limit"] = None
        (self.op_dir / "optimization" / "search_state.json").write_text(json.dumps(st))
        for i, tile in enumerate((8, 16, 24)):
            self._record(tile, 110.0 + i, 1700, self._tight(110.0 + i))
        res, _ = run_cli("stop", "--op-dir", str(self.op_dir))
        self.assertFalse(res["stop"], res)
        self.assertEqual(res["hv_stagnation"], 3, "measured either way")

    def test_the_default_limit_clears_one_tile_block(self):
        self.assertGreater(harness.DEFAULT_CONFIG["hv_stagnation_limit"], 24,
                           "a block that buys no front point must not end a run "
                           "in the middle of an action")

    def _record(self, tile, p, mib, samples=None):
        self.f.write_text(KERNEL.replace("64, 128", f"{tile}, 128"), encoding="utf-8")
        h = file_hash(self.f)
        args = ["record", "--op-dir", str(self.op_dir), "--parent", "u1",
                "--code-hash", h, "--s", "1", "--p", str(p),
                "--workspace-b", str(mib << 20)]
        if samples:
            args += ["--samples", ",".join(str(x) for x in samples)]
        run_cli(*args)
        return h

    def _tight(self, p):
        return [p + i * 0.05 for i in range(9)]


if __name__ == "__main__":
    unittest.main()
