# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Dominance on the measured objective is an interval, not a median.

Two points a fraction of a percent apart on a noisy kernel are not ordered,
and a front that claims they are is a claim the readings do not support.

An undetermined comparison is a TIE. Dominance asks two things -- no worse on
every objective, better on at least one -- and `undetermined` answers the
first, not the second. So a point that is not measurably slower and uses
strictly less workspace dominates, and two points that differ only by an
unresolved latency stay on the front together.

It used to BLOCK dominance, and a recorded front showed the cost: a point
worse on BOTH objectives stayed on it, kept alive by a couple of readings far
below its own median. The price of the change is that the fastest point can
leave the front when something not measurably slower is cheaper -- it does not
leave the RUN, because `impl.py` and the ratchet are not the archive.

The way to settle a pair that matters is more readings of the same program --
`record --remeasure`, which is not charged, because `eval_budget` counts
distinct programs measured.
"""
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import CHIP_ARGS, DEMO_KERNEL as KERNEL, file_hash, harness, new_op_dir, read_state, run_cli  # noqa: E402

LW = {"objectives": ["aicore_e2e", "workspace"]}
OBJS = harness.parse_objectives(LW)
ALPHA = 0.05


def _pt(h, p, w, samples=()):
    return {"code_hash": h, "latency_us": p, "workspace_b": w,
            "values": {"aicore_e2e": p, "workspace": w}, "samples": list(samples)}


class OrderTest(unittest.TestCase):

    def test_a_clear_gap_is_ordered(self):
        v, ci = harness.measured_order([90.0, 90.5, 91.0, 90.2, 90.8],
                                       [100.0, 99.5, 100.5, 100.2, 99.8])
        self.assertEqual(v, "better")
        self.assertLess(ci["ci_high"], 1.0)

    def test_an_overlap_is_undetermined(self):
        noisy = [90.0, 110.0, 95.0, 105.0, 100.0]
        v, ci = harness.measured_order(noisy, [x + 0.3 for x in noisy])
        self.assertEqual(v, "undetermined")
        self.assertLessEqual(ci["ci_low"], 1.0)
        self.assertGreaterEqual(ci["ci_high"], 1.0)

    def test_one_reading_is_not_an_interval(self):
        v, ci = harness.measured_order([90.0], [100.0])
        self.assertEqual(v, "undetermined", "no interval, so nothing is claimed")
        self.assertIsNone(ci)


class DominanceTest(unittest.TestCase):

    def test_an_unresolved_latency_lets_a_strictly_smaller_point_dominate(self):
        """Not measurably slower, and cheaper, is what a reader means by it."""
        noisy = [90.0, 110.0, 95.0, 105.0, 100.0]
        fast = _pt("a" * 64, 100.0, 1600 << 20, noisy)
        small = _pt("b" * 64, 100.3, 380 << 20, [x + 0.3 for x in noisy])
        self.assertTrue(harness.dominates_measured(OBJS, small, fast, ALPHA))
        self.assertFalse(harness.dominates_measured(OBJS, fast, small, ALPHA),
                         "the larger workspace is strictly worse")

    def test_a_point_worse_on_every_objective_is_dominated(self):
        """A recorded front held exactly this pair and kept both.

        The slower point lost on latency AND on workspace; it survived because
        a couple of its readings sat far below its own median, so the interval
        could not order the pair and `undetermined` blocked the dominance.
        """
        tight = [768.1, 759.7, 780.1, 773.8, 757.1]
        wide = [790.9, 773.8, 615.2, 795.7, 611.3]
        good = _pt("a" * 64, 768.06, 334233600, tight)
        bad = _pt("b" * 64, 773.79, 368836608, wide)
        self.assertEqual(harness.measured_order(tight, wide)[0], "undetermined")
        self.assertTrue(harness.dominates_measured(OBJS, good, bad, ALPHA))
        arc, _ = harness.archive_insert(OBJS, [], good, ALPHA)
        arc, verdict = harness.archive_insert(OBJS, arc, bad, ALPHA)
        self.assertEqual(verdict, "dominated")
        self.assertEqual(len(arc), 1)

    def test_a_real_gap_still_dominates_at_equal_workspace(self):
        a = _pt("a" * 64, 90.0, 1 << 20, [90.0, 90.5, 91.0, 90.2, 90.8])
        b = _pt("b" * 64, 100.0, 1 << 20, [100.0, 99.5, 100.5, 100.2, 99.8])
        self.assertTrue(harness.dominates_measured(OBJS, a, b, ALPHA))
        self.assertFalse(harness.dominates_measured(OBJS, b, a, ALPHA))

    def test_without_readings_it_falls_back_to_the_median(self):
        a = _pt("a" * 64, 90.0, 1 << 20)
        b = _pt("b" * 64, 90.3, 1 << 20)
        self.assertTrue(harness.dominates_measured(OBJS, a, b, ALPHA),
                        "refusing to order two points nobody sampled would fill "
                        "the front with duplicates")

    def test_an_unresolved_pair_at_the_same_workspace_both_stay(self):
        """Nothing separates them, so neither dominates and the front keeps both.

        This is the case `undetermined` exists for: two programs at the same
        corner, told apart only by noise.
        """
        noisy = [90.0, 110.0, 95.0, 105.0, 100.0]
        arc, v1 = harness.archive_insert(
            OBJS, [], _pt("a" * 64, 100.0, 380 << 20, noisy), ALPHA)
        arc, v2 = harness.archive_insert(
            OBJS, arc, _pt("b" * 64, 100.3, 380 << 20, [x + 0.3 for x in noisy]), ALPHA)
        self.assertEqual([v1, v2], ["inserted", "inserted"])
        self.assertEqual(len(arc), 2)

    def test_the_cheaper_point_takes_the_front_when_latency_is_unresolved(self):
        """The price of the change, pinned so it is a decision and not a drift.

        The FASTEST point can leave the front. It does not leave the run:
        `impl.py`, `best_aicore_e2e_us` and the ratchet are not the archive.
        """
        noisy = [90.0, 110.0, 95.0, 105.0, 100.0]
        arc, _ = harness.archive_insert(
            OBJS, [], _pt("a" * 64, 100.0, 1600 << 20, noisy), ALPHA)
        arc, verdict = harness.archive_insert(
            OBJS, arc, _pt("b" * 64, 100.3, 380 << 20, [x + 0.3 for x in noisy]), ALPHA)
        self.assertEqual(verdict, "inserted")
        self.assertEqual([e["code_hash"] for e in arc], ["b" * 64])

    def test_repruning_drops_a_point_the_new_readings_dominate(self):
        tight_fast = [90.0, 90.2, 90.1, 90.3, 90.05]
        arc = [_pt("a" * 64, 90.0, 1 << 20, tight_fast),
               _pt("b" * 64, 100.0, 1 << 20, [100.0, 99.9, 100.1, 100.2, 99.8])]
        self.assertEqual(len(harness.archive_reprune(OBJS, arc, ALPHA)), 1)


class BlockSamplesTest(unittest.TestCase):
    """A block point must carry its readings into the cache and the archive.

    Without them the front compares medians, which on a noisy kernel means
    archive points whose `samples` are empty deciding what stays on the front.
    """

    def test_a_block_trial_carries_its_readings(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        op_dir = Path(tmp.name)
        (op_dir / "optimization").mkdir()
        st = {"op": "demo", "eval_cache": {}, "archive": [],
              "config": dict(harness.DEFAULT_CONFIG, objectives=["aicore_e2e", "workspace"]),
              "target": {"P_ref": 50}, "progress": {"evals_used": 1},
              "nodes": {"open": [], "closed": []}}
        a = SimpleNamespace(parent="u1", op_dir=str(op_dir))
        ctx = {"faults": [], "trials": [], "meter": {"device_trials": 0},
               "winners": {}, "base_src": KERNEL}
        harness.block_recorder(a, st, ctx)(
            {"status": "ok",
             # What the trial was RANKED on, which is `p`.
             "objective_us": 90.0,
             "config": {"vec#0": {"d0": 32, "d1": 128}},
             "metrics": {"workspace_b": 1 << 20, "workspace_state": "measured"},
             "samples": [89.0, 90.0, 91.0]})
        entry = next(iter(st["eval_cache"].values()))
        self.assertEqual(entry["samples"], [89.0, 90.0, 91.0])
        self.assertEqual(st["archive"][0]["samples"], [89.0, 90.0, 91.0],
                         "and they reach the point the front judges")

    def test_promoting_the_winner_does_not_destroy_its_readings(self):
        """The winner is promoted by a `record` with no `--samples`.

        `m` was recovered from the block (`block_metrics`); the readings were
        not, and the promotion's cache write dropped them, so the delivered
        program held `samples: []` while the archive row for the same code hash
        still carried the readings the block had taken. `verify` and
        `record --remeasure` both read the cache.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        op_dir = Path(tmp.name) / "demo"
        (op_dir / "optimization").mkdir(parents=True)
        f = op_dir / "demo_impl.py"
        f.write_text(KERNEL, encoding="utf-8")
        run_cli("init", "--op-dir", str(op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", file_hash(f),
                "--preopt-workspace-b", str(1 << 20), "--op-file", str(f),
                "--allow-unversioned", *CHIP_ARGS,
                "--config", json.dumps({"objectives": ["aicore_e2e", "workspace"]}))
        f.write_text(KERNEL.replace("64, 128", "32, 128"), encoding="utf-8")
        h = file_hash(f)
        state = op_dir / "optimization" / "search_state.json"
        st = json.loads(state.read_text())
        # What a block leaves behind for its winner: measured, with readings.
        st["eval_cache"][h] = {"s": 1, "p": 90.0, "via": "block",
                               "samples": [89.0, 90.0, 91.0],
                               "m": {"workspace_b": 1 << 20,
                                     "workspace_state": "measured"}}
        state.write_text(json.dumps(st), encoding="utf-8")
        # The promotion: no --samples, because the optimizer did not re-measure.
        run_cli("record", "--op-dir", str(op_dir), "--parent", "u1",
                "--code-hash", h, "--s", "1", "--p", "90.0",
                "--workspace-b", str(1 << 20))
        entry = json.loads(state.read_text())["eval_cache"][h]
        self.assertEqual(entry["samples"], [89.0, 90.0, 91.0])

    def test_readings_the_caller_passes_win_over_the_block_s(self):
        """A real re-measurement replaces them; it does not merge."""
        st = {"eval_cache": {"a" * 64: {"samples": [1.0, 2.0]}}}
        self.assertEqual(harness.block_samples(st, "a" * 64), [1.0, 2.0])
        self.assertEqual(harness.block_samples(st, "b" * 64), [])
        self.assertEqual(harness.block_samples(st, None), [])


class RemeasureTest(unittest.TestCase):

    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", self.pre,
                "--preopt-workspace-b", str(1000 << 20),
                "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS,
                "--config", json.dumps(LW))
        self.point = self._record(128, 90.0, 1600 << 20, [89.0, 90.0, 91.0])

    def test_more_readings_are_pooled_and_not_charged(self):
        before = read_state(self.op_dir)["progress"]["evals_used"]
        res, rc = run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                          "--code-hash", self.point, "--s", "1", "--p", "90",
                          "--remeasure", "--samples", "90.5,89.5,90.2")
        self.assertEqual(rc, 0, res)
        self.assertFalse(res["charged"])
        self.assertEqual(res["readings"], 6)
        st = read_state(self.op_dir)
        self.assertEqual(st["progress"]["evals_used"], before, "the budget is untouched")
        self.assertEqual(st["progress"]["remeasurements"], 1)
        self.assertEqual(len(st["eval_cache"][self.point]["samples"]), 6)
        point = next(e for e in st["archive"] if e["code_hash"] == self.point)
        self.assertEqual(len(point["samples"]), 6)
        # `aicore_e2e_us` in the payload is THIS point's pooled reading.
        self.assertEqual(point["values"]["aicore_e2e"], res["aicore_e2e_us"])

    def test_the_ratchet_is_not_touched(self):
        n_before = (read_state(self.op_dir)["progress"].get("refine") or {}).get("n")
        run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                "--code-hash", self.point, "--s", "1", "--p", "90",
                "--remeasure", "--samples", "90.5,89.5,90.2")
        self.assertEqual((read_state(self.op_dir)["progress"].get("refine") or {}).get("n"),
                         n_before)

    def test_a_program_that_was_never_measured_is_refused(self):
        res, rc = run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                          "--code-hash", "f" * 64, "--s", "1", "--p", "90",
                          "--remeasure", "--samples", "90.5,89.5")
        self.assertEqual(rc, 3)
        self.assertEqual(res["error"], "not_a_front_point")

    def test_readings_are_mandatory(self):
        res, rc = run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                          "--code-hash", self.point, "--s", "1", "--p", "90",
                          "--remeasure")
        self.assertEqual(rc, 3)
        self.assertEqual(res["error"], "no_samples")

    def test_the_run_headline_follows_the_pooled_median_for_its_own_best(self):
        """`progress` and the front row describe the SAME program.

        Left alone, `stop` and the report quoted `best_aicore_e2e_us` from the
        original readings while the archive row for that code_hash carried the
        pooled median -- two latencies for one kernel, and the speedup computed
        from the stale one.
        """
        # `stop --force` runs the ratchet on the open action's best, which is
        # how a budget-ended run gets a best_node at all (`_promote_open_refine`).
        run_cli("stop", "--op-dir", str(self.op_dir), "--force", "budget")
        pr = read_state(self.op_dir)["progress"]
        self.assertEqual(pr["best_aicore_e2e_us"], 90.0)
        res, rc = run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                          "--code-hash", self.point, "--s", "1", "--p", "90",
                          "--remeasure", "--samples", "99.0,99.0,99.0")
        self.assertEqual(rc, 0, res)
        pr = read_state(self.op_dir)["progress"]
        point = next(e for e in read_state(self.op_dir)["archive"]
                     if e["code_hash"] == self.point)
        self.assertEqual(pr["best_aicore_e2e_us"], res["aicore_e2e_us"])
        self.assertEqual(pr["best_aicore_e2e_us"], point["aicore_e2e_us"],
                         "the headline and the front row are the same program")
        self.assertEqual(pr["best_aicore_e2e_us"], point["values"]["aicore_e2e"])
        self.assertEqual(pr["best_aicore_e2e_us"], 95.0, "the pooled median")
        self.assertEqual(pr["best_J"], point["J"])

    def test_a_remeasured_point_that_is_not_the_best_leaves_the_headline_alone(self):
        other = self._record(256, 95.0, 380 << 20, [95.0, 95.0, 95.0])
        before = read_state(self.op_dir)["progress"]["best_aicore_e2e_us"]
        run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                "--code-hash", other, "--s", "1", "--p", "95",
                "--remeasure", "--samples", "99.0,99.0")
        self.assertEqual(read_state(self.op_dir)["progress"]["best_aicore_e2e_us"], before)

    def test_the_front_is_re_derived_from_the_pooled_readings(self):
        res, _ = run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                         "--code-hash", self.point, "--s", "1", "--p", "90",
                         "--remeasure", "--samples", "90.1,90.0,89.9")
        self.assertEqual(res["archive_points"], len(read_state(self.op_dir)["archive"]))
        log = next((self.op_dir / "optimization").glob("*.md")).read_text()
        self.assertIn(f"- remeasure {self.point[:8]}: +3 readings", log)

    def _record(self, tile, p, ws, samples):
        self.f.write_text(KERNEL.replace("64, 128", f"{tile}, 128"), encoding="utf-8")
        h = file_hash(self.f)
        run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                "--code-hash", h, "--s", "1", "--p", str(p), "--workspace-b", str(ws),
                "--samples", ",".join(str(x) for x in samples))
        return h


if __name__ == "__main__":
    unittest.main()
