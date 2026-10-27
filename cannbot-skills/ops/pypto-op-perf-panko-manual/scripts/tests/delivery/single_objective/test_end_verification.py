# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""End-of-run verification: the speedup the run reports is checked once, at the end.

Search-time acceptance is a point comparison of two median-of-9 readings, so the
reported speedup is the maximum over many noisy comparisons and is biased
upwards. `verify` re-measures the two programs the run hands over -- the preopt
and the delivered kernel -- in an ABBA order, and puts a bootstrap interval on
the speedup. The report says whether it held.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import (CHIP_ARGS, DEMO_KERNEL as KERNEL, file_hash, harness, INIT_ARGS, new_op_dir,  # noqa: E402
                      record_measurement, run_cli)

FASTER = KERNEL.replace("64, 128", "128, 128")


def _csv(xs):
    return ",".join(str(x) for x in xs)


class BootstrapTest(unittest.TestCase):
    def test_a_clear_speedup_is_confirmed(self):
        pre = [100 + (i % 5) for i in range(30)]
        dele = [80 + (i % 5) for i in range(30)]
        ci = harness.bootstrap_speedup(pre, dele)
        self.assertGreater(ci["ci_low"], 1.0)
        self.assertEqual(harness.ci_verdict(ci), "confirmed")

    def test_noise_alone_is_not_confirmed(self):
        pre = [100 + (i * 7 % 11) for i in range(30)]
        dele = [100 + (i * 5 % 11) for i in range(30)]
        self.assertEqual(harness.ci_verdict(harness.bootstrap_speedup(pre, dele)),
                         "unconfirmed")

    def test_a_regression_is_named(self):
        pre = [80 + (i % 5) for i in range(30)]
        dele = [100 + (i % 5) for i in range(30)]
        self.assertEqual(harness.ci_verdict(harness.bootstrap_speedup(pre, dele)),
                         "regressed")

    def test_it_is_deterministic(self):
        pre = [100 + (i * 7 % 13) for i in range(30)]
        dele = [95 + (i * 3 % 13) for i in range(30)]
        self.assertEqual(harness.bootstrap_speedup(pre, dele),
                         harness.bootstrap_speedup(pre, dele))

    def test_too_few_readings_give_no_interval(self):
        self.assertIsNone(harness.bootstrap_speedup([1.0], [1.0, 2.0]))


class VerifyFrontTest(unittest.TestCase):
    """Every retained front point gets its own ABBA pass.

    A front is handed over as a set of choices. A point nobody re-measured is a
    search-time point estimate -- which is the thing `verify` exists to stop the
    run reporting about its winner.
    """

    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "100",
                # Not 100: a preopt that already meets the target is refused
                # (test_target.py). The 80 us candidate meets it, which is what
                # releases the close below.
                "--preopt-p", "120", "--preopt-hash", self.pre,
                "--preopt-workspace-b", str(1000 << 20),
                "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS)
        self.fast = self._record("128, 128", 80.0, 1600 << 20)      # kept
        self.small = self._record("32, 128", 95.0, 300 << 20)       # trade-off
        run_cli("close", "--op-dir", str(self.op_dir), "--action", "u1")
        run_cli("stop", "--op-dir", str(self.op_dir), "--force", "manual")

    def test_the_front_is_verified_point_by_point(self):
        first, rec = self._pass([80 + i % 3 for i in range(15)],
                                [100 + i % 3 for i in range(15)])
        self.assertEqual(first, self.fast, "the delivered point goes first")
        self.assertEqual(rec["status"], "pending", "the front is not done")
        self.assertEqual(rec["point"], self.small)
        st = self._state()
        self.assertEqual(st["verifications"][0]["status"], "confirmed")
        self.assertEqual(st["verification"]["arms"]["delivered"]["code_hash"],
                         self.small)
        self.assertEqual(st["verification"]["arms"]["preopt"]["code_hash"], self.pre)

        second, rec2 = self._pass([95 + i % 3 for i in range(15)],
                                  [100 + i % 3 for i in range(15)])
        self.assertEqual(second, self.small)
        self.assertEqual(rec2["status"], "confirmed")
        st = self._state()
        self.assertEqual([v["status"] for v in st["verifications"]],
                         ["confirmed"], "the earlier pass is kept")
        self.assertEqual(st["verification"]["status"], "confirmed",
                         "and the last one stays where the summaries read it")
        self.assertEqual(st["verify_queue"], [])
        self.assertEqual(file_hash(self.f), self.fast,
                         "impl.py is the delivered point throughout")

    def test_the_frontier_records_each_point_s_verdict(self):
        self._pass([80 + i % 3 for i in range(15)], [100 + i % 3 for i in range(15)])
        self._pass([95 + i % 3 for i in range(15)], [100 + i % 3 for i in range(15)])
        run_cli("report", "--op-dir", str(self.op_dir))
        front = json.loads((self.op_dir / "optimization" / "frontier"
                            / "frontier.json").read_text(encoding="utf-8"))
        by_hash = {r["code_hash"]: r for r in front["front"]}
        self.assertEqual(by_hash[self.fast]["verified"]["status"], "confirmed")
        self.assertEqual(by_hash[self.small]["verified"]["status"], "confirmed")
        self.assertNotIn(self.pre, by_hash,
                         "the preopt is dominated here; it is the other arm of "
                         "every pass, never a point to verify")
        log = next((self.op_dir / "optimization").glob("*.md")).read_text()
        self.assertIn(f"- verification[{self.small[:8]}]:", log)

    def test_it_can_be_switched_off(self):
        self.op_dir = Path(self.tmp_dir.name) / "off"
        self.op_dir.mkdir(parents=True)
        self.f = self.op_dir / "demo_impl.py"
        self.f.write_text(KERNEL, encoding="utf-8")
        self.pre = file_hash(self.f)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "100",
                "--preopt-p", "120", "--preopt-hash", self.pre,
                "--preopt-workspace-b", str(1000 << 20),
                "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS,
                "--config", json.dumps({"verify_front": False}))
        self.fast = self._record("128, 128", 80.0, 1600 << 20)
        self._record("32, 128", 95.0, 300 << 20)
        run_cli("close", "--op-dir", str(self.op_dir), "--action", "u1")
        run_cli("stop", "--op-dir", str(self.op_dir), "--force", "manual")
        self.assertEqual(self._state()["verify_queue"], [])
        _, rec = self._pass([80 + i % 3 for i in range(15)],
                            [100 + i % 3 for i in range(15)])
        self.assertEqual(rec["status"], "confirmed", "and it stops there")
        self.assertEqual(self._state().get("verifications", []), [])

    def _record(self, tile, p, ws):
        return record_measurement(self, KERNEL.replace("64, 128", tile), p, ws)[0]

    def _state(self):
        return json.loads((self.op_dir / "optimization" / "search_state.json")
                          .read_text(encoding="utf-8"))

    def _pass(self, fast_vals, slow_vals):
        """One whole ABBA pass; returns the hash it verified and the last reply."""
        point = None
        for _ in range(4):
            staged = run_cli("verify", "--op-dir", str(self.op_dir))[0]
            point = point or self._state()["verification"]["arms"]["delivered"]["code_hash"]
            vals = slow_vals if staged["arm"] == "preopt" else fast_vals
            rec = run_cli("verify", "--op-dir", str(self.op_dir),
                          "--samples", _csv(vals))[0]
        return point, rec


class VerifyCliTest(unittest.TestCase):
    def setUp(self):
        self.op_file = new_op_dir(self, KERNEL)
        self.pre_hash = self.pre
        res = run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo",
                      "--p-ref", "100", "--preopt-p", "100",
                      # p_ref == preopt_p: no target, see test_target.py.
                      "--config", json.dumps({"success_stop": False}),
                      "--preopt-hash", self.pre_hash,
                      "--op-file", str(self.op_file),
                      "--allow-unversioned", *INIT_ARGS)[0]
        self.assertTrue(res["initialized"], res)

    def test_stop_asks_for_verification(self):
        res = self._improve_and_stop()
        self.assertTrue(res["verify"]["required"])
        self.assertEqual(self._state()["verification"]["status"], "pending")

    def test_abba_order_and_a_confirmed_result(self):
        self._improve_and_stop()
        arms = []
        for _ in range(4):
            arm, rec = self._step([80 + i % 3 for i in range(15)],
                                  [100 + i % 3 for i in range(15)])
            arms.append(arm)
        self.assertEqual(arms, ["preopt", "delivered", "delivered", "preopt"])
        self.assertEqual(rec["status"], "confirmed")
        self.assertEqual(rec["result"]["n_preopt"], 30)
        self.assertEqual(rec["search_speedup"], 1.25)
        run_cli("report", "--op-dir", str(self.op_dir))
        log = next((self.op_dir / "optimization").glob("*.md")).read_text()
        self.assertIn("verification: CONFIRMED", log)

    def test_samples_for_the_wrong_program_are_refused(self):
        self._improve_and_stop()
        # Nothing staged: the working file holds the delivered kernel, but the
        # first ABBA step is the preopt.
        res, rc = run_cli("verify", "--op-dir", str(self.op_dir), "--samples", "1,2,3")
        self.assertNotEqual(rc, 0)
        self.assertEqual(res["error"], "wrong_program")

    def test_verify_before_stop_is_refused(self):
        res, rc = run_cli("verify", "--op-dir", str(self.op_dir))
        self.assertNotEqual(rc, 0)
        self.assertEqual(res["error"], "no_stop")

    def test_no_change_needs_no_verification(self):
        res = run_cli("stop", "--op-dir", str(self.op_dir), "--force", "manual")[0]
        self.assertFalse(res["verify"]["required"])
        self.assertEqual(self._state()["verification"]["status"], "no_change")

    def test_skipped_is_reported_as_unverified(self):
        self._improve_and_stop()
        run_cli("verify", "--op-dir", str(self.op_dir))
        run_cli("verify", "--op-dir", str(self.op_dir), "--skip", "device_unavailable")
        self.assertEqual(file_hash(self.op_file), self.best_hash)
        rep = run_cli("report", "--op-dir", str(self.op_dir))[0]
        self.assertEqual(rep["verification"]["status"], "skipped")

    def test_a_repeated_stop_does_not_reset_progress(self):
        self._improve_and_stop()
        self._step([80] * 15, [100] * 15)
        run_cli("stop", "--op-dir", str(self.op_dir), "--force", "manual")
        self.assertEqual(self._state()["verification"]["step"], 1)

    def _state(self):
        return json.loads((self.op_dir / "optimization" / "search_state.json")
                          .read_text(encoding="utf-8"))

    def _improve_and_stop(self):
        self.op_file.write_text(FASTER, encoding="utf-8")
        self.best_hash = file_hash(self.op_file)
        r = run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                    "--code-hash", self.best_hash, "--s", "1", "--p", "80")[0]
        self.assertTrue(r["improved"], r)
        run_cli("close", "--op-dir", str(self.op_dir), "--action", "u1")
        return run_cli("stop", "--op-dir", str(self.op_dir), "--force", "manual")[0]

    def _step(self, fast, slow):
        staged = run_cli("verify", "--op-dir", str(self.op_dir))[0]
        self.assertEqual(file_hash(self.op_file), staged["code_hash"],
                         "staging puts the arm's program in the working file")
        vals = fast if staged["arm"] == "delivered" else slow
        rec = run_cli("verify", "--op-dir", str(self.op_dir), "--samples", _csv(vals))[0]
        self.assertEqual(file_hash(self.op_file), self.best_hash,
                         "the deliverable is put back after every measurement")
        return staged["arm"], rec


if __name__ == "__main__":
    unittest.main()
