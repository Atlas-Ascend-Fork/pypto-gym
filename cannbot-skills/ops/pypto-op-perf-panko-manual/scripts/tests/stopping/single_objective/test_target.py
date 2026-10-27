# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""`P_ref` as a target, and what happens when the preopt already meets it.

`success` was `best_J >= 100`, and `J = 100 * P_ref / p`, so it says "the
primary objective reached P_ref". Two things were wrong with how it was asked:

  * it read `best_J`, which only moves at an action's `close` and had already
    been shown to drift from the measurements (see test_delivery.py). A run
    that met the target inside an action still open would not have seen it. It
    now reads the program the run would DELIVER;
  * nothing checked `j0` at `init`. A campaign passed a `P_ref` its
    own preopt was FASTER than, so `j0` started above 100, `success` was
    already true, and the run ended at evaluation 0 and reported it as a
    success. The number was right and the conclusion was wrong.

`init` refuses that configuration and asks, because only the caller can tell a
mistyped target from a kernel that genuinely beats the reference.
`config.success_stop: false` is how to say the second.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import CHIP_ARGS, DEMO_KERNEL as KERNEL, file_hash, harness, new_op_dir, read_state, run_cli  # noqa: E402


class InitTest(unittest.TestCase):
    def test_the_met_target_is_refused_with_both_numbers(self):
        _, _, res, rc = self.init_run(4464.6, 4188.0)
        self.assertEqual(rc, 3)
        self.assertEqual(res["error"], "target_already_met")
        self.assertEqual((res["p_ref"], res["preopt_p"]), (4464.6, 4188.0))
        self.assertGreater(res["j0"], 100.0)
        self.assertEqual([o["id"] for o in res["ask_user"]["options"]],
                         ["fix_p_ref", "no_target"])

    def test_exactly_at_the_target_is_also_refused(self):
        """`j0 == 100` means `success` is true on the first `stop`."""
        _, _, res, rc = self.init_run(100.0, 100.0)
        self.assertEqual((rc, res["error"]), (3, "target_already_met"))

    def test_saying_there_is_no_target_lets_it_run(self):
        op_dir, _, res, rc = self.init_run(4464.6, 4188.0, {"success_stop": False})
        self.assertEqual(rc, 0, res)
        stop, _ = run_cli("stop", "--op-dir", str(op_dir))
        self.assertFalse(stop["stop"], "the budget and the wall clock remain")

    def test_an_ordinary_target_is_untouched(self):
        op_dir, _, res, rc = self.init_run(50.0, 100.0)
        self.assertEqual(rc, 0, res)
        self.assertEqual(res["best_J"], 50.0)
        stop, _ = run_cli("stop", "--op-dir", str(op_dir))
        self.assertFalse(stop["stop"])

    def init_run(self, p_ref, preopt_p, config=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        op_dir = Path(tmp.name) / "demo"
        op_dir.mkdir(parents=True)
        f = op_dir / "demo_impl.py"
        f.write_text(KERNEL, encoding="utf-8")
        args = ["init", "--op-dir", str(op_dir), "--op", "demo",
                "--preopt-p", str(preopt_p),
                "--preopt-hash", file_hash(f),
                "--preopt-workspace-b", str(1000 << 20),
                "--op-file", str(f), "--allow-unversioned", *CHIP_ARGS]
        if p_ref is not None:
            args += ["--p-ref", str(p_ref)]
        if config:
            args += ["--config", json.dumps(config)]
        res, rc = run_cli(*args)
        return op_dir, f, res, rc


class NoReferenceTest(unittest.TestCase):
    """`--p-ref` is optional, and leaving it out means there is no target.

    It was REQUIRED, so a campaign with no reference kernel had to put a number
    there -- and the run above is what that produced. The caller could
    not say "there is nothing to reach"; it could only guess, and a guess on
    the wrong side of the preopt ends the run at evaluation 0.
    """

    init_run = InitTest.init_run

    def test_without_it_the_preopt_is_the_reference(self):
        _, _, res, rc = self.init_run(None, 189.56)
        self.assertEqual(rc, 0, res)
        self.assertEqual((res["p_ref"], res["p_ref_via"]), (189.56, "preopt"))

    def test_and_j_reads_as_a_percentage_of_the_baseline(self):
        """
        `J = 100 * P_ref / p`, so `J = 150` is 1.5x the preopt. The search
        is unchanged: `J` is monotone in `1/p` for any fixed `P_ref`.
        """
        _, _, res, _ = self.init_run(None, 189.56)
        self.assertEqual(res["best_J"], 100.0)

    def test_and_success_stop_goes_off_with_it(self):
        """
        `j0` is exactly 100 by construction, so leaving the stop on would
        refuse every run that did not name a target.
        """
        op_dir, _, res, rc = self.init_run(None, 189.56)
        self.assertEqual(rc, 0, res)
        self.assertIs(res["success_stop"], False)
        stop, _ = run_cli("stop", "--op-dir", str(op_dir))
        self.assertFalse(stop["stop"], "the budget and the wall clock remain")

    def test_the_state_records_where_the_reference_came_from(self):
        """A report that quotes a speedup has to be able to say what against."""
        op_dir, _, _, _ = self.init_run(None, 189.56)
        st = json.loads((op_dir / "optimization" / "search_state.json").read_text())
        self.assertEqual(st["target"]["P_ref_via"], "preopt")
        self.assertEqual(st["target"]["P_ref"], 189.56)

    def test_a_named_reference_says_so_too(self):
        op_dir, _, res, _ = self.init_run(50.0, 100.0)
        st = json.loads((op_dir / "optimization" / "search_state.json").read_text())
        self.assertEqual(st["target"]["P_ref_via"], "explicit:cli")
        self.assertEqual(res["p_ref_via"], "explicit:cli")
        self.assertIs(res["success_stop"], True)

    def test_asking_to_stop_at_a_target_it_did_not_name_is_refused(self):
        """
        Not overridden silently: the caller asked for two things that cannot
        both hold.
        """
        _, _, res, rc = self.init_run(None, 189.56, {"success_stop": True})
        self.assertEqual((rc, res["error"]), (3, "target_already_met"))
        self.assertEqual(res["p_ref_via"], "preopt")
        self.assertIn("no --p-ref", res["reason"])

    def test_an_explicit_false_is_left_alone(self):
        _, _, res, rc = self.init_run(None, 189.56, {"success_stop": False})
        self.assertEqual(rc, 0, res)
        self.assertIs(res["success_stop"], False)


class ReachedTest(unittest.TestCase):
    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo",
                "--p-ref", "50", "--preopt-p", "100",
                "--preopt-hash", file_hash(self.f),
                "--preopt-workspace-b", str(1000 << 20),
                "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS)

    def test_the_target_is_read_off_the_delivered_program(self):
        """
        No `close` has run, so `best_J` has not moved -- and the condition
        still fires, which is the point.
        """
        h = self._record(128, 49.0)
        st = read_state(self.op_dir)
        self.assertLess(st["progress"]["best_J"], 100.0, "best_J has not moved")
        self.assertTrue(harness.target_met(st, str(self.op_dir)))
        stop, _ = run_cli("stop", "--op-dir", str(self.op_dir))
        self.assertEqual((stop["stop"], stop["reason"]), (True, "success"))
        self.assertEqual(stop["best_code_hash"], h)

    def test_short_of_the_target_does_not_fire(self):
        self._record(128, 51.0)
        stop, _ = run_cli("stop", "--op-dir", str(self.op_dir))
        self.assertFalse(stop["stop"], stop)

    def test_success_stop_false_ignores_a_met_target(self):
        st = read_state(self.op_dir)
        st["config"]["success_stop"] = False
        (self.op_dir / "optimization" / "search_state.json").write_text(json.dumps(st))
        self._record(128, 49.0)
        self.assertTrue(harness.target_met(read_state(self.op_dir), str(self.op_dir)))
        stop, _ = run_cli("stop", "--op-dir", str(self.op_dir))
        self.assertFalse(stop["stop"], "measured, not enforced")

    def test_no_p_ref_is_no_target(self):
        st = read_state(self.op_dir)
        st["target"]["P_ref"] = 0
        self.assertFalse(harness.target_met(st, str(self.op_dir)))

    def _record(self, tile, p, mib=900):
        self.f.write_text(KERNEL.replace("64, 128", f"{tile}, 128"), encoding="utf-8")
        h = file_hash(self.f)
        run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                "--code-hash", h, "--s", "1", "--p", str(p),
                "--workspace-b", str(mib << 20),
                "--samples", ",".join(str(p + i * 0.05) for i in range(9)))
        return h


if __name__ == "__main__":
    unittest.main()
