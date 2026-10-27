# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""A tile draw that is too big is shrunk until it fits, not rejected.

Nearly every static rejection in a block is a buffer overflow, and each one
costs the block a measurement it could have had. Rejecting hands TPE a cliff;
repairing measures the nearest feasible point and reports it for the drawn one.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import CUBE_KERNEL as KERNEL, SCRIPTS_DIR  # noqa: E402,F401

from bayesian_optimization import block, space  # noqa: E402


def _hw():
    return space.HW(calibrated=False, dtype_bytes=2)


class RepairUnitTest(unittest.TestCase):
    def test_an_l0a_overflow_shrinks_its_own_axes(self):
        hw = _hw()
        cfg = {"cube#0": {"mL0": 512, "kL0": 512, "nL0": 64,
                          "mL1": 512, "kL1": 512, "nL1": 64}}
        self.assertFalse(space.static_feasible(cfg, hw)[0])
        fixed, steps = space.repair(cfg, self._check(hw))
        self.assertIsNotNone(fixed)
        self.assertTrue(space.static_feasible(fixed, hw)[0])
        self.assertEqual(fixed["cube#0"]["nL0"], 64, "L0A does not involve n")
        self.assertGreater(steps, 0)

    def test_it_stops_at_the_first_point_that_fits(self):
        """Nearest, not smallest: one rung below the boundary, not the floor."""
        hw = _hw()
        cfg = {"cube#0": {"mL0": 256, "kL0": 256, "nL0": 16,
                          "mL1": 256, "kL1": 256, "nL1": 16}}
        fixed, _ = space.repair(cfg, self._check(hw))
        c = fixed["cube#0"]
        self.assertLessEqual(space.ceil16(c["mL0"]) * space.ceil16(c["kL0"]) * 2, 64 * 1024)
        # one more rung up on the axis that moved would not fit
        self.assertGreater(c["mL0"] * c["kL0"], 64 * 1024 // 2 // 2)

    def test_the_l1_multiplier_shrinks_before_l0(self):
        hw = _hw()
        cfg = {"cube#0": {"mL0": 128, "kL0": 128, "nL0": 128,
                          "mL1": 512, "kL1": 512, "nL1": 512}}
        self.assertIn("L1 tile", space.static_feasible(cfg, hw)[1])
        fixed, _ = space.repair(cfg, self._check(hw))
        c = fixed["cube#0"]
        self.assertEqual((c["mL0"], c["kL0"], c["nL0"]), (128, 128, 128))
        for ax in "mkn":
            self.assertEqual(c[f"{ax}L1"] % c[f"{ax}L0"], 0)

    def test_a_ub_overflow_shrinks_the_vec_tile(self):
        # One operation holding three tiles of vec#0. `verify` gates on the
        # operation groups, so a domain without them states no bound at all.
        dom = {"ub_bytes": 192 * 1024, "dtype_bytes": 2,
               "vec_residency": {"vec#0": 3},
               "vec_op_groups": [[["vec#0", None]] * 3]}
        from bayesian_optimization import domain as D

        def check(c):
            return D.verify(dom, c, 2)
        cfg = {"vec#0": {"d0": 256, "d1": 1024}}
        self.assertFalse(check(cfg)[0])
        fixed, _ = space.repair(cfg, check, space.RepairContext(domain=dom))
        self.assertTrue(check(fixed)[0])
        self.assertEqual(fixed["vec#0"]["d1"] % 16, 0)

    def test_a_rule_that_is_not_capacity_is_not_repaired(self):
        fixed, _ = space.repair({"vec#0": {"d0": 8, "d1": 100}},
                                lambda c: space.static_feasible(c, _hw()))
        self.assertIsNone(fixed, "an alignment violation is not a size problem")

    def test_a_tied_leading_dimension_moves_together(self):
        dom = {"ub_bytes": 192 * 1024, "dtype_bytes": 2,
               "vec_residency": {"vec#0": 1, "vec#1": 1}}
        from bayesian_optimization import domain as D
        cfg = {"vec#0": {"d0": 512, "d1": 256}, "vec#1": {"d0": 512, "d1": 16}}
        fixed, _ = space.repair(cfg, lambda c: D.verify(dom, c, 2),
                                space.RepairContext(domain=dom,
                                                    tie=[["vec#0", "vec#1"]]))
        self.assertTrue(D.verify(dom, fixed, 2)[0])
        self.assertEqual(fixed["vec#0"]["d0"], fixed["vec#1"]["d0"])

    def _check(self, hw):
        return lambda c: space.static_feasible(c, hw)


class RepairBlockTest(unittest.TestCase):
    """Against a block whose objective refuses nothing the static rules pass."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.op_file = Path(self._tmp.name) / "k.py"
        self.op_file.write_text(KERNEL, encoding="utf-8")

    def test_repair_measures_more_and_never_ships_the_draw(self):
        off, _ = self._run(False)
        on, _ = self._run(True)
        rejected_off = sum(1 for h in off["history"] if h["status"] == "infeasible_static")
        self.assertGreater(rejected_off, 0, "the fixture must exercise the gate")
        self.assertGreater(on["n_repaired"], 0)
        self.assertLess(sum(1 for h in on["history"] if h["status"] == "infeasible_static"),
                        rejected_off)
        self.assertTrue(space.static_feasible(on["best_config"],
                                              space.HW(calibrated=False, dtype_bytes=2))[0],
                        "best_config must be what was measured, not the repaired draw")
        for h in on["history"]:
            if h.get("repaired_from"):
                self.assertIn("repair_steps", h)

    def test_the_study_learns_the_value_at_the_measured_point(self):
        """
        The drawn point gets the penalty a static rejection would have given
        it; the measurement is filed as its own trial at the params that ran.
        """
        import optuna
        from bayesian_optimization import driver as D2
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study()
        trial = study.ask({"x": optuna.distributions.CategoricalDistribution([16, 32, 64])})
        ctx = {"objective": object(), "study": study, "history": [],
               "opts": block.BlockOptions(), "state": {"seen": {}},
               "repair": {"from": {"x": 64}, "steps": 1, "why": "too big",
                          "drawn_key": "drawn", "measured_params": {"x": 16}}}
        value = D2.record_ok(ctx, {"x": 16}, "measured", 42.0, trial)
        self.assertEqual(value, D2.PENALTY, "the draw cannot run, whatever it stood for")
        self.assertEqual(ctx["state"]["seen"]["drawn"], D2.PENALTY)
        self.assertEqual(ctx["state"]["seen"]["measured"], 42.0)
        # The asked trial may itself have drawn x=16, so filter on the value:
        # what is under test is that the MEASUREMENT is filed at these params.
        added = [t for t in study.trials
                 if t.params == {"x": 16} and t.value is not None]
        self.assertEqual([t.value for t in added], [42.0])

    def test_the_value_is_filed_under_the_params_that_were_measured(self):
        """
        Reporting a repaired draw's value for the DRAWN point tells the
        surrogate the infeasible region is fine, so it keeps drawing there -- and
        with repair on, every such draw costs an evaluation.
        """
        on, _ = self._run(True)
        study = on.get("study") or on.get("_study")
        del study
        drawn = [h for h in on["history"] if h.get("repaired_from")]
        self.assertTrue(drawn, "the fixture must exercise repair")
        hw = space.HW(calibrated=False, dtype_bytes=2)
        for h in drawn:
            self.assertTrue(space.static_feasible(h["config"], hw)[0],
                            "history records the measured point")
            self.assertFalse(space.static_feasible(h["repaired_from"], hw)[0],
                             "and the draw it stands for, which cannot run")

    def test_off_is_the_old_behaviour(self):
        res, _ = self._run(False)
        self.assertEqual(res["n_repaired"], 0)

    def _run(self, repair):
        hw = space.HW(calibrated=False, dtype_bytes=2)
        seen = []

        def obj(cfg):
            ok, why = space.static_feasible(cfg, hw)
            assert ok, f"an infeasible config reached the device: {why}"
            seen.append(cfg)
            c = cfg["cube#0"]
            return 1, float(2000 - c["mL0"] - c["kL0"] - c["nL0"])
        res = block.run(str(self.op_file), obj, hw,
                        block.BlockOptions(stagnation_k=99, max_trials=30,
                                           current_latency=1000.0, seed=0,
                                           repair=repair))
        return res, seen


if __name__ == "__main__":
    unittest.main()
