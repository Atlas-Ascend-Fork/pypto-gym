# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The multi-objective margin, pinned against the scalar one it replaces.

Moving acceptance from `j > bar` to the front is safe only if the number the
STOPPING rules read does not change with it. SPRT, `stage_min_gain` and the
stagnation counters all consume `log(J/bar)`; this file fixes that
`epsilon_margin` returns exactly that quantity whenever the run has one
objective and the archive holds one point, so those rules carry over untouched.

The last two cases are runs where the delivered kernel and the front
disagreed. They are here as facts, not illustrations: whatever acceptance rule
replaces the ratchet has to get them right.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import SCRIPTS_DIR  # noqa: E402,F401

import mo_margin  # noqa: E402

LATENCY = [{"name": "aicore_e2e", "direction": "min"}]
LATENCY_AND_WORKSPACE = [{"name": "aicore_e2e", "direction": "min"},
                         {"name": "workspace", "direction": "min"}]


def current_margin(p_ref, p_best, p_cand):
    """
    What `_apply_ratchet` computed before the front: `log(J/bar)`
    with `J = 100 * P_ref / p`.
    """
    return math.log((100.0 * p_ref / p_cand) / (100.0 * p_ref / p_best))


class OneObjectiveReduction(unittest.TestCase):
    """The property the whole refactor rests on."""

    def test_it_is_the_margin_the_harness_computes_today(self):
        # Measured pairs: a preopt, two later bests, and the front point that
        # ties one of them on latency.
        for p_best, p_cand in ((189.78, 144.78), (144.78, 87.12),
                               (87.12, 89.64), (250.28, 168.42), (100.0, 100.0)):
            with self.subTest(best=p_best, cand=p_cand):
                ours = mo_margin.epsilon_margin(
                    LATENCY, [{"aicore_e2e": p_best}], {"aicore_e2e": p_cand})
                self.assertAlmostEqual(
                    ours, current_margin(80.6, p_best, p_cand), places=12)

    def test_it_carries_no_reference_point(self):
        # `J` does: a run with `P_ref` unresolved at the 1.0 floor had a
        # `best_J` near zero and `success` unreachable. A ratio of two
        # measurements cannot inherit that.
        a, c = [{"aicore_e2e": 4172.56}], {"aicore_e2e": 3696.6}
        ours = mo_margin.epsilon_margin(LATENCY, a, c)
        for p_ref in (1.0, 80.6, 4464.6):
            self.assertAlmostEqual(
                ours, current_margin(p_ref, 4172.56, 3696.6), places=12)

    def test_an_improvement_is_positive_and_a_regression_negative(self):
        faster = mo_margin.epsilon_margin(
            LATENCY, [{"aicore_e2e": 100.0}], {"aicore_e2e": 50.0})
        slower = mo_margin.epsilon_margin(
            LATENCY, [{"aicore_e2e": 100.0}], {"aicore_e2e": 200.0})
        self.assertAlmostEqual(faster, math.log(2.0), places=12)
        self.assertAlmostEqual(slower, math.log(0.5), places=12)


class WhyThisIsNotTheStoppingMargin(unittest.TestCase):
    """The limitation, pinned so nobody wires this into SPRT by mistake.

    `max_i` reads the axis on which the candidate is CLOSEST to the archive
    point, which is the right answer to "how much improvement would clear the
    front" and the wrong input to "is this action worth another evaluation".
    """

    def test_a_tie_on_any_axis_collapses_the_margin_to_zero(self):
        arch = [{"aicore_e2e": 100.0, "workspace": 100.0}]
        near = mo_margin.epsilon_margin(
            LATENCY_AND_WORKSPACE, arch,
            {"aicore_e2e": 101.0, "workspace": 100.0})
        disaster = mo_margin.epsilon_margin(
            LATENCY_AND_WORKSPACE, arch,
            {"aicore_e2e": 400.0, "workspace": 100.0})
        # One percent slower and four times slower are both one epsilon of
        # workspace from the front, and the margin says so. Neither is accepted,
        # which is correct; SPRT reading these as the same evidence is not.
        self.assertEqual(near, 0.0)
        self.assertEqual(disaster, 0.0)

    def test_the_nearest_axis_hides_every_other_one(self):
        # Not only on exact ties. Against (100, 100), both of these are one
        # percent of workspace from the front, so `max_i` returns the workspace
        # ratio for both and the 4x latency regression never enters the number.
        arch = [{"aicore_e2e": 100.0, "workspace": 100.0}]
        near = mo_margin.epsilon_margin(
            LATENCY_AND_WORKSPACE, arch,
            {"aicore_e2e": 101.0, "workspace": 101.0})
        disaster = mo_margin.epsilon_margin(
            LATENCY_AND_WORKSPACE, arch,
            {"aicore_e2e": 400.0, "workspace": 101.0})
        self.assertLess(near, 0.0)
        self.assertEqual(near, disaster)


class FrontBehaviour(unittest.TestCase):

    def test_an_empty_archive_is_cleared_by_anything(self):
        self.assertEqual(
            mo_margin.epsilon_margin(LATENCY_AND_WORKSPACE, [],
                                     {"aicore_e2e": 1.0, "workspace": 1.0}),
            mo_margin.INF)

    def test_it_must_clear_every_point_not_just_one(self):
        arch = [{"aicore_e2e": 100.0, "workspace": 900.0},
                {"aicore_e2e": 300.0, "workspace": 100.0}]
        # Beats the first on workspace and the second on latency: a real corner.
        ok, m = mo_margin.extends_front(
            LATENCY_AND_WORKSPACE, arch,
            {"aicore_e2e": 200.0, "workspace": 500.0})
        self.assertTrue(ok)
        self.assertGreater(m, 0.0)
        # Beaten by the first on both: no corner, whatever the second says.
        ok, m = mo_margin.extends_front(
            LATENCY_AND_WORKSPACE, arch,
            {"aicore_e2e": 150.0, "workspace": 950.0})
        self.assertFalse(ok)
        self.assertLess(m, 0.0)

    def test_a_maximised_objective_inverts_rather_than_negates(self):
        objs = [{"name": "throughput", "direction": "max"}]
        better = mo_margin.epsilon_margin(
            objs, [{"throughput": 100.0}], {"throughput": 200.0})
        worse = mo_margin.epsilon_margin(
            objs, [{"throughput": 100.0}], {"throughput": 50.0})
        self.assertAlmostEqual(better, math.log(2.0), places=12)
        self.assertAlmostEqual(worse, math.log(0.5), places=12)

    def test_zero_is_a_reading_not_a_refusal(self):
        # `workspace` has a measured `zero` state; a kernel that reserves none
        # wins that axis outright rather than breaking the comparison.
        objs = LATENCY_AND_WORKSPACE
        m = mo_margin.epsilon_margin(
            objs, [{"aicore_e2e": 100.0, "workspace": 500.0}],
            {"aicore_e2e": 100.0, "workspace": 0.0})
        self.assertEqual(m, mo_margin.INF)
        tie = mo_margin.epsilon_margin(
            objs, [{"aicore_e2e": 100.0, "workspace": 0.0}],
            {"aicore_e2e": 100.0, "workspace": 0.0})
        self.assertAlmostEqual(tie, 0.0, places=12)

    def test_a_missing_objective_refuses_rather_than_guesses(self):
        self.assertIsNone(mo_margin.epsilon_margin(
            LATENCY_AND_WORKSPACE, [{"aicore_e2e": 100.0, "workspace": 100.0}],
            {"aicore_e2e": 90.0}))
        ok, m = mo_margin.extends_front(
            LATENCY_AND_WORKSPACE, [{"aicore_e2e": 100.0, "workspace": 100.0}],
            {"aicore_e2e": 90.0})
        self.assertFalse(ok)
        self.assertIsNone(m)


class TheTwoCampaignsThatDisagreed(unittest.TestCase):
    """Recorded facts. Any acceptance rule has to get these right."""

    def test_the_delivered_kernel_was_dominated(self):
        # One campaign. `impl.py` held a point the rank above it beat
        # outright, at the SAME workspace to the byte. The harness could not
        # see it: `delivered_is_best` compares the file against the RATCHET's
        # best, and the ratchet reads one objective.
        rank2 = {"aicore_e2e": 3689.8, "workspace": 1039.79}
        delivered = {"aicore_e2e": 3696.6, "workspace": 1039.79}
        # Zero, not negative: the two tie on workspace, so the delivered point
        # is an epsilon from the front while being strictly worse. Dominated
        # either way -- `extends_front` refuses it -- but see
        # `WhyThisIsNotTheStoppingMargin`.
        self.assertEqual(
            mo_margin.epsilon_margin(LATENCY_AND_WORKSPACE, [rank2], delivered), 0.0)
        self.assertFalse(
            mo_margin.extends_front(LATENCY_AND_WORKSPACE, [rank2], delivered)[0])
        self.assertGreater(
            mo_margin.epsilon_margin(LATENCY_AND_WORKSPACE, [delivered], rank2), 0.0)

    def test_the_sparse_trade_is_not_a_domination(self):
        # A second campaign. `impl.py` held a marginally faster kernel
        # at many times the workspace of the front's ship candidate. Neither
        # dominates, so both clear the other and the choice is the caller's.
        # The ratchet made it on latency alone.
        delivered = {"aicore_e2e": 168.42, "workspace": 4.36 * 1024}
        ship = {"aicore_e2e": 171.72, "workspace": 321.8}
        self.assertGreater(
            mo_margin.epsilon_margin(LATENCY_AND_WORKSPACE, [ship], delivered), 0.0)
        self.assertGreater(
            mo_margin.epsilon_margin(LATENCY_AND_WORKSPACE, [delivered], ship), 0.0)


if __name__ == "__main__":
    unittest.main()
