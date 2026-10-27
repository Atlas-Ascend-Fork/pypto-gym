# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""`prior_gain` supplied by the model must be bounded, like `V` already is.

Selection is a product, `V_sel = w_stage * prior_gain * novelty * V`, and the
staged prior is an ORDERING claim that holds only while every factor is bounded
by one -- the argument `clamp_belief` documents for V. A catalogue node's prior
comes from `action_catalog.json` and is 0.9, 0.6 or 0.3. An inserted node's came
from the model's own JSON through a bare `float()`, unbounded, so a single
`prior_gain: 10` outranked every catalogue action by eleven times and
`novelty = 1/(1+tried)` needed ten firings to pull it back under.

These call the internals rather than the CLI: the defect is the missing bound,
not which command owns the record.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import harness  # noqa: E402

# The ceiling every catalogue action shares: the top `prior_gain` in
# action_catalog.json, at full belief, untried, inside the active stage.
CATALOG_CEILING = 0.9


def _insert(**kw):
    ins = {"parent": "x1", "stage": "frontend", "delta": "do a thing"}
    ins.update(kw)
    return {"insert": [ins]}


def _fresh_state():
    return {"op": "demo", "nodes": {"open": [], "held": [], "closed": []},
            "progress": {"active_stage": "frontend"}, "config": {}}


class ClampPriorTest(unittest.TestCase):

    def test_above_one_is_clamped_and_named(self):
        value, note = harness.clamp_prior(10.0)
        self.assertEqual(value, 1.0)
        self.assertIsNotNone(note)
        self.assertIn("10", note)

    def test_below_zero_is_clamped_and_named(self):
        value, note = harness.clamp_prior(-1.0)
        self.assertEqual(value, 0.0)
        self.assertIsNotNone(note)

    def test_in_range_passes_through_silently(self):
        for v in (0.0, 0.3, 0.5, 0.6, 0.9, 1.0):
            value, note = harness.clamp_prior(v)
            self.assertEqual(value, v)
            self.assertIsNone(note, f"{v} should need no adjustment")

    def test_unreadable_falls_back_to_the_default_not_to_zero(self):
        """
        A zero prior is ABSORBING in the product: the node would sit in the
        frontier and never be selectable, which is indistinguishable from bad
        luck. The documented default is the safe landing, and it is reported.
        """
        for bad in ("high", None, [], {}):
            value, note = harness.clamp_prior(bad)
            self.assertEqual(value, harness.CATALOG_PRIOR_DEFAULT)
            self.assertNotEqual(value, 0.0)
            self.assertIsNotNone(note)

    def test_nan_falls_back_to_the_default(self):
        value, note = harness.clamp_prior(float("nan"))
        self.assertFalse(math.isnan(value))
        self.assertEqual(value, harness.CATALOG_PRIOR_DEFAULT)
        self.assertIsNotNone(note)

    def test_default_is_below_the_catalogue_ceiling(self):
        """
        An action the catalogue does not contain has no measured standing, so
        its default must not outrank a catalogue action of maximal prior.
        """
        self.assertLess(harness.CATALOG_PRIOR_DEFAULT, CATALOG_CEILING)


class ApplyInsertsTest(unittest.TestCase):

    def test_out_of_range_prior_is_bounded_on_the_node(self):
        st = _fresh_state()
        notes = harness.apply_inserts(st, st["nodes"]["open"], _insert(prior_gain=10.0))
        node = st["nodes"]["open"][0]
        # The model's value is bounded and kept; under learned g it no longer
        # sets `prior_gain`, which must still be a prior.
        self.assertEqual(node["prior_gain_model"], 1.0)
        self.assertTrue(0.0 <= node["prior_gain"] <= 1.0)
        self.assertEqual(len(notes), 1)
        self.assertIn(node["id"], notes[0])

    def test_absent_prior_takes_the_default_without_a_note(self):
        st = _fresh_state()
        notes = harness.apply_inserts(st, st["nodes"]["open"], _insert())
        self.assertEqual(st["nodes"]["open"][0]["prior_gain_model"],
                         harness.CATALOG_PRIOR_DEFAULT)
        self.assertEqual(notes, [])

    def test_belief_is_still_bounded(self):
        st = _fresh_state()
        harness.apply_inserts(st, st["nodes"]["open"], _insert(V=1.4))
        self.assertEqual(st["nodes"]["open"][0]["V"], 1.0)

    def test_every_insert_is_reported_once(self):
        st = _fresh_state()
        edits = {"insert": [
            {"parent": "x1", "stage": "frontend", "delta": "a", "prior_gain": 10.0},
            {"parent": "x1", "stage": "frontend", "delta": "b", "prior_gain": 0.6},
            {"parent": "x1", "stage": "frontend", "delta": "c", "prior_gain": -3},
        ]}
        notes = harness.apply_inserts(st, st["nodes"]["open"], edits)
        self.assertEqual(len(st["nodes"]["open"]), 3)
        self.assertEqual(len(notes), 2, "the in-range insert needs no note")


class SelectionBoundTest(unittest.TestCase):
    """The point of the bound: the advantage an insert can buy is now finite."""

    def test_an_insert_cannot_outrank_the_catalogue_by_more_than_the_bound(self):
        st = _fresh_state()
        harness.apply_inserts(st, st["nodes"]["open"],
                              _insert(prior_gain=10.0, V=1.0))
        inserted = self._score(st["nodes"]["open"][0])
        best_catalogue = self._score({"stage": "frontend", "V": 1.0,
                                      "prior_gain": CATALOG_CEILING, "tried": 0})
        self.assertLessEqual(inserted, 1.0)
        # Before the bound this ratio was 10.0 / 0.9 = 11.1.
        self.assertLessEqual(inserted / best_catalogue, 1.0 / CATALOG_CEILING + 1e-9)

    def test_a_negative_prior_does_not_make_the_score_negative(self):
        st = _fresh_state()
        harness.apply_inserts(st, st["nodes"]["open"],
                              _insert(prior_gain=-5.0, V=1.0))
        self.assertGreaterEqual(self._score(st["nodes"]["open"][0]), 0.0)

    def _score(self, node):
        return harness.pypto_action_priority(node, "frontend", True)


class MigratePriorsTest(unittest.TestCase):
    """The state file is the world model, and a run re-hydrates from it.

    Bounding the value where it enters does nothing for a search resumed from a
    file an earlier build wrote, whose frontier can still hold a prior the
    selector would multiply by.
    """

    def test_load_bounds_a_prior_already_on_disk(self):
        st = harness.migrate_keys({
            "op": "demo",
            "nodes": {"open": [{"id": "u1", "stage": "frontend",
                                "prior_gain": 10.0, "V": 1.0, "tried": 0}],
                      "held": [{"id": "u2", "stage": "swimlane",
                                "prior_gain": -2.0, "V": 0.5, "tried": 0}]},
        })
        self.assertEqual(st["nodes"]["open"][0]["prior_gain"], 1.0)
        self.assertEqual(st["nodes"]["held"][0]["prior_gain"], 0.0)

    def test_migration_leaves_a_catalogue_prior_untouched(self):
        st = harness.migrate_keys({
            "op": "demo",
            "nodes": {"open": [{"id": "u1", "stage": "frontend",
                                "prior_gain": 0.9, "V": 1.0, "tried": 0}]},
        })
        self.assertEqual(st["nodes"]["open"][0]["prior_gain"], 0.9)

    def test_migration_survives_a_state_with_no_nodes(self):
        harness.migrate_keys({"op": "demo"})
        harness.migrate_keys({"op": "demo", "nodes": {}})


if __name__ == "__main__":
    unittest.main()
