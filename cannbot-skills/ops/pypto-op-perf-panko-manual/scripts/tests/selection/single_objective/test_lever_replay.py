# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""The study replay, and what a failed proposal may cost.

A recorded campaign died here. The kernel's G_TILE is 12, which is not on
the vector ladder; the seed trial therefore held `vec#0.d0 = 12`, and once the
core moved the anchor to 24 the replayed distribution no longer contained 12.
The ValueError came out of `record`, before `save`: the run's evaluations went
unpersisted and the working file was left holding a candidate nobody recorded.

Two things were wrong. The replay must admit the values its own trials hold,
and a proposal that cannot be made must never cost the measurement that asked
for it.

Fixing the replay alone was not enough, and the next campaign said so: the
rebuild succeeded and the very next DRAW raised the same error, because the
ask built its distribution from the anchor alone while the study held trials
replayed against a wider one. Guarded, that is no longer a crash -- it is the
lever silently dead for the rest of the action, which is worse to diagnose.
Every distribution this study is handed has to be built from the same `held`.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import harness  # noqa: E402
from panko import selection  # noqa: E402

from bayesian_optimization import apply, lever, space  # noqa: E402

SITES = [apply.Site(id="vec#0", kind="vec", ordinal=0,
                    func_text="set_vec_tile_shapes", start=0, end=0)]
ANCHOR_12 = {"vec#0": {"d0": 12, "d1": 128}}
ANCHOR_24 = {"vec#0": {"d0": 24, "d1": 128}}
RECORD = {"trials": [{"params": {"vec#0.d0": 12, "vec#0.d1": 128}, "value": 856.5},
                     {"params": {"vec#0.d0": 24, "vec#0.d1": 128}, "value": 1007.8}]}


class ReplayTest(unittest.TestCase):

    def test_an_off_ladder_value_is_not_on_the_ladder(self):
        self.assertNotIn(12, space.VEC_CHOICES,
                         "the premise: G_TILE=12 is off the sampling grid")

    def test_the_replay_keeps_a_trial_the_anchor_has_moved_away_from(self):
        record = {"trials": list(RECORD["trials"])}
        study = lever.rebuild_study(record, SITES, None, ANCHOR_24)
        self.assertEqual(len(study.trials), 2, "both observations survive")
        self.assertEqual(record.get("replay_dropped"), None, "and none was dropped")
        self.assertIn(12, [t.params["vec#0.d0"] for t in study.trials])

    def test_it_still_works_when_the_anchor_holds_the_value(self):
        study = lever.rebuild_study({"trials": list(RECORD["trials"])}, SITES, None, ANCHOR_12)
        self.assertEqual(len(study.trials), 2)

    def test_a_trial_the_study_refuses_is_dropped_and_counted(self):
        record = {"trials": [{"params": {"vec#0.d0": 12, "vec#0.d1": 128}, "value": 1.0},
                             {"params": {"no_site_id": 4}, "value": 2.0}]}
        study = lever.rebuild_study(record, SITES, None, ANCHOR_24)
        self.assertEqual(len(study.trials), 1)
        self.assertIn("replay_dropped", record)
        self.assertEqual(record.get("replay_dropped"), 1)


class DrawTest(unittest.TestCase):
    """The ask has to agree with the replay about what the parameter IS."""

    def test_a_draw_after_the_replay_does_not_raise(self):
        record = {"trials": list(RECORD["trials"])}
        study = lever.rebuild_study(record, SITES, None, ANCHOR_24)
        ctx = {"rec": record, "sites": SITES, "anchor": ANCHOR_24}
        trial, cfg, sid = lever.draw_one_site(study, ctx)
        self.assertEqual(sid, "vec#0")
        self.assertIn("vec#0.d0", trial.params)
        self.assertIn("d0", cfg["vec#0"])

    def test_the_ask_offers_the_off_ladder_value_the_study_holds(self):
        record = {"trials": list(RECORD["trials"])}
        study = lever.rebuild_study(record, SITES, None, ANCHOR_24)
        ctx = {"rec": record, "sites": SITES, "anchor": ANCHOR_24}
        trial, _, _ = lever.draw_one_site(study, ctx)
        self.assertIn(12, trial.distributions["vec#0.d0"].choices,
                      "the study holds a trial at 12; the draw must know that")

    def test_held_values_reads_complete_trials_only(self):
        record = {"trials": [{"params": {"vec#0.d0": 12}, "value": 1.0},
                             {"params": {"vec#0.d0": 999}, "value": None}]}
        self.assertEqual(lever.held_values(record), {"vec#0.d0": {12}})


class ProposalBlastRadiusTest(unittest.TestCase):
    """A proposal is not a measurement."""

    def test_a_failing_lever_is_reported_not_raised(self):
        st = {"nodes": {"open": [{"id": "u1", "catalog_id": "F-10"}]},
              "progress": {}}
        # The seam is the proposer as `bayes_next` looks it up, which is the
        # module that defines them both.
        boom = selection.tile_bayes_propose

        def explode(*_a, **_k):
            raise ValueError("'12' not in (1, 2, 4, 8, 16, 24)")
        selection.tile_bayes_propose = explode
        try:
            got = harness.bayes_next(st, "/nowhere", "u1")
        finally:
            selection.tile_bayes_propose = boom
        self.assertFalse(got["proposed"])
        self.assertIn("'12' not in", got["error"])
        self.assertEqual(st["progress"]["lever_errors"], 1)
        self.assertIn("ValueError", st["progress"]["lever_last_error"])

    def test_an_action_that_is_not_a_lever_proposes_nothing(self):
        st = {"nodes": {"open": []}, "progress": {}}
        self.assertIsNone(harness.bayes_next(st, "/nowhere", "u9"))


if __name__ == "__main__":
    unittest.main()
