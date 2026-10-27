# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The 3-objective path: aicore_e2e alongside mix_aic and workspace."""
import os
import sys
import unittest
from typing import Any, NamedTuple

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from _support import harness as H, harness_source  # noqa: E402


class Args:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        """Anything `init` reads that this case did not set reads as unset."""
        return self.__dict__.get(name)


KERNEL_SRC = """import pypto


@pypto.frontend.jit()
def demo(a, b):
    pypto.set_vec_tile_shapes(64, 128)
    return a + b
"""

OBJS3 = H.parse_objectives({"objectives": ["aicore_e2e", "mix_aic", "workspace"]})


class TestAlias(unittest.TestCase):
    def test_aicore_e2e_is_p(self):
        """
        `p` IS the profiler's AICore End-to-End Time. It is the run's
        primary latency, so it needs no indirection through `m`.
        """
        o = H.parse_objectives({"objectives": ["aicore_e2e"]})[0]
        self.assertEqual(o["from"], "p")
        self.assertEqual(o["samples_from"], "samples")
        self.assertEqual(o["direction"], "min")

    def test_mix_aic_is_a_second_reading_on_m(self):
        o = H.parse_objectives({"objectives": ["mix_aic"]})[0]
        self.assertEqual(o["from"], "m.mix_aic")
        self.assertEqual(o["samples_from"], "m.mix_aic_samples")

    def test_three_objectives_are_distinct_sources(self):
        self.assertEqual([o["from"] for o in OBJS3],
                         ["p", "m.mix_aic", "m.workspace_b"])

    def test_workspace_declares_no_samples(self):
        ws = [o for o in OBJS3 if o["name"] == "workspace"][0]
        self.assertNotIn("samples_from", ws)

    def test_e2e_and_mix_aic_are_not_one_objective(self):
        """Different `from`, so the duplicate guard must NOT fire."""
        H.parse_objectives({"objectives": ["aicore_e2e", "mix_aic"]})

    def test_user_may_declare_a_sample_source(self):
        o = H.parse_objectives({"objectives": [
            {"name": "dm", "from": "m.dm_bytes", "samples_from": "m.dm_samples"}]})[0]
        self.assertEqual(o["samples_from"], "m.dm_samples")

    def test_bad_sample_source_is_refused(self):
        with self.assertRaises(ValueError):
            H.parse_objectives({"objectives": [
                {"name": "dm", "from": "m.dm_bytes", "samples_from": "dm_samples"}]})


class TestValues(unittest.TestCase):
    def test_all_three_read(self):
        vals, missing = H.objective_values(
            OBJS3, 84.72,
            {"mix_aic": 176.314, "workspace_b": 308319744,
             "workspace_state": "measured"})
        self.assertEqual(missing, [])
        self.assertEqual(vals["aicore_e2e"], 84.72)
        self.assertEqual(vals["mix_aic"], 176.314)

    def test_a_missing_objective_is_reported_not_skipped(self):
        vals, missing = H.objective_values(
            OBJS3, 84.72,
            {"util": 40.12, "workspace_b": 405798912,
             "workspace_state": "measured"})
        self.assertEqual([m["objective"] for m in missing], ["mix_aic"])
        self.assertNotIn("mix_aic", vals)


class TestPerObjectiveSamples(unittest.TestCase):
    def test_collected_from_their_declared_sources(self):
        got = H.objective_samples(
            OBJS3, {"samples": [84.0, 85.0, 84.5]},
            {"mix_aic_samples": [176.0, 177.0, 175.0]})
        self.assertEqual(sorted(got), ["aicore_e2e", "mix_aic"])
        self.assertEqual(got["aicore_e2e"], [84.0, 85.0, 84.5])
        self.assertEqual(got["mix_aic"], [176.0, 177.0, 175.0])

    def test_workspace_never_collects(self):
        got = H.objective_samples(OBJS3, {"samples": [1.0, 2.0]},
                                  {"workspace_b": [1, 2]})
        self.assertNotIn("workspace", got)

    def test_legacy_row_still_feeds_the_p_objective(self):
        o = H.parse_objectives({"objectives": ["aicore_e2e"]})[0]
        self.assertEqual(H.row_samples(o, {"samples": [1.0, 2.0]}), [1.0, 2.0])

    def test_row_prefers_its_own_per_objective_list(self):
        o = H.parse_objectives({"objectives": ["aicore_e2e"]})[0]
        row = {"samples": [1.0, 2.0],
               "samples_by_objective": {"aicore_e2e": [9.0, 9.5]}}
        self.assertEqual(H.row_samples(o, row), [9.0, 9.5])

    def test_mix_aic_never_borrows_p_samples(self):
        """The defect this replaces: one list served every objective."""
        o = H.parse_objectives({"objectives": ["mix_aic"]})[0]
        self.assertEqual(H.row_samples(o, {"samples": [84.0, 85.0]}), [])


def _row(e2e, mix, ws, s_e2e=None, s_mix=None):
    samples = {}
    for name, vals in (("aicore_e2e", s_e2e), ("mix_aic", s_mix)):
        if vals:
            samples[name] = vals
    return {"code_hash": "h", "values": {"aicore_e2e": e2e, "mix_aic": mix,
                                         "workspace": ws},
            "samples_by_objective": samples}


class TestDominance(unittest.TestCase):
    def test_better_on_all_three_dominates(self):
        self.assertTrue(H.dominates_measured(
            OBJS3, _row(80.0, 170.0, 100), _row(90.0, 180.0, 200)))

    def test_e2e_better_but_mix_aic_worse_does_not_dominate(self):
        """The case the user observed on device, and why mix_aic stays."""
        self.assertFalse(H.dominates_measured(
            OBJS3, _row(80.0, 190.0, 100), _row(90.0, 180.0, 100)))

    def test_e2e_is_judged_on_its_own_readings(self):
        """Overlapping e2e intervals are a tie, so workspace decides."""
        a = _row(84.0, 176.0, 100, s_e2e=[84.0, 84.5, 83.5, 84.2, 84.1])
        b = _row(84.1, 176.0, 200, s_e2e=[84.1, 84.4, 83.6, 84.3, 84.0])
        self.assertTrue(H.dominates_measured(OBJS3, a, b, alpha=0.05))

    def test_a_clearly_slower_e2e_blocks_dominance(self):
        a = _row(120.0, 176.0, 100, s_e2e=[120.0, 121.0, 119.0, 120.5, 120.2])
        b = _row(84.0, 176.0, 200, s_e2e=[84.0, 84.5, 83.5, 84.2, 84.1])
        self.assertFalse(H.dominates_measured(OBJS3, a, b, alpha=0.05))


class TestMeasurementFlag(unittest.TestCase):
    def test_carries_every_reading_without_naming_them(self):
        m = H.cli_measurement(Args(measurement=(
            '{"p": 84.72, "samples": [84.0, 85.0], "mix_aic": 176.314,'
            ' "mix_aic_samples": [176.0, 177.0], "aiv_time": 179.11,'
            ' "cube_utilization": 96.104, "something_new_2027": 1.5}')))
        self.assertEqual(m["mix_aic"], 176.314)
        self.assertEqual(m["mix_aic_samples"], [176.0, 177.0])
        self.assertEqual(m["something_new_2027"], 1.5)

    def test_drops_what_has_its_own_flag(self):
        m = H.cli_measurement(Args(
            measurement='{"p": 176.3, "samples": [1.0], "util": 42.0}'))
        self.assertNotIn("p", m)
        self.assertNotIn("samples", m)
        self.assertEqual(m["util"], 42.0)

    def test_drops_structures_and_nulls(self):
        m = H.cli_measurement(Args(
            measurement='{"pipes": {"AIC": {"util": 1}}, "aic_util": null,'
                        ' "workspace_b": 123}'))
        self.assertEqual(m, {"workspace_b": 123})

    def test_absent_or_malformed_is_empty(self):
        self.assertEqual(H.cli_measurement(Args()), {})
        self.assertEqual(H.cli_measurement(Args(measurement="not json")), {})
        self.assertEqual(H.cli_measurement(Args(measurement="[1,2]")), {})


if __name__ == "__main__":
    unittest.main()


class TestBlockInstrument(unittest.TestCase):
    """The tile block and the ratchet climb the SAME reading."""

    def test_the_package_and_the_harness_name_the_same_objective(self):
        from bayesian_optimization import evaluator as ev_mod
        self.assertEqual(ev_mod.J_OBJECTIVE, H.J_OBJECTIVE)

    def test_a_block_trial_keeps_p_on_what_it_ranked(self):
        """
        `p` is the ranked reading -- the same column every other `record`
        writes. MIX_AIC rides on `m` under its own name.
        """
        import pathlib
        import tempfile
        import types
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        op_dir = pathlib.Path(tmp.name)
        (op_dir / "optimization").mkdir()
        st = {"op": "demo", "eval_cache": {}, "archive": [],
              "config": dict(H.DEFAULT_CONFIG,
                             objectives=["aicore_e2e", "mix_aic", "workspace"]),
              "target": {"P_ref": 50}, "progress": {"evals_used": 1},
              "nodes": {"open": [], "closed": []}}
        a = types.SimpleNamespace(parent="u1", op_dir=str(op_dir))
        ctx = {"faults": [], "trials": [], "meter": {"device_trials": 0},
               "winners": {}, "base_src": KERNEL_SRC}
        H.block_recorder(a, st, ctx)(
            {"status": "ok", "objective_us": 45.0,
             "config": {"vec#0": {"d0": 32, "d1": 256}},
             "metrics": {"mix_aic": 90.0, "workspace_b": 1 << 20,
                         "workspace_state": "measured"},
             "samples": [44.0, 45.0, 46.0]})
        entry = next(iter(st["eval_cache"].values()))
        self.assertEqual(entry["p"], 45.0, "p is what the block ranked on")
        self.assertEqual(entry["m"]["mix_aic"], 90.0)
        self.assertEqual(st["archive"][0]["values"]["aicore_e2e"], 45.0)
        self.assertEqual(st["archive"][0]["values"]["mix_aic"], 90.0)


class TestDefaultObjectives(unittest.TestCase):
    """The DEFAULT is ONE; `mix_aic` stays available and stays recorded.

    `test_default_mode.py` is where the default itself is pinned. Here it only
    has to be the thing the other objectives are reached FROM: naming a second
    axis is what adds it, and nothing is inherited.
    """

    def test_the_default_is_aicore_e2e_alone(self):
        objs = H.parse_objectives({})
        self.assertEqual([o["name"] for o in objs], ["aicore_e2e"])
        self.assertEqual([o["from"] for o in objs], ["p"])

    def test_workspace_is_reached_by_naming_it(self):
        objs = H.parse_objectives({"objectives": ["aicore_e2e", "workspace"]})
        self.assertEqual([o["name"] for o in objs], ["aicore_e2e", "workspace"])
        self.assertEqual([o["from"] for o in objs], ["p", "m.workspace_b"])

    def test_mix_aic_can_still_be_declared(self):
        objs = H.parse_objectives({"objectives": ["aicore_e2e", "mix_aic", "workspace"]})
        self.assertEqual([o["name"] for o in objs],
                         ["aicore_e2e", "mix_aic", "workspace"])

    def test_mix_aic_readings_are_carried_whether_or_not_it_is_an_objective(self):
        """
        Dropping it from the default must not stop it being MEASURED: the
        comparison that showed what it costs was only possible because a run
        that did not optimise it still recorded it.
        """
        from bayesian_optimization import evaluator as ev_mod
        for k in ("mix_aic", "mix_aic_samples", "mix_aic_sigma_log"):
            self.assertIn(k, ev_mod.CARRIED_METRICS)
        m = H.cli_measurement(Args(measurement=(
            '{"p": 84.7, "samples": [84.0, 85.0], "mix_aic": 176.3,'
            ' "mix_aic_samples": [176.0, 177.0]}')))
        self.assertEqual(m["mix_aic"], 176.3)
        self.assertEqual(m["mix_aic_samples"], [176.0, 177.0])


class TestOneClock(unittest.TestCase):
    """Every place that compares a latency uses `p`, never `m.mix_aic`."""

    def test_the_block_bar_is_on_the_clock_the_block_ranks_with(self):
        """
        A tile study ranks its trials on `p`, so the incumbent it measures them
        against must be `p` too -- and it must be the measurement of the
        program ON DISK. Both bars read `best_mix_aic_us` while `p` was
        MIX_AIC; then the global best on `p`, which under more than one
        objective is not the front base `select` restores.

        Neither bar may reach into `progress` for it: that is `best_node`'s
        number whatever it is called.
        """
        src = harness_source()
        self.assertNotIn("current_latency=st[", src)
        self.assertNotIn("current_latency=(st[", src)
        self.assertEqual(src.count('current_latency=incumbent_latency(st, ctx["base_src"])'), 1)
        self.assertEqual(src.count("current_latency=incumbent_latency(st, src)"), 1)


class _GrantKnobs(NamedTuple):
    """The optional budget knobs one grant is computed under."""

    budget: Any = None
    share: float = 0.5
    cap: Any = 12
    max_trials: Any = None


class TestBlockBudget(unittest.TestCase):
    """The tile guard must not vanish when a run declines to name a budget."""

    def test_the_share_binds_with_no_run_budget(self):
        """
        `budget * share` is undefined without a total, so `eval_budget: null`
        used to remove the guard. A ratio against `evals_used` does not.
        """
        self.assertEqual(self._grant(used=24, spent=12), 0)
        self.assertEqual(self._grant(used=52, spent=26), 0)

    def test_the_first_block_is_still_usable(self):
        """
        Early in a run the ratio alone grants almost nothing; the per-block
        cap is the floor as well as the ceiling.
        """
        self.assertEqual(self._grant(used=3, spent=0), 12)
        self.assertEqual(self._grant(used=8, spent=0), 12)

    def test_the_cap_bounds_one_block_however_much_is_left(self):
        self.assertEqual(self._grant(used=400, spent=0, knobs=_GrantKnobs(budget=1000)), 12)

    def test_the_floor_does_not_top_up_a_block_that_overspent(self):
        """
        `max(remaining, cap)` would have handed more trials to a run whose
        blocks had already taken nearly its whole budget.
        """
        self.assertEqual(self._grant(used=71, spent=60, knobs=_GrantKnobs(budget=120)), 0)

    def test_a_spent_run_budget_still_wins(self):
        self.assertEqual(self._grant(used=120, spent=0, knobs=_GrantKnobs(budget=120)), 0)

    def test_max_trials_still_bounds(self):
        self.assertEqual(self._grant(used=100, spent=0, knobs=_GrantKnobs(max_trials=3)), 3)

    def test_the_cap_can_be_turned_off(self):
        self.assertEqual(self._grant(used=3, spent=0, knobs=_GrantKnobs(cap=None)), 1)

    def _grant(self, used, spent, knobs=_GrantKnobs()):
        import types
        return H.block_budget(
            {"evals_used": used, "block_evals": spent},
            {"eval_budget": knobs.budget, "block_budget_share": knobs.share,
             "block_max_trials": knobs.cap},
            types.SimpleNamespace(max_trials=knobs.max_trials))
