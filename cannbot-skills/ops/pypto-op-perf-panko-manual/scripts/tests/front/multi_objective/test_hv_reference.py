# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""The hypervolume's reference point, and why it is fixed.

It was the worst value among the current points plus 10%, recomputed every
time. That moves: a point with a worse workspace pushes it outward and inflates
the volume of every point already on the front, so a sequence of real latency
improvements reads as gain under the threshold because the box grew underneath
it -- and an improvement can fail to appear in the hypervolume at all.

It is now fixed for the run: the preopt on the primary objective, the chip's
memory on a byte-valued objective, and a byte axis is measured as log(bytes)
so that one threshold can serve both axes.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import (CHIP_ARGS, CHIP_ENVELOPE_OFFLINE, DEMO_KERNEL as KERNEL, file_hash, harness,  # noqa: E402
                      new_op_dir, run_cli)

MIB = 1 << 20


def _state(objectives=("aicore_e2e", "workspace"), preopt=(100.0, 1000 * MIB),
           memory_mib=65536):
    st = {"config": {"objectives": list(objectives)},
          "nodes": {"closed": [{"id": "root", "code_hash": "p" * 64}]},
          "eval_cache": {"p" * 64: {"s": 1, "p": preopt[0], "m": {
              "mix_aic": preopt[0], "workspace_b": preopt[1],
              "workspace_state": "measured"}}}}
    if memory_mib is not None:
        st["chip_envelope"] = {"memory_mib": memory_mib}
    return st


class ReferenceTest(unittest.TestCase):
    def test_the_primary_axis_is_the_preopt(self):
        _, ref = self._ref()
        self.assertEqual(ref[0], 100.0)

    def test_a_byte_axis_is_the_chips_memory_in_log(self):
        _, ref = self._ref()
        self.assertAlmostEqual(ref[1], math.log(65536 * MIB))

    def test_it_does_not_depend_on_the_points(self):
        """The whole point. Nothing but the state's own constants is read."""
        st = _state()
        objs = harness.parse_objectives(st["config"])
        first = harness.hv_reference_point(st, objs)
        st["archive"] = [{"code_hash": "x" * 64,
                          "values": {"aicore_e2e": 50.0,
                                     "workspace": 40000.0 * MIB}}]
        self.assertEqual(harness.hv_reference_point(st, objs), first)

    def test_without_a_ceiling_there_is_no_reference(self):
        """
        A SKU whose ini carries no `memory_size` and no --chip-memory-mib.
        Inventing a bound would be the moving reference again.
        """
        _, ref = self._ref(memory_mib=None)
        self.assertIsNone(ref)

    def test_an_objective_may_carry_its_own(self):
        st = _state(objectives=({"name": "aicore_e2e", "reference": 200.0},
                                "workspace"))
        ref = harness.hv_reference_point(st, harness.parse_objectives(st["config"]))
        self.assertEqual(ref[0], 200.0)

    def test_a_bad_reference_is_refused_at_parse(self):
        for bad in ("abc", 0, -1):
            with self.assertRaises(ValueError):
                harness.parse_objectives(
                    {"objectives": [{"name": "aicore_e2e", "reference": bad}]})

    def _ref(self, **kw):
        st = _state(**kw)
        return st, harness.hv_reference_point(st, harness.parse_objectives(st["config"]))


class LogAxisTest(unittest.TestCase):
    """
    Why the byte axis is log: on a linear axis the share a 1% reduction
    occupies collapses as the reference grows, so one threshold cannot serve
    both axes.
    """

    def test_the_linear_share_collapses_and_the_log_share_does_not(self):
        lin = [self._share(r, False) for r in (4000, 65536, 1000000)]
        log = [self._share(r, True) for r in (4000, 65536, 1000000)]
        self.assertGreater(lin[0] / lin[-1], 100, "linear: 250x across the range")
        self.assertLess(log[0] / log[-1], 5, "log: under 4x")

    def test_a_byte_axis_is_recognised_by_its_source_or_unit(self):
        self.assertTrue(harness.is_bytes({"from": "m.workspace_b"}))
        self.assertTrue(harness.is_bytes({"from": "m.x", "unit": "B"}))
        self.assertFalse(harness.is_bytes({"from": "p"}))
        self.assertFalse(harness.is_bytes({"from": "m.mix_aic"}))

    def test_zero_bytes_is_not_log_of_zero(self):
        o = {"name": "workspace", "from": "m.workspace_b", "direction": "min"}
        self.assertEqual(harness.hv_axis(o, 0.0), 0.0)

    def _share(self, ref_mib, logw):
        o = {"name": "workspace", "from": "m.workspace_b", "direction": "min"}
        if not logw:
            o = dict(o, from_="")                      # force the linear branch
            lo, hi, r = 400.0 * MIB, 396.0 * MIB, ref_mib * MIB
            return (lo - hi) / (r - lo)
        f = harness.hv_axis
        lo, hi = f(o, 400.0 * MIB), f(o, 396.0 * MIB)
        return (lo - hi) / (f(o, ref_mib * MIB) - lo)


class DegradedTest(unittest.TestCase):
    """
    With no ceiling the test falls back to the axis records rather than
    switching off.
    """

    def test_the_gain_is_skipped_but_the_records_still_answer(self):
        st = _state(memory_mib=None)
        st["progress"] = {}
        before = [{"code_hash": "a" * 64,
                   "values": {"aicore_e2e": 90.0, "workspace": 1600.0 * MIB}}]
        after = [{"code_hash": "b" * 64,
                  "values": {"aicore_e2e": 80.0, "workspace": 1600.0 * MIB}}]
        self.assertIsNone(harness.hv_gain(st, before, after))
        d = harness.hv_observe(st, before, after)
        self.assertIsNone(d["hv"])
        self.assertEqual(d["axis_records_moved"], ["aicore_e2e"])
        self.assertTrue(d["counted_as_gain"])


class CliTest(unittest.TestCase):
    def setUp(self):
        self.f = new_op_dir(self, KERNEL)

    def test_the_chips_memory_is_the_default(self):
        res, rc = self._init()
        self.assertEqual(rc, 0, res)
        self.assertEqual(self._env()["memory_mib"], 65536)

    def test_init_says_what_the_box_is(self):
        """
        In the objective's own units, not the axis's: the byte axis is
        log(bytes), so printing the coordinate raw says the chip has 24.6
        bytes of memory.
        """
        res, rc = self._init()
        self.assertEqual(rc, 0, res)
        self.assertEqual(res["hv_reference"],
                         {"aicore_e2e": 100.0, "workspace": 65536 << 20})

    def test_the_budget_can_be_stated(self):
        """
        The default assumes the kernel runs alone, which it does not in
        deployment -- weights, KV cache and other operators share the device.
        """
        res, rc = self._init("--chip-memory-mib", "4096")
        self.assertEqual(rc, 0, res)
        env = self._env()
        self.assertEqual((env["memory_mib"], env["memory_mib_via"]),
                         (4096, "explicit:cli"))

    def test_a_nonsense_budget_is_refused(self):
        for bad in ("0", "-1", "abc"):
            res, rc = self._init("--chip-memory-mib", bad)
            self.assertEqual(rc, 3, bad)
            self.assertIn("--chip-memory-mib", res["reason"])

    def test_an_envelope_without_it_is_accepted_for_one_objective(self):
        """
        Nothing degrades: there is no volume to measure, and the ratchet
        ranks on the primary objective.
        """
        res, rc = self._no_mib("--config", json.dumps({"objectives": ["aicore_e2e"]}))
        self.assertEqual(rc, 0, res)
        self.assertNotIn("memory_mib", self._env())
        self.assertIsNone(res["hv_reference"])

    def test_but_a_multi_objective_run_is_refused_without_a_ceiling(self):
        """
        It used to start. `front_base` then had all-zero contributions, so
        the base was "least tried, then fastest" -- a single-objective search
        reporting a multi-objective front, with nothing said about it.
        """
        res, rc = self._no_mib()
        self.assertEqual(rc, 4, res)
        self.assertEqual(res["error"], "hv_reference_unavailable")
        self.assertEqual(res["byte_objectives"], ["workspace"])
        self.assertEqual([o["id"] for o in res["ask_user"]["options"]],
                         ["chip_memory", "axis_reference", "single_objective"])

    def test_the_cli_budget_answers_the_refusal(self):
        res, rc = self._no_mib("--chip-memory-mib", "4096")
        self.assertEqual(rc, 0, res)
        self.assertEqual(res["hv_reference"]["workspace"], 4096 << 20)

    def test_an_axis_reference_answers_it_too(self):
        """A bound this kernel must stay under, rather than the chip's."""
        res, rc = self._no_mib("--config", json.dumps({"objectives": [
            {"name": "aicore_e2e", "from": "p", "direction": "min"},
            {"name": "workspace", "from": "m.workspace_b", "direction": "min",
             "reference": 2 << 30}]}))
        self.assertEqual(rc, 0, res)
        self.assertEqual(res["hv_reference"]["workspace"], 2 << 30)

    def test_a_capacity_is_still_required(self):
        env = {k: v for k, v in CHIP_ENVELOPE_OFFLINE.items() if k != "ub_kb"}
        res, rc = run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo",
                          "--p-ref", "50", "--preopt-p", "100",
                          "--preopt-hash", file_hash(self.f),
                          "--op-file", str(self.f), "--allow-unversioned",
                          "--chip-envelope", json.dumps(env))
        self.assertEqual(rc, 3)
        self.assertIn("ub_kb", res["reason"])

    def _init(self, *extra):
        return run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo",
                       "--p-ref", "50", "--preopt-p", "100",
                       "--preopt-hash", file_hash(self.f),
                       "--preopt-workspace-b", str(1000 << 20),
                       "--op-file", str(self.f), "--allow-unversioned",
                       *CHIP_ARGS, *extra)

    def _env(self):
        st = json.loads((self.op_dir / "optimization" / "search_state.json").read_text())
        return st["chip_envelope"]

    def _no_mib(self, *extra):
        """An envelope that names every capacity and no budget: it is a budget,
        not a capacity, so the ini does not always carry it and a caller naming
        the buffers has no reason to name it."""
        env = {k: v for k, v in CHIP_ENVELOPE_OFFLINE.items() if k != "memory_mib"}
        return run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo",
                       "--p-ref", "50", "--preopt-p", "100",
                       "--preopt-hash", file_hash(self.f),
                       "--preopt-workspace-b", str(1000 << 20),
                       "--op-file", str(self.f), "--allow-unversioned",
                       "--chip-envelope", json.dumps(env), *extra)


if __name__ == "__main__":
    unittest.main()
