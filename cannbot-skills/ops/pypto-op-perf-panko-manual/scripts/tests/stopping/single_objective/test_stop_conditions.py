# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""What halts a run, under the SHIPPED defaults (C2).

The phase switch is gone, and with it the only stagnation condition that fired
in practice. `test_front_phase_trigger.py` made the point that a feature
reachable only through a test config is dead on the device; the same argument
now applies to the halt, so this file asserts what the defaults actually do
rather than what a test config can make them do.

Under the shipped defaults: `global_stagnation_limit` is None (it ended runs
well short of their budget), `stage_convergence` is False, and
`frontier_exhausted` waits on the optimizer. So a run ends on `success`, the
budget, the wall clock, or the optimizer declaring the frontier exhausted --
and nothing else. The stagnation counters stay as diagnostics for C3.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import CHIP_ARGS, DEMO_KERNEL as KERNEL, harness, new_op_dir, run_cli  # noqa: E402


class DefaultsTest(unittest.TestCase):
    def test_no_stagnation_halt_is_configured(self):
        cfg = harness.DEFAULT_CONFIG
        self.assertIsNone(cfg["global_stagnation_limit"])
        self.assertFalse(cfg["stage_convergence"])

    def test_the_phase_machinery_is_gone(self):
        for k in ("mo_phases", "greedy_stagnation_limit", "front_stagnation_limit"):
            self.assertNotIn(k, harness.DEFAULT_CONFIG)
        for name in ("_phase", "_enter_front_phase", "_greedy_or_front",
                     "_front_budget_share", "GREEDY_EXHAUSTED",
                     "FRONT_PHASE_MIN_SHARE"):
            self.assertFalse(hasattr(harness, name), name)


class StopTest(unittest.TestCase):
    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", self.pre,
                "--preopt-workspace-b", str(1000 << 20),
                "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS)

    def test_a_long_run_of_unproductive_closes_does_not_halt(self):
        self._patch(progress={"global_stagnation": 99})
        self.assertFalse(self._stop()["stop"])

    def test_a_configured_limit_still_halts(self):
        """
        Gone from the defaults, not from the code: a campaign that wants the
        halt can still ask for it.
        """
        self._patch(progress={"global_stagnation": 5},
                    config={"global_stagnation_limit": 5})
        res = self._stop()
        self.assertEqual((res["stop"], res["reason"]), (True, "global_stagnation"))

    def test_the_budget_halts(self):
        self._patch(progress={"evals_used": 7}, config={"eval_budget": 7})
        self.assertEqual(self._stop()["reason"], "eval_budget")

    def test_the_wall_clock_halts(self):
        self._patch(config={"wall_clock_limit_s": 0})
        self.assertEqual(self._stop()["reason"], "wall_clock")

    def test_convergence_is_recorded_but_does_not_halt(self):
        self._patch(progress={"converged": True})
        self.assertEqual(self._stop()["reason"], "converged",
                         "`converged` itself is still a halt -- what is off by "
                         "default is the machinery that SETS it")
        self._patch(progress={"converged": False})
        self.assertFalse(self._stop()["stop"])

    def _path(self):
        return self.op_dir / "optimization" / "search_state.json"

    def _state(self):
        return json.loads(self._path().read_text())

    def _patch(self, progress=None, config=None):
        st = self._state()
        st["progress"].update(progress or {})
        st["config"].update(config or {})
        self._path().write_text(json.dumps(st), encoding="utf-8")

    def _stop(self):
        return run_cli("stop", "--op-dir", str(self.op_dir))[0]


if __name__ == "__main__":
    unittest.main()
