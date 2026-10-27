# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""A candidate the readings cannot order is not evidence about the action.

`stagnation_K` counted every non-accepted candidate alike: candidates clearly
slower and candidates inside the measurement's own spread closed an action on
the same count. A large share of all measured candidates are the second kind,
on the quietest chip as much as on a noisy one, because the search proposes
near-misses whatever the noise is -- so an action could be closed on almost
nothing.

So the rejection is re-asked with the readings, through the archive's own
interval test. `clear` advances `n`; `undetermined` advances `n_undetermined`,
which has its own bound so an action still terminates on a kernel whose noise
swallows every candidate.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import CHIP_ARGS, DEMO_KERNEL as KERNEL, file_hash, harness, new_op_dir, read_state, run_cli  # noqa: E402


def _tight(p):
    return [p + i * 0.05 for i in range(9)]


def _wide(p):
    return [p * (1 + 0.06 * ((i % 3) - 1)) for i in range(9)]


class ClarityTest(unittest.TestCase):
    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", self.pre,
                "--preopt-workspace-b", str(1000 << 20),
                "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS,
                "--preopt-samples", ",".join(str(x) for x in _tight(100.0)))

    def test_a_clear_failure_advances_n(self):
        self._front()
        _, res = self._record(8, 130.0, 1600, _tight(130.0))
        self.assertEqual(res["acceptance"]["clarity"], "clear")
        self.assertEqual((res["n"], res["n_undetermined"]), (1, 0))

    def test_the_tied_workspace_shape_is_undetermined(self):
        """
        Identical workspace to the byte, latency inside the readings' own
        spread -- a recorded front hit exactly this.
        """
        self._front()
        _, res = self._record(16, 92.0, 1600, _wide(92.0))
        self.assertEqual(res["acceptance"]["clarity"], "undetermined")
        self.assertEqual((res["n"], res["n_undetermined"]), (0, 1))

    def test_a_worse_deterministic_axis_is_clear_however_noisy_the_other_is(self):
        """
        The workspace probe returns one value, not a sample. A candidate
        that is 100 MiB bigger IS off the front, whatever its latency does.
        """
        self._front()
        _, res = self._record(16, 92.0, 1700, _wide(92.0))
        self.assertEqual(res["acceptance"]["clarity"], "clear")

    def test_slower_than_the_preopt_splits_on_the_readings_too(self):
        clear = self._record(8, 140.0, 300, _tight(140.0))[1]
        self.assertEqual(clear["acceptance"]["clarity"], "clear")
        self.assertFalse(clear["acceptance"]["within_reference"])
        noisy = self._record(40, 101.0, 290, _wide(101.0))[1]
        self.assertEqual(noisy["acceptance"]["clarity"], "undetermined")
        self.assertFalse(noisy["acceptance"]["within_reference"])

    def test_without_readings_nothing_changes(self):
        """A run that passes no --samples counts exactly as it did before."""
        self._record(128, 90.0, 1600)
        _, res = self._record(16, 92.0, 1600)
        self.assertEqual(res["acceptance"]["clarity"], "clear")
        self.assertEqual(res["n"], 1)

    def test_an_acceptance_clears_the_undetermined_count(self):
        self._front()
        self._record(16, 92.0, 1600, _wide(92.0))
        self.assertEqual(read_state(self.op_dir)["progress"]["refine"]["n_undetermined"], 1)
        _, res = self._record(48, 85.0, 1600, _tight(85.0))
        self.assertTrue(res["improved"])
        self.assertEqual(res["n_undetermined"], 0)

    def _record(self, tile, p, mib, samples=None):
        self.f.write_text(KERNEL.replace("64, 128", f"{tile}, 128"), encoding="utf-8")
        h = file_hash(self.f)
        args = ["record", "--op-dir", str(self.op_dir), "--parent", "u1",
                "--code-hash", h, "--s", "1", "--p", str(p),
                "--workspace-b", str(mib << 20)]
        if samples:
            args += ["--samples", ",".join(str(x) for x in samples)]
        res, _ = run_cli(*args)
        return h, res

    def _front(self):
        return self._record(128, 90.0, 1600, _tight(90.0))[1]


class BoundTest(ClarityTest):
    def test_its_own_bound_closes_the_action_and_close_honours_it(self):
        st = read_state(self.op_dir)
        st["config"]["undetermined_K"] = 3
        (self.op_dir / "optimization" / "search_state.json").write_text(json.dumps(st))
        self._front()
        for i, tile in enumerate((16, 24, 32)):
            _, res = self._record(tile, 92.0 - i * 0.1, 1600, _wide(92.0 - i * 0.1))
            self.assertEqual(res["acceptance"]["clarity"], "undetermined", res)
        self.assertEqual((res["n"], res["n_undetermined"]), (0, 3))
        self.assertTrue(res["stop_refine"])
        self.assertEqual(res["stop_cause"], "undetermined_K")
        # The harness told the optimizer to close; it must not then refuse it.
        closed, rc = run_cli("close", "--op-dir", str(self.op_dir), "--action", "u1")
        self.assertEqual(rc, 0, closed)

    def test_the_default_bound_clears_the_worst_recorded_action(self):
        self.assertGreaterEqual(harness.DEFAULT_CONFIG["undetermined_K"], 7,
                                "sparse's F-1 had 6 undetermined candidates in 7")

    def test_the_refusal_names_both_counters(self):
        self._front()
        self._record(16, 92.0, 1600, _wide(92.0))
        res, rc = run_cli("close", "--op-dir", str(self.op_dir), "--action", "u1")
        self.assertEqual(rc, 3)
        self.assertIn("n_undetermined=1 of 7", res["reason"])


if __name__ == "__main__":
    unittest.main()
