# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The learned ceiling outlives the action that paid for it.

`memory_key` carries the action id because its payload is latencies, and a
latency is comparable only within one program shape. The ceiling rode along in
the same blob, and it is not a latency: it says which tiles the compiler will
accept, which the derived tile domain decides.

A recorded campaign is the bill. Two of its blocks stored the same domain hash
under two action ids, so the second drew its predecessor's candidates back one
for one and was charged for all of them -- capacity refusals the first block
had already bought included -- and the kernel's tile blocks aborted with
`n_feasible: 0`.

The contrast sits in the same campaign set: where consecutive blocks share an
action id the ceiling carries, and the later blocks refuse nothing at all.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import harness_source, SCRIPTS_DIR  # noqa: E402,F401

from _support import harness as H  # noqa: E402
from bayesian_optimization import block, driver, space  # noqa: E402

# Two domains that differ only in a view extent, which is an input of the
# derivation and so a real change of the tile space.
DOM_A = {"views": [[320, 1]], "capacity_elems": 9830, "dtype_bytes": 2,
         "ub_bytes": 196608, "vec_residency": {"vec#0": 3},
         "vec_dtype_bytes": {"vec#0": 4}}
DOM_B = dict(DOM_A, views=[[640, 1]])


class KeyTest(unittest.TestCase):

    def test_the_ceiling_key_ignores_the_action(self):
        """The case at issue: same domain, two actions, one fact."""
        self.assertEqual(block.ceiling_key(DOM_A), block.ceiling_key(DOM_A))
        self.assertNotEqual(block.memory_key("u1", DOM_A),
                            block.memory_key("u33", DOM_A))

    def test_the_memory_key_still_carries_the_action(self):
        """
        Latencies are not comparable across program shapes, so this one must
        keep separating them.
        """
        self.assertTrue(block.memory_key("u1", DOM_A).startswith("u1|"))
        self.assertIn(block.ceiling_key(DOM_A), block.memory_key("u1", DOM_A))

    def test_a_real_domain_change_still_separates_the_ceiling(self):
        """
        A view extent is an input of the derivation. The residency and the
        extents are what a capacity refusal depends on, so when they move the
        refusal does not carry -- which is the scope `_carried_refusals` asked
        for and could not state.
        """
        self.assertNotEqual(block.ceiling_key(DOM_A), block.ceiling_key(DOM_B))


class DriverSeedTest(unittest.TestCase):
    """Where `initial_state` takes the ceiling from."""

    def test_the_domain_ceiling_is_used(self):
        self.assertEqual(self._state({}, 221184)["ceiling"], 221184)

    def test_the_memory_ceiling_still_resumes_an_older_state_file(self):
        """State written before the split keeps its ceiling inside `memory`."""
        self.assertEqual(self._state({"ceiling": 221184}, None)["ceiling"], 221184)

    def test_the_domain_ceiling_wins_over_the_memory_s(self):
        self.assertEqual(self._state({"ceiling": 3538944}, 221184)["ceiling"],
                         221184)

    def test_no_ceiling_anywhere_is_unbounded(self):
        self.assertEqual(self._state({}, None)["ceiling"], float("inf"))

    def _state(self, memory, ceiling):
        opts = driver.SearchOptions(ceiling=ceiling, dtype_bytes=4)
        return driver.initial_state(memory or {}, None, opts)


class FoldTest(unittest.TestCase):
    """What `block_fold` stores."""

    def test_a_first_ceiling_is_stored(self):
        self.assertEqual(self._fold(None, 221184), 221184)

    def test_a_lower_ceiling_replaces_it(self):
        self.assertEqual(self._fold(3538944, 221184), 221184)

    def test_a_higher_ceiling_does_not(self):
        """
        4718592 is what the second block learned on its own. Letting it
        overwrite 3538944 would be the block unlearning what it paid for.
        """
        self.assertEqual(self._fold(3538944, 4718592), 3538944)

    def test_a_block_that_refused_nothing_retracts_nothing(self):
        """`ceiling: None` is the absence of evidence, not a retraction."""
        self.assertEqual(self._fold(221184, None), 221184)

    def _fold(self, held, learned):
        st = {"progress": {"evals_used": 0}, "eval_cache": {}}
        ceilings = {"d": held} if held is not None else {}
        ctx = {"base_src": "x", "memories": {}, "mem_key": "k",
               "ceilings": ceilings, "ceiling_key": "d",
               "refusal_rows": [], "structure": "s", "domain": {},
               "winner_hash": None, "remaining": 4, "charged": 0,
               "meter": {"device_trials": 0}, "trials": [], "faults": [],
               "started_at": "2026-10-06T00:00:00+00:00"}

        class FoldAction:
            parent, reason, live_tiles = "u33", "", 4

        H.block_fold(FoldAction(), st, ctx, {"ceiling": learned}, {})
        return ctx["ceilings"].get("d")


class WiringTest(unittest.TestCase):
    """The context carries both stores, and the block is handed the ceiling."""

    def test_the_block_options_pass_the_ceiling_through(self):
        self.assertIn("ceiling", block.BlockOptions.__dataclass_fields__)
        self.assertIn("ceiling", driver.SearchOptions.__dataclass_fields__)

    def test_the_harness_reads_the_ceiling_store_by_its_own_key(self):
        src = harness_source()
        self.assertIn('ceiling=ctx["ceilings"].get(ctx["ceiling_key"])', src)
        self.assertIn('"bayesian_optimization_ceilings"', src)


class CarriedCeilingSafetyTest(unittest.TestCase):
    """A carried ceiling may not retire the value the kernel is running.

    `_lower_ceiling` checks the floor when it LEARNS a ceiling, and while a
    ceiling died with the action that bought it that was the whole story. It no
    longer does: `ceiling_key` carries it across every block on the same tile
    domain, and the incumbent moves underneath it, so the guard is applied
    again where the ceiling is USED.
    """

    def test_a_draw_above_both_is_rejected(self):
        ctx = self._ctx(ceiling=200, floor=100)
        v = driver.static_reason(ctx, {"vec#0": {"d0": 1000, "d1": 1}})
        self.assertIsNotNone(v)
        self.assertEqual(v[0], "over_learned_ceiling")

    def test_a_ceiling_at_or_below_the_floor_is_not_acted_on(self):
        """
        The carried case: the ceiling is below what the kernel is running, so
        every draw at or under the incumbent would otherwise be retired.
        """
        ctx = self._ctx(ceiling=100, floor=4000)
        v = driver.static_reason(ctx, {"vec#0": {"d0": 1000, "d1": 1}})
        self.assertIsNone(v, "this retires the running value")

    def test_both_gates_carry_the_guard(self):
        src = Path(driver.__file__).read_text(encoding="utf-8")
        self.assertEqual(
            src.count('nb >= ctx["state"]["ceiling"] and nb > ctx["state"]["floor"]'),
            2, "static_reason and _static_verdict")
        self.assertNotIn('if nb and nb >= ctx["state"]["ceiling"]:', src)

    def _ctx(self, ceiling, floor):
        return {"opts": driver.SearchOptions(dtype_bytes=4, domain=None),
                "hw": space.HW(calibrated=False),
                "state": {"ceiling": float(ceiling), "floor": float(floor)}}


if __name__ == "__main__":
    unittest.main()
