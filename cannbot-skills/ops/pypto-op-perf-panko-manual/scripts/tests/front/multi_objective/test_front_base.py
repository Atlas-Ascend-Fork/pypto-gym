# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Where the next action starts, in every phase (B2).

Since the ratchet keeps what extends the front (B1), the next action's base is
a question about the front in phase 1 too. `_base_hash` now returns the point
`_advance_front_base` chose at the last close whatever the phase, and
`front_base` chooses among `base_candidates`: on disk, no slower than the
preopt, and non-dominated on the medians.

What these pin down:
  * noise twins the bootstrap archive keeps are not bases;
  * a point slower than the preopt is not a base;
  * with one objective the base is the fastest point -- greedy accumulation;
  * an action that only accepted a trade-off is progress for the halt, and
    the next action starts from the front, not from the fastest program.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import (CHIP_ARGS, DEMO_KERNEL as KERNEL, file_hash, harness, new_op_dir, read_state,  # noqa: E402
                      record_measurement, run_cli)


def _pt(h, p, w=None, tried=0):
    vals = {"aicore_e2e": p} if w is None else {"aicore_e2e": p, "workspace": w}
    return {"code_hash": h, "tried_from": tried, "values": vals}


class CandidateTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.op_dir = Path(self.tmp_dir.name)
        self.nodes = self.op_dir / "optimization" / "nodes"
        self.nodes.mkdir(parents=True)

    def test_a_noise_twin_is_not_a_base(self):
        """The bootstrap archive keeps a 1%-slower copy it cannot separate."""
        st = self._state(["aicore_e2e", "workspace"],
                         [_pt("a" * 64, 90.0, 1000.0), _pt("t" * 64, 91.0, 1000.0),
                          _pt("b" * 64, 95.0, 380.0)])
        self.assertEqual(self._hashes(st), ["a", "b"])

    def test_a_point_slower_than_the_preopt_is_not_a_base(self):
        st = self._state(["aicore_e2e", "workspace"],
                         [_pt("a" * 64, 90.0, 1600.0), _pt("c" * 64, 105.0, 200.0)],
                         preopt=(100.0, 1000))
        self.assertEqual(self._hashes(st), ["a"])

    def test_one_objective_is_greedy(self):
        """The untried twin does not win on fewest-tried: it is not a candidate."""
        st = self._state(["aicore_e2e"],
                         [_pt("a" * 64, 90.0, tried=3), _pt("t" * 64, 90.5)])
        base = harness.front_base(st, str(self.op_dir))
        self.assertEqual(base["code_hash"], "a" * 64)

    def _state(self, objectives, archive, preopt=None):
        for e in archive:
            (self.nodes / f"{e['code_hash']}.py").write_text(KERNEL, encoding="utf-8")
        st = {"config": {"objectives": objectives}, "archive": archive}
        if preopt is not None:
            p, w = preopt
            st["nodes"] = {"closed": [{"id": "root", "code_hash": "p" * 64}]}
            st["eval_cache"] = {"p" * 64: {"s": 1, "p": p, "m": {
                "workspace_b": w, "workspace_state": "measured"}}}
        return st

    def _hashes(self, st):
        return sorted(e["code_hash"][0] for e in harness.base_candidates(st, str(self.op_dir)))


class CloseTest(unittest.TestCase):
    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", self.pre,
                "--preopt-workspace-b", str(1000 << 20),
                "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS)

    def test_a_trade_off_only_action_is_progress_and_the_next_base(self):
        st = read_state(self.op_dir)
        st["progress"]["global_stagnation"] = 3
        (self.op_dir / "optimization" / "search_state.json").write_text(json.dumps(st))
        # As fast as the preopt and far smaller: accepted (on the front, within
        # the reference), but not faster, so `close` does not promote it.
        trade = self._record(256, 100.0, 380 << 20)
        for i, tile in enumerate((8, 16, 24, 32, 40, 48, 56)):
            self._record(tile, 101.0 + i, 1700 << 20)
        res, rc = run_cli("close", "--op-dir", str(self.op_dir), "--action", "u1")
        self.assertEqual(rc, 0, res)
        st = read_state(self.op_dir)
        self.assertEqual(st["progress"]["global_stagnation"], 0,
                         "it extended the front, so the run is not stagnating")
        self.assertEqual(st["progress"]["best_aicore_e2e_us"], 100.0,
                         "the fastest program is still the preopt")
        self.assertEqual(st["progress"]["base_code_hash"], trade)
        self.assertEqual(st["progress"]["base_selection"]["code_hash"], trade)
        self.assertEqual(st["progress"]["base_selection"]["reference"]["aicore_e2e"],
                         100.0, "the primary axis is referenced at the preopt")
        sel = run_cli("select", "--op-dir", str(self.op_dir))[0]["selected"]
        self.assertEqual(sel["restore_code_hash"], trade)
        self.assertEqual(file_hash(self.f), trade)

    def _record(self, tile, p, ws):
        return record_measurement(
            self, KERNEL.replace("64, 128", f"{tile}, 128"), p, ws)[0]


if __name__ == "__main__":
    unittest.main()
