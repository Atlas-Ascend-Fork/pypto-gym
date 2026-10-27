# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""What `impl.py` holds at stop (C1).

`ratchet_best_hash` is `best_node`: a scalar on `J` that only moves at an action's
`close`. The front it is reported next to is built on the bootstrap interval,
so the two drifted apart and a recorded run shipped a program another point
beat -- once a kernel slower than a point with far less workspace, once one
marginally slower than rank 2 at the same workspace to the byte.

Delivery is now `delivery_point`: the best primary objective among every
correct measurement whose source is on disk, ties broken by the remaining
objectives. Both cases are reproduced below.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import CHIP_ARGS, DEMO_KERNEL as KERNEL, file_hash, harness, new_op_dir, read_state, run_cli  # noqa: E402


class PointTest(unittest.TestCase):
    """`delivery_point` on a state built by hand, so `best_node` can be wrong."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.op_dir = Path(self.tmp_dir.name)
        self.nodes = self.op_dir / "optimization" / "nodes"
        self.nodes.mkdir(parents=True)

    def test_a_tie_on_workspace_goes_to_the_faster(self):
        """
        A fraction of a percent apart on latency, identical workspace: the
        bootstrap cannot order them and the archive keeps both.
        """
        st = self._st([("a" * 64, 1004.0, 1090000000), ("b" * 64, 1002.2, 1090000000)])
        self.assertEqual(self._pick(st), "b" * 64)

    def test_the_sparse_case_the_faster_and_smaller_point_wins(self):
        st = self._st([("a" * 64, 250.0, 13500 << 20), ("b" * 64, 245.1, 1000 << 20)])
        self.assertEqual(self._pick(st), "b" * 64)

    def test_the_primary_objective_decides_before_the_others(self):
        """AICore E2E first: a 5x smaller program that is slower does not ship."""
        st = self._st([("a" * 64, 90.0, 5000 << 20), ("b" * 64, 95.0, 1000 << 20)])
        self.assertEqual(self._pick(st), "a" * 64)

    def test_a_point_with_no_source_cannot_be_delivered(self):
        st = self._st([("a" * 64, 95.0, 1000), ("b" * 64, 90.0, 1000)],
                      on_disk={"a" * 64})
        self.assertEqual(self._pick(st), "a" * 64)

    def test_an_incorrect_measurement_is_not_a_candidate(self):
        st = self._st([("a" * 64, 95.0, 1000), ("b" * 64, 90.0, 1000)])
        st["eval_cache"]["b" * 64]["s"] = 0
        self.assertEqual(self._pick(st), "a" * 64)

    def test_a_tie_on_the_primary_prefers_a_known_trade_off(self):
        st = self._st([("a" * 64, 90.0, None), ("b" * 64, 90.0, 1000)])
        self.assertEqual(self._pick(st), "b" * 64)

    def test_nothing_deliverable_is_no_point(self):
        st = self._st([], on_disk=set())
        h, detail = harness.delivery_point(st, str(self.op_dir))
        self.assertIsNone(h)
        self.assertEqual(detail["why"], "no_deliverable_measurement")

    def _st(self, points, on_disk=None, objectives=("aicore_e2e", "workspace")):
        cache = {}
        for h, p, w in points:
            if on_disk is None or h in on_disk:
                (self.nodes / f"{h}.py").write_text(KERNEL, encoding="utf-8")
            m = {"workspace_state": "measured"}
            if w is not None:
                m["workspace_b"] = w
            cache[h] = {"s": 1, "p": p, "m": m}
        return {"config": {"objectives": list(objectives)}, "eval_cache": cache,
                "nodes": {"closed": []}}

    def _pick(self, st):
        return harness.delivery_point(st, str(self.op_dir))[0]


class StopTest(unittest.TestCase):
    """End to end: a remeasurement moves a median past `best_node`.

    `best_node` moves only at `close`, and `--remeasure` deliberately does not
    run the ratchet -- so pooling readings into an archived point is how the
    deliverable and the fastest measurement come apart.
    """

    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", self.pre,
                "--preopt-workspace-b", str(1000 << 20),
                "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS)

    def test_stop_ships_the_fastest_measurement_not_best_node(self):
        fast = self._record(128, 90.0, 1600 << 20, samples=[90.0] * 9)
        self._exhaust()
        res, rc = run_cli("close", "--op-dir", str(self.op_dir), "--action", "u1")
        self.assertEqual(rc, 0, res)
        self.assertEqual(read_state(self.op_dir)["best_node"], res["closed"])
        # A trade-off, archived; then pooled readings drop its median below the
        # program `best_node` names. `--remeasure` runs no ratchet by design.
        trade = self._record(256, 95.0, 380 << 20, samples=[95.0] * 9)
        run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                "--code-hash", trade, "--s", "1", "--remeasure",
                "--samples", ",".join(["85.0"] * 30))
        st = read_state(self.op_dir)
        self.assertEqual(st["eval_cache"][trade]["p"], 85.0)
        self.assertEqual(st["progress"]["best_aicore_e2e_us"], 90.0,
                         "best_node still names the 90 us program")
        res, _ = run_cli("stop", "--op-dir", str(self.op_dir), "--force", "wall_clock")
        self.assertTrue(res["stop"], res)
        self.assertEqual(res["best_code_hash"], trade)
        self.assertEqual(res["best_node_code_hash"], fast)
        self.assertTrue(res["delivered_is_best"])
        self.assertEqual(file_hash(self.f), trade, "the file holds the fastest")
        st = read_state(self.op_dir)
        self.assertEqual(st["progress"]["delivery_outranked_best_node"]["delivered"],
                         trade)
        rep, _ = run_cli("report", "--op-dir", str(self.op_dir))
        self.assertEqual(rep["best_code_hash"], trade)
        impl = [r for r in rep["frontier"]["front"] if r["is_impl"]]
        self.assertEqual([r["code_hash"] for r in impl], [trade],
                         "is_impl marks the row the file carries")
        log = next((self.op_dir / "optimization").glob("*.md")).read_text()
        self.assertIn("is beaten by it on the measurements", log)

    def test_verification_measures_the_program_the_stop_delivered(self):
        """The ABBA pass has to be about the kernel that is being shipped.

        The arm was `ratchet_best_hash` -- `best_node` -- while `stop` ships
        `delivery_point`. A `--remeasure` moves one and not the other, so the
        pass measured the 90 us program, compared it against that program's
        speedup, and then wrote it over the 85 us one the stop had just put in
        the working file. The selected winner was lost by verifying it.
        """
        fast = self._record(128, 90.0, 1600 << 20, samples=[90.0] * 9)
        self._exhaust()
        run_cli("close", "--op-dir", str(self.op_dir), "--action", "u1")
        trade = self._record(256, 95.0, 380 << 20, samples=[95.0] * 9)
        run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                "--code-hash", trade, "--s", "1", "--remeasure",
                "--samples", ",".join(["85.0"] * 30))
        res, _ = run_cli("stop", "--op-dir", str(self.op_dir), "--force", "wall_clock")
        self.assertEqual(res["best_code_hash"], trade)
        self.assertEqual(res["best_node_code_hash"], fast)
        st = read_state(self.op_dir)
        self.assertEqual(st["progress"]["delivered_code_hash"], trade)
        v = st["verification"]
        self.assertEqual(v["arms"]["delivered"]["code_hash"], trade)
        self.assertAlmostEqual(v["search_speedup"], round(100.0 / 85.0, 4),
                               msg="the claim quoted is the delivered program's")
        staged, _ = run_cli("verify", "--op-dir", str(self.op_dir))
        self.assertEqual(staged["code_hash"], v["arms"][staged["arm"]]["code_hash"])
        run_cli("verify", "--op-dir", str(self.op_dir),
                "--samples", ",".join(["99.0"] * 5))
        self.assertEqual(file_hash(self.f), trade,
                         "between passes the file holds the delivered program")

    def test_a_skipped_verification_leaves_the_delivered_program(self):
        """`--skip` restores too, and restored `best_node` for the same reason."""
        self._record(128, 90.0, 1600 << 20, samples=[90.0] * 9)
        self._exhaust()
        run_cli("close", "--op-dir", str(self.op_dir), "--action", "u1")
        trade = self._record(256, 95.0, 380 << 20, samples=[95.0] * 9)
        run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                "--code-hash", trade, "--s", "1", "--remeasure",
                "--samples", ",".join(["85.0"] * 30))
        run_cli("stop", "--op-dir", str(self.op_dir), "--force", "wall_clock")
        run_cli("verify", "--op-dir", str(self.op_dir))
        run_cli("verify", "--op-dir", str(self.op_dir), "--skip", "device_unavailable")
        self.assertEqual(file_hash(self.f), trade)

    def test_an_ordinary_run_delivers_best_node(self):
        fast = self._record(128, 90.0, 1600 << 20, samples=[90.0] * 9)
        self._exhaust()
        run_cli("close", "--op-dir", str(self.op_dir), "--action", "u1")
        res, _ = run_cli("stop", "--op-dir", str(self.op_dir), "--force", "wall_clock")
        self.assertEqual(res["best_code_hash"], fast)
        self.assertNotIn("best_node_code_hash", res)
        self.assertTrue(res["delivered_is_best"])
        self.assertNotIn("delivery_outranked_best_node", read_state(self.op_dir)["progress"])

    def _record(self, tile, p, ws, samples=None):
        self.f.write_text(KERNEL.replace("64, 128", f"{tile}, 128"), encoding="utf-8")
        h = file_hash(self.f)
        args = ["record", "--op-dir", str(self.op_dir), "--parent", "u1",
                "--code-hash", h, "--s", "1", "--p", str(p), "--workspace-b", str(ws)]
        if samples:
            args += ["--samples", ",".join(str(x) for x in samples)]
        run_cli(*args)
        return h

    def _exhaust(self):
        """`close` refuses an untested action, so spend its stagnation_K."""
        for i, tile in enumerate((8, 16, 24, 32, 40, 48, 56)):
            self._record(tile, 110.0 + i, 1700 << 20, samples=[110.0 + i] * 9)


if __name__ == "__main__":
    unittest.main()
