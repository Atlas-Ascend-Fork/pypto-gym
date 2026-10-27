# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""The non-dominated archive, and the frontier it delivers.

The ratchet keeps one program: whatever is fastest. A candidate that is slower
but needs less workspace is measured and thrown away though nothing dominates
it. The archive keeps those points in EVERY
run (the search does not read it yet), and `stop` / `report` write them out with
their sources, because a front point is a deliverable.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import (CHIP_ARGS, DEMO_KERNEL as KERNEL, file_hash, harness, new_op_dir, read_state,  # noqa: E402
                      record_measurement, run_cli)

# Two objectives, so these stay about the archive rather than about the
# default's third axis. The default is exercised through the CLI tests below.
OBJS = harness.parse_objectives({"objectives": ["mix_aic", "workspace"]})


def _pt(h, p, w):
    return {"code_hash": h, "latency_us": p, "workspace_b": w,
            "values": {"mix_aic": p, "workspace": w}}


def _insert(archive, point):
    return harness.archive_insert(OBJS, archive, point)


class ArchiveTest(unittest.TestCase):
    def test_a_slower_but_smaller_point_is_kept(self):
        arc, v = _insert([_pt("a", 100.0, 1000)], _pt("b", 120.0, 400))
        self.assertEqual(v, "inserted")
        self.assertEqual([e["code_hash"] for e in arc], ["b", "a"])   # by workspace

    def test_a_dominated_point_is_refused(self):
        arc, v = _insert([_pt("a", 100.0, 1000)], _pt("b", 120.0, 1000))
        self.assertEqual(v, "dominated")
        self.assertEqual(len(arc), 1)

    def test_a_new_point_drops_what_it_dominates(self):
        arc, v = _insert([_pt("a", 100.0, 1000), _pt("b", 120.0, 400)],
                         _pt("c", 90.0, 400))
        self.assertEqual(v, "inserted")
        self.assertEqual([e["code_hash"] for e in arc], ["c"])

    def test_the_same_program_is_not_stored_twice(self):
        arc, v = _insert([_pt("a", 100.0, 1000)], _pt("a", 100.0, 1000))
        self.assertEqual((v, len(arc)), ("duplicate", 1))

    def test_a_tie_on_both_keeps_the_first(self):
        """
        Formally non-dominated, but two rows a reader cannot choose between
        are noise in a list of deliverables.
        """
        arc, v = _insert([_pt("a", 100.0, 1000)], _pt("b", 100.0, 1000))
        self.assertEqual((v, [e["code_hash"] for e in arc]), ("tied", ["a"]))


class CliTest(unittest.TestCase):
    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", self.pre,
                "--preopt-workspace-b", str(1000 << 20),
                "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS)

    def test_the_preopt_opens_the_archive(self):
        arc = read_state(self.op_dir)["archive"]
        self.assertEqual([e["code_hash"] for e in arc], [self.pre])
        self.assertEqual(arc[0]["workspace_b"], 1000 << 20)

    def test_a_trade_off_is_kept_and_stays_in_the_archive(self):
        """
        Front acceptance: a slower point that needs less workspace, and that
        is no slower than the preopt, is kept -- the scalar ratchet reverted it.
        """
        fast = self._record(128, 90.0, 1600 << 20)      # kept
        small = self._record(256, 95.0, 380 << 20)      # kept, non-dominated
        st = read_state(self.op_dir)
        hashes = [e["code_hash"] for e in st["archive"]]
        self.assertIn(small, hashes)
        self.assertIn(fast, hashes)
        self.assertEqual(st["progress"]["archive_kept_slower"], 1)
        self.assertEqual(st["eval_cache"][small]["verdict"], "kept")

    def test_a_trade_off_slower_than_the_preopt_is_reverted_but_archived(self):
        """
        The reference point: on the front, but slower than the program the
        run started from, so the ratchet does not continue from it.
        """
        self._record(128, 90.0, 1600 << 20)
        slow = self._record(256, 105.0, 380 << 20)
        st = read_state(self.op_dir)
        self.assertIn(slow, [e["code_hash"] for e in st["archive"]])
        self.assertEqual(st["eval_cache"][slow]["verdict"], "reverted")

    def test_the_trajectory_says_what_the_front_did(self):
        """
        `reverted` alone cannot say whether a candidate was a discarded
        trade-off or genuinely dominated.
        """
        self._record(128, 90.0, 1600 << 20)             # kept, inserted
        dom = self._record(256, 95.0, 1600 << 20)       # reverted, dominated
        trade = self._record(32, 105.0, 380 << 20)      # reverted, inserted
        log = next((self.op_dir / "optimization").glob("*.md")).read_text()
        rows = {}
        for ln in log.splitlines():
            if not ln.startswith("| "):
                continue
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows[cells[1].strip("`")] = cells
        # code | aicore_e2e | workspace | verdict | archive | ...
        self.assertIn(dom[:8], rows)
        self.assertIn(trade[:8], rows)
        dom_row, trade_row = rows.get(dom[:8]), rows.get(trade[:8])
        self.assertEqual(dom_row[3], "1600.0", "the unit is in the header")
        self.assertEqual(dom_row[4:6], ["reverted", "dominated"])
        self.assertEqual(trade_row[4:6], ["reverted", "inserted"],
                         "slower than the preopt: the front keeps it, the "
                         "ratchet does not continue from it")

    def test_record_reports_the_objective_vector(self):
        res = run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                      "--code-hash", "1" * 64, "--s", "1", "--p", "95",
                      "--workspace-b", str(380 << 20))[0]
        self.assertEqual(res["objectives"], {"aicore_e2e": 95.0,
                                             "workspace": float(380 << 20)})
        self.assertEqual(res["archive"], "inserted")
        # It is faster AND smaller than the preopt, so the preopt is dropped.
        self.assertEqual(res["archive_points"], 1)
        self.assertEqual(read_state(self.op_dir)["progress"]["archive_dropped"], 1)

    def test_every_measured_candidate_keeps_its_source(self):
        """
        Without this a front point cannot be delivered: only kept candidates
        used to be snapshotted.
        """
        small = self._record(256, 95.0, 380 << 20)
        self.assertTrue((self.op_dir / "optimization" / "nodes" / f"{small}.py").exists())

    def test_a_measurement_missing_an_objective_halts_the_run(self):
        """
        Every dominance judgement rests on the objective vector, so a
        component nobody measured ends the run rather than being skipped.
        """
        self.f.write_text(KERNEL.replace("64, 128", "32, 128"), encoding="utf-8")
        res, rc = run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                          "--code-hash", file_hash(self.f), "--s", "1", "--p", "80",
                          "--workspace-state", "unavailable")
        self.assertNotEqual(rc, 0)
        self.assertEqual(res["error"], "objective_unmeasured")
        self.assertEqual([m["objective"] for m in res["missing"]], ["workspace"])
        st = read_state(self.op_dir)
        self.assertEqual(st["progress"]["stop_reason"], "objective_unmeasured")
        self.assertEqual(len(st["archive"]), 1, "nothing was recorded")

    def test_an_objective_has_to_name_its_instrument(self):
        """
        `latency` meant `p`, and `p` has been two instruments over this
        repo's life, so the name did not say what was being optimised. It is
        refused with both instruments named rather than resolved to one.
        """
        with self.assertRaises(ValueError) as c:
            harness.parse_objectives({"objectives": ["latency", "workspace"]})
        msg = str(c.exception)
        self.assertIn("aicore_e2e", msg)
        self.assertIn("mix_aic", msg)

    def test_two_names_for_one_reading_are_still_one_objective(self):
        objs = harness.parse_objectives({"objectives": ["mix_aic", "workspace"]})
        self.assertEqual([o["from"] for o in objs], ["m.mix_aic", "m.workspace_b"])
        with self.assertRaises(ValueError) as c:
            harness.parse_objectives({"objectives": [
                "aicore_e2e", {"name": "same_p", "from": "p"}]})
        self.assertIn("one objective under two names", str(c.exception))

    def test_a_resubmission_of_a_measured_program_does_not_halt_the_run(self):
        """
        A re-record carries no numbers because the numbers are already in the
        cache. Halting on it ended a recorded campaign on the hash of its own
        incumbent.
        """
        h = self._record(128, 90.0, 1600 << 20)
        res, rc = run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                          "--code-hash", h, "--s", "1", "--p", "90")
        self.assertEqual(rc, 0, res)
        self.assertTrue(res["duplicate"], res)
        st = read_state(self.op_dir)
        self.assertIsNone(st["progress"]["stop_reason"])
        self.assertNotIn("objective_unmeasured", st["progress"])

    def test_a_new_program_still_halts(self):
        res, rc = run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                          "--code-hash", "e" * 64, "--s", "1", "--p", "90",
                          "--workspace-state", "unavailable")
        self.assertEqual(rc, 4)
        self.assertEqual(res["error"], "objective_unmeasured")

    def test_a_latency_only_run_does_not_need_the_workspace(self):
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", self.pre,
                "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS,
                "--config", json.dumps({"objectives": ["mix_aic"]}))
        self.f.write_text(KERNEL.replace("64, 128", "32, 128"), encoding="utf-8")
        res, rc = run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                          "--code-hash", file_hash(self.f), "--s", "1", "--p", "80",
                          "--workspace-state", "unavailable")
        self.assertEqual(rc, 0, res)
        self.assertEqual(read_state(self.op_dir)["archive"][0]["values"], {"mix_aic": 80.0})

    def test_stop_and_report_write_the_frontier(self):
        fast = self._record(128, 40.0, 1600 << 20)      # J >= 100, so stop fires
        small = self._record(256, 95.0, 380 << 20)
        run_cli("close", "--op-dir", str(self.op_dir), "--action", "u1")
        res = run_cli("stop", "--op-dir", str(self.op_dir), "--force", "manual")[0]
        self.assertGreaterEqual(res["frontier"]["points"], 2)
        run_cli("verify", "--op-dir", str(self.op_dir), "--skip", "test")
        rep = run_cli("report", "--op-dir", str(self.op_dir))[0]
        front = rep["frontier"]
        self.assertEqual(front["sources_missing"], 0)
        impl = [r for r in front["front"] if r["is_impl"]]
        self.assertEqual([r["code_hash"] for r in impl], [fast], "impl.py is the fastest")
        for r in front["front"]:
            self.assertTrue((self.op_dir / r["source"]).exists())
        log = next((self.op_dir / "optimization").glob("*.md")).read_text()
        self.assertIn("## Frontier", log)
        self.assertIn("impl.py is the FASTEST point", log)
        self.assertIn(small[:8], log)

    def _record(self, tile, p, ws):
        return record_measurement(
            self, KERNEL.replace("64, 128", f"{tile}, 128"), p, ws)[0]


if __name__ == "__main__":
    unittest.main()
