# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""A front can ask for a DIRECTION, and a single objective cannot.

The single-objective reason to look outside the catalogue is negative --
"nothing here makes it faster any more" -- and a model answers it with more of
the same. An inserted intent, which has no catalogue standing of its own, is
then rarely reached at all.

With a front the request becomes positive and nameable: "lower the workspace, a
latency regression is acceptable". That is answerable with a rewrite the
catalogue does not contain, and it is search control rather than preference --
WHICH point to ship stays the reader's.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import CHIP_ARGS, DEMO_KERNEL as KERNEL, file_hash, harness, new_op_dir, run_cli  # noqa: E402

OBJS = [{"name": "mix_aic", "from": "p", "direction": "min"},
        {"name": "workspace", "from": "m.workspace_b", "direction": "min"}]


def _st(archive, live, seeded=("F-17", "S-14")):
    return {"config": {"objectives": ["mix_aic", "workspace"]},
            "objective_actions": {"workspace": list(seeded)},
            "nodes": {"open": list(live), "closed": [
                {"id": "root", "eval": {"p": 100.0,
                                        "m": {"workspace_b": 400 << 20,
                                              "workspace_state": "measured"}}}]},
            "archive": [{"code_hash": h, "values": v} for h, v in archive]}


def _node(nid, moves, tried=0):
    return {"id": nid, "catalog_id": nid, "stage": "frontend", "V": 0.5,
            "prior_gain": 0.6, "tried": tried, "moves": list(moves)}


class DemandTest(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.op_dir = self.tmp_dir.name
        self.front = [("a" * 64, {"mix_aic": 90.0, "workspace": float(300 << 20)}),
                      ("b" * 64, {"mix_aic": 120.0, "workspace": float(180 << 20)})]

    def test_a_catalogue_action_still_untried_answers_it_first(self):
        """
        Asking while the core was about to try F-17 spends the model's
        attention on what the core already had.
        """
        st = _st(self.front, [_node("F-17", ["workspace"])])
        self.assertIsNone(self._demand(st))

    def test_the_spent_catalogue_is_what_makes_the_request_worth_making(self):
        st = _st(self.front, [_node("F-17", ["workspace"], tried=3),
                              _node("F-1", [])])
        d = self._demand(st)
        self.assertEqual(d["objective"], "workspace")
        self.assertEqual(d["direction"], "min")
        self.assertEqual(d["beyond"], float(180 << 20), "the front's own best")
        self.assertEqual(d["catalog_spent"], ["F-17", "S-14"])

    def test_the_primary_objective_never_asks(self):
        """The greedy phase IS the request for latency, and best_J answers it."""
        st = _st(self.front, [_node("F-17", ["workspace"], tried=1)])
        st["objective_actions"] = {"mix_aic": ["F-1"]}
        self.assertIsNone(self._demand(st))

    def test_a_single_objective_run_has_no_direction_to_name(self):
        st = _st(self.front, [_node("F-17", ["workspace"], tried=1)])
        st["config"]["objectives"] = ["mix_aic"]
        self.assertIsNone(self._demand(st), "a single objective has no direction to ask for")

    def test_no_front_is_no_direction(self):
        st = _st([], [_node("F-17", ["workspace"], tried=1)])
        self.assertIsNone(self._demand(st))

    def test_the_largest_gap_is_reported_as_context_not_as_the_trigger(self):
        st = _st(self.front, [_node("F-17", ["workspace"], tried=1)])
        d = self._demand(st)
        self.assertEqual(d["largest_gap"], [float(180 << 20), float(300 << 20)])
        self.assertEqual(d["front_points"], 2)
        self.assertEqual(d["preopt"], float(400 << 20))

    def test_the_hint_asks_for_the_thing_the_catalogue_cannot_give(self):
        st = _st(self.front, [_node("F-17", ["workspace"], tried=1)])
        hint = harness.demand_hint_text(self._demand(st), OBJS)
        self.assertIn("180.0 MiB", hint)
        self.assertIn("LOWERS `workspace`", hint)
        self.assertIn("REGRESSION IS ACCEPTABLE", hint)
        self.assertIn("F-17", hint)

    def _demand(self, st):
        return harness.front_demand(st, self.op_dir)


class CatalogTest(unittest.TestCase):

    def test_the_catalogue_says_which_actions_aim_at_the_workspace(self):
        from _support import CATALOG
        items = json.loads(Path(CATALOG).read_text(encoding="utf-8"))
        tagged = sorted(a["id"] for a in items if "workspace" in (a.get("moves") or []))
        self.assertIn("F-17", tagged)
        self.assertIn("S-14", tagged)
        self.assertIn("S-21", tagged)
        self.assertGreaterEqual(len(tagged), 7,
                                "without these the request can never be made")


class CliTest(unittest.TestCase):
    """Through `select` and `evolve`, which is where the model reads it."""

    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", file_hash(self.f),
                "--preopt-workspace-b", str(400 << 20),
                "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS,
                "--config", json.dumps({"objectives": ["mix_aic", "workspace"]}))

    def test_init_records_which_catalogue_actions_aim_at_each_objective(self):
        seeded = self._state()["objective_actions"]["workspace"]
        self.assertIn("F-17", seeded)
        self.assertIn("S-14", seeded)

    def test_select_hands_back_a_direction_instead_of_a_shrug(self):
        self._empty_frontier_with_a_front()
        res, _ = run_cli("select", "--op-dir", str(self.op_dir))
        self.assertEqual(res["reason"], "expansion_required")
        self.assertEqual(res["demand"]["objective"], "workspace")
        self.assertIn("REGRESSION IS ACCEPTABLE", res["hint"])

    def test_evolve_records_the_direction_with_the_request(self):
        self._empty_frontier_with_a_front()
        res, _ = run_cli("evolve", "--op-dir", str(self.op_dir), "--edits",
                         json.dumps({"reasons": ["nothing left in the catalogue"]}))
        self.assertTrue(res["expansion_required"])
        self.assertEqual(res["demand"]["objective"], "workspace")
        self.assertIn("LOWERS `workspace`", res["demand_hint"])
        self.assertEqual(self._state()["progress"]["expansion_demand"]["objective"],
                         "workspace")
        log = next((self.op_dir / "optimization").glob("*.md")).read_text()
        self.assertIn("DIRECTION: lower `workspace` past", log)

    def test_an_insert_clears_the_demand(self):
        self._empty_frontier_with_a_front()
        run_cli("evolve", "--op-dir", str(self.op_dir), "--edits",
                json.dumps({"reasons": ["none"]}))
        run_cli("evolve", "--op-dir", str(self.op_dir), "--edits",
                json.dumps({"insert": [{"parent": "root", "stage": "frontend",
                                        "delta": "keep the scores in bf16",
                                        "V": 0.7, "prior_gain": 0.6}],
                            "reasons": ["answering the demand"]}))
        pr = self._state()["progress"]
        self.assertFalse(pr["expansion_required"])
        self.assertIsNone(pr.get("expansion_demand"))

    def _state(self):
        return json.loads(
            (self.op_dir / "optimization" / "search_state.json").read_text())

    def _write(self, st):
        (self.op_dir / "optimization" / "search_state.json").write_text(
            json.dumps(st), encoding="utf-8")

    def _empty_frontier_with_a_front(self):
        st = self._state()
        st["nodes"]["open"] = []
        st["nodes"]["held"] = []
        st["archive"] = [
            {"code_hash": "a" * 64,
             "values": {"mix_aic": 90.0, "workspace": float(300 << 20)}},
            {"code_hash": "b" * 64,
             "values": {"mix_aic": 120.0, "workspace": float(180 << 20)}}]
        self._write(st)


if __name__ == "__main__":
    unittest.main()
