# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""The closed loop: outcomes are recorded, the table is rebuilt, the next run reads it.

`prior_gain` is a hint about which actions work, and it should learn from every
campaign. What keeps the loop honest is tested here: a run's table never changes
under it, a catalogue entry whose text changed is a different action, old
evidence fades, and a pinned experiment is left alone.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import CHIP_ENVELOPE_OFFLINE, DEMO_KERNEL as KERNEL, file_hash, INIT_ARGS, run_cli  # noqa: E402
import prior_store as PS  # noqa: E402


class _RowKnobs(NamedTuple):
    """The optional fields of one synthetic outcome row."""

    cid: Any = "F-1"
    kept: bool = False
    text: str = "tile it"
    s: int = 1
    gain: float = 0.0
    prop: Any = None


def _row(camp, start, knobs=_RowKnobs()):
    return {"campaign": camp, "campaign_start": start, "catalog_id": knobs.cid,
            "delta_hash": PS.delta_hash(knobs.text), "kept": knobs.kept, "s": knobs.s,
            "log_gain": knobs.gain, "propensity": knobs.prop,
            "predicates": {"has_cube_op": True, "determined": True}}


CATALOG = [{"id": "F-1", "delta": "tile it"}, {"id": "F-2", "delta": "fuse it"}]


class PerChipTest(unittest.TestCase):
    """
    What an action is worth is a property of the hardware, so the store and
    the table are per chip.
    """

    def test_the_store_and_the_table_are_named_after_the_chip(self):
        self.assertTrue(PS.store_dir("/h", "Ascend910B3").endswith("outcomes/ascend910b3"))
        self.assertTrue(PS.learned_table_path("/h", "Ascend910C")
                        .endswith("action_prior_table.ascend910c.json"))
        self.assertTrue(PS.store_dir("/h").endswith("outcomes"),
                        "the legacy flat store keeps its path")
        self.assertTrue(PS.learned_table_path("/h").endswith("action_prior_table.json"))

    def test_phase_two_rows_are_not_evidence_about_kept(self):
        """The table models "this action made the program faster".

        A phase-2 candidate is applied to a front point and is EXPECTED to be
        slower than the global best, so it always arrives `kept: false`. Left
        in, every action's learned rate would fall in proportion to how well
        phase 2 worked.
        """
        rows = [dict(_row("c1", 1, _RowKnobs(kept=True)), phase="greedy"),
                dict(_row("c2", 2, _RowKnobs(kept=False)), phase="front", front_insert=True),
                dict(_row("c3", 3, _RowKnobs(kept=False)), phase="front", front_insert=False)]
        t = PS.build(rows, CATALOG, PS.BuildOptions(half_life=0))
        self.assertEqual(t["by_catalog"]["F-1"]["n_meas"], 1)
        self.assertEqual(t["by_catalog"]["F-1"]["n_kept"], 1)
        self.assertEqual(t["meta"]["rows_dropped_front_phase"], 2)

    def test_a_row_with_no_phase_is_read_as_before(self):
        """Rows written before `phase` existed are greedy-phase rows."""
        t = PS.build([_row("c1", 1, _RowKnobs(kept=True))], CATALOG, PS.BuildOptions(half_life=0))
        self.assertEqual(t["by_catalog"]["F-1"]["n_meas"], 1)
        self.assertEqual(t["meta"]["rows_dropped_front_phase"], 0)

    def test_another_chips_rows_do_not_count(self):
        rows = [dict(_row("c1", 1, _RowKnobs(kept=True)), soc="Ascend910B3"),
                dict(_row("c2", 2, _RowKnobs(kept=True)), soc="Ascend910C"),
                dict(_row("c3", 3, _RowKnobs(kept=True)), soc="Ascend910C")]
        t = PS.build(rows, CATALOG, PS.BuildOptions(half_life=0, soc="Ascend910C"))
        self.assertEqual(t["by_catalog"]["F-1"]["n_meas"], 2)
        self.assertEqual(t["meta"]["rows_dropped_other_chip"], 1)
        self.assertEqual(t["meta"]["soc"], "ascend910c")

    def test_rows_from_before_the_split_are_dropped_not_guessed(self):
        rows = [_row("c1", 1, _RowKnobs(kept=True)), dict(_row("c2", 2, _RowKnobs(kept=True)), soc="Ascend910C")]
        t = PS.build(rows, CATALOG, PS.BuildOptions(half_life=0, soc="Ascend910C"))
        self.assertEqual(t["by_catalog"]["F-1"]["n_meas"], 1)
        self.assertEqual(t["meta"]["rows_unlabelled_chip"], 1)

    def test_legacy_soc_claims_them_for_one_chip(self):
        rows = [_row("c1", 1, _RowKnobs(kept=True)), dict(_row("c2", 2, _RowKnobs(kept=True)), soc="Ascend910C")]
        t = PS.build(rows, CATALOG, PS.BuildOptions(half_life=0, soc="Ascend910B3",
                                                    legacy_soc="Ascend910B3"))
        self.assertEqual(t["by_catalog"]["F-1"]["n_meas"], 1, "the unlabelled row")
        self.assertEqual(t["meta"]["rows_dropped_other_chip"], 1)
        self.assertEqual(t["meta"]["legacy_soc"], "ascend910b3")


class BuildTest(unittest.TestCase):
    def test_counts_and_predicate_cells(self):
        rows = [_row("c1", 1, _RowKnobs(kept=True, gain=0.1)), _row("c1", 1), _row("c1", 1, _RowKnobs(s=0))]
        t = PS.build(rows, CATALOG, PS.BuildOptions(half_life=0))
        self.assertEqual(t["by_catalog"]["F-1"]["n_meas"], 3)
        self.assertEqual(t["by_catalog"]["F-1"]["n_kept"], 1)
        self.assertEqual(t["by_catalog"]["F-1"]["n_s0"], 1)
        self.assertIn("F-1|has_cube_op=True", t["by_catalog_predicate"])
        self.assertNotIn("F-1|determined=True", t["by_catalog_predicate"])

    def test_a_changed_catalogue_text_is_a_different_action(self):
        rows = [_row("c1", 1, _RowKnobs(text="tile it")), _row("c1", 1, _RowKnobs(text="the OLD wording"))]
        t = PS.build(rows, CATALOG, PS.BuildOptions(half_life=0))
        self.assertEqual(t["by_catalog"]["F-1"]["n_meas"], 1)
        self.assertEqual(t["meta"]["rows_dropped_catalog_text_changed"], 1)

    def test_old_evidence_fades(self):
        rows = [_row("old", 1, _RowKnobs(kept=True))] + [_row(f"c{i}", 10 + i) for i in range(4)]
        t = PS.build(rows, CATALOG, PS.BuildOptions(half_life=1))
        self.assertLess(t["by_catalog"]["F-1"]["n_kept"], 0.1)

    def test_inserts_pool_into_one_class(self):
        t = PS.build([_row("c1", 1, _RowKnobs(cid=None, kept=True))], CATALOG, PS.BuildOptions(half_life=0))
        self.assertEqual(t["by_catalog"]["insert"]["n_kept"], 1)

    def test_ipw_upweights_rarely_selected_rows(self):
        rows = [_row("c1", 1, _RowKnobs(kept=True, prop=0.1)), _row("c1", 1, _RowKnobs(prop=0.9)),
                _row("c1", 1, _RowKnobs(prop=0.9))]
        plain = PS.build(rows, CATALOG, PS.BuildOptions(half_life=0))["by_catalog"]["F-1"]
        w = PS.build(rows, CATALOG, PS.BuildOptions(half_life=0, ipw=True))["by_catalog"]["F-1"]
        self.assertAlmostEqual(w["n_meas"], plain["n_meas"], places=3)
        self.assertGreater(w["n_kept"], plain["n_kept"])

    def test_the_base_table_is_merged(self):
        base = {"global": {"n_meas": 100, "n_kept": 10},
                "by_catalog": {"F-2": {"n_meas": 50, "n_kept": 5}}}
        t = PS.build([_row("c1", 1)], CATALOG, PS.BuildOptions(base=base, half_life=0))
        self.assertEqual(t["by_catalog"]["F-2"]["n_meas"], 50)
        self.assertEqual(t["global"]["n_meas"], 101)

    def test_malformed_lines_are_skipped(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with open(os.path.join(tmp.name, "x.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(_row("c1", 1)) + "\n{broken\n")
        rows, bad = PS.read_rows([tmp.name])
        self.assertEqual((len(rows), bad), (1, 1))


class LoopCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name) / "home"
        self.env = dict(os.environ, PANKO_HOME=str(self.home))

    def test_a_campaign_feeds_the_next_one(self):
        op1, st1, rep1 = self._campaign("a")
        # Nothing ships, so the first campaign on a fresh machine has no table.
        self.assertEqual(st1["prior_table"]["origin"], "none")
        self.assertIsNone(st1["prior_table"]["snapshot"])
        rows = (op1 / "optimization" / "action_outcomes.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 1)
        self.assertTrue(json.loads(rows[0])["kept"])
        self.assertTrue(rep1["prior_table"]["rebuilt"], rep1)
        self.assertTrue((self.home / "action_prior_table.json").exists())
        _, st2, _ = self._campaign("b")
        self.assertEqual(st2["prior_table"]["origin"], "learned")
        self.assertEqual(st2["prior_table"]["sha256"],
                         PS.file_sha(st2["prior_table"]["snapshot"]))

    def test_the_rows_and_the_table_land_under_the_chip(self):
        """A 910C campaign must not write into what 910B3 measured."""
        chip = dict(CHIP_ENVELOPE_OFFLINE, soc="Ascend910C")
        op_dir = Path(self._tmp.name) / "c"
        op_dir.mkdir()
        f = op_dir / "demo_impl.py"
        f.write_text(KERNEL, encoding="utf-8")
        run_cli("init", "--op-dir", str(op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", file_hash(f), "--op-file", str(f),
                "--allow-unversioned", "--chip-envelope", json.dumps(chip),
                "--preopt-workspace-b", str(1 << 20),
                "--config", json.dumps({"action_prior_table": ""}), env=self.env)
        sel = run_cli("select", "--op-dir", str(op_dir), env=self.env)[0]["selected"]
        f.write_text(KERNEL.replace("64, 128", "128, 128"), encoding="utf-8")
        run_cli("record", "--op-dir", str(op_dir), "--parent", sel["id"],
                "--code-hash", file_hash(f), "--s", "1", "--p", "40", env=self.env)
        run_cli("close", "--op-dir", str(op_dir), "--action", sel["id"], env=self.env)
        row = json.loads((op_dir / "optimization" / "action_outcomes.jsonl")
                         .read_text().splitlines()[0])
        self.assertEqual(row["soc"], "Ascend910C")
        self.assertTrue((self.home / "outcomes" / "ascend910c").is_dir())
        run_cli("stop", "--op-dir", str(op_dir), "--force", "manual", env=self.env)
        run_cli("verify", "--op-dir", str(op_dir), "--skip", "test", env=self.env)
        rep = run_cli("report", "--op-dir", str(op_dir), env=self.env)[0]
        self.assertEqual(rep["prior_table"]["soc"], "ascend910c", rep)
        self.assertTrue((self.home / "action_prior_table.ascend910c.json").exists())
        self.assertFalse((self.home / "action_prior_table.json").exists(),
                         "the other chip's table is untouched")

    def test_a_running_campaign_keeps_its_snapshot(self):
        _, st1, _ = self._campaign("a")
        snap = st1["prior_table"]["snapshot"]
        before = PS.file_sha(snap)
        self._campaign("b")                      # rebuilds the machine's table
        self.assertEqual(PS.file_sha(snap), before)

    def test_a_pinned_experiment_neither_reads_nor_rebuilds(self):
        pinned = str(Path(__file__).parents[4] / "references"
                     / "action_prior_table.json")
        _, st, rep = self._campaign("a", {"action_prior_table": pinned})
        self.assertEqual(st["prior_table"]["origin"], "explicit")
        self.assertFalse(rep["prior_table"]["rebuilt"])
        self.assertFalse((self.home / "action_prior_table.json").exists())

    def test_thompson_records_a_propensity(self):
        op1, _, _ = self._campaign("a", {"selection": "thompson"})
        row = json.loads((op1 / "optimization" / "action_outcomes.jsonl")
                         .read_text().splitlines()[0])
        self.assertIsNotNone(row["propensity"])
        self.assertTrue(0 < row["propensity"] <= 1)

    def _campaign(self, name, config=None, p=40):
        op_dir = Path(self._tmp.name) / name
        op_dir.mkdir()
        f = op_dir / "demo_impl.py"
        f.write_text(KERNEL, encoding="utf-8")
        # This file is about the LOOP -- a campaign's outcomes rebuilding the
        # machine's table for the next one -- so it must NOT inherit the
        # suite's pinned synthetic table, which turns the rebuild off. An
        # empty `action_prior_table` is "not pinned".
        cfg = dict({"action_prior_table": ""}, **(config or {}))
        args = ["init", "--op-dir", str(op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", file_hash(f), "--op-file", str(f),
                "--allow-unversioned", *INIT_ARGS, "--config", json.dumps(cfg)]
        run_cli(*args, env=self.env)
        sel = run_cli("select", "--op-dir", str(op_dir), env=self.env)[0]["selected"]
        f.write_text(KERNEL.replace("64, 128", "128, 128"), encoding="utf-8")
        run_cli("record", "--op-dir", str(op_dir), "--parent", sel["id"],
                "--code-hash", file_hash(f), "--s", "1", "--p", str(p), env=self.env)
        run_cli("close", "--op-dir", str(op_dir), "--action", sel["id"], env=self.env)
        run_cli("stop", "--op-dir", str(op_dir), "--force", "manual", env=self.env)
        run_cli("verify", "--op-dir", str(op_dir), "--skip", "test", env=self.env)
        rep = run_cli("report", "--op-dir", str(op_dir), env=self.env)[0]
        st = json.loads((op_dir / "optimization" / "search_state.json").read_text())
        return op_dir, st, rep


if __name__ == "__main__":
    unittest.main()
