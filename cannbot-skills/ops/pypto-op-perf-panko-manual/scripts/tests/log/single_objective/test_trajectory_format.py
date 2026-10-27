# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""The trajectory file is how a run is READ.

One long key=value line per event put every field on the same footing and lost
the only structure that matters: which action the events belong to. These tests
pin the shape -- a section per action, its candidates as a table, its notes in
the order they happened -- not the prose.

The table is GitHub-flavoured Markdown, so the file renders as a table in a
preview instead of a wall of aligned spaces. A Markdown table ends at the first
line that is not a row, so a note between two candidates has to BE a row or it
splits the table in half; that is why notes carry `·` in the eval column.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import CHIP_ARGS, DEMO_KERNEL as KERNEL, file_hash, new_op_dir, record_measurement, run_cli  # noqa: E402


class TrajectoryTest(unittest.TestCase):

    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "100",
                "--preopt-p", "810.745", "--preopt-hash", file_hash(self.f),
                "--preopt-workspace-b", str(367 << 20),
                "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS,
                "--config", json.dumps({"objectives": ["mix_aic", "workspace"],
                                        "eval_budget": 120,
                                        "global_stagnation_limit": 5,
                                        "stagnation_K": 2}))
        self.sel = run_cli("select", "--op-dir", str(self.op_dir))[0]["selected"]

    def test_the_baseline_names_every_objective_not_just_the_first(self):
        """The whole run is quoted against the baseline.

        Under two objectives a line that names only the latency cannot say
        what the front started from, and a reader had to open
        `search_state.json` to find the preopt's workspace.
        """
        head = self._log().split("## Trajectory")[0]
        line = next(ln for ln in head.splitlines() if ln.startswith("- preopt:"))
        self.assertEqual(line, "- preopt: mix_aic 810.75 us · workspace 367.0 MiB")
        self.assertIn("- P_ref 100.0 us · j0 ", head)

    def test_the_opening_is_a_block_not_one_long_line(self):
        head = self._log().split("## Trajectory")[0]
        self.assertIn("- predicates: ", head)
        self.assertIn("objectives mix_aic, workspace · budget 120 evals", head)
        self.assertTrue(all(len(ln) < 400 for ln in head.splitlines()),
                        "the predicates used to push everything off the edge")

    def test_a_section_per_action_with_a_table_of_its_candidates(self):
        fast = self._record("128, 128", 555.193, 519 << 20)
        slow = self._record("96, 128", 681.738, 1440 << 20)
        log = self._log().splitlines()
        self.assertIn(f"## {self.sel['id']} · ", self._log())
        i = next(k for k, ln in enumerate(log) if ln.startswith("| eval |"))
        self.assertEqual(
            self._cells(log[i]),
            ["eval", "cand", "mix_aic (us)", "workspace (MiB)",
             "ratchet", "front", "n", "note"],
            "the unit belongs in the header so the cells stay numeric")
        self.assertTrue(all(set(c) <= set("-:") for c in self._cells(log[i + 1])),
                        "the separator row is what makes the line above a header")

        rows = {self._cells(ln)[1].strip("`"): self._cells(ln)
                for ln in log if ln.startswith("| ")}
        self.assertEqual(rows[fast[:8]],
                         [str(2), f"`{fast[:8]}`", "555.19", "519.0",
                          "kept ↑", "inserted", "0", ""])
        self.assertEqual(rows[slow[:8]][4:7], ["reverted", "dominated", "1"])
        # Every row has the same number of cells, or it is not a table.
        widths = {len(self._cells(ln)) for ln in log if ln.startswith("| ")}
        self.assertEqual(widths, {8})

    def test_a_note_is_a_row_so_it_keeps_its_place_in_the_sequence(self):
        h = self._record("128, 128", 555.193, 519 << 20)
        run_cli("record", "--op-dir", str(self.op_dir), "--parent", self.sel["id"],
                "--code-hash", h, "--s", "1", "--p", "555.193",
                "--workspace-b", str(519 << 20))
        note = next(ln for ln in self._log().splitlines() if "duplicate" in ln)
        cells = self._cells(note)
        self.assertEqual(cells[0], "·")
        self.assertEqual(cells[1:-1], [""] * 6, "a note fills no data column")
        self.assertIn(h[:8], cells[-1])
        self.assertEqual(len(cells), 8)

    def test_an_incorrect_candidate_is_charged_and_says_so_in_its_own_row(self):
        """
        s=0 is a measurement. The device ran the test and the answer was
        wrong; only a candidate that never reached the device is free.

        The note rides in the candidate's own row rather than in one below it:
        a separate row for a fact about the row above reads as if something
        else happened in between.
        """
        before = json.loads(
            (self.op_dir / "optimization" / "search_state.json").read_text()
        )["progress"]["evals_used"]
        self.f.write_text(KERNEL.replace("64, 128", "16, 128"), encoding="utf-8")
        h = file_hash(self.f)
        run_cli("record", "--op-dir", str(self.op_dir), "--parent", self.sel["id"],
                "--code-hash", h, "--s", "0")
        after = json.loads(
            (self.op_dir / "optimization" / "search_state.json").read_text()
        )["progress"]["evals_used"]
        self.assertEqual(after, before + 1, "an incorrect candidate is charged")

        row = next(self._cells(ln) for ln in self._log().splitlines()
                   if ln.startswith("| ") and h[:8] in ln)
        self.assertEqual(row, [str(after), f"`{h[:8]}`", "—", "—",
                               "reverted", "—", "1", "incorrect (s=0)"])
        self.assertNotIn("| · ", self._log(),
                         "it needs no note row of its own")

    def test_a_note_cell_can_never_break_the_table(self):
        from _support import harness
        self.assertEqual(harness.md_cell("a | b"), "a \\| b")
        self.assertEqual(harness.md_cell("two\nlines"), "two lines")

    def test_a_close_is_followed_by_a_status_line(self):
        self._close_an_action()
        status = next(ln for ln in self._log().splitlines() if ln.startswith("— "))
        self.assertIn("/120 evals", status)
        self.assertIn("best 555.193 us", status)
        self.assertIn("front ", status)
        self.assertIn("stagnation ", status)
        self.assertIn("wall ", status)

    def test_evolve_reasons_get_a_line_each(self):
        self._close_an_action()
        run_cli("evolve", "--op-dir", str(self.op_dir), "--edits",
                json.dumps({"update": [{"id": "u2", "V": 0.3}],
                            "reasons": ["the S2T axis is settled",
                                        "the vector pipe is the imbalance"]}))
        lines = [ln for ln in self._log().splitlines() if ln.strip().startswith("> ")]
        self.assertEqual([ln.strip() for ln in lines],
                         ["> the S2T axis is settled",
                          "> the vector pipe is the imbalance"])

    def test_one_reason_given_as_a_string_is_not_spelt_out_letter_by_letter(self):
        self._close_an_action()
        run_cli("evolve", "--op-dir", str(self.op_dir), "--edits",
                json.dumps({"update": [], "reasons": "one whole sentence"}))
        lines = [ln.strip() for ln in self._log().splitlines() if ln.strip().startswith("> ")]
        self.assertEqual(lines, ["> one whole sentence"])

    def _log(self):
        return next((self.op_dir / "optimization").glob("*.md")).read_text()

    def _close_an_action(self):
        """`n` counts measurements that did not improve, so two are needed."""
        self._record("128, 128", 555.193, 519 << 20)
        self._record("96, 128", 681.738, 1440 << 20)
        self._record("32, 128", 698.311, 1500 << 20)
        run_cli("close", "--op-dir", str(self.op_dir), "--action", self.sel["id"])

    def _record(self, tile, p, ws):
        return record_measurement(self, KERNEL.replace("64, 128", tile), p, ws,
                                  parent=self.sel.get("id"))[0]

    def _cells(self, line):
        return [c.strip() for c in line.strip().strip("|").split("|")]


if __name__ == "__main__":
    unittest.main()
