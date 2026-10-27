# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Every measurement can be attributed to the catalogue action that produced it.

A closed node kept only its x-id and no log line named the catalogue id, so the
u-id -> action mapping had to be rebuilt from mutually inconsistent seeding
orders, and some measurements could not be attributed at all.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import DEMO_KERNEL as KERNEL, file_hash, INIT_ARGS, new_op_dir, read_state, run_cli  # noqa: E402


class AttributionTest(unittest.TestCase):
    def setUp(self):
        self.op_file = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo",
                "--p-ref", "50", "--preopt-p", "100",
                "--preopt-hash", file_hash(self.op_file),
                "--op-file", str(self.op_file), "--allow-unversioned", *INIT_ARGS)

    def test_select_opens_a_section_for_the_action(self):
        sel = run_cli("select", "--op-dir", str(self.op_dir))[0]["selected"]
        node = next(n for n in read_state(self.op_dir)["nodes"]["open"] if n["id"] == sel["id"])
        log = self._log()
        self.assertIn(f"## {sel['id']} · {node.get('catalog_id') or 'insert'} · ", log)
        self.assertIn("selected at eval ", log)

    def test_select_records_every_candidate_it_scored(self):
        """
        The decision, not only its winner: `did X change what was selected?`
        was unanswerable without this.
        """
        res = run_cli("select", "--op-dir", str(self.op_dir))[0]
        sel = res["selected"]
        selects = self.op_dir / "optimization" / "selects.jsonl"
        rows = [json.loads(ln) for ln in selects.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        rec = rows[0]
        self.assertEqual(rec["chosen"], sel["id"])
        self.assertEqual(res["candidates"], len(rec["candidates"]))
        live = {n["id"] for n in read_state(self.op_dir)["nodes"]["open"]}
        self.assertEqual({c["id"] for c in rec["candidates"]}, live,
                         "the whole frontier it chose from")
        for key in ("V", "g", "w_stage", "V_sel", "stage"):
            self.assertIn(key, rec["candidates"][0])
        top = max(rec["candidates"], key=lambda c: c["V_sel"])
        self.assertEqual(top["id"], sel["id"], "the record explains the argmax")
        self.assertNotIn("phase", rec, "there is one search since C2")

    def test_each_select_appends_one_record(self):
        run_cli("select", "--op-dir", str(self.op_dir))
        sel = run_cli("select", "--op-dir", str(self.op_dir))[0]["selected"]
        rows = (self.op_dir / "optimization" / "selects.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 2)
        self.assertEqual(json.loads(rows[-1])["chosen"], sel["id"])

    def test_the_log_line_carries_the_runner_up(self):
        run_cli("select", "--op-dir", str(self.op_dir))
        rec = json.loads((self.op_dir / "optimization" / "selects.jsonl")
                         .read_text().splitlines()[0])
        line = next(ln for ln in self._log().splitlines()
                    if ln.strip().startswith("selected at eval"))
        self.assertIn(f"of {len(rec['candidates'])} candidates", line)
        up = max((c for c in rec["candidates"] if c["id"] != rec["chosen"]),
                 key=lambda c: c["V_sel"])
        self.assertIn(f"next {up['catalog_id'] or up['id']} {up['V_sel']}", line)

    def test_a_closed_node_keeps_its_action(self):
        sel = run_cli("select", "--op-dir", str(self.op_dir))[0]["selected"]
        cat = next(n for n in read_state(self.op_dir)["nodes"]["open"]
                   if n["id"] == sel["id"]).get("catalog_id")
        self.op_file.write_text(KERNEL.replace("64, 128", "128, 128"), encoding="utf-8")
        run_cli("record", "--op-dir", str(self.op_dir), "--parent", sel["id"],
                "--code-hash", file_hash(self.op_file), "--s", "1", "--p", "40")
        res, rc = run_cli("close", "--op-dir", str(self.op_dir), "--action", sel["id"])
        self.assertEqual(rc, 0, res)
        node = next(n for n in read_state(self.op_dir)["nodes"]["closed"] if n["id"] == res["closed"])
        self.assertEqual(node["action"], sel["id"])
        self.assertEqual(node["catalog_id"], cat)
        self.assertEqual(node["delta"], sel["delta"])
        log = self._log()
        self.assertIn(f"## {sel['id']} · {cat or 'insert'} · ", log,
                      "the close belongs to the action's own section")
        self.assertIn("closed → ", log)

    def _log(self):
        return next((self.op_dir / "optimization").glob("*.md")).read_text()


if __name__ == "__main__":
    unittest.main()
