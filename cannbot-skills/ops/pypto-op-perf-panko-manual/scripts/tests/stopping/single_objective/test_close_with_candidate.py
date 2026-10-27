# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""`close` on an action that produced a candidate -- the ordinary case.

657bf83 renamed the local `J` to `j` and, with it, the dict key read in the
close log line, so every close that had a best candidate raised KeyError after
saving the state and before emitting its directive. No test closed an action
with a candidate, so nothing noticed.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import DEMO_KERNEL as KERNEL, file_hash, INIT_ARGS, new_op_dir, run_cli  # noqa: E402


class CloseWithCandidateTest(unittest.TestCase):
    def setUp(self):
        self.op_file = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo",
                # The preopt must not already meet the target, or `init`
                # refuses (test_target.py). The candidate below measures 80 and
                # does meet it, which is what releases the close.
                "--p-ref", "100", "--preopt-p", "120",
                "--preopt-hash", file_hash(self.op_file),
                "--op-file", str(self.op_file),
                "--allow-unversioned", *INIT_ARGS)

    def test_close_emits_its_directive_and_promotes_the_winner(self):
        self.op_file.write_text(KERNEL.replace("64, 128", "128, 128"), encoding="utf-8")
        h = file_hash(self.op_file)
        run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                "--code-hash", h, "--s", "1", "--p", "80")
        res, rc = run_cli("close", "--op-dir", str(self.op_dir), "--action", "u1")
        self.assertEqual(rc, 0)
        self.assertEqual(res["best_aicore_e2e_us"], 80.0)
        st = json.loads((self.op_dir / "optimization" / "search_state.json")
                        .read_text(encoding="utf-8"))
        self.assertEqual(st["best_node"], res["closed"])
        log = next((self.op_dir / "optimization").glob("*.md")).read_text()
        self.assertIn(f"closed → {res['closed']} · best 80.0 us (j 125.0)", log)


if __name__ == "__main__":
    unittest.main()
