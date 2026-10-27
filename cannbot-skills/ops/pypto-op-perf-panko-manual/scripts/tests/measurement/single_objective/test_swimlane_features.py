# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""`pipe_usage.csv` reaches the swimlane readout, not just its parser.

`test_pypto_reuse.py` covers `pipe_usage` and `find_pipe_usage` directly, and
kept passing while `features()` had stopped calling them: the pipe utilisation
was gone from every normal output and no test noticed, because none of them
went in through the entry points a caller uses. These do -- `features()` and
the CLI -- so a parser wired to nothing fails here.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))
import swimlane  # noqa: E402

SWIMLANE = Path(__file__).parents[3] / "swimlane.py"

# One task on each pipe, and the metadata records that name their lanes.
TRACE = {"traceEvents": [
    {"ph": "M", "name": "thread_name", "pid": 1, "tid": 1, "args": {"name": "AIC_0"}},
    {"ph": "M", "name": "thread_name", "pid": 1, "tid": 2, "args": {"name": "AIV_0"}},
    {"ph": "X", "pid": 1, "tid": 1, "ts": 0.0, "dur": 300.0, "name": "cube"},
    {"ph": "X", "pid": 1, "tid": 2, "ts": 0.0, "dur": 160.0, "name": "vec"},
]}

# pypto's `calculate_pipe_usage` output, abridged to the block this reads.
PIPE_USAGE_CSV = "\n".join([
    "Total Pipe Usage",
    "Pipe, AverageTime, TotalExecuteTime, AverageUsage",
    "CUBE, 305.2, 412.5, 73.99%",
    "VECTOR_ALU, 160.9, 412.5, 39.01%",
    "MTE_IN, 363.0, 412.5, 88.0%",
    "MTE1, 49.5, 412.5, 12.0%",
    "MTE_OUT, 86.6, 412.5, 21.0%",
    "",
])

EXPECTED = {"pipe_cube_frac": 0.7399, "pipe_vector_alu_frac": 0.3901,
            "pipe_mte_in_frac": 0.88, "pipe_mte1_frac": 0.12,
            "pipe_mte_out_frac": 0.21}


class FeaturesTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        self.trace = self.out / "merged_swimlane.json"
        self.trace.write_text(json.dumps(TRACE), encoding="utf-8")

    def test_features_carries_the_pipe_usage_beside_the_trace(self):
        (self.out / "pipe_usage.csv").write_text(PIPE_USAGE_CSV, encoding="utf-8")
        got = swimlane.features(str(self.trace))
        self.assertEqual({k: got.get(k) for k in EXPECTED}, EXPECTED)
        self.assertEqual(got["pipe_usage_source"], "pipe_usage.csv")

    def test_the_cli_reports_them_too(self):
        """The readout a caller actually runs."""
        (self.out / "pipe_usage.csv").write_text(PIPE_USAGE_CSV, encoding="utf-8")
        r = subprocess.run([sys.executable, str(SWIMLANE), "--trace", str(self.trace)],
                           capture_output=True, text=True)
        feats = json.loads(r.stdout)["features"]
        self.assertEqual({k: feats.get(k) for k in EXPECTED}, EXPECTED)

    def test_without_the_file_the_trace_features_still_come_back(self):
        """`--gen_exe_topo_json` is not on by default, so absence is the norm."""
        got = swimlane.features(str(self.trace))
        self.assertIsNotNone(got["span_us"])
        self.assertNotIn("pipe_usage_source", got)
        for k in EXPECTED:
            self.assertNotIn(k, got)


if __name__ == "__main__":
    unittest.main()
