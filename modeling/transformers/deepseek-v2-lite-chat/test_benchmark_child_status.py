# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host regression using real child processes; no inference or NPU is loaded."""

import itertools
import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).with_name("benchmark_bs_comparison.py").resolve()
CHILD = """import json
import sys
from pathlib import Path

args = sys.argv[1:]
Path("request.json").write_text(json.dumps(args), encoding="utf-8")
stream, code = args[args.index("--prompt") + 1].split(":")
if stream != "missing":
    target = sys.stderr if stream == "stderr" else sys.stdout
    print("吞吐量: 5.0 tokens/s\\n推理耗时: 2.0 s\\n生成token数: 10", file=target)
sys.exit(int(code))
"""


class BenchmarkChildStatusTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.model = Path(temporary.name)
        self.addCleanup(os.chdir, Path.cwd())
        environment = patch.dict(os.environ, {"MODEL_PATH": str(self.model)})
        environment.start()
        self.addCleanup(environment.stop)
        scripts = self.model / "scripts"
        scripts.mkdir()
        (scripts / "ask_DeepSeek-V2-Lite-Chat.py").write_text(CHILD, encoding="utf-8")
        self.benchmark = runpy.run_path(str(SOURCE))

    def run_case(self, code, fused, stream):
        result = self.benchmark["run_test"](fused, 10, f"{stream}:{code}", device=3)
        args = json.loads((self.model / "request.json").read_text(encoding="utf-8"))
        self.assertEqual("--use_kv_fusion" in args, fused)
        self.assertIn("--use_acl_graph", args)
        self.assertEqual(args[args.index("--device") + 1], "3")
        self.assertEqual(args[args.index("--output_length") + 1], "10")
        self.assertEqual(
            result["mode"], "PyPTO ACLGraph" if fused else "Baseline ACLGraph"
        )
        return result

    def test_nonzero_child_exit_rejects_even_complete_metrics(self):
        for code, fused, stream in itertools.product(
            (1, 7, 137), (False, True), ("stdout", "stderr")
        ):
            with self.subTest(code=code, fused=fused, stream=stream):
                result = self.run_case(code, fused, stream)
                self.assertFalse(result["success"])
                self.assertIn(str(code), result["error"])
                self.assertNotIn("throughput", result)
                self.assertNotIn("inference_time", result)
                self.assertNotIn("tokens", result)

    def test_successful_child_keeps_metrics_from_both_streams(self):
        for fused, stream in itertools.product((False, True), ("stdout", "stderr")):
            with self.subTest(fused=fused, stream=stream):
                result = self.run_case(0, fused, stream)
                self.assertTrue(result["success"])
                self.assertEqual(result["throughput"], 5.0)
                self.assertEqual(result["inference_time"], 2.0)
                self.assertEqual(result["tokens"], 10)
                self.assertGreaterEqual(result["total_time"], 0.0)

    def test_zero_exit_without_metrics_remains_a_failure(self):
        for fused in (False, True):
            with self.subTest(fused=fused):
                result = self.run_case(0, fused, "missing")
                self.assertFalse(result["success"])
                self.assertEqual(result["error"], "Failed to parse metrics")


if __name__ == "__main__":
    unittest.main()
