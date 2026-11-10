# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify the filtered void check used by --ops kernel config generation."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "cmake/scripts/util/ascendc_ops_config.py"
sys.path.insert(0, str(SCRIPT.parent))
try:
    spec = importlib.util.spec_from_file_location("gym_ops_config", SCRIPT)
    ops_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops_config)
finally:
    sys.path.remove(str(SCRIPT.parent))


class OpsConfigVoidTest(unittest.TestCase):
    def test_selected_root_ignores_stale_sibling(self):
        with tempfile.TemporaryDirectory(prefix="gym_ops_config_") as temp:
            bin_dir = Path(temp) / "ascend950" / "bin"
            selected = bin_dir / "selected_op"
            selected.mkdir(parents=True)
            (selected / "kernel.json").write_text("{}\n")
            (bin_dir / "stale_empty_op").mkdir()

            # A filtered build only validates its own operator directory.
            ops_config.check_single_op_is_void(str(selected))

            # An empty selected directory, or the whole unfiltered tree, still fails.
            with self.assertRaisesRegex(RuntimeError, "not any obj compile success"):
                ops_config.check_single_op_is_void(str(bin_dir / "stale_empty_op"))
            with self.assertRaisesRegex(RuntimeError, "not any obj compile success"):
                ops_config.check_single_op_is_void(str(bin_dir))


if __name__ == "__main__":
    unittest.main()
