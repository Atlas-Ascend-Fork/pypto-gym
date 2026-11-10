# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify the installed pypto_gym namespace imports without torch."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "torch_extension" / "pypto_gym"


class TorchBindingLazyImportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gym_lazy_binding_")
        self.addCleanup(self.temp.cleanup)
        self.site = Path(self.temp.name) / "site-packages"
        self.pkg = self.site / "pypto_gym"
        self.pkg.mkdir(parents=True)
        for name in ("__init__.py", "_loader.py"):
            shutil.copy2(PACKAGE / name, self.pkg / name)
        (self.pkg / "operators.json").write_text(json.dumps(["example_op"]) + "\n")
        frontends = self.pkg / "_frontends"
        frontends.mkdir()
        (frontends / "example_op.py").write_text("def example_op(*args, **kwargs):\n"
                                                 "    return 'loaded'\n")

    def test_import_without_torch_is_allowed(self):
        result = self._run(
            "import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            raise ImportError('torch is blocked for this test')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Block())\n"
            "import pypto_gym\n"
            "print('IMPORT_OK', pypto_gym.__all__)\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("IMPORT_OK", result.stdout)
        self.assertIn("example_op", result.stdout)

    def test_lazy_access_reports_missing_torch(self):
        result = self._run(
            "import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            raise ImportError('torch is blocked for this test')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Block())\n"
            "import pypto_gym\n"
            "try:\n"
            "    pypto_gym.example_op\n"
            "except ImportError as error:\n"
            "    print('LAZY_IMPORT_ERROR', 'blocked' in str(error))\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("LAZY_IMPORT_ERROR True", result.stdout)

    def _run(self, body):
        return subprocess.run(
            [sys.executable, "-c", body], cwd=self.temp.name,
            env={"PYTHONPATH": str(self.site)},
            capture_output=True, text=True)

if __name__ == "__main__":
    unittest.main()
