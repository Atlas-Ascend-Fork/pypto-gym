# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check the repository license policy and Python source headers."""

from pathlib import Path
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET

PLUGIN = Path(__file__).resolve().parents[1]
REPO = PLUGIN.parents[2]


class LicenseScopeTests(unittest.TestCase):
    def test_oat_uses_one_repository_rule(self):
        root = ET.parse(REPO / "OAT.xml").getroot()
        self.assertEqual(root.findtext(".//licensefile"), "LICENSE")
        rules = [
            item
            for item in root.findall(".//policyitem")
            if item.attrib.get("type") == "license"
        ]
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].attrib["name"], "CANN-2.0")
        self.assertEqual(rules[0].attrib["path"], ".*")

    def test_plugin_python_files_have_repository_header(self):
        git = shutil.which("git")
        self.assertIsNotNone(git)
        paths = subprocess.check_output(
            [git, "-C", str(REPO), "ls-files", "-z", "--", "*.py", "*.pyi"]
        )
        for name in filter(None, paths.decode().split("\0")):
            if not name.startswith(
                (
                    "cannbot-skills/plugins-official/pypto-pro-op-orchestrator/",
                    "cannbot-skills/ops/pypto-pro-kernel/",
                )
            ):
                continue
            path = REPO / name
            if not path.is_file():
                continue
            lines = path.read_text(encoding="utf-8").splitlines()[:25]
            header = "\n".join(lines)
            with self.subTest(path=name):
                self.assertIn(
                    "CANN Open Software License Agreement Version 2.0", header
                )
                self.assertIn("This program is free software", header)
                self.assertTrue(any(line.startswith("# Copyright") for line in lines))
                self.assertTrue(
                    any(
                        line.startswith(("# See LICENSE", "# See the License"))
                        for line in lines
                    )
                )


if __name__ == "__main__":
    unittest.main()
