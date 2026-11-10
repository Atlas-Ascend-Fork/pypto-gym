# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the run installer in temporary CANN and custom directories."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class RunInstallTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gym_run_install_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cann = self.root / "cann"
        self.opp = self.cann / "opp"
        self.opp.mkdir(parents=True)
        (self.cann / "set_env.sh").write_text("# fake CANN environment\n")
        self.site = self.cann / "python/site-packages"
        self.site.mkdir(parents=True)
        self.public = self.site / "pypto_gym"
        self.vendor = self.opp / "vendors/pypto_gym"
        self.payload = self.root / "payload"
        self.scripts = self.payload / "share/info/pypto_gym/scripts"
        shutil.copytree(ROOT / "scripts/package/pypto_gym/scripts", self.scripts)
        os.chmod(self.scripts / "install.sh", 0o755)
        self.binding = self.payload / "packages/vendors/pypto_gym/python/pypto_gym"
        self.binding.mkdir(parents=True)
        (self.binding / "__init__.py").write_text('VALUE = "installed"\n')
        (self.binding / "_loader.py").write_text("# loader fixture\n")
        (self.binding / "operators.json").write_text('["example_op"]\n')
        (self.binding / "_native").mkdir()
        (self.binding / "_frontends").mkdir()
        self.env = dict(os.environ, ASCEND_HOME_PATH=str(self.cann))

    def run_installer(self, *args, success=True):
        result = subprocess.run(
            [str(self.scripts / "install.sh"), "--fixture.run",
             "--" + str(self.root), *args], env=self.env,
            capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_standard_cann_import_without_vendor_pythonpath(self):
        self.run_installer()
        for name in ("__init__.py", "_loader.py", "_native", "_frontends", "operators.json"):
            link = self.public / name
            self.assertTrue(link.is_symlink(), link)
            self.assertFalse(Path(os.readlink(link)).is_absolute())
            self.assertEqual(link.resolve(), (self.vendor / "python/pypto_gym" / name).resolve())
        result = subprocess.run(
            [sys.executable, "-c", "import pypto_gym; print(pypto_gym.VALUE)"],
            cwd=self.root, env=dict(self.env, PYTHONPATH=str(self.site)),
            check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "installed")
        self.run_installer("--uninstall")
        self.assertFalse(self.public.exists())
        self.assertFalse(self.vendor.exists())

    def test_custom_install_does_not_publish_into_cann(self):
        custom = self.root / "custom"
        custom.mkdir()
        self.run_installer(f"--install-path={custom}")
        self.assertFalse(self.public.exists())
        vendor = custom / "vendors/pypto_gym"
        self.assertTrue((vendor / "python/pypto_gym/operators.json").is_file())
        self.assertIn(str(vendor / "python"), (vendor / "bin/set_env.bash").read_text())
        self.run_installer("--uninstall", f"--install-path={custom}")
        self.assertFalse(vendor.exists())

    def test_unwritable_cann_degrades_to_vendor_only(self):
        # Simulate a non-root install into a root-owned CANN: the vendor tree is
        # writable, but python/site-packages is not. Installation must still
        # succeed and point the user at the vendor set_env.bash.
        setpriv = shutil.which("setpriv")
        if os.geteuid() != 0 or not setpriv:
            self.skipTest("requires root and setpriv to drop privileges")
        os.chmod(self.root, 0o755)
        os.chmod(self.cann, 0o755)
        os.chmod(self.opp, 0o777)
        os.chmod(self.site, 0o555)
        self.addCleanup(os.chmod, self.site, 0o755)
        result = subprocess.run(
            [setpriv, "--reuid=65534", "--regid=65534", "--clear-groups",
             str(self.scripts / "install.sh"), "--fixture.run",
             "--" + str(self.root)], env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.public.exists())
        self.assertTrue((self.vendor / "python/pypto_gym/operators.json").is_file())
        self.assertIn("not registered", result.stdout + result.stderr)
        self.assertIn(str(self.vendor / "python"), (self.vendor / "bin/set_env.bash").read_text())

    def test_public_registration_is_readable_with_restrictive_umask(self):
        previous_umask = os.umask(0o077)
        try:
            result = subprocess.run(
                [str(self.scripts / "install.sh"), "--fixture.run", "--" + str(self.root),
                 "--install-for-all"], env=self.env, capture_output=True, text=True)
        finally:
            os.umask(previous_umask)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.public.stat().st_mode & 0o777, 0o755)
        self.run_installer("--uninstall")

    def test_existing_kernel_namespace_survives_upgrade_and_standalone_uninstall(self):
        ops = self.public / "ops"
        ops.mkdir(parents=True)
        marker = ops / "wheel.py"
        marker.write_text("# existing kernel wheel\n")
        (self.opp / "vendors").mkdir()
        (self.opp / "vendors/config.ini").write_text("load_priority=other_vendor\n")
        self.run_installer()
        self.run_installer()  # Upgrade removes and recreates only owned links.
        result = subprocess.run(
            [str(self.vendor / "scripts/uninstall.sh")],
            env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(marker.is_file())
        self.assertFalse((self.public / "__init__.py").is_symlink())
        self.assertFalse(self.vendor.exists())
        self.assertEqual((self.opp / "vendors/config.ini").read_text(), "load_priority=other_vendor\n")

    def test_unrelated_python_package_is_not_overwritten(self):
        self.public.mkdir()
        existing = self.public / "__init__.py"
        existing.write_text("# another package\n")
        self.vendor.mkdir(parents=True)
        old = self.vendor / "old-version"
        old.write_text("keep\n")
        result = self.run_installer(success=False)
        self.assertIn("already owned by another package", result.stderr)
        self.assertEqual(existing.read_text(), "# another package\n")
        self.assertEqual(old.read_text(), "keep\n")

    def test_upgrade_without_bindings_removes_old_registration(self):
        self.run_installer()
        shutil.rmtree(self.binding)
        self.run_installer()
        self.assertFalse(self.public.exists())
        self.assertTrue(self.vendor.is_dir())

    def test_no_bindings_preserves_unowned_empty_namespace(self):
        self.public.mkdir()
        shutil.rmtree(self.binding)
        self.run_installer()
        self.run_installer("--uninstall")
        self.assertTrue(self.public.is_dir())


if __name__ == "__main__":
    unittest.main()
