# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise the run installer (pip shell-wheel registration) in temporary CANN
and custom directories. Decision 29/30: the run payload embeds a python shell
wheel; the installer pip-installs it into <cann_root>/python/site-packages for
a standard CANN layout, and degrades with a warning otherwise."""
import base64
import hashlib
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]

_WHEEL_NAME = "pypto_gym-0.0.0-py3-none-any.whl"


def make_shell_wheel(path: Path, init_body: str = 'VALUE = "installed"\n'):
    """Hand-craft a minimal valid wheel (shell package + dist-info with RECORD)."""
    dist_info = "pypto_gym-0.0.0.dist-info"
    files = {
        "pypto_gym/__init__.py": init_body,
        "pypto_gym/_bindings/operators.json": '["example_op"]\n',
    }
    metadata = "Metadata-Version: 2.1\nName: pypto-gym\nVersion: 0.0.0\n"
    wheel_meta = ("Wheel-Version: 1.0\nGenerator: gym-test\n"
                  "Root-Is-Purelib: true\nTag: py3-none-any\n")
    records = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for arc, content in list(files.items()) + [
                (f"{dist_info}/METADATA", metadata), (f"{dist_info}/WHEEL", wheel_meta)]:
            zf.writestr(arc, content)
            data = content.encode()
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            records.append(f"{arc},sha256={digest},{len(data)}")
        records.append(f"{dist_info}/RECORD,,")
        zf.writestr(f"{dist_info}/RECORD", "\n".join(records) + "\n")


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
        (self.binding / "_loader").mkdir()
        (self.binding / "_loader" / "__init__.py").write_text("# loader fixture\n")
        (self.binding / "_bindings").mkdir()
        (self.binding / "_bindings" / "operators.json").write_text('["example_op"]\n')
        (self.binding / "_bindings" / "csrc").mkdir()
        (self.binding / "_bindings" / "frontends").mkdir()
        make_shell_wheel(self.payload / f"packages/vendors/pypto_gym/python/{_WHEEL_NAME}")
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

    def test_standard_cann_pip_registration(self):
        self.run_installer()
        init = self.public / "__init__.py"
        self.assertTrue(init.is_file())
        self.assertFalse(init.is_symlink())
        self.assertTrue(list(self.site.glob("pypto_gym-*.dist-info")))
        result = subprocess.run(
            [sys.executable, "-c", "import pypto_gym; print(pypto_gym.VALUE)"],
            cwd=self.root, env=dict(self.env, PYTHONPATH=str(self.site)),
            check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "installed")
        self.run_installer("--uninstall")
        self.assertFalse(self.public.exists())
        self.assertFalse(list(self.site.glob("pypto_gym-*.dist-info")))
        self.assertFalse(self.vendor.exists())

    def test_custom_install_keeps_wheel_for_manual_pip(self):
        custom = self.root / "custom"
        custom.mkdir()
        self.run_installer(f"--install-path={custom}")
        self.assertFalse(self.public.exists())
        vendor = custom / "vendors/pypto_gym"
        self.assertTrue((vendor / f"python/{_WHEEL_NAME}").is_file())
        self.assertTrue((vendor / "python/pypto_gym/_bindings/operators.json").is_file())
        set_env = (vendor / "bin/set_env.bash").read_text()
        self.assertIn("ASCEND_CUSTOM_OPP_PATH", set_env)
        self.assertNotIn("PYTHONPATH", set_env)  # decision 29: pip carries the Python layer
        self.run_installer("--uninstall", f"--install-path={custom}")
        self.assertFalse(vendor.exists())

    def test_missing_whl_degrades_with_warning(self):
        (self.payload / f"packages/vendors/pypto_gym/python/{_WHEEL_NAME}").unlink()
        result = self.run_installer()
        self.assertFalse(self.public.exists())
        self.assertTrue((self.vendor / "python/pypto_gym/_bindings/operators.json").is_file())
        self.assertIn("not registered", result.stdout + result.stderr)

    def test_unwritable_site_packages_degrades_to_vendor_only(self):
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
        self.assertTrue((self.vendor / "python/pypto_gym/_bindings/operators.json").is_file())
        self.assertIn("not registered", result.stdout + result.stderr)

    def test_install_for_all_readable_with_restrictive_umask(self):
        previous_umask = os.umask(0o077)
        try:
            result = subprocess.run(
                [str(self.scripts / "install.sh"), "--fixture.run", "--" + str(self.root),
                 "--install-for-all"], env=self.env, capture_output=True, text=True)
        finally:
            os.umask(previous_umask)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        mode = (self.public / "__init__.py").stat().st_mode
        self.assertTrue(mode & 0o004, f"installed file is not world-readable: {oct(mode)}")
        self.run_installer("--uninstall")

    def test_uninstall_preserves_foreign_files(self):
        # RECORD-precise removal: files not owned by our distribution survive
        # uninstall (namespace replacement at install time is intended,
        # decisions 24/25; uninstall must still not rm -rf blindly).
        (self.opp / "vendors").mkdir()
        (self.opp / "vendors/config.ini").write_text("load_priority=other_vendor\n")
        self.run_installer()
        self.assertTrue((self.public / "__init__.py").is_file())
        foreign = self.public / "foreign.txt"
        foreign.write_text("not ours\n")
        # Runtime imports leave nested bytecode caches that are not in RECORD.
        for sub in ("", "_loader/", "_bindings/frontends/"):
            pycache = self.public / sub / "__pycache__"
            pycache.mkdir(parents=True, exist_ok=True)
            (pycache / "mod.cpython-312.pyc").write_bytes(b"\x00")
        result = subprocess.run(
            [str(self.vendor / "scripts/uninstall.sh")],
            env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(foreign.is_file())              # foreign file survives
        self.assertFalse((self.public / "__init__.py").exists())
        self.assertFalse((self.public / "_loader").exists())
        self.assertFalse(list(self.site.glob("pypto_gym-*.dist-info")))
        self.assertFalse(self.vendor.exists())
        self.assertEqual((self.opp / "vendors/config.ini").read_text(),
                         "load_priority=other_vendor\n")

    def test_unrelated_python_package_is_not_overwritten(self):
        self.public.mkdir()
        existing = self.public / "__init__.py"
        existing.write_text("# another package\n")
        # Registration degrades to a warning (non-fatal); the vendor tree is
        # ours and is replaced as usual on install/upgrade.
        result = self.run_installer()
        self.assertIn("occupied by another package", result.stdout + result.stderr)
        self.assertEqual(existing.read_text(), "# another package\n")
        self.assertTrue((self.vendor / "python/pypto_gym/_bindings/operators.json").is_file())

    def test_upgrade_without_bindings_removes_old_registration(self):
        self.run_installer()
        self.assertTrue((self.public / "__init__.py").is_file())
        shutil.rmtree(self.binding.parent)  # python/ tree incl. operators.json + whl
        self.run_installer()
        self.assertFalse(self.public.exists())
        self.assertTrue(self.vendor.is_dir())

    def test_legacy_symlink_tree_is_replaced_by_pip_install(self):
        legacy_target = self.root / "legacy"
        legacy_target.mkdir()
        (legacy_target / "__init__.py").write_text('VALUE = "legacy"\n')
        self.public.mkdir(parents=True)
        (self.public / "__init__.py").symlink_to(legacy_target / "__init__.py")
        self.run_installer()
        init = self.public / "__init__.py"
        self.assertFalse(init.is_symlink())
        self.assertEqual(init.read_text(), 'VALUE = "installed"\n')
        self.run_installer("--uninstall")
        self.assertFalse(self.public.exists())

    def test_no_bindings_preserves_unowned_empty_namespace(self):
        self.public.mkdir()
        shutil.rmtree(self.binding.parent)
        self.run_installer()
        self.run_installer("--uninstall")
        self.assertTrue(self.public.is_dir())


if __name__ == "__main__":
    unittest.main()
