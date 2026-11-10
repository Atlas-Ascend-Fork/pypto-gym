# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Release scope checks without compiling kernels or requiring an NPU."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "cmake/scripts/select_run_ops.py"
spec = importlib.util.spec_from_file_location("gym_run_scope", SCRIPT)
scope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scope)

# Importing the CLI does not build or initialize a device.
sys.path.insert(0, str(ROOT))
from build_ci import BuildCtrl


class RunBuildScopeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gym_run_scope_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        for name in ("example_op", "mega_mhc", "unlisted_op"):
            op = self.root / "ops/pro/experimental" / name
            (op / "op_host").mkdir(parents=True)
            (op / "tests").mkdir()
            (op / "CMakeLists.txt").write_text("# binary operator fixture\n")
        self.manifest = self.root / "ops/pro/build_ops.yaml"
        self.manifest.write_text("- experimental/example_op\n- experimental/mega_mhc\n")
        script_dir = self.root / "cmake/scripts"
        script_dir.mkdir(parents=True)
        shutil.copy2(SCRIPT, script_dir / SCRIPT.name)

    def test_all_only_includes_uncommented_entries(self):
        self.assertEqual(scope.select_run_ops(self.root), ["example_op", "mega_mhc"])
        self.manifest.write_text("- experimental/example_op\n# - experimental/mega_mhc\n")
        self.assertEqual(scope.select_run_ops(self.root, "all"), ["example_op"])

    def test_explicit_list_bypasses_manifest(self):
        self.manifest.unlink()
        self.assertEqual(scope.select_run_ops(self.root, "mega_mhc,example_op,mega_mhc"),
                         ["example_op", "mega_mhc"])
        self.assertEqual(scope.select_run_ops(self.root, "mega_mhc;example_op"),
                         ["example_op", "mega_mhc"])

    def test_missing_and_empty_scope_fail_closed(self):
        self.manifest.unlink()
        with self.assertRaises(ValueError):
            scope.select_run_ops(self.root)
        for text in ("", "# - experimental/example_op\n# - experimental/mega_mhc\n"):
            with self.subTest(text=text):
                self.manifest.write_text(text)
                with self.assertRaisesRegex(ValueError, "no operators listed"):
                    scope.select_run_ops(self.root)

    def test_comments_blank_lines_and_duplicates(self):
        self.manifest.write_text("# release scope\n\n- experimental/example_op # included\n"
                                 "# - experimental/mega_mhc\n- experimental/example_op\n")
        self.assertEqual(scope.select_run_ops(self.root), ["example_op"])

    def test_malformed_manifest_is_rejected(self):
        for text in ("experimental/example_op\n",
                     "operators: [experimental/example_op]\n",
                     "- [broken\n", "- true\n", "- 123\n", "- null\n", "- ''\n",
                     "- /experimental/example_op\n"):
            with self.subTest(text=text):
                self.manifest.write_text(text)
                with self.assertRaises(ValueError):
                    scope.select_run_ops(self.root)

    def test_unknown_and_unready_operators_are_rejected(self):
        for name in ("typo_op", "../example_op"):
            with self.subTest(name=name):
                self.manifest.write_text(f"- {name}\n")
                with self.assertRaises(ValueError):
                    scope.select_run_ops(self.root)
        (self.root / "ops/pro/experimental/mega_mhc/CMakeLists.txt").unlink()
        with self.assertRaisesRegex(ValueError, "no binary build directory"):
            scope.select_run_ops(self.root, "mega_mhc")

    def test_paths_disambiguate_all_but_explicit_names_must_be_unique(self):
        duplicate = self.root / "ops/tensor/experimental/mega_mhc"
        (duplicate / "op_host").mkdir(parents=True)
        (duplicate / "CMakeLists.txt").write_text("# ambiguous fixture\n")
        self.assertEqual(scope.select_run_ops(self.root), ["example_op", "mega_mhc"])
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            scope.select_run_ops(self.root, "mega_mhc")

    def test_stest_uses_run_manifest_and_keeps_whl_scope(self):
        ctrl = BuildCtrl.__new__(BuildCtrl)
        ctrl.src_root = self.root
        ctrl.feature = SimpleNamespace(ops="ALL", pkg_type_run=True)
        self.manifest.write_text("# - experimental/example_op\n- experimental/mega_mhc\n")
        self.assertEqual([p.name for p in ctrl.selected_op_dirs()], ["mega_mhc"])
        ctrl.feature.ops = "example_op"
        self.assertEqual([p.name for p in ctrl.selected_op_dirs()], ["example_op"])
        ctrl.feature = SimpleNamespace(ops="ALL", pkg_type_run=False)
        self.assertEqual([p.name for p in ctrl.selected_op_dirs()],
                         ["example_op", "mega_mhc", "unlisted_op"])

    def test_default_bindings_fall_back_to_aclnn_when_unsupported(self):
        ctrl = BuildCtrl.__new__(BuildCtrl)
        ctrl.src_root = self.root
        ctrl.feature = SimpleNamespace(ops="example_op", pkg_type_run=True,
                                       torch_bindings=True, torch_bindings_explicit=False)
        ctrl.resolve_torch_bindings()
        self.assertFalse(ctrl.feature.torch_bindings)

        binding = self.root / "ops/pro/experimental/example_op/torch_extension"
        binding.mkdir()
        (binding / "CMakeLists.txt").write_text("# binding fixture\n")
        ctrl.feature.torch_bindings = True
        ctrl.resolve_torch_bindings()
        self.assertTrue(ctrl.feature.torch_bindings)

        # An explicit --torch-bindings request is still validated by CMake.
        ctrl.feature = SimpleNamespace(ops="mega_mhc", pkg_type_run=True,
                                       torch_bindings=True, torch_bindings_explicit=True)
        ctrl.resolve_torch_bindings()
        self.assertTrue(ctrl.feature.torch_bindings)

    def test_cmake_ignores_unselected_orphans(self):
        cmake = BuildCtrl.which_cmake()
        if not cmake:
            self.skipTest("CMake is unavailable")
        # Sort the selected operator last so the previous discovery loop leaves
        # _op_dir pointing at it. The orphan check must use its own directory.
        selected = self.root / "ops/pro/experimental/z_selected"
        (selected / "op_host").mkdir(parents=True)
        (selected / "CMakeLists.txt").write_text("# selected operator\n")
        orphan = self.root / "ops/pro/experimental/a_unlisted"
        (orphan / "op_host").mkdir(parents=True)
        (orphan / "op_host/CMakeLists.txt").write_text("# incomplete operator\n")
        self.manifest.write_text("- experimental/z_selected\n")
        (self.root / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.18)
project(scope_check NONE)
set(Python3_EXECUTABLE "{sys.executable}")
include("{ROOT / 'cmake/func.cmake'}")
gym_op_discovery(OP_LIST OP_DIR_LIST)
file(WRITE "${{CMAKE_BINARY_DIR}}/selected.txt" "${{OP_LIST}}")
''')
        build = Path(self.temp.name) / "orphan_build"
        for requested in ("ALL", "z_selected"):
            with self.subTest(requested=requested):
                result = subprocess.run(
                    [str(cmake), "-S", str(self.root), "-B", str(build),
                     f"-DASCEND_OP_NAME={requested}"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual((build / "selected.txt").read_text(), "z_selected")

        # Ignoring an unselected directory must not allow publishing it later.
        self.manifest.write_text("- experimental/a_unlisted\n")
        for requested in ("ALL", "a_unlisted"):
            with self.subTest(requested=requested):
                result = subprocess.run(
                    [str(cmake), "-S", str(self.root), "-B", str(build),
                     f"-DASCEND_OP_NAME={requested}"], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("no binary build directory",
                              " ".join((result.stdout + result.stderr).split()))

    def test_cmake_tracks_manifest_changes(self):
        cmake = BuildCtrl.which_cmake()
        if not cmake:
            self.skipTest("CMake is unavailable")
        (self.root / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.18)
project(scope_check NONE)
set(Python3_EXECUTABLE "{sys.executable}")
include("{ROOT / 'cmake/func.cmake'}")
gym_op_discovery(OP_LIST OP_DIR_LIST)
file(WRITE "${{CMAKE_BINARY_DIR}}/selected.txt" "${{OP_LIST}}")
''')
        build = Path(self.temp.name) / "build"
        subprocess.run([str(cmake), "-S", str(self.root), "-B", str(build)],
                       check=True, capture_output=True, text=True)
        self.assertEqual((build / "selected.txt").read_text(), "example_op;mega_mhc")
        # CMake's Makefile timestamp comparison may have one-second resolution.
        time.sleep(1.1)
        self.manifest.write_text("- experimental/example_op\n# - experimental/mega_mhc\n")
        subprocess.run([str(cmake), "--build", str(build)],
                       check=True, capture_output=True, text=True)
        self.assertEqual((build / "selected.txt").read_text(), "example_op")
        time.sleep(1.1)
        self.manifest.write_text("# - experimental/example_op\n# - experimental/mega_mhc\n")
        result = subprocess.run([str(cmake), "--build", str(build)],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no operators listed", " ".join((result.stdout + result.stderr).split()))
        subprocess.run([str(cmake), "-S", str(self.root), "-B", str(build),
                        "-DASCEND_OP_NAME=mega_mhc"], check=True, capture_output=True, text=True)
        self.assertEqual((build / "selected.txt").read_text(), "mega_mhc")


if __name__ == "__main__":
    unittest.main()
