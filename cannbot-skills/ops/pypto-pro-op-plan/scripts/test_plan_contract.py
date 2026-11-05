#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Regression checks for PyPTO-Pro planning artifacts and their consumers."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[2]
INTENT = OPS_ROOT / "pypto-pro-intent-understand"
MATERIAL = OPS_ROOT / "pypto-pro-material-explore"
PLAN = OPS_ROOT / "pypto-pro-op-plan"
KERNEL = OPS_ROOT / "pypto-pro-kernel"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _report_with_paths(template: str, samples: list[str], tutorials: list[str]) -> str:
    template = template.replace("{kernel_owner_path}", "pypto-pro-kernel/SKILL.md")
    report = re.sub(r"\{[^{}\r\n]+\}", "filled", template).replace(
        "op_name: filled", "op_name: demo",
    )
    report = re.sub(r"(?m)^\|[^\n]*`filled`[^\n]*\n?", "", report)
    sample_rows = "\n".join(
        f"| {index} | `{path}` | `vf.add` |" for index, path in enumerate(samples, 1)
    )
    tutorial_rows = "\n".join(f"| `{path}` | guide | pattern | applicable |" for path in tutorials)
    report = report.replace(
        "### 4.1 全量样例参考（按 cube/vec 组成分类）",
        "### 4.1 全量样例参考（按 cube/vec 组成分类）\n" + sample_rows,
        1,
    )
    return report.replace(
        "### 5.1 适用的设计模式",
        "### 5.1 适用的设计模式\n" + tutorial_rows,
        1,
    )


class PlanContractTests(unittest.TestCase):
    def test_material_scans_all_pro_guide_ranges_and_rejects_missing_sources(self) -> None:
        material = self._load_module(MATERIAL / "scripts/build_material_index.py")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        devkit, guides = self._create_devkit_fixture(root)
        index, counts = material.build_index(devkit, material.manifest_path())
        self.assertEqual(re.findall(r"`(docs/guide/[^`]+\.md)`", index), sorted(guides))
        self.assertEqual(counts[2], len(guides))
        self.assertNotIn("docs/pypto_pro/tutorials/old.md", index)
        for relative in (
            "docs/guide/programming_guide/pro", "docs/guide/quick_start/pro",
            "docs/guide/introduction.md",
            "docs/guide/programming_guide/pro/index.md", "docs/guide/quick_start/pro/index.md",
        ):
            with self.subTest(missing=relative):
                source, backup = devkit / relative, root / "missing source"
                source.rename(backup)
                try:
                    path_pattern = re.escape(relative).replace("/", r"[/\\]")
                    self.assertRaisesRegex(
                        material.MaterialIndexError, path_pattern,
                        material.build_index, devkit, material.manifest_path(),
                    )
                finally:
                    backup.rename(source)

    def test_spec_template_has_no_operator_specific_defaults(self) -> None:
        text = _read(INTENT / "templates" / "spec-template.md")
        for forbidden in ("bfloat16", "1024", "eps", "min_v", "max_v"):
            self.assertNotIn(forbidden, text)
        self.assertEqual(text.count("```json machine-contract"), 1)
        self.assertRegex(text, r'"formula":\s*"\{\{[^{}]+\}\}"')
        self.assertRegex(text, r'"supported_dtypes":\s*\[\]')
        self.assertNotIn('"p0_shapes"', text)
        self.assertRegex(text, r'"default_params":\s*\{\}')
        self.assertTrue(re.search(r"\{\{[^{}]+\}\}", text))

    def test_plan_commands_use_canonical_python_entrypoint(self) -> None:
        paths = list(INTENT.rglob("*.md")) + list(MATERIAL.rglob("*.md")) + list(PLAN.rglob("*.md"))
        for path in paths:
            self.assertNotIn("python3", _read(path), path.as_posix())

    def test_interface_and_current_workflow_handoffs_are_explicit(self) -> None:
        intent = _read(INTENT / "SKILL.md")
        plan = _read(PLAN / "SKILL.md")
        material = _read(MATERIAL / "SKILL.md")
        self.assertIn("公开接口的 rank、shape", intent)
        self.assertIn("rank-0", intent)
        self.assertIn("../pypto-pro-kernel/SKILL.md", plan)
        self.assertIn("../pypto-pro-kernel/references/index.md", plan)
        self.assertIn("适用前提", plan)
        self.assertIn("不受冻结参考清单限制", plan)
        self.assertIn("单一 VF API", material)

    def test_plan_uses_one_plan_validator(self) -> None:
        plan = _read(PLAN / "SKILL.md")
        validator_path = PLAN / "scripts" / "validate_plan.py"
        self.assertTrue(validator_path.is_file())
        self.assertIn("scripts/validate_plan.py", plan)
        self.assertIn("未指定时默认 A5", plan)
        self.assertNotIn("python -c '", plan)

    def test_kernel_owner_references_accept_direct_paths_links_and_anchors(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_plan.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kernel = self._create_kernel_fixture(root)
            op_dir = root / "custom/demo"
            op_dir.mkdir(parents=True)
            report = op_dir / "EXPLORE_REPORT.md"
            report.write_text(
                "`pypto-pro-kernel/references/api/demo.md#lane-map`\n"
                "[second](pypto-pro-kernel/references/api/demo.md#lane-map-1)\n"
                "[entry](pypto-pro-kernel/SKILL.md)\n", encoding="utf-8",
            )
            self.assertEqual(validator.check_kernel_references(op_dir, kernel), [])
            self.assertEqual(sorted(path.name for path in op_dir.iterdir()), ["EXPLORE_REPORT.md"])

    def test_kernel_owner_references_reject_missing_escaping_and_bad_anchors(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_plan.py")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        kernel = self._create_kernel_fixture(root)
        op_dir = root / "custom/demo"
        op_dir.mkdir(parents=True)
        report = op_dir / "EXPLORE_REPORT.md"
        for reference, diagnostic in (
            ("pypto-pro-kernel/references/api/missing.md", "file does not exist"),
            ("pypto-pro-kernel/../outside.md", "escapes its root"),
            ("pypto-pro-kernel/%2E%2E/outside.md", "escapes its root"),
            ("../pypto-pro-kernel/SKILL.md", "must start with"),
            ("pypto-pro-kernel/references/api/demo.md#missing", "anchor does not exist"),
            ("pypto-pro-kernel/scripts/check.py#missing", "requires a Markdown owner"),
        ):
            for citation in (f"`{reference}`", f"[owner]({reference})"):
                with self.subTest(citation=citation):
                    report.write_text(citation, encoding="utf-8")
                    errors = validator.check_kernel_references(op_dir, kernel)
                    self.assertTrue(any(diagnostic in error for error in errors), errors)
        for text in (
            "no sources",
            "<!-- `pypto-pro-kernel/SKILL.md` -->",
            "```text\n`pypto-pro-kernel/SKILL.md`\n```",
            "`cat pypto-pro-kernel/SKILL.md`",
        ):
            with self.subTest(unexecuted_reference=text):
                report.write_text(text, encoding="utf-8")
                self.assertIn("must cite", "\n".join(validator.check_kernel_references(op_dir, kernel)))

    def test_kernel_owner_references_reject_symlink_escape(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_plan.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kernel = self._create_kernel_fixture(root)
            outside = root / "outside.md"
            outside.write_text("# outside\n", encoding="utf-8")
            try:
                (kernel / "references/outside.md").symlink_to(outside)
            except OSError as error:
                self.skipTest(f"filesystem does not permit symlink fixture: {error}")
            op_dir = root / "demo"
            op_dir.mkdir()
            (op_dir / "EXPLORE_REPORT.md").write_text(
                "`pypto-pro-kernel/references/outside.md`", encoding="utf-8",
            )
            errors = validator.check_kernel_references(op_dir, kernel)
            self.assertTrue(any("resolves outside" in error for error in errors), errors)

    def test_explicit_owner_link_is_not_ignored_as_a_shell_excerpt(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_plan.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kernel = self._create_kernel_fixture(root)
            op_dir = root / "demo"
            op_dir.mkdir()
            report = op_dir / "EXPLORE_REPORT.md"
            valid = "`pypto-pro-kernel/SKILL.md`\n"
            for citation, accepted in (
                ("`cat pypto-pro-kernel/references/missing.md`", True),
                ("[bad](<../cache root/pypto-pro-kernel/references/missing.md>)", False),
                ("[bad](<../cache/pypto-pro-kernel/references/missing.md>)", False),
                ("[good](pypto-pro-kernel/references/api/demo.md#lane-map)", True),
            ):
                with self.subTest(citation=citation):
                    report.write_text(valid + citation, encoding="utf-8")
                    errors = validator.check_kernel_references(op_dir, kernel)
                    self.assertEqual(not errors, accepted, errors)

    def test_owner_path_decoding_preserves_filename_and_fragment_boundaries(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_plan.py")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        kernel = self._create_kernel_fixture(root)
        (kernel / "references/api/demo#variant.md").write_text("## Lane map\n", encoding="utf-8")
        op_dir = root / "demo"
        op_dir.mkdir()
        for reference, expected in (
            ("demo%23variant.md#lane-map", ""),
            ("demo%23variant.md#lane%2Dmap", ""),
            ("demo%23variant.md#missing", "anchor does not exist"),
            ("%00.md", "invalid kernel reference"),
        ):
            with self.subTest(reference=reference):
                (op_dir / "EXPLORE_REPORT.md").write_text(
                    f"`pypto-pro-kernel/references/api/{reference}`", encoding="utf-8",
                )
                errors = validator.check_kernel_references(op_dir, kernel)
                if expected:
                    self.assertIn(expected, "\n".join(errors))
                else:
                    self.assertEqual(errors, [])

    def test_report_validator_uses_template_and_index_contents(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_plan.py")
        op_dir, paths, report, index = self._create_report_fixture()
        self.assertEqual(validator.check_report(op_dir, "demo", index), [])
        for path in paths:
            with self.subTest(missing=path):
                (op_dir / "EXPLORE_REPORT.md").write_text(
                    report.replace(f"`{path}`", f"`cat {path}`"), encoding="utf-8",
                )
                errors = validator.check_report(op_dir, "demo", index)
                self.assertTrue(any("not covered" in error and path in error for error in errors), errors)
        tutorial_row = f"| `{paths[1]}` | guide | pattern | applicable |"
        misplaced = report.replace(tutorial_row, "").replace("### 5.2 来自教程的关键约束与建议", "")
        (op_dir / "EXPLORE_REPORT.md").write_text(
            misplaced.replace("## 6. Stage 3 设计事实输入", "## 6. Stage 3 设计事实输入\n" + tutorial_row),
            encoding="utf-8",
        )
        self.assertEqual(validator.check_report(op_dir, "demo", index), [])
        (op_dir / "EXPLORE_REPORT.md").write_text(
            report.replace(f"| `{paths[1]}` |", "| `docs/guide/unlisted.md` |") + "\n" + paths[1],
            encoding="utf-8",
        )
        errors = validator.check_report(op_dir, "demo", index)
        self.assertTrue(any("guide referenced paths absent" in error for error in errors))
        report = _report_with_paths(
            _read(MATERIAL / "templates/explore_report.md"),
            [paths[0], "pro_ops/a5/vector/unlisted.py"],
            [paths[1], "docs/guide/programming_guide/tensor/unlisted.md"],
        )
        (op_dir / "EXPLORE_REPORT.md").write_text(report, encoding="utf-8")
        errors = "\n".join(validator.check_report(op_dir, "demo", index))
        self.assertIn("sample referenced paths absent", errors)
        self.assertIn("guide referenced paths absent", errors)

    def test_report_validator_checks_file_citations(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_plan.py")
        op_dir, _, report, index = self._create_report_fixture()
        api = "docs/pypto_pro/api/add.md"
        guide = "docs/guide/programming_guide/pro/tail block.md"
        sample = "pro_ops/a5/vector/test.v1#variant.py"
        index += "\n" + "\n".join(f"- `{path}`" for path in (api, guide, sample))
        report += f"\n`{guide}#tail-blocks.1`、`{sample}`\n"
        for citation, valid in (
            (api, True), (guide + "#tail-blocks.1", True), (sample, True),
            ("docs/pypto_pro/api/unlisted.md", False),
            ("docs/guide/programming_guide/pro/missing.mdd", False),
            (sample.replace("variant", "missing"), False),
            (r"C:\cache root\docs\guide\programming_guide\pro\tail block.md", False),
        ):
            with self.subTest(citation=citation):
                (op_dir / "EXPLORE_REPORT.md").write_text(
                    report + f"\n补充证据：`{citation}`\n", encoding="utf-8",
                )
                errors = validator.check_report(op_dir, "demo", index)
                if valid:
                    self.assertEqual(errors, [])
                else:
                    self.assertTrue(any("referenced paths absent" in error for error in errors), errors)

    def test_report_validator_parses_frontmatter(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_plan.py")
        op_dir, _, report, index = self._create_report_fixture()
        for field, values, valid in (
            ("op_name: demo", ('"demo"', "'demo' # operator"), True),
            ("schema_version: 2", ('"2" # version', "'2'", "2 # version"), True),
            ("feasibility: filled", ('"feasible # literal"', "'it''s feasible'", "feasible#literal"), True),
            ("op_name: demo", ('"other"', '"demo', "'demo\"", "demo\nop_name: demo"), False),
            ("schema_version: 2", ("999", "2#not-a-comment"), False),
            ("feasibility: filled", ("", "# no value", '""', "' ' # empty", '"bad\\q"'), False),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    replacement = field.split(":", 1)[0] + ": " + value
                    (op_dir / "EXPLORE_REPORT.md").write_text(
                        report.replace(field, replacement), encoding="utf-8",
                    )
                    errors = validator.check_report(op_dir, "demo", index)
                    self.assertEqual(not errors, valid, errors)
        (op_dir / "EXPLORE_REPORT.md").write_text(
            report.replace("op_name: demo", "# note\n\nop_name: 'it''s # feasible' # comment"),
            encoding="utf-8",
        )
        self.assertEqual(validator.check_report(op_dir, "it's # feasible", index), [])

    def test_report_validator_ignores_commands_and_non_devkit_references(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_plan.py")
        op_dir, _, report, index = self._create_report_fixture()
        report += (
            "\n`docs/guide/`、`docs/guide/v1.0/pro`\n"
            "`examples/samples/vector_kernels/reduce_sum_impl.py`\n"
            "`cat docs/guide/missing.md`\n"
            "`/usr/bin/cat /cache/docs/guide/missing.md`\n"
        )
        (op_dir / "EXPLORE_REPORT.md").write_text(report, encoding="utf-8")
        self.assertEqual(validator.check_report(op_dir, "demo", index), [])

    def test_plan_validator_accepts_a_complete_fixture(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_plan.py")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        op_dir, devkit, kernel, index, report = self._create_plan_fixture(root)
        results = validator.validate(op_dir, devkit, kernel)
        self.assertEqual([name for name, _ in results], ["SPEC", "INDEX", "REPORT", "REFERENCES"])
        self.assertTrue(all(not errors for _, errors in results), results)

        env = os.environ.copy()
        for key in ("PYPTO_DEVKIT_DIR", "CANNBOT_CONFIG_ROOT", "TILE_FWK_DEVICE_ID"):
            env.pop(key, None)
        command = [sys.executable, "-B", str(PLAN / "scripts/validate_plan.py"),
                   "--op-dir", str(op_dir), "--devkit", str(devkit), "--kernel-root", str(kernel)]
        completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        for artifact in ("SPEC", "INDEX", "REPORT", "REFERENCES"):
            self.assertIn(f"[PASS] {artifact}", completed.stdout)

        # The complete CLI must fail on each actual artifact defect, even
        # though no retired knowledge package or selection JSON exists.
        self.assertEqual([path.name for path in (root / "config").iterdir()],
                         ["pypto-pro-kernel"])
        for artifact, filename, invalid in (
            ("SPEC", "SPEC.md", "not a machine contract\n"),
            ("INDEX", "PRO_MATERIAL_INDEX.md", index + "\nextra stale entry\n"),
            ("REPORT", "EXPLORE_REPORT.md",
             report.replace("feasibility: filled", "feasibility: unsupported")),
            ("REFERENCES", "EXPLORE_REPORT.md", report +
             "\n`pypto-pro-kernel/references/api/demo.md#missing`\n"),
        ):
            with self.subTest(broken_artifact=artifact):
                target = op_dir / filename
                original = _read(target)
                try:
                    target.write_text(invalid, encoding="utf-8")
                    result = subprocess.run(command, cwd=root, env=env,
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertIn(f"[FAIL] {artifact}", result.stdout)
                finally:
                    target.write_text(original, encoding="utf-8")

        # The ordinary entry needs no explicit knowledge-root argument,
        # resolves its sibling installation from any working directory,
        # and checks an actual current owner rather than our fixture.
        actual_report = report.replace(
            "pypto-pro-kernel/references/api/demo.md#lane-map",
            "pypto-pro-kernel/references/api/memory-allocation.md#aliasing-and-mutual-exclusion",
        )
        (op_dir / "EXPLORE_REPORT.md").write_text(actual_report, encoding="utf-8")
        self.assertTrue(all(not errors for _, errors in validator.validate(op_dir, devkit)))
        result = subprocess.run(command[:-2], cwd=root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def _create_plan_fixture(self, root: Path):
        op_dir = root / "chosen outputs/demo"
        devkit, _ = self._create_devkit_fixture(root)

        op_dir.mkdir(parents=True)
        contract = {
            "schema_version": 1, "op_name": "demo", "formula": "y = x",
            "supported_dtypes": ["float32"],
            "inputs": [{"name": "x", "shape": [4], "dtype": "float32", "value_range": [-1, 1]}],
            "outputs": [{"name": "y", "shape": [4], "dtype": "float32", "value_range": [-1, 1]}],
            "default_params": {}, "tolerance": {"atol": 0.001, "rtol": 0.001},
            "dynamic_axes_ranges": {}, "shape_constraints": [],
            "p0_cases": [{"name": "p0", "params": {}, "input_shapes": {"x": [4]}, "output_shapes": {"y": [4]}}],
        }
        (op_dir / "SPEC.md").write_text(
            "```json machine-contract\n" + json.dumps(contract) +
            "\n```\n\n## kernel 契约补充\n已确认。\n", encoding="utf-8",
        )
        material = self._load_module(MATERIAL / "scripts/build_material_index.py")
        index, _ = material.build_index(devkit, material.manifest_path())
        (op_dir / "PRO_MATERIAL_INDEX.md").write_text(index, encoding="utf-8")
        template = _read(MATERIAL / "templates" / "explore_report.md")
        covered = re.findall(
            r"`((?:pro_ops/[^`]+\.py|docs/guide/[^`]+\.md))`", index,
        )
        report = _report_with_paths(
            template,
            [path for path in covered if path.startswith("pro_ops/")],
            [path for path in covered if path.startswith("docs/")],
        )
        kernel = self._create_kernel_fixture(root)
        # Consume the template's source field without an extra citation.
        report = report.replace("pypto-pro-kernel/SKILL.md",
                                "pypto-pro-kernel/references/api/demo.md#lane-map")
        (op_dir / "EXPLORE_REPORT.md").write_text(report, encoding="utf-8")
        return op_dir, devkit, kernel, index, report

    def _load_module(self, path: Path):
        spec = importlib.util.spec_from_file_location(f"_plan_test_{path.stem}", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _create_devkit_fixture(self, root: Path) -> tuple[Path, set[str]]:
        devkit = root / "devkit cache"
        guides = {
            "docs/guide/programming_guide/pro/index.md",
            "docs/guide/programming_guide/pro/development/tile guide.md",
            "docs/guide/quick_start/pro/index.md",
            "docs/guide/quick_start/pro/SIMD/Add_operator.md",
            "docs/guide/introduction.md",
        }
        excluded = {
            "docs/guide/programming_guide/tensor/index.md",
            "docs/guide/quick_start/tensor/index.md",
            "docs/guide/introduction_tensor.md",
            "docs/pypto_pro/tutorials/old.md",
        }
        manifest = _read(MATERIAL / "references/official_samples.md")
        samples = re.findall(r"`(pro_ops/[^`|]+\.py)`", manifest)
        for relative in {"docs/pypto_pro/api/index.md", *guides, *excluded, *samples}:
            target = devkit / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# fixture\n", encoding="utf-8")
        return devkit, guides

    def _create_report_fixture(self) -> tuple[Path, tuple[str, str], str, str]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        op_dir = Path(temporary.name) / "demo"
        op_dir.mkdir()
        paths = ("pro_ops/a5/vector/test_demo.py", "docs/guide/programming_guide/pro/guide.md")
        report = _report_with_paths(_read(MATERIAL / "templates/explore_report.md"), [paths[0]], [paths[1]])
        (op_dir / "EXPLORE_REPORT.md").write_text(report, encoding="utf-8")
        index = "\n".join(f"- `{path}`" for path in paths)
        return op_dir, paths, report, index

    def _create_kernel_fixture(self, root: Path) -> Path:
        kernel = root / "config/pypto-pro-kernel"
        files = {
            "SKILL.md": "# Current workflow\n",
            "references/api/demo.md": "# Candidate\n## Lane map\n## Lane map\n",
            "scripts/check.py": "# entry\n",
        }
        for relative, content in files.items():
            target = kernel / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        # Exercise the real shared Markdown parser, not a second test implementation.
        checks = kernel / "scripts/_checks"
        checks.mkdir()
        for name in ("__init__.py", "links.py", "paths.py", "source.py"):
            shutil.copy2(KERNEL / "scripts/_checks" / name, checks / name)
        return kernel

if __name__ == "__main__":
    unittest.main()
