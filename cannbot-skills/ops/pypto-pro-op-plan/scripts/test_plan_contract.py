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

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[2]
INTENT = OPS_ROOT / "pypto-pro-intent-understand"
MATERIAL = OPS_ROOT / "pypto-pro-material-explore"
PLAN = OPS_ROOT / "pypto-pro-op-plan"
KB = OPS_ROOT / "pypto-pro-op-kb"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _ref(path: Path, rel: str) -> dict[str, str]:
    return {
        "path": rel,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "reason": "specific design reason",
    }


def _report_with_paths(template: str, samples: list[str], tutorials: list[str]) -> str:
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

    def test_interface_and_kb_handoffs_are_explicit(self) -> None:
        intent = _read(INTENT / "SKILL.md")
        plan = _read(PLAN / "SKILL.md")
        material = _read(MATERIAL / "SKILL.md")
        self.assertIn("公开接口的 rank、shape", intent)
        self.assertIn("rank-0", intent)
        self.assertIn('class_id` 必须为字面量 `"."`', plan)
        self.assertIn("<op-dir>/<class>/KB_SELECTION.json", plan)
        self.assertIn("单一 VF API", material)

    def test_kb_documents_define_zero_or_multi_topology_semantics(self) -> None:
        plan = _read(PLAN / "SKILL.md")
        router = _read(KB / "ROUTER.md")
        router_flat = " ".join(router.split())
        contract = _read(KB / "CONTRACT.md")
        selection_rule = json.loads(
            _read(KB / "topology-map.json")
        )["selection_rule"]

        self.assertIn("Route zero or more topologies matched by the formula", router)
        self.assertIn("record `topologies: []` rather than force a best-fit category", router)
        self.assertIn("property, target and mandatory routing still runs", router)
        self.assertIn("does not mean unknown or skipped", router)
        self.assertIn("every actual topology match must be recorded", router_flat)
        self.assertIn("union of constraints routed by every matched topology", router)
        self.assertIn("applicable property modifier", router)
        self.assertIn("contains topologies/properties", router)
        self.assertNotIn("For the selected topology", router)
        self.assertIn("possibly empty", contract)
        self.assertIn("never force a best-fit category", contract)
        self.assertIn("never use an empty array to mean unknown or skipped", contract)
        self.assertIn("omitting one is invalid", contract)
        self.assertIn("property, target and mandatory routing still applies", contract)
        self.assertIn("routed constraints is mandatory", contract)
        self.assertIn("applicable property modifiers in the candidate pool", contract)
        self.assertIn("optional-pattern relevance rules", contract)
        self.assertIn("may match zero or more topologies", selection_rule)
        self.assertIn("omitting an actual match is invalid", selection_rule)
        self.assertIn("must not mean unknown or skipped", selection_rule)
        self.assertIn(
            "then add every applicable constraint from property modifiers, "
            "the target gate and mandatory constraints",
            selection_rule,
        )
        self.assertIn("matched topologies and applicable property modifiers", selection_rule)
        self.assertIn("retain all and only candidates", selection_rule)
        self.assertIn("不能表示未知、未分析或跳过", plan)
        topology_map = json.loads(_read(KB / "topology-map.json"))
        self.assertTrue(topology_map["property_modifiers"]["is_list"]["patterns"])
        self.assertTrue(topology_map["target_gated"])
        self.assertTrue(topology_map["mandatory_constraints"])
        self.assertNotRegex(plan, r"contract v\d+")

    def test_plan_uses_one_plan_validator(self) -> None:
        plan = _read(PLAN / "SKILL.md")
        validator_path = PLAN / "scripts" / "validate_plan.py"
        self.assertTrue(validator_path.is_file())
        self.assertIn("scripts/validate_plan.py", plan)
        self.assertIn("未指定时默认 A5", plan)
        self.assertNotIn("python -c '", plan)

    def test_kb_validator_accepts_flat_split_and_dynamic_topologies(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_plan.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb, selection = self._create_kb_fixture(root)
            op_dir = root / "custom" / "demo"
            op_dir.mkdir(parents=True)

            flat = op_dir / "KB_SELECTION.json"
            flat.write_text(json.dumps(selection), encoding="utf-8")
            self.assertEqual(validator.check_kb(op_dir, kb, "demo"), [])

            flat.unlink()
            split = op_dir / "class_a" / "KB_SELECTION.json"
            split.parent.mkdir()
            selection["class_id"] = "class_a"
            split.write_text(json.dumps(selection), encoding="utf-8")
            self.assertEqual(validator.check_kb(op_dir, kb, "demo"), [])

            selection["topologies"] = []
            selection["optional_patterns"] = []
            selection["no_matching_pattern"] = True
            split.write_text(json.dumps(selection), encoding="utf-8")
            self.assertEqual(validator.check_kb(op_dir, kb, "demo"), [])

    def test_kb_validator_rejects_stale_unknown_and_escaping_references(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_plan.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb, selection = self._create_kb_fixture(root)
            op_dir = root / "custom" / "demo"
            op_dir.mkdir(parents=True)
            path = op_dir / "KB_SELECTION.json"
            selection["topologies"] = ["made-up"]
            selection["properties"] = {"unknown": True}
            selection["optional_patterns"][0]["reason"] = ""
            selection["optional_patterns"][0]["path"] = "patterns/../constraints/demo.md"
            selection["required_constraints"][0]["sha256"] = "sha256:stale"
            path.write_text(json.dumps(selection), encoding="utf-8")
            errors = "\n".join(validator.check_kb(op_dir, kb, "demo"))
            for expected in (
                "unknown, duplicate, or non-string topologies", "unknown properties", "reason must be non-empty",
                "escapes patterns/", "stale sha256",
            ):
                self.assertIn(expected, errors)

            path.write_text('{"op":"demo","op":"other"}', encoding="utf-8")
            self.assertIn("duplicate JSON key", "\n".join(validator.check_kb(op_dir, kb, "demo")))

            (kb / "topology-map.json").write_text(
                '{"contract":[],"topologies":{}}', encoding="utf-8",
            )
            with self.assertRaises(validator.PlanConfigurationError):
                validator.check_kb(op_dir, kb, "demo")

    def test_kb_validator_requires_boolean_flags_and_integer_contract_versions(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_plan.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb, selection = self._create_kb_fixture(root)
            op_dir = root / "custom/demo"
            op_dir.mkdir(parents=True)
            path = op_dir / "KB_SELECTION.json"
            for flag in (0, 1, None, "false", True):
                with self.subTest(flag=flag):
                    selection["no_matching_pattern"] = flag
                    path.write_text(json.dumps(selection), encoding="utf-8")
                    self.assertEqual(
                        validator.check_kb(op_dir, kb, "demo"),
                        ["KB_SELECTION.json no_matching_pattern is invalid"],
                    )
            mapping_path = kb / "topology-map.json"
            mapping = json.loads(_read(mapping_path))
            for version in (True, False, 1.0, "1", None):
                with self.subTest(contract_version=version):
                    mapping["contract"]["contract_version"] = version
                    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
                    self.assertRaisesRegex(
                        validator.PlanConfigurationError, "contract_version must be an integer",
                        validator.check_kb, op_dir, kb, "demo",
                    )

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
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            (op_dir / "EXPLORE_REPORT.md").write_text(
                report, encoding="utf-8",
            )
            kb, selection = self._create_kb_fixture(root)
            (op_dir / "KB_SELECTION.json").write_text(json.dumps(selection), encoding="utf-8")
            results = validator.validate(op_dir, devkit, kb)
            self.assertEqual([name for name, _ in results], ["SPEC", "INDEX", "REPORT", "KB"])
            self.assertTrue(all(not errors for _, errors in results), results)

            env = os.environ.copy()
            for key in ("PYPTO_DEVKIT_DIR", "CANNBOT_CONFIG_ROOT", "TILE_FWK_DEVICE_ID"):
                env.pop(key, None)
            command = [sys.executable, "-B", str(PLAN / "scripts/validate_plan.py"),
                       "--op-dir", str(op_dir), "--devkit", str(devkit), "--kb-root", str(kb)]
            completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            for artifact in ("SPEC", "INDEX", "REPORT", "KB"):
                self.assertIn(f"[PASS] {artifact}", completed.stdout)

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

    def _create_kb_fixture(self, root: Path) -> tuple[Path, dict[str, object]]:
        production_version = json.loads(
            _read(KB / "topology-map.json")
        )["contract"]["contract_version"]
        contract_version = production_version + 1000
        kb = root / "config" / "pypto-pro-op-kb"
        files = {
            "patterns/demo.md": b"pattern\n",
            "constraints/demo.md": b"constraint\n",
        }
        for relative, content in files.items():
            target = kb / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        (kb / "topology-map.json").write_text(json.dumps({
            "contract": {
                "contract_version": contract_version,
                "property_keys": ["dtypes", "is_list"],
            },
            "topologies": {"elementwise": {}, "row-reduction": {}},
        }), encoding="utf-8")
        selection = {
            "schema_version": contract_version,
            "op": "demo",
            "class_id": ".",
            "topologies": ["elementwise", "row-reduction"],
            "properties": {"dtypes": ["float16"]},
            "optional_patterns": [
                _ref(kb / "patterns/demo.md", "patterns/demo.md")
            ],
            "required_constraints": [
                _ref(kb / "constraints/demo.md", "constraints/demo.md")
            ],
            "no_matching_pattern": False,
        }
        return kb, selection

if __name__ == "__main__":
    unittest.main()
