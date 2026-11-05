# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check generated CPU Golden inputs and exit codes."""

import importlib.util
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

OPS = Path(__file__).resolve().parents[3] / "ops"
spec = importlib.util.spec_from_file_location(
    "golden_scaffold_test",
    OPS / "pypto-pro-golden-generate/scripts/gen_golden_scaffold.py",
)
scaffold = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaffold)


class GoldenScaffoldTests(unittest.TestCase):
    def test_golden_validation_rejects_framework_imports_before_execution(self):
        runner_spec = importlib.util.spec_from_file_location(
            "golden_validation_test",
            OPS / "pypto-pro-golden-generate/scripts/validate_golden_once.py",
        )
        runner = importlib.util.module_from_spec(runner_spec)
        runner_spec.loader.exec_module(runner)
        imports = (
            "import pypto_pro.language as pl",
            "from pypto import frontend",
            "from pypto_pro.language import *",
            "from . import pypto_pro",
            "__import__('pypto_pro.language')",
            "import importlib as il\nil.import_module('pypto.frontend')",
            "from importlib import import_module as load\nload('pypto_pro')",
            "def nested():\n    import pypto_pro",
        )
        with tempfile.TemporaryDirectory() as temporary:
            golden = Path(temporary) / "probe_golden.py"
            for role, target in (("npu", "probe_golden"), ("cpu", "probe_golden_cpu")):
                functions = f"def {target}(): pass\ndef _validate(): pass\ndef _make_inputs(): pass\n"
                for source in imports:
                    with self.subTest(role=role, source=source):
                        golden.write_text(source + "\n" + functions)
                        with self.assertRaisesRegex(runner.ReceiptError, "independent of PyPTO"):
                            runner._golden_contract(golden, "probe", role)
                allowed = "import torch\n" + ("import torch_npu\n" if role == "npu" else "")
                golden.write_text(allowed + functions)
                runner._golden_contract(golden, "probe", role)
            golden.write_text("def unrelated(): pass\n")
            with self.assertRaisesRegex(runner.ReceiptError, "missing functions"):
                runner._golden_contract(golden, "probe", "cpu")

    def test_spec_consumers_work_in_a_relocated_bundle_and_report_missing_core(self):
        contract = {
            "schema_version": 1,
            "op_name": "probe",
            "formula": "y = x + 1",
            "supported_dtypes": ["float32"],
            "inputs": [
                {"name": "x", "shape": [1], "dtype": "float32", "value_range": [0, 1]}
            ],
            "outputs": [
                {"name": "y", "shape": [1], "dtype": "float32", "value_range": [1, 2]}
            ],
            "default_params": {},
            "tolerance": {"rtol": 0, "atol": 0},
            "dynamic_axes_ranges": {},
            "shape_constraints": [],
            "p0_cases": [
                {
                    "name": "p0",
                    "params": {},
                    "input_shapes": {"x": [1]},
                    "output_shapes": {"y": [1]},
                }
            ],
            "exit_criteria": {
                "id": "latency",
                "metric": "latency_us",
                "operator": "<=",
                "threshold": 10,
                "unit": "us",
                "basis": "model",
                "cases": ["p0"],
                "aggregation": "each",
                "source": "test_model",
            },
        }

        def load(name, path):
            module_spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            return module

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skills = root / "config with spaces" / "skills"
            for relative in (
                "pypto-pro-kernel/scripts/validate_spec.py",
                "pypto-pro-kernel/scripts/exit_criteria.py",
                "pypto-pro-golden-generate/scripts/gen_golden_scaffold.py",
                "pypto-pro-golden-generate/scripts/validate_golden_once.py",
                "pypto-pro-golden-generate/templates/golden-template.py.tmpl",
            ):
                target = skills / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(OPS / relative, target)
            spec_path = root / "SPEC.md"
            spec_path.write_text(
                "```json machine-contract\n" + json.dumps(contract) + "\n```\n"
            )
            generator = load(
                "bundled_scaffold",
                skills / "pypto-pro-golden-generate/scripts/gen_golden_scaffold.py",
            )
            runner = load(
                "bundled_golden_runner",
                skills / "pypto-pro-golden-generate/scripts/validate_golden_once.py",
            )
            template = (
                skills / "pypto-pro-golden-generate/templates/golden-template.py.tmpl"
            )
            generated = ast.parse(generator.generate(str(spec_path), str(template)))
            entry = next(
                node
                for node in generated.body
                if isinstance(node, ast.FunctionDef) and node.name == "probe_golden"
            )
            self.assertEqual([argument.arg for argument in entry.args.args], ["x"])
            self.assertEqual(runner._op_name(spec_path), "probe")
            validator = skills / "pypto-pro-kernel/scripts/validate_spec.py"
            result = subprocess.run(
                [sys.executable, str(validator), str(spec_path)],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            validator.unlink()
            with self.assertRaisesRegex(ValueError, "pypto-pro-kernel"):
                generator.generate(str(spec_path), str(template))
            with self.assertRaisesRegex(runner.ReceiptError, "pypto-pro-kernel"):
                runner._op_name(spec_path)

    def test_case_factory_uses_per_case_dtype(self):
        tensors = [{"name": "x", "dtype": "float16"}]
        cases = [
            {"name": "fp16", "input_shapes": {"x": [2]}, "params": {}},
            {
                "name": "bf16",
                "input_shapes": {"x": [2]},
                "input_dtypes": {"x": "bfloat16"},
                "params": {},
            },
        ]
        source = scaffold._build_make_inputs(tensors, cases)
        self.assertIn("dtype=torch.float16", source)
        self.assertIn("dtype=torch.bfloat16", source)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "Torch is needed to execute the generated CPU Golden",
    )
    def test_generated_cpu_footer_accepts_success_and_rejects_bad_output(self):
        contract = {
            "schema_version": 1,
            "op_name": "probe",
            "formula": "y = x + 1",
            "supported_dtypes": ["float32"],
            "inputs": [
                {"name": "x", "shape": [1], "dtype": "float32", "value_range": [0, 1]}
            ],
            "outputs": [
                {"name": "y", "shape": [1], "dtype": "float32", "value_range": [1, 2]}
            ],
            "default_params": {},
            "tolerance": {"rtol": 0, "atol": 0},
            "dynamic_axes_ranges": {},
            "shape_constraints": [],
            "p0_cases": [
                {
                    "name": "p0",
                    "params": {},
                    "input_shapes": {"x": [1]},
                    "output_shapes": {"y": [1]},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "SPEC.md").write_text(
                "```json machine-contract\n" + json.dumps(contract) + "\n```\n"
            )
            source = scaffold.generate(
                str(root / "SPEC.md"),
                str(
                    OPS
                    / "pypto-pro-golden-generate/templates/golden_cpu_template.py.tmpl"
                ),
            )
            source = scaffold._replace_block(
                source,
                "def _make_inputs(device):",
                "def _make_inputs(device):\n    return [torch.zeros(1, device=device)], {}\n",
            )
            for expression, expected_code in (
                ("x + 1", 0),
                ("torch.full_like(x, float('nan'))", 1),
            ):
                candidate = scaffold._replace_block(
                    source,
                    "def probe_golden_cpu(",
                    "def probe_golden_cpu(x):\n    return " + expression + "\n",
                )
                path = root / "probe_golden_cpu.py"
                path.write_text(candidate)
                result = subprocess.run(
                    [sys.executable, str(path)],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                        "OMP_NUM_THREADS": "1",
                    },
                    timeout=45,
                )
                self.assertEqual(
                    result.returncode, expected_code, result.stdout + result.stderr
                )


if __name__ == "__main__":
    unittest.main()
