# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import csv
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from itertools import product
from pathlib import Path
from unittest.mock import patch


MSPROF_SCRIPT = Path(__file__).parents[1] / "msprof_perf_summary.py"
MSPROF_SPEC = importlib.util.spec_from_file_location(
    "msprof_perf_summary_ut", MSPROF_SCRIPT
)
MSPROF = importlib.util.module_from_spec(MSPROF_SPEC)
assert MSPROF_SPEC and MSPROF_SPEC.loader
sys.modules[MSPROF_SPEC.name] = MSPROF
MSPROF_SPEC.loader.exec_module(MSPROF)
CLI = importlib.import_module("evidence_cli")
CONTRACT = importlib.import_module("golden_contract")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    assert rows
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PerformanceRoutingAndAdapterTest(unittest.TestCase):
    def test_compare_and_quick_seed_defaults_preserve_legacy(self) -> None:
        for mode, manifest, seed in product(("compare", "quick"), (False, True), (None, 0, 42, 7, -1)):
            with self.subTest(mode=mode, manifest=manifest, seed=seed):
                argv = [str(MSPROF_SCRIPT), f"--{mode}", "--output-dir=.", "--device=0"]
                if manifest:
                    argv.append("--case-manifest=unused.json")
                if seed is not None:
                    argv.append(f"--seed={seed}")
                valid = not manifest or seed in (None, 0, 42)
                with patch.object(MSPROF, f"_run_{mode}_mode", return_value=0) as collect, \
                        patch(f"legacy_compare._run_{mode}_mode_legacy", return_value=0) as legacy, \
                        patch.object(MSPROF, f"run_{mode}_mode",
                                     wraps=getattr(MSPROF, f"run_{mode}_mode")) as route, \
                        patch.object(sys, "argv", argv), patch.object(sys, "exit") as exit_call:
                    MSPROF.main()
                exit_call.assert_called_once_with(0 if valid else 1)
                self.assertEqual(collect.called, manifest and valid)
                self.assertEqual(legacy.called, not manifest)
                expected_seed = None if manifest and seed in (None, 0) else (seed or 0)
                self.assertEqual(route.call_args.args[0].seed, expected_seed)

    def test_timeline_seed_defaults_reach_evidence_preflight(self) -> None:
        for seed in (None, 0, 42, 7, -1):
            with self.subTest(seed=seed):
                argv = [str(MSPROF_SCRIPT), "--timeline", "--output-dir=.", "--device=0",
                        "--case-manifest=unused.json", "--case-id=p0"]
                if seed is not None:
                    argv.append(f"--seed={seed}")
                # Stop at the first evidence read; no profiler or NPU is needed.
                with patch.object(CLI, "_validated_performance_cases", return_value=None) as evidence, \
                        patch.object(CLI, "run_timeline_mode", wraps=CLI.run_timeline_mode) as route, \
                        patch.object(sys, "argv", argv), patch.object(sys, "exit") as exit_call:
                    MSPROF.main()
                exit_call.assert_called_once_with(1)
                self.assertEqual(evidence.called, seed in (None, 0, 42))
                self.assertEqual(route.call_args.args[0].seed, None if seed in (None, 0) else seed)

    def test_bound_route_uses_core_time_and_keeps_overlap_unverified(self) -> None:
        diagnosis = CLI.diagnose_bound_route({
            "aicore_time(us)": "2.0",
            "aiv_time(us)": "5.0",
            "aic_mte2_ratio": "0.99",
            "aiv_vec_ratio": "0.20",
            "aiv_mte2_ratio": "0.91",
            "aiv_scalar_ratio": "0.10",
        })
        self.assertEqual(diagnosis["primary_core"], "aiv")
        self.assertEqual(diagnosis["routing_label"], "data_movement_candidate")
        self.assertFalse(diagnosis["scalar_route_triggered"])
        self.assertEqual(diagnosis["pipeline_overlap_status"], "unverified")
        self.assertEqual(diagnosis["roofline_terminal_status"], "insufficient_evidence")
        self.assertFalse(diagnosis["completion_eligible"])

    def test_scalar_and_mixed_routes_are_not_healthy_terminal_states(self) -> None:
        scalar = CLI.diagnose_bound_route({
            "aicore_time(us)": "0",
            "aiv_time(us)": "3",
            "aiv_scalar_ratio": "0.85",
            "aiv_vec_ratio": "0.20",
        })
        self.assertEqual(scalar["routing_label"], "scalar_candidate")
        self.assertTrue(scalar["scalar_route_triggered"])
        mixed = CLI.diagnose_bound_route({
            "aicore_time(us)": "0",
            "aiv_time(us)": "3",
            "aiv_vec_ratio": "0.82",
            "aiv_mte2_ratio": "0.83",
            "aiv_scalar_ratio": "0.05",
        })
        self.assertEqual(mixed["routing_label"], "mixed")
        self.assertFalse(mixed["completion_eligible"])

    def test_pipe_ratio_unit_is_fraction_and_percent_input_fails_closed(self) -> None:
        self.assertEqual(CLI.normalized_pipe_ratio("0.91"), 0.91)
        for invalid in ("91", "nan", "garbage"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                CLI.diagnose_bound_route({
                    "aiv_time(us)": "1.0",
                    "aiv_vec_ratio": invalid,
                })

    def test_manifest_function_adapter_selects_one_existing_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "called.txt"
            test_script = root / "test_sample.py"
            test_script.write_text(
                "from pathlib import Path\n"
                f"MARKER = Path({str(marker)!r})\n"
                "def test_first():\n    MARKER.write_text('first', encoding='utf-8')\n"
                "def test_second():\n    MARKER.write_text('second', encoding='utf-8')\n",
                encoding="utf-8",
            )
            manifest = root / "PERFORMANCE_CASES.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "cases": [
                    {"id": "p0", "shape": "[1]", "dtype": "fp16",
                     "test_function": "test_second"},
                ],
            }), encoding="utf-8")
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(
                    CLI.run_case_function(
                        str(test_script), str(manifest), "p0"
                    ),
                    0,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "second")
            self.assertEqual(stream.getvalue().strip(), "PYPTO_PERF_SELECTED_CASE=p0")
            with self.assertRaisesRegex(ValueError, "no test_function"):
                CLI.run_case_function(
                    str(test_script), str(manifest), "unknown"
                )

    def test_function_adapter_requires_complete_valid_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "PERFORMANCE_CASES.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "cases": [
                    {"id": "p0", "shape": "[1]", "dtype": "fp16",
                     "test_function": "test_p0"},
                    {"id": "p1", "shape": "[2]", "dtype": "fp16"},
                ],
            }), encoding="utf-8")
            cases, source, error = CONTRACT.parse_case_manifest(str(manifest))
            self.assertIsNone(error)
            args = Namespace(
                performance_cases=source, case_arg=None, case_env=None,
            )
            self.assertFalse(MSPROF.uses_function_adapter(args, ["p0", "p1"]))
            self.assertIn(
                "explicit per-case selector",
                MSPROF.profile_case_contract_error(cases, args),
            )

    def test_discovery_lists_exact_lowering_names_without_guessing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_dir = (
                root / "PROF_PipeUtilization" / "PROF_1" /
                "device_0" / "summary" / "mindstudio_profiler_output"
            )
            csv_dir.mkdir(parents=True)
            write_csv(csv_dir / "op_summary_1.csv", [
                {"Op Name": "_Ztarget", "Task Type": "AI_CORE",
                 "Task Duration(us)": "3.0"},
                {"Op Name": "helper", "Task Type": "AI_CORE",
                 "Task Duration(us)": "1.0"},
            ])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    CLI.list_op_names_mode(Namespace(prof_group_dir=str(root))),
                    0,
                )
            self.assertIn("_Ztarget", output.getvalue())
            self.assertIn("exact_op_name", output.getvalue())


class L2CacheCleanTest(unittest.TestCase):
    """L2 清刷开关：默认开启、可关闭，并透传到采集命令。"""

    def test_flag_defaults_on_and_can_be_disabled(self) -> None:
        parser = getattr(MSPROF, "_build_arg_parser")()
        argv = ["--compare", "--output-dir=."]
        self.assertTrue(parser.parse_args(argv).l2_clean)
        self.assertFalse(parser.parse_args(argv + ["--no-l2-clean"]).l2_clean)

    def test_standard_command_carries_l2_flag(self) -> None:
        captured: dict[str, list] = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        run_standard = getattr(MSPROF, "_run_msprof_standard")
        for enabled, flag in ((True, "--l2-clean"), (False, "--no-l2-clean")):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "PROF_GROUP_x").mkdir()
                request = MSPROF.RunRequest("runner.py", tmp, warmup=1, l2_clean=enabled)
                captured.clear()
                with patch.object(MSPROF.subprocess, "run", side_effect=fake_run):
                    _, err = run_standard(request)
                self.assertIsNone(err)
                self.assertIn(flag, captured["cmd"])

    def test_summary_records_l2_clean(self) -> None:
        args = Namespace(
            warmup=1, repeats=1, seed=42, l2_clean=False, op_name="k",
            case_arg=None, case_env=None, case_manifest=None, quick=False,
            performance_cases=None, collection_id=None, deep_round_dir=None,
            golden_iterations=None,
        )
        csi = MSPROF.CompareSummaryInput(
            out_dir=Path("/tmp/x"), rows=[], speedups=[], ref_times=[],
            asc_times=[], n_cases=0, args=args, device_id=0, device_src="cli",
        )
        self.assertIs(MSPROF.compute_compare_summary(csi)["l2_clean"], False)


class SelectorPreflightEnvTest(unittest.TestCase):
    """manifest 预检必须使用与正式 profiling 一致的设备环境。"""

    def test_manifest_preflight_uses_tile_fwk_env(self) -> None:
        seen: list[bool] = []

        def fake_env(device_id, seed=None, case_env=None, case_name=None, tile_fwk=False):
            seen.append(tile_fwk)
            return {}

        args = Namespace(
            seed=42, case_arg=None, case_env=None, case_manifest="cases.json",
            performance_cases={"type": "case_manifest",
                               "case_functions": {"c1": "f1"}},
        )
        with patch.object(CLI, "_profile_env", side_effect=fake_env), \
                patch.object(CLI, "_test_command", return_value=["true"]), \
                patch.object(CLI.subprocess, "run",
                             return_value=types.SimpleNamespace(returncode=1, stdout="", stderr="")):
            getattr(CLI, "_selector_preflight")("t.py", ["c1"], args, 0)
        self.assertEqual(seen, [True, True])


if __name__ == "__main__":
    unittest.main()
