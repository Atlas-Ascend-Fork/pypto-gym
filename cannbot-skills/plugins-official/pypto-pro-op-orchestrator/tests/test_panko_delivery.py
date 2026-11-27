# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host regressions for Pro acquisition and multi-file delivery contracts."""
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[3] / "ops/pypto-pro-op-perf-panko-manual/scripts"
CAPACITY = dict.fromkeys(("Vec", "Mat", "Left", "Right", "Acc"), 65536)
ENVELOPE = dict(ub_kb=248, l1_kb=512, l0a_kb=64, l0b_kb=64, l0c_kb=256, cube_cores=28, vector_cores=56, soc="a5")
HOST_HELPER = "def host_shape():\n    return [16, 32]\n"
KERNEL = """import pypto_pro.language as pl
@pl.jit(auto_mutex=False)
def kernel(a):
    tile = pl.make_tile_group(type=pl.TileType(shape=[16, 32], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Vec), addrs=0, depth=2, mutex_ids=[0, 1])
    return a
"""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(os.name == "posix", "Pro campaign locking requires fcntl")
class ProContractsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.env = dict(os.environ, PANKO_HOME=str(self.root / "home"), PYTHONPATH=str(SCRIPTS))

    def run_cli(self, *args, status=0):
        result = subprocess.run([sys.executable, *map(str, args)], cwd=self.root, env=self.env,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, status, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def prepare_package(self, scripts):
        self.package = self.root / "mypkg"
        inner = self.package / "inner"
        inner.mkdir(parents=True)
        (self.package / "__init__.py").write_text("BIAS = 5\n")
        (inner / "__init__.py").touch()
        (inner / "kernel.py").write_text("from .helper import compute\n" + KERNEL)
        (inner / "helper.py").write_text(
            "import pypto_pro.language as pl\nfrom mypkg import BIAS\n"
            "@pl.jit(auto_mutex=False)\ndef compute(x):\n    return x + BIAS\n")
        (self.root / "precision.py").write_text("from mypkg.inner import kernel\nassert kernel.compute(7) == 12\n")
        view = self.root / "view.py"
        prepare = [scripts / "source_analysis/program_bundle.py", "prepare", "--view", view,
                   "--manifest", "manifest.json", "--root"]
        rejected = subprocess.run([sys.executable, *prepare, inner], cwd=self.root, env=self.env,
                                  capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("Prepare the outermost package 'mypkg' as --root", rejected.stderr)
        self.assertFalse(view.exists())
        self.assertFalse(Path(str(view) + ".runtime.json").exists())
        self.assertFalse((self.root / "manifest.json").exists())
        self.run_cli(*prepare, self.package)
        return view

    def baseline_receipt(self):
        view = self.root / "kernel.py"
        view.write_text(KERNEL)
        precision = self.root / "precision.py"
        precision.touch()
        self.template = dict(
            schema="panko-pro-npu-v1", code_hash=sha(view),
            program_files={str(view): sha(view)},
            contract=dict(test_files={str(precision): sha(precision)}, device="0", op_name="kernel",
                          warmup=2, runs=9,
                          argv=["python3", "precision.py"], cwd=str(self.root), env={"TILE_FWK_DEVICE_ID": "0"},
                          program=None, memory_capacity=CAPACITY),
            correctness_passed=True, workspace_b=4096, workspace_by_space={"Vec": 4096}, workspace_state="measured")
        return view, self.reading("baseline", 100)

    def initialize(self, scripts=SCRIPTS):
        view, baseline = self.baseline_receipt()
        self.run_cli(scripts / "panko_harness.py", "init", "--op-dir", self.root, "--op", "kernel",
                     "--op-file", view, "--device", "0", "--chip-envelope", json.dumps(ENVELOPE),
                     "--config", '{"bayesian_optimization_tile_search":false,"eval_budget":1}',
                     "--measurement-file", baseline)
        return view, baseline

    def test_normalization_evidence_matches_baseline_environment(self):
        view, baseline = self.baseline_receipt()
        original = self.root / "original.py"
        original.write_text(KERNEL.replace("shape=[16,", "shape=[a.shape[0],"))

        def arm(path):
            return dict(source_sha256=sha(path), correctness=dict(all_passed=True, case_ids=["17"]),
                        latency_samples_us=[10] * 5)

        evidence = dict(schema_version=1, measurement_kind="device", before=arm(original), after=arm(view),
                        case_context_sha256=hashlib.sha256(b"{}").hexdigest())
        receipt = self.root / "normalization.json"
        init = [SCRIPTS / "panko_harness.py", "init", "--op-dir", self.root, "--op", "kernel", "--op-file", view,
                "--device", "0", "--chip-envelope", json.dumps(ENVELOPE), "--measurement-file", baseline,
                "--normalization-before", original, "--normalization-evidence", receipt]
        command = "python3 precision.py"
        state = self.root / "optimization/search_state.json"
        for device, test_command, reason in (("1", command, "device"), ("0", "python3 other.py", "command"),
                                             ("0", "MODE=other " + command, "command"),
                                             ("0", "cd other && " + command, "command")):
            with self.subTest(device=device, command=test_command):
                receipt.write_text(json.dumps(dict(evidence, device=device, test_command=test_command)))
                self.assertIn(reason, self.run_cli(*init, status=3)["error"])
                self.assertFalse(state.exists())
        receipt.write_text(json.dumps(dict(evidence, device="0", test_command=command)))
        self.run_cli(*init)
        self.assertTrue(state.exists())

    def reading(self, name, us):
        profile = self.root / name / "op_summary.csv"
        profile.parent.mkdir()
        profile.write_text("Op Name,Task Type,Task Duration(us)\n" + f"kernel,AI_CORE,{us}\n" * 11)
        receipt = self.root / (name + ".json")
        receipt.write_text(json.dumps(dict(self.template, p=us, samples=[us] * 9,
                                           msprof_dir=str(profile.parent), profile_files={str(profile): sha(profile)})))
        return receipt

    def prepare_profiled_kernel(self, source, precision_check=""):
        self.env.pop("TILE_FWK_DEVICE_ID", None)
        self.env["PATH"] = str(self.root) + os.pathsep + self.env["PATH"]
        kernel = self.root / "kernel.py"
        kernel.write_text(source + "\n" + HOST_HELPER)
        # On the host only the decorator is needed to import the kernel.
        language = self.root / "pypto_pro/language.py"
        language.parent.mkdir()
        language.write_text("def jit(**kwargs): return lambda fn: fn\n")
        (self.root / "precision.py").write_text(
            "import inspect, os, kernel\nassert os.environ['TILE_FWK_DEVICE_ID'] == '0'\n"
            f"assert inspect.getsource(kernel.host_shape) == {HOST_HELPER!r}\n" + precision_check)
        profiler = self.root / "msprof"
        profiler.write_text("""#!/usr/bin/env python3
import subprocess, sys
from pathlib import Path
args = iter(sys.argv[1:])
out = Path(next(args).split("=", 1)[1])
python = next(arg for arg in args if not arg.startswith("--"))
subprocess.run([python, *args], check=True)
launches = int(sys.argv[sys.argv.index("--launches") + 1])
latency = Path("latency.txt").read_text() if Path("latency.txt").exists() else "12"
(out / "op_summary.csv").write_text("Op Name,Task Type,Task Duration(us)\\n" + f"kernel,AI_CORE,{latency}\\n" * launches)
""")
        profiler.chmod(0o755)
        return kernel

    def prepare_profiled_package(self, kernel):
        package = self.root / "package"
        package.mkdir()
        runtime = package / kernel.name
        kernel.rename(runtime)
        source = runtime.read_text().replace("return a", "return compute(a)")
        runtime.write_text("from helper import compute\n" + source)
        (package / "helper.py").write_text("def compute(x):\n    return x + 1\n")
        (package / "unused.py").write_text("unused = True\n")
        precision = self.root / "precision.py"
        precision.write_text(f"import sys\nsys.path.insert(0, {str(package)!r})\n" + precision.read_text())
        view = self.root / "view.py"
        self.run_cli(SCRIPTS / "source_analysis/program_bundle.py", "prepare", "--root", package,
                     "--view", view, "--manifest", "manifest.json")
        return view

    def start_profiled_campaign(self, *, tile_search=False, precision_check="", capacity=None, source=KERNEL,
                                catalog=None, command="python3 precision.py", package=False,
                                address_limits=None, **config):
        capacity = CAPACITY if capacity is None else capacity
        kernel = self.prepare_profiled_kernel(source, precision_check)
        if package:
            kernel = self.prepare_profiled_package(kernel)
        harness = SCRIPTS / "panko_harness.py"
        target = ["--op-dir", self.root, "--op-file", kernel]
        baseline = self.run_cli(SCRIPTS / "measure.py", *target, "--device", "0", "--test-command", command,
                                "--test-files", '["precision.py"]', "--out", "baseline.json",
                                "--memory-capacity", json.dumps(capacity),
                                *(["--address-limits", json.dumps(address_limits)] if address_limits else []))
        self.run_cli(harness, "init", *target, "--device", "0", "--op", "kernel", "--test-command", command,
                     "--measurement-file", "baseline.json", "--chip-envelope", json.dumps(ENVELOPE),
                     *(["--catalog", catalog] if catalog else []),
                     "--config", json.dumps(dict(bayesian_optimization_tile_search=tile_search,
                                                 eval_budget=4 if tile_search else 2, selection_seed=0) | config))
        return kernel, baseline

    def test_step_preserves_precision_warning_filter(self):
        command = "python3 -Werror precision.py"
        kernel, baseline = self.start_profiled_campaign(command=command, precision_check=(
            "import warnings\nif 'shape=[16, 16]' in inspect.getsource(kernel.kernel):\n"
            "    warnings.warn('precision warning')\n"))
        harness = SCRIPTS / "panko_harness.py"
        parent = self.run_cli(harness, "select", "--op-dir", self.root)["selected"]["id"]
        kernel.write_text(kernel.read_text().replace("shape=[16, 32]", "shape=[16, 16]"))
        result = self.run_cli(harness, "step", "--op-dir", self.root, "--op-file", kernel,
                              "--parent", parent, "--dispatch-id", "warning", "--test-command", command)
        reading = json.loads(Path(result["measurement_file"]).read_text())
        self.assertEqual(reading["contract"]["argv"], command.split())
        self.assertFalse(reading["correctness_passed"])
        self.assertTrue(reading["test_failed"])
        self.assertIsNone(reading["p"])
        campaign = json.loads((self.root / "optimization/search_state.json").read_text())
        self.assertEqual([row["code_hash"] for row in campaign["archive"]], [baseline["code_hash"]])

    def test_init_refuses_receipt_with_unsupported_interpreter_option(self):
        kernel, baseline = self.baseline_receipt()
        reading = json.loads(baseline.read_text())
        reading["contract"]["argv"].insert(1, "-i")
        baseline.write_text(json.dumps(reading))
        result = self.run_cli(SCRIPTS / "panko_harness.py", "init", "--op-dir", self.root, "--op", "kernel",
                              "--op-file", kernel, "--device", "0", "--chip-envelope", json.dumps(ENVELOPE),
                              "--measurement-file", baseline, status=3)
        self.assertIn("Unsupported Python interpreter option '-i'", result["error"])
        self.assertFalse((self.root / "optimization/search_state.json").exists())

    def test_auto_timing_target_bound_to_baseline(self):
        kernel, baseline = self.start_profiled_campaign()
        harness = SCRIPTS / "panko_harness.py"
        parent = self.run_cli(harness, "select", "--op-dir", self.root)["selected"]["id"]
        kernel.write_text(kernel.read_text().replace("[16, 32]", "[16, 16]", 1))
        profiler = self.root / "msprof"
        script = profiler.read_text()
        state = self.root / "optimization/search_state.json"
        self.assertEqual(json.loads(state.read_bytes())["measurement_contract"]["timed_target"],
                         {"op_name": "kernel", "rows_per_call": 1})
        original = state.read_bytes()
        for name, count in (("other", 2), ("other", 1), ("kernel", 2), ("kernel", 1)):
            if (name, count) == ("kernel", 2):
                legacy = json.loads(original)
                del legacy["measurement_contract"]["timed_target"]
                state.write_text(json.dumps(legacy))
                original = state.read_bytes()
            profiler.write_text(script.replace("kernel,AI_CORE,", f"{name},AI_CORE,")
                                .replace("* launches)", f"* (launches * {count}))"))
            filename = f"{name}-{count}.json"
            result = self.run_cli(SCRIPTS / "measure.py", "--op-file", kernel, "--op-dir", self.root,
                                  "--test-command", "python3 precision.py", "--device", "0", "--test-files",
                                  '["precision.py"]', "--memory-capacity", json.dumps(CAPACITY), "--out", filename)
            with self.subTest(name=name, count=count):
                self.assertEqual((result["op_name"], result["rows_per_call"]), (name, count))
                same = (name, count) == (baseline["op_name"], baseline["rows_per_call"])
                reply = self.run_cli(harness, "record", "--op-dir", self.root, "--parent", parent,
                                     "--measurement-file", filename, status=0 if same else 3)
                if not same:
                    self.assertIn("timing target", reply["error"])
                    self.assertEqual(state.read_bytes(), original)

    def test_package_remeasurement_requires_baseline_tasks_per_call(self):
        view, baseline = self.start_profiled_campaign(package=True)
        profiler = self.root / "msprof"
        script = profiler.read_text()
        record = [SCRIPTS / "panko_harness.py", "record", "--op-dir", self.root,
                  "--parent", "root", "--remeasure"]
        state = self.root / "optimization/search_state.json"
        original = state.read_bytes()
        self.assertEqual(json.loads(original)["measurement_contract"]["timed_target"],
                         {"tasks_per_call": {"kernel": 1}})
        for name, count in (("kernel", 2), ("other", 1), ("kernel", 1)):
            if name == "other":
                legacy = json.loads(original)
                del legacy["measurement_contract"]["timed_target"]
                state.write_text(json.dumps(legacy))
                original = state.read_bytes()
            profiler.write_text(script.replace("kernel,AI_CORE,", f"{name},AI_CORE,")
                                .replace("* launches)", f"* (launches * {count}))"))
            path = f"{name}-{count}.json"
            self.run_cli(SCRIPTS / "measure.py", "--op-file", view, "--op-dir", self.root,
                         "--test-command", "python3 precision.py", "--device", "0", "--test-files",
                         '["precision.py"]', "--memory-capacity", json.dumps(CAPACITY), "--out", path)
            same = (name, count) == ("kernel", 1)
            reply = self.run_cli(*record, "--measurement-file", path, status=0 if same else 3)
            if not same:
                self.assertIn("timing target", reply["error"])
                self.assertEqual(state.read_bytes(), original)
                path = self.root / path
                edited = json.loads(path.read_bytes())
                edited["tasks_per_call"] = {"kernel": 1}
                edited["contract"]["timed_target"] = {"tasks_per_call": {"kernel": 1}}
                path.write_text(json.dumps(edited))
                self.assertIn("timing CSV", self.run_cli(*record, "--measurement-file", path, status=3)["error"])
        self.assertEqual(len(json.loads(state.read_bytes())["eval_cache"][baseline["code_hash"]]["samples"]), 18)

    def test_address_gap_reaches_measurement_and_pareto_record(self):
        source = """import pypto_pro.language as pl
@pl.jit(auto_mutex=False)
def kernel(a):
    typ = pl.TileType(shape=[16, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    src = pl.make_tile_group(type=typ, addrs=0, mutex_ids=[0])
    dst = pl.make_tile_group(type=typ, addrs=4096, mutex_ids=[1])
    return a
"""
        kernel, baseline = self.start_profiled_campaign(source=source)
        self.assertEqual((baseline["workspace_by_space"], baseline["workspace_b"]), ({"Vec": 6144}, 6144))
        harness = SCRIPTS / "panko_harness.py"
        target = ["--op-dir", self.root]
        parent = self.run_cli(harness, "select", *target)["selected"]["id"]
        kernel.write_text(kernel.read_text().replace("addrs=4096", "addrs=2048"))
        reply = self.run_cli(harness, "step", *target, "--parent", parent, "--op-file", kernel,
                             "--dispatch-id", "pack", "--test-command", "python3 precision.py")
        candidate = json.loads(Path(reply["measurement_file"]).read_text())
        self.assertEqual((candidate["workspace_by_space"], candidate["workspace_b"]), ({"Vec": 4096}, 4096))
        campaign = json.loads((self.root / "optimization/search_state.json").read_bytes())
        self.assertEqual([(p["code_hash"], p["workspace_b"]) for p in campaign["archive"]],
                         [(candidate["code_hash"], 4096)])

    def test_feasible_allows_moving_only_the_second_slot(self):
        kernel, baseline = self.start_profiled_campaign(source=KERNEL.replace("addrs=0", "addrs=[0, 4096]"))
        self.assertEqual(baseline["workspace_b"], 6144)
        harness = SCRIPTS / "panko_harness.py"
        parent = self.run_cli(harness, "select", "--op-dir", self.root)["selected"]["id"]
        target = ["--op-dir", self.root, "--op-file", kernel, "--parent", parent]
        kernel.write_text(kernel.read_text().replace("[0, 4096]", "[0, 2048]"))
        self.assertTrue(self.run_cli(harness, "feasible", *target)["feasible"])
        reply = self.run_cli(harness, "step", *target, "--dispatch-id", "pack-slots",
                             "--test-command", "python3 precision.py")
        self.assertEqual(json.loads(Path(reply["measurement_file"]).read_text())["workspace_b"], 4096)

    def test_feasible_recognizes_equivalent_scalar_and_list_addresses(self):
        kernel, _ = self.start_profiled_campaign()
        harness = SCRIPTS / "panko_harness.py"
        parent = self.run_cli(harness, "select", "--op-dir", self.root)["selected"]["id"]
        kernel.write_text(kernel.read_text().replace("addrs=0", "addrs=[0, 2048]"))
        reply = self.run_cli(harness, "feasible", "--op-dir", self.root, "--op-file", kernel, "--parent", parent)
        self.assertTrue(reply["semantic_noop"])

    def test_partial_workspace_is_diagnostic_only(self):
        source = KERNEL.replace("    return a", """    extra = pl.make_tile_group(
        type=pl.TileType(shape=[n, 32], dtype=pl.DT_FP32,
                               target_memory=pl.MemorySpace.Vec), addrs=4096, mutex_ids=[2])
    return a""").replace("kernel(a)", "kernel(a, n)")
        kernel = self.prepare_profiled_kernel(source)
        target = ["--op-dir", self.root, "--op-file", kernel, "--device", "0", "--test-command", "python3 precision.py"]
        measure = [SCRIPTS / "measure.py", *target, "--test-files", '["precision.py"]',
                   "--memory-capacity", json.dumps(CAPACITY)]
        harness = SCRIPTS / "panko_harness.py"
        init = [harness, "init", *target, "--op", "kernel", "--chip-envelope", json.dumps(ENVELOPE),
                "--config", '{"bayesian_optimization_tile_search":false}', "--measurement-file"]
        partial = self.run_cli(*measure, "--out", "partial.json")
        self.assertTrue(partial["correctness_passed"])
        self.assertEqual(partial["workspace_state"], "unavailable")
        self.assertIsNone(partial["workspace_b"])
        self.assertTrue(partial["workspace_unsized"])
        self.assertEqual(partial["workspace_known_by_space"], {"Vec": 4096})
        self.assertNotIn("workspace_by_space", partial)
        self.assertIn("memory", self.run_cli(*init, "partial.json", status=3)["error"])
        legacy = dict(partial, workspace_b=4096, workspace_state="measured", workspace_by_space={"Vec": 4096})
        (self.root / "legacy.json").write_text(json.dumps(legacy))
        self.assertIn("memory", self.run_cli(*init, "legacy.json", status=3)["error"])
        state = self.root / "optimization/search_state.json"
        self.assertFalse(state.exists())

        original = kernel.read_text()
        kernel.write_text(original.replace("shape=[n, 32]", "shape=[16, 32]"))
        complete = self.run_cli(*measure, "--out", "complete.json")
        self.assertEqual((complete["workspace_state"], complete["workspace_b"]), ("measured", 6144))
        self.run_cli(*init, "complete.json")
        parent = self.run_cli(harness, "select", "--op-dir", self.root)["selected"]["id"]
        before = state.read_bytes()
        kernel.write_text(original)
        refused = self.run_cli(harness, "record", "--op-dir", self.root, "--parent", parent,
                               "--measurement-file", "partial.json", status=3)
        self.assertIn("memory", refused["error"])
        self.assertEqual(state.read_bytes(), before)

    def test_single_tiles_share_group_workspace_accounting(self):
        source = KERNEL.replace("depth=2, mutex_ids=[0, 1]", "depth=1, mutex_ids=[0]")
        tile_type = "pl.TileType(shape=[16, 32], dtype=pl.DT_FP32)"
        source = source.replace("    return a", f"    typ = {tile_type}\n    dst = ALLOCATION\n    return a")
        kernel = self.prepare_profiled_kernel(source)
        for index, (allocation, expected) in enumerate((
            ("pl.make_tile_group(type=typ, addrs=4096, mutex_ids=[1])", 6144),
            ("pl.make_tile(typ, addr=4096)", 6144),
            ("pl.make_tile(pl.TileType([16, 32], pl.DT_FP32), addr=2048)", 4096),
            ("pl.make_tile(typ, addr=a.offset)", None),
            ("pl.make_tile(a.tile_type, addr=4096)", None),
        )):
            with self.subTest(allocation=allocation):
                kernel.write_text(source.replace("ALLOCATION", allocation) + "\n" + HOST_HELPER)
                receipt = self.run_cli(SCRIPTS / "measure.py", "--op-dir", self.root, "--op-file", kernel,
                                       "--device", "0", "--test-command", "python3 precision.py",
                                       "--test-files", '["precision.py"]', "--memory-capacity", json.dumps(CAPACITY),
                                       "--out", f"{index}.json")
                self.assertTrue(receipt["correctness_passed"])
                self.assertEqual(receipt["workspace_b"], expected)
                if expected is None:
                    self.assertEqual(receipt["workspace_state"], "unavailable")
                    self.assertEqual(receipt["workspace_known_by_space"], {"Vec": 2048})
                    result = self.run_cli(SCRIPTS / "panko_harness.py", "init", "--op-dir", self.root, "--op", "kernel",
                                          "--op-file", kernel, "--device", "0", "--chip-envelope", json.dumps(ENVELOPE),
                                          "--measurement-file", f"{index}.json", status=3)
                    self.assertIn("memory", result["error"])
                    self.assertFalse((self.root / "optimization/search_state.json").exists())
                else:
                    self.assertEqual(receipt["workspace_state"], "measured")
                    self.assertEqual(receipt["workspace_by_space"], {"Vec": expected})

    def test_declared_vec_address_requires_a_verified_limit(self):
        kernel = self.prepare_profiled_kernel(KERNEL)
        capacity = dict(CAPACITY, Vec=248 * 1024)
        target = ["--op-dir", self.root, "--op-file", kernel, "--device", "0"]
        measure = [SCRIPTS / "measure.py", *target, "--test-command", "python3 precision.py",
                   "--test-files", '["precision.py"]', "--memory-capacity", json.dumps(capacity)]
        init = [SCRIPTS / "panko_harness.py", "init", *target, "--op", "kernel",
                "--chip-envelope", json.dumps(ENVELOPE), "--config", '{"bayesian_optimization_tile_search":false}',
                "--measurement-file"]
        for index, (end, limit, expected) in enumerate((
            (248 * 1024, None, "measured"), (260096, None, "unavailable"),
            (256 * 1024 + 32, 256 * 1024, "unavailable"), (248 * 1024, 240 * 1024, "unavailable"),
            (256 * 1024, 256 * 1024, "measured"), (260096, 256 * 1024, "measured"),
        )):
            with self.subTest(end=end, limit=limit):
                kernel.write_text(KERNEL.replace("addrs=0", f"addrs={end - 4096}") + "\n" + HOST_HELPER)
                limits = ["--address-limits", json.dumps({"Vec": limit})] if limit else []
                path = f"{index}.json"
                receipt = self.run_cli(*measure, *limits, "--out", path)
                self.assertEqual(receipt["workspace_state"], expected)
                if expected == "measured":
                    self.assertEqual(receipt["workspace_headroom"]["Vec"], capacity["Vec"] - end)
                    self.assertEqual(receipt["workspace_b"], end)
                else:
                    self.assertIsNone(receipt["workspace_b"])
                    self.assertIn("memory", self.run_cli(*init, path, status=3)["error"])
        self.run_cli(*init, path)
        state = json.loads((self.root / "optimization/search_state.json").read_text())
        self.assertEqual(state["measurement_contract"]["address_limits"], {"Vec": 256 * 1024})
        self.assertEqual(state["measurement_contract"]["memory_capacity"], capacity)

    def test_vec_budget_alone_does_not_authorize_a_reserved_tail(self):
        kernel = self.root / "kernel.py"
        probe = ("import json; from evaluation.workspace_probe import profile_workspace; "
                 "print(json.dumps(profile_workspace('kernel.py', {'Vec': 248 * 1024})))")
        kernel.write_text(KERNEL.replace("addrs=0", "addrs=249856"))
        at_limit = self.run_cli("-c", probe)
        self.assertEqual((at_limit["workspace_state"], at_limit["workspace_b"]), ("measured", 253952))
        self.assertEqual(at_limit["workspace_headroom"]["Vec"], 0)

        kernel.write_text(KERNEL.replace("addrs=0", "addrs=256000"))
        over_limit = self.run_cli("-c", probe)
        self.assertEqual(over_limit["workspace_state"], "unavailable")
        self.assertIsNone(over_limit["workspace_b"])
        self.assertIn("Vec needs 260096 bytes, budget is 253952", over_limit["workspace_error"])

    def test_depth_slots_use_declared_addresses(self):
        kernel = self.root / "kernel.py"
        probe = ("import json; from evaluation.workspace_probe import profile_workspace; "
                 "print(json.dumps(profile_workspace('kernel.py', {'Vec': 8192})))")
        for addresses, expected in (("1024", 5120), ("[4096, 0]", 6144), ("[0, 0]", 2048),
                                    ("[0, 8192]", None), ("[0]", None), ("unknown_base", None)):
            with self.subTest(addresses=addresses):
                kernel.write_text(KERNEL.replace("addrs=0", "addrs=" + addresses))
                result = self.run_cli("-c", probe)
                self.assertEqual(result["workspace_b"], expected)
                if expected is None:
                    self.assertEqual(result["workspace_state"], "unavailable")

    def test_single_file_crlf_snapshot_and_frontier_export(self):
        kernel = self.prepare_profiled_kernel(KERNEL)
        baseline_bytes = kernel.read_bytes().replace(b"\n", b"\r\n")
        kernel.write_bytes(baseline_bytes)
        harness = SCRIPTS / "panko_harness.py"
        target = ["--op-dir", self.root, "--op-file", kernel]
        command = "python3 precision.py"
        baseline = self.run_cli(SCRIPTS / "measure.py", *target, "--test-command", command, "--device", "0",
                                "--test-files", '["precision.py"]', "--memory-capacity", json.dumps(CAPACITY),
                                "--out", "baseline.json")
        self.run_cli(harness, "init", *target, "--op", "kernel", "--device", "0", "--test-command", command,
                     "--measurement-file", "baseline.json", "--chip-envelope", json.dumps(ENVELOPE),
                     "--config", json.dumps(dict(eval_budget=2, bayesian_optimization_tile_search=False)))
        kernel.write_bytes(b"# pending edit")
        selected = self.run_cli(harness, "select", "--op-dir", self.root)["selected"]
        self.assertTrue(selected["restored"])
        self.assertEqual(sha(kernel), baseline["code_hash"])
        self.assertEqual(kernel.read_bytes(), baseline_bytes)
        kernel.write_bytes(baseline_bytes.replace(b"[16, 32]", b"[16, 64]", 1))
        (self.root / "latency.txt").write_text("24")
        self.run_cli(harness, "step", *target, "--parent", selected["id"],
                     "--dispatch-id", "larger", "--test-command", command)
        exported, = (self.root / "optimization/frontier").glob("*.py")
        self.assertEqual(sha(exported), baseline["code_hash"])

    def test_step_commits_stop_when_baseline_spent_the_budget(self):
        kernel, _ = self.initialize()
        baseline = sha(kernel)
        cli = SCRIPTS / "panko_harness.py"
        target = ["--op-dir", self.root]
        parent = self.run_cli(cli, "select", *target)["selected"]["id"]
        kernel.write_text(KERNEL.replace("[16, 32]", "[16, 64]"))
        step = [cli, "step", *target, "--op-file", kernel, "--parent", parent,
                "--dispatch-id", "exhausted", "--test-command", "python3 precision.py"]
        self.assertEqual(self.run_cli(*step)["next"], "stop")
        path = self.root / "optimization/search_state.json"
        state = json.loads(path.read_bytes())
        self.assertEqual(state["verification"]["status"], "no_change")
        self.assertEqual(state["progress"]["stop_reason"], "eval_budget")
        self.assertEqual(state["progress"]["evals_used"], 1)
        self.assertEqual(state["progress"]["delivered_code_hash"], baseline)
        self.assertEqual(sha(kernel), baseline)
        verified = self.run_cli(cli, "verify", *target, "--auto")
        self.assertTrue(verified["done"])
        self.assertEqual(verified["status"], "no_change")
        saved = path.read_bytes()
        self.assertEqual(self.run_cli(*step)["next"], "stop")
        self.assertEqual(path.read_bytes(), saved)

    def test_step_uses_baseline_device_without_shell_export(self):
        kernel, baseline = self.start_profiled_campaign()
        harness = SCRIPTS / "panko_harness.py"
        target = ["--op-dir", self.root, "--op-file", kernel]
        command = "python3 precision.py"
        selected = self.run_cli(harness, "select", "--op-dir", self.root)["selected"]["id"]
        kernel.write_text(kernel.read_text().replace("[16, 32]", "[16, 16]", 1))
        step = [harness, "step", *target, "--parent", selected, "--dispatch-id", "first", "--test-command"]
        refused = self.run_cli(*step, "TILE_FWK_DEVICE_ID=7 " + command, status=3)
        self.assertIn("device", refused["error"])
        self.assertEqual(len(list((self.root / "optimization/profiles").iterdir())), 1)
        receipt = json.loads(Path(self.run_cli(*step, command)["measurement_file"]).read_text())
        self.assertNotEqual(receipt["code_hash"], baseline["code_hash"])
        self.assertEqual(receipt["contract"], baseline["contract"])
        self.assertEqual(receipt["contract"]["env"]["TILE_FWK_DEVICE_ID"], "0")
        self.assertTrue(receipt["correctness_passed"])

    def test_step_retries_failed_candidate_acquisition(self):
        kernel, _ = self.start_profiled_campaign()
        harness = SCRIPTS / "panko_harness.py"
        selected = self.run_cli(harness, "select", "--op-dir", self.root)["selected"]["id"]
        kernel.write_text(kernel.read_text().replace("[16, 32]", "[16, 16]", 1))
        step = [harness, "step", "--op-dir", self.root, "--op-file", kernel, "--parent", selected,
                "--dispatch-id", "retry", "--test-command", "python3 precision.py"]
        profiler = self.root / "msprof"
        original = profiler.read_bytes()
        profiler.write_text("#!/bin/sh\nexit 1\n")
        self.run_cli(*step, status=3)
        state_file = self.root / "optimization/search_state.json"
        self.assertEqual(json.loads(state_file.read_bytes())["progress"]["evals_used"], 1)
        profiler.write_bytes(original)
        result = self.run_cli(*step)
        self.assertTrue(Path(result["measurement_file"]).is_file())
        self.assertEqual(self.run_cli(*step), result)
        self.assertEqual(json.loads(state_file.read_bytes())["progress"]["evals_used"], 2)

    def stopped_verification_campaign(self):
        kernel, baseline = self.start_profiled_campaign(precision_check=(
            "from pathlib import Path\n"
            "Path('latency.txt').write_text('10' if 'shape=[16, 16]' in inspect.getsource(kernel.kernel) else '20')\n"))
        harness = SCRIPTS / "panko_harness.py"
        target = ["--op-dir", self.root]
        parent = self.run_cli(harness, "select", *target)["selected"]["id"]
        kernel.write_text(kernel.read_text().replace("shape=[16, 32]", "shape=[16, 16]"))
        candidate = sha(kernel)
        step = self.run_cli(harness, "step", *target, "--op-file", kernel, "--parent", parent,
                            "--dispatch-id", "first", "--test-command", "python3 precision.py")
        self.assertTrue(step["stop"]["stop"])
        search = {"preopt": self.root / "baseline.json", "delivered": Path(step["measurement_file"])}
        return kernel, baseline, candidate, search

    def test_verification_refuses_auto_target_drift(self):
        self.stopped_verification_campaign()
        verify = [SCRIPTS / "panko_harness.py", "verify", "--op-dir", self.root]
        stage = self.run_cli(*verify)["stage"]
        profiler = self.root / "msprof"
        profiler.write_text(profiler.read_text().replace("* launches)", "* (2 * launches))"))
        for mode in ("--acquire", "--auto"):
            self.assertIn("timing target", self.run_cli(*verify, mode, status=3)["error"])
            self.assertEqual(self.run_cli(*verify)["stage"], stage)

    def test_auto_verification_binds_each_arm_to_its_runtime_package(self):
        limits = {"Vec": 4096}
        view, _ = self.start_profiled_campaign(package=True, capacity=dict(CAPACITY, Vec=2048),
                                              address_limits=limits, precision_check=(
            "from pathlib import Path\n"
            "Path('latency.txt').write_text('10' if 'shape=[16, 16]' in inspect.getsource(kernel.kernel) else '20')\n"))
        package = self.root / "package"
        files = [package / name for name in ("kernel.py", "helper.py", "unused.py")]
        baseline = {str(path): sha(path) for path in files}
        cli = SCRIPTS / "panko_harness.py"
        target = ["--op-dir", self.root]
        parent = self.run_cli(cli, "select", *target)["selected"]["id"]
        kernel = package / "kernel.py"
        kernel.write_text(kernel.read_text().replace("shape=[16, 32]", "shape=[16, 16]"))
        candidate = {str(path): sha(path) for path in files}
        self.assertNotEqual(baseline, candidate)
        profiler = self.root / "msprof"
        original = profiler.read_text()
        step = [cli, "step", *target, "--op-file", view, "--parent", parent, "--test-command", "python3 precision.py"]
        profiler.write_text(original.replace("kernel,AI_CORE,", "other,AI_CORE,"))
        self.assertIn("timing target", self.run_cli(*step, "--dispatch-id", "drift", status=3)["error"])
        profiler.write_text(original)
        self.run_cli(*step, "--dispatch-id", "smaller")
        profiler.write_text(original.replace("kernel,AI_CORE,", "other,AI_CORE,"))
        for mode in ("--auto", "--acquire"):
            self.assertIn("timing target", self.run_cli(cli, "verify", *target, mode, status=3)["error"])
            self.assertEqual(self.run_cli(cli, "verify", *target)["stage"], 0)
        profiler.write_text(original)
        self.assertEqual(self.run_cli(cli, "verify", *target, "--auto")["status"], "confirmed")
        verification = json.loads((self.root / "optimization/search_state.json").read_bytes())["verification"]
        self.assertEqual([r["program_files"] for r in verification["receipts"]],
                         [baseline, candidate, candidate, baseline])
        for receipt in verification["receipts"]:
            self.assertEqual(receipt["contract"]["address_limits"], limits)
            self.assertEqual(receipt["workspace_state"], "measured")
            self.assertEqual(receipt["workspace_b"], 4096 if receipt["arm"] == "preopt" else 2048)

    def test_manual_verification_requires_one_new_acquisition_per_stage(self):
        kernel, baseline, candidate, search = self.stopped_verification_campaign()
        harness = SCRIPTS / "panko_harness.py"
        target = ["--op-dir", self.root]
        verify = [harness, "verify", *target]
        state_file = self.root / "optimization/search_state.json"
        previous = {}
        for i, arm in enumerate(("preopt", "delivered", "delivered", "preopt")):
            prompt = self.run_cli(*verify)
            self.assertEqual((prompt["stage"], prompt["arm"]), (i, arm))
            state_before = state_file.read_bytes()
            for stale in [search[arm], *previous.get(arm, [])]:
                self.run_cli(*verify, "--measurement-file", stale, status=3)
                self.assertEqual(state_file.read_bytes(), state_before)
            path = Path(self.run_cli(*verify, "--acquire")["measurement_file"])
            reading = json.loads(path.read_text())
            self.assertEqual(len(reading["samples"]), prompt["runs"])
            self.assertEqual(reading["code_hash"], baseline["code_hash"] if arm == "preopt" else candidate)
            forged = self.root / "wrong-count.json"
            forged.write_text(json.dumps(dict(reading, samples=reading["samples"][:-1])))
            pending = state_file.read_bytes()
            self.run_cli(*verify, "--measurement-file", forged, status=3)
            self.assertEqual(state_file.read_bytes(), pending)
            if i == 0:
                csv, = Path(reading["msprof_dir"]).rglob("op_summary.csv")
                original = csv.read_bytes()
                csv.write_bytes(original + b"\n")
                self.run_cli(*verify, "--measurement-file", path, status=3)
                self.assertEqual(state_file.read_bytes(), pending)
                csv.write_bytes(original)
            result = self.run_cli(*verify, "--measurement-file", path)
            previous.setdefault(arm, []).append(path)
        self.assertEqual(result["status"], "confirmed")
        check = json.loads(state_file.read_bytes())["verification"]
        self.assertEqual([r["step"] for r in check["receipts"]], list(range(4)))
        self.assertEqual(len({r["msprof_dir"] for r in check["receipts"]}), 4)
        self.assertEqual(sha(kernel), candidate)

    def test_partial_legacy_manual_pass_reacquires_from_first_arm(self):
        self.stopped_verification_campaign()
        state_file = self.root / "optimization/search_state.json"
        campaign = json.loads(state_file.read_bytes())
        old = campaign["verification"]
        old["step"] = 1
        old["arms"]["preopt"].update(samples=[20] * 9, reads=[20])
        state_file.write_text(json.dumps(campaign))
        acquire = self.run_cli(SCRIPTS / "panko_harness.py", "verify", "--acquire", "--op-dir", self.root)
        self.assertEqual((acquire["stage"], acquire["arm"]), (0, "preopt"))
        updated = json.loads(state_file.read_bytes())
        self.assertEqual(updated["superseded_verifications"], [old])
        self.assertEqual(updated["verification"]["arms"]["preopt"]["samples"], [])

    def test_direct_block_refuses_changes_outside_the_measured_kernel(self):
        kernel, _ = self.start_profiled_campaign(tile_search=True)
        harness = SCRIPTS / "panko_harness.py"
        selected = self.run_cli(harness, "select", "--op-dir", self.root)["selected"]["id"]
        source = kernel.read_text()
        optimization = self.root / "optimization"
        state_file = optimization / "search_state.json"
        saved_state = state_file.read_bytes()
        for candidate, reason in (
            (source + "\nHOST_SIDE_CHANGE = 123\n", "host wrapper"),
            (source.replace("auto_mutex=False", "auto_mutex=False, debug_options={}"), "debug_options"),
        ):
            with self.subTest(reason=reason):
                kernel.write_text(candidate)
                reply = self.run_cli(harness, "block", "--op-dir", self.root, "--parent", selected,
                                     "--test-command", "python3 precision.py", "--max-trials", "2", status=3)
                self.assertTrue(reply["refused"])
                self.assertIn(reason, reply["error"])
                self.assertEqual(state_file.read_bytes(), saved_state)
                self.assertFalse((optimization / "blocks/0").exists())

    def test_vector_coverage_uses_target_capacity(self):
        readings = self.run_cli("-c", """
import json
from bayesian_optimization import seed, space
cfg = {"vec#0": {"d0": 16, "d1": 32}}
hw = space.HW(dtype_bytes=4, ub_budget_kb=64)
print(json.dumps([
    seed.features(cfg, hw),
    seed.features(cfg, hw, {"ub_bytes": 32768, "vec_residency": {"vec#0": 2}}),
    seed.features(cfg, space.HW(vector_cores=4), {"capacity_elems": 4096}),
]))
""")
        self.assertEqual(readings, [{"vec#0.ub": 2048 / 65536}, {"vec#0.ub": 4096 / 32768},
                                        {"vec#0.trips": 8, "vec#0.waves": 2}])

    def test_tile_block_measures_shape_candidates_within_budget(self):
        kernel, baseline = self.start_profiled_campaign(tile_search=True)
        harness = SCRIPTS / "panko_harness.py"
        selected = self.run_cli(harness, "select", "--op-dir", self.root)["selected"]["id"]
        reply = self.run_cli(harness, "block", "--op-dir", self.root, "--parent", selected,
                             "--test-command", "python3 precision.py", "--max-trials", "2")
        self.assertTrue(reply["ran"])
        self.assertLessEqual(reply["granted"], 2)
        receipts = [json.loads(p.read_bytes()) for p in (self.root / "optimization/blocks/0").glob("tile-[0-9]*.json")]
        self.assertGreater(reply["device_trials"], 0)
        self.assertLessEqual(reply["device_trials"], reply["granted"])
        self.assertTrue(any(r["code_hash"] != baseline["code_hash"] for r in receipts))
        self.assertTrue(all(r["contract"] == baseline["contract"] and r["correctness_passed"] for r in receipts))
        self.assertEqual(kernel.read_text(), KERNEL + "\n" + HOST_HELPER)

    def test_tile_front_sources_survive_export_and_auto_verification(self):
        precision_check = (
            "import ast, math\nfrom pathlib import Path\n"
            "tree = ast.parse(inspect.getsource(kernel.kernel))\n"
            "shape = next(k.value for k in ast.walk(tree) if isinstance(k, ast.keyword) and k.arg == 'shape')\n"
            "Path('latency.txt').write_text(str(100000 / math.prod(ast.literal_eval(shape))))\n")
        kernel, baseline = self.start_profiled_campaign(
            tile_search=True, capacity=dict(CAPACITY, Vec=248 * 1024), precision_check=precision_check)
        harness = SCRIPTS / "panko_harness.py"
        target = ["--op-dir", self.root]
        parent = self.run_cli(harness, "select", *target)["selected"]["id"]
        block = self.run_cli(harness, "block", *target, "--parent", parent,
                             "--test-command", "python3 precision.py", "--max-trials", "3")
        if block.get("receipt"):
            state_file = self.root / "optimization/search_state.json"
            self.assertEqual(json.loads(state_file.read_bytes())["progress"]["evals_used"], 4)
            self.run_cli(harness, "record", *target, "--parent", parent, "--measurement-file", block["receipt"])
            self.assertEqual(json.loads(state_file.read_bytes())["progress"]["evals_used"], 4)
        stop = self.run_cli(harness, "stop", *target)
        self.assertTrue(stop["stop"], stop)
        frontier = json.loads((self.root / "optimization/frontier/frontier.json").read_text())
        rows = frontier["front"]
        fastest = min(rows, key=lambda row: row["aicore_e2e_us"])
        self.assertTrue(any(r["aicore_e2e_us"] > fastest["aicore_e2e_us"]
                            and r["workspace_b"] < fastest["workspace_b"]
                            and r["code_hash"] != baseline["code_hash"] for r in rows), rows)
        verified = self.run_cli(harness, "verify", *target, "--auto")
        self.assertEqual(set(verified["points"]), {r["code_hash"] for r in rows} - {baseline["code_hash"]})
        self.assertEqual(frontier["sources_missing"], 0)
        for row in rows:
            self.assertEqual(sha(self.root / row["source"]), row["code_hash"])
            self.assertEqual(sha(self.root / "optimization/nodes" / (row["code_hash"] + ".py")), row["code_hash"])
        campaign = json.loads((self.root / "optimization/search_state.json").read_text())
        for check in [*campaign.get("verifications", []), campaign["verification"]]:
            point = check["arms"]["delivered"]["code_hash"]
            self.assertTrue(all(r["point"] == point for r in check["receipts"]))
            self.assertEqual([r["code_hash"] for r in check["receipts"]],
                             [baseline["code_hash"], point, point, baseline["code_hash"]])
        self.assertEqual(sha(kernel), frontier["impl_code_hash"])

    def check_tile_profile_retry(self, direct):
        kernel, _ = self.start_profiled_campaign(tile_search=True)
        harness = SCRIPTS / "panko_harness.py"
        target = ["--op-dir", self.root]
        parent = self.run_cli(harness, "select", *target)["selected"]["id"]
        args = [harness, "block" if direct else "step", *target, "--parent", parent,
                "--test-command", "python3 precision.py"]
        root = self.root / "optimization"
        if direct:
            directory = root / "blocks/0"
            args += ["--max-trials", "3"]
        else:
            directory = root / "steps/retry"
            args += ["--op-file", kernel, "--dispatch-id", "retry"]
            kernel.write_text(kernel.read_text().replace("depth=2, mutex_ids=[0, 1]", "depth=1, mutex_ids=[0]"))
        base = sha(kernel)
        profiler = self.root / "msprof"
        original = profiler.read_text()
        profiler.write_text(original.replace("args = iter", "counter = Path('calls')\n"
            "n = int(counter.read_text()) + 1 if counter.exists() else 1\n"
            "counter.write_text(str(n))\nif n == 2: sys.exit(1)\nargs = iter"))
        self.run_cli(*args, status=3)
        self.assertEqual(sha(kernel), base)
        first = directory / "tile-000.json"
        progress = json.loads((root / "search_state.json").read_text())["progress"]
        self.assertEqual((progress["evals_used"], progress["block_evals"]), (2, 1))
        if not direct:
            self.assertIn("original", self.run_cli(*args[:-1], "another", status=3)["error"])
        original_receipt = first.read_bytes()
        first.write_bytes(original_receipt + b" ")
        self.assertIn("receipt changed", self.run_cli(*args, status=3)["error"])
        self.assertEqual((self.root / "calls").read_text(), "2")
        first.write_bytes(original_receipt)
        stamp = first.stat().st_mtime_ns
        result = self.run_cli(*args)
        if not direct:
            self.assertEqual(self.run_cli(*args), result)
        self.assertEqual(first.stat().st_mtime_ns, stamp)
        summary = json.loads((directory / "tile-block.json").read_text())
        self.assertEqual(summary["device_trials"], 3)
        self.assertEqual(int((self.root / "calls").read_text()), 4 + summary["confirm"].get("trials", 0))
        progress = json.loads((root / "search_state.json").read_text())["progress"]
        self.assertEqual(progress["block_evals"], 3)
        self.assertEqual(progress["evals_used"], 4)

    def test_direct_block_resumes_after_profile_failure(self):
        self.check_tile_profile_retry(direct=True)

    def test_step_resumes_after_tile_profile_failure(self):
        self.check_tile_profile_retry(direct=False)

    def prepare_retune_step(self):
        kernel, baseline = self.start_profiled_campaign(
            tile_search=True, precision_check="assert 'depth=1' not in inspect.getsource(kernel.kernel)\n")
        harness = SCRIPTS / "panko_harness.py"
        parent = self.run_cli(harness, "select", "--op-dir", self.root)["selected"]["id"]
        kernel.write_text(kernel.read_text().replace("depth=2, mutex_ids=[0, 1]", "depth=1, mutex_ids=[0]"))
        step = [harness, "step", "--op-dir", self.root, "--op-file", kernel, "--parent", parent,
                "--dispatch-id", "first", "--test-command", "python3 precision.py"]
        return kernel, baseline, step

    def test_step_measures_the_pending_f2_trial_once(self):
        self.check_pending_f2_trial(package=False)

    def test_package_step_replay_preserves_pending_f2_trial(self):
        self.check_pending_f2_trial(package=True)

    def check_pending_f2_trial(self, package):
        catalog = self.root / "actions.json"
        actions = json.loads((SCRIPTS.parent / "references/action_catalog.json").read_text())
        catalog.write_text(json.dumps([a for a in actions if a["id"] == "F-2"]))
        kernel, _ = self.start_profiled_campaign(tile_search=True, catalog=catalog, package=package,
                                                capacity=dict(CAPACITY, Vec=248 * 1024), eval_budget=8)
        harness = SCRIPTS / "panko_harness.py"
        target = ["--op-dir", self.root]
        selection = self.run_cli(harness, "select", *target)
        parent = selection["selected"]["id"]
        step = [harness, "step", *target, "--op-file", kernel, "--parent", parent,
                "--test-command", "python3 precision.py", "--dispatch-id"]
        choices = self.root / "optimization/selects.jsonl"
        before = choices.read_bytes()
        reply = self.run_cli(*step, "trial")
        proposal = reply["bayesian_optimization"]
        self.assertTrue(proposal["applied"])
        self.assertEqual(proposal["trial"], selection["bayesian_optimization"]["trial"] + 1)
        self.assertEqual(sha(kernel), proposal["code_hash"])
        self.assertEqual((reply["next"], reply["parent"]), ("evaluate", parent))
        self.assertNotIn("selection", reply)
        staged = [kernel]
        if package:
            staged.extend([self.root / "view.py.runtime.json", *(self.root / "package").glob("*.py")])
        hashes = [sha(path) for path in staged]
        again = self.run_cli(*step, "trial")
        self.assertEqual(again, reply)
        self.assertEqual([sha(path) for path in staged], hashes)
        self.assertEqual(choices.read_bytes(), before)
        self.assertEqual(sha(kernel), proposal["code_hash"])
        if package:
            helper = self.root / "package/helper.py"
            original = helper.read_bytes()
            helper.write_bytes(original + b"\n# pending coder edit\n")
            edited = [sha(path) for path in staged]
            self.assertEqual(self.run_cli(*step, "trial"), reply)
            self.assertEqual([sha(path) for path in staged], edited)
            helper.write_bytes(original)
        following = self.run_cli(*step, "next")
        reading = json.loads(Path(following["measurement_file"]).read_text())
        self.assertEqual(reading["code_hash"], proposal["code_hash"])

    def test_step_honors_record_close_before_selecting(self):
        kernel, baseline = self.start_profiled_campaign(stagnation_K=1, eval_budget=4)
        harness = SCRIPTS / "panko_harness.py"
        target = ["--op-dir", self.root]
        parent = self.run_cli(harness, "select", *target)["selected"]["id"]
        choices = self.root / "optimization/selects.jsonl"
        before = choices.read_bytes()
        kernel.write_text(kernel.read_text().replace("shape=[16, 32]", "shape=[32, 32]"))
        (self.root / "latency.txt").write_text("100")
        result = self.run_cli(harness, "step", *target, "--op-file", kernel, "--parent", parent,
                              "--dispatch-id", "slower", "--test-command", "python3 precision.py")
        self.assertFalse(result["stop"]["stop"])
        self.assertTrue(result["stop_refine"])
        self.assertEqual((result["next"], result["parent"]), ("close", parent))
        self.assertEqual(sha(kernel), baseline["code_hash"])
        self.assertEqual(choices.read_bytes(), before)
        self.run_cli(harness, "close", *target, "--action", parent)

    def test_step_rejects_first_incorrect_retune_trial_once(self):
        kernel, baseline, step = self.prepare_retune_step()
        rejected = sha(kernel)
        result = self.run_cli(*step)
        receipt = Path(result["measurement_file"])
        self.assertEqual(receipt.name, "tile-000.json")
        reading = json.loads(receipt.read_text())
        self.assertTrue(reading["test_failed"])
        self.assertEqual(reading["code_hash"], rejected)
        self.assertFalse(result["improved"])
        self.assertEqual(sha(kernel), baseline["code_hash"])
        state_file = self.root / "optimization/search_state.json"
        saved = state_file.read_bytes()
        campaign = json.loads(saved)
        self.assertEqual(campaign["eval_cache"][rejected]["s"], 0)
        self.assertEqual([point["code_hash"] for point in campaign["archive"]], [baseline["code_hash"]])
        self.assertEqual(campaign["progress"]["evals_used"], 2)
        self.assertEqual(campaign["progress"]["block_evals"], 1)
        self.assertFalse(result["stop"]["stop"])
        self.assertTrue(result["selection"]["selected"]["id"])
        self.assertEqual(self.run_cli(*step), result)
        self.assertEqual(state_file.read_bytes(), saved)
        self.assertEqual(list(receipt.parent.glob("tile-[0-9]*.json")), [receipt])

    def test_retune_without_precision_evidence_does_not_charge_a_rejection(self):
        kernel, _, step = self.prepare_retune_step()
        (self.root / "msprof").write_text("#!/bin/sh\nexit 1\n")
        self.run_cli(*step, status=3)
        campaign = json.loads((self.root / "optimization/search_state.json").read_text())
        self.assertNotIn(sha(kernel), campaign["eval_cache"])
        self.assertEqual(campaign["progress"]["evals_used"], 1)
        self.assertEqual(campaign["progress"].get("block_evals", 0), 0)

    def test_single_file_init_requires_csv_samples(self):
        kernel, baseline = self.baseline_receipt()
        baseline.write_text(json.dumps(dict(json.loads(baseline.read_bytes()), samples=[99] + [100] * 8)))
        reply = self.run_cli(SCRIPTS / "panko_harness.py", "init", "--op-dir", self.root,
                             "--op", "kernel", "--op-file", kernel, "--device", "0",
                             "--chip-envelope", json.dumps(ENVELOPE),
                             "--config", '{"bayesian_optimization_tile_search":false}',
                             "--measurement-file", baseline, status=3)
        self.assertIn("samples differ from the timing CSV", reply["error"])
        self.assertFalse((self.root / "optimization/search_state.json").exists())

    def test_single_file_record_checks_each_original_csv(self):
        kernel, _ = self.start_profiled_campaign()
        harness = SCRIPTS / "panko_harness.py"
        parent = self.run_cli(harness, "select", "--op-dir", self.root)["selected"]["id"]
        kernel.write_text(kernel.read_text().replace("[16, 32]", "[16, 16]", 1))
        measure = [SCRIPTS / "measure.py", "--op-dir", self.root, "--op-file", kernel, "--device", "0",
                   "--test-command", "python3 precision.py", "--test-files", '["precision.py"]',
                   "--memory-capacity", json.dumps(CAPACITY)]
        first, last = (self.root / name for name in ("first.json", "last.json"))
        original = self.run_cli(*measure, "--out", first)
        (self.root / "latency.txt").write_text("10")
        self.run_cli(*measure, "--out", last)
        envelope = self.root / "winner.json"

        def write_confirmation():
            readings = [{"path": str(path), "sha256": sha(path)} for path in (first, last)]
            envelope.write_text(json.dumps(dict(schema="panko-confirmed-winner-v1", op_file=str(kernel),
                                                readings=readings)))

        record = [harness, "record", "--op-dir", self.root, "--parent", parent, "--measurement-file"]
        state = self.root / "optimization/search_state.json"
        before = state.read_bytes()
        first.write_text(json.dumps(dict(original, p=1, samples=[1] * 9)))
        write_confirmation()
        for receipt in (first, envelope):
            with self.subTest(receipt=receipt.name):
                reply = self.run_cli(*record, receipt, status=3)
                self.assertIn("samples differ from the timing CSV", reply["error"])
                self.assertEqual(state.read_bytes(), before)

        first.write_text(json.dumps(original))
        write_confirmation()
        self.run_cli(*record, envelope)
        self.assertEqual(json.loads(state.read_bytes())["eval_cache"][sha(kernel)]["p"], 11)

    def test_init_checks_baseline_samples_against_csv(self):
        kernel, baseline = self.baseline_receipt()
        raw = json.loads(baseline.read_bytes())
        baseline.write_text(json.dumps(dict(raw, p=1, samples=[1] * 9)))
        reply = self.run_cli(SCRIPTS / "panko_harness.py", "init", "--op-dir", self.root, "--op", "kernel",
                             "--op-file", kernel, "--device", "0", "--chip-envelope", json.dumps(ENVELOPE),
                             "--measurement-file", baseline, status=3)
        self.assertIn("samples differ from the timing CSV", reply["error"])
        self.assertFalse((self.root / "optimization/search_state.json").exists())

    def test_record_checks_each_original_acquisition_before_confirmation(self):
        kernel, _ = self.initialize()
        harness = SCRIPTS / "panko_harness.py"
        parent = self.run_cli(harness, "select", "--op-dir", self.root)["selected"]["id"]
        kernel.write_text(KERNEL.replace("return a", "return a + 0"))
        self.template.update(code_hash=sha(kernel), program_files={str(kernel): sha(kernel)})
        first, last = self.reading("first", 12), self.reading("last", 14)
        raw = first.read_bytes()
        first.write_text(json.dumps(dict(json.loads(raw), p=1, samples=[1] * 9)))
        state = self.root / "optimization/search_state.json"
        before = state.read_bytes()
        record = [harness, "record", "--op-dir", self.root, "--parent", parent, "--measurement-file"]
        envelope = self.root / "winner.json"

        def write_envelope():
            readings = [{"path": str(path), "sha256": sha(path)} for path in (first, last)]
            envelope.write_text(json.dumps(dict(schema="panko-confirmed-winner-v1",
                                                op_file=str(kernel), readings=readings)))

        write_envelope()
        for receipt in (first, envelope):
            with self.subTest(receipt=receipt.name):
                reply = self.run_cli(*record, receipt, status=3)
                self.assertIn("samples differ from the timing CSV", reply["error"])
                self.assertEqual(state.read_bytes(), before)

        first.write_bytes(raw)
        write_envelope()
        self.run_cli(*record, envelope)
        self.assertEqual(json.loads(state.read_bytes())["eval_cache"][self.template["code_hash"]]["p"], 13)

    def test_record_remeasurement(self):
        kernel, baseline = self.initialize()
        harness = SCRIPTS / "panko_harness.py"
        record = [harness, "record", "--op-dir", self.root, "--parent", "root", "--remeasure"]
        state = self.root / "optimization/search_state.json"
        before = state.read_bytes()
        raw = ["--code-hash", sha(kernel), "--s", "1", "--p", "100", "--samples", "1,1,1"]
        fresh = self.reading("fresh", 12)
        edited = self.root / "edited.json"
        edited.write_text(json.dumps(dict(json.loads(fresh.read_bytes()), samples=[1] + [12] * 8)))
        for flags in (raw, [*raw, "--unevaluated", "oom"], ["--measurement-file", edited],
                      ["--measurement-file", baseline], ["--measurement-file", fresh, "--samples=1,1,1"]):
            self.run_cli(*record, *flags, status=3)
            self.assertEqual(state.read_bytes(), before)
        result = self.run_cli(*record, "--measurement-file", fresh)
        self.assertFalse(result["charged"])
        self.assertEqual(result["aicore_e2e_us"], 56)
        after = state.read_bytes()
        self.assertEqual(json.loads(after)["eval_cache"][sha(kernel)]["samples"], [100] * 9 + [12] * 9)
        self.run_cli(*record, "--measurement-file", fresh, status=3)
        self.assertEqual(state.read_bytes(), after)

    def test_remeasuring_baseline_reprunes_tradeoff(self):
        kernel, baseline = self.start_profiled_campaign(precision_check=(
            "from pathlib import Path\n"
            "Path('latency.txt').write_text(Path('timing').read_text() if Path('timing').exists() else '100')\n"))
        harness = SCRIPTS / "panko_harness.py"
        target = ["--op-dir", self.root]
        parent = self.run_cli(harness, "select", *target)["selected"]["id"]
        original = kernel.read_bytes()
        kernel.write_bytes(original.replace(b"shape=[16, 32]", b"shape=[32, 32]"))
        (self.root / "timing").write_text("50")
        self.run_cli(harness, "step", *target, "--op-file", kernel, "--parent", parent,
                     "--dispatch-id", "tradeoff", "--test-command", "python3 precision.py")
        state = self.root / "optimization/search_state.json"
        archive = json.loads(state.read_bytes())["archive"]
        self.assertEqual(sorted((p["aicore_e2e_us"], p["workspace_b"]) for p in archive), [(50, 8192), (100, 4096)])
        kernel.write_bytes(original)
        (self.root / "timing").write_text("10")
        for i in range(3):
            receipt = f"recheck-{i}.json"
            self.run_cli(SCRIPTS / "measure.py", *target, "--op-file", kernel, "--device", "0",
                         "--test-command", "python3 precision.py", "--test-files", '["precision.py"]',
                         "--memory-capacity", json.dumps(CAPACITY), "--out", receipt)
            self.run_cli(harness, "record", *target, "--parent", parent, "--remeasure", "--measurement-file", receipt)
            if i == 1:
                # Resume with the stale front row and cache written by older versions.
                legacy = json.loads(state.read_bytes())
                del legacy["eval_cache"][baseline["code_hash"]]["reads"]
                next(p for p in legacy["archive"] if p["code_hash"] == baseline["code_hash"])["reads"] = [100]
                state.write_text(json.dumps(legacy))
        campaign = json.loads(state.read_bytes())
        self.assertEqual(campaign["progress"]["preopt_aicore_e2e_us"], 10)
        self.assertEqual([p["code_hash"] for p in campaign["archive"]], [baseline["code_hash"]])
        self.assertEqual(campaign["archive"][0]["reads"], [100, 10, 10, 10])
        self.assertEqual(campaign["progress"]["evals_used"], 2)

    def test_installed_frontier_exports_single_file(self):
        installer = Path(__file__).resolve().parents[1] / "scripts/install_opencode.py"
        runpy.run_path(str(installer))["install"](self.root)
        scripts = self.root / ".opencode/skills/pypto-pro-op-perf-panko-manual/scripts"
        self.env["PYTHONPATH"] = str(scripts)
        source, _ = self.initialize(scripts)
        stopped = self.run_cli(scripts / "panko_harness.py", "stop", "--op-dir", self.root)
        self.assertTrue(stopped["stop"])
        self.assertEqual(stopped["frontier"]["points"], 1)
        view, = (self.root / "optimization/frontier").glob("*.py")
        self.assertEqual(view.read_bytes(), source.read_bytes())

    def test_package_factory_width_survives_nested_global(self):
        package = self.root / "package"
        package.mkdir()
        (package / "factory.py").write_text("""import pypto_pro.language as pl
WIDTH = 16

def make(WIDTH):
    def reset():
        global WIDTH
        WIDTH += 1

    @pl.jit(auto_mutex=False)
    def kernel(a):
        tile = pl.make_tile_group(type=pl.TileType(shape=[16, WIDTH], dtype=pl.DT_FP32,
                                  target_memory=pl.MemorySpace.Vec), addrs=0, mutex_ids=[0])
        return a
    return kernel
""")
        (package / "entry.py").write_text("import factory\nkernel = factory.make(64)\n")
        view = self.root / "view.py"
        self.run_cli(SCRIPTS / "source_analysis/program_bundle.py", "prepare", "--root", package,
                     "--view", view, "--manifest", "manifest.json")
        probe = ("import json; from evaluation.workspace_probe import profile_workspace; "
                 "print(json.dumps(profile_workspace('view.py', {'Vec': 8192})))")
        self.assertEqual(self.run_cli("-c", probe)["workspace_by_space"], {"Vec": 4096})
        functions = (node for node in ast.walk(ast.parse(view.read_text())) if isinstance(node, ast.FunctionDef))
        reset = next(node for node in functions if node.name == "reset")
        declaration, increment = reset.body
        self.assertEqual(declaration.names, [increment.target.id])
        self.assertNotEqual(increment.target.id, "WIDTH")

    def test_prepare_resolves_initializer_reexports_in_source_order(self):
        package = self.root / "package"
        package.mkdir()
        (package / "__init__.py").write_text(
            "width = 7\nfrom . import helper, width as initial\nfrom . import helper as public\n"
            "result = helper.value + initial\nhelper = 11\nfrom . import helper as after\n")
        (package / "helper.py").write_text("value = 35\n")
        (package / "width.py").write_text("value = 999\n")
        (package / "consumer.py").write_text("from . import public as h, after\nresult = h.value + after\n")
        expected = self.run_cli("-c", "from package import consumer, result; print([result, consumer.result])")
        self.assertEqual(expected, [42, 46])
        view = self.root / "view.py"
        manifest = self.run_cli(SCRIPTS / "source_analysis/program_bundle.py", "prepare", "--root", package,
                                "--view", view, "--manifest", "manifest.json", "--entry-file", "consumer.py")
        symbols = manifest["symbol_map"]
        actual = self.run_cli("-c", f"import view; print([view.{symbols['__init__.py']['result']}, "
                                    f"view.{symbols['consumer.py']['result']}])")
        self.assertEqual(actual, expected)

    def test_stop_restores_package_after_interrupted_helper_edit(self):
        view, baseline = self.start_profiled_campaign(package=True, eval_budget=2)
        helper = self.root / "package/helper.py"
        helper.write_text("def compute(")
        stop = [SCRIPTS / "panko_harness.py", "stop", "--op-dir", self.root]
        poll = self.run_cli(*stop)
        self.assertFalse(poll["stop"])
        self.assertFalse(poll["delivered_is_best"])
        self.assertEqual(helper.read_text(), "def compute(")
        stop.extend(["--force", "interrupted"])
        nodes = self.root / "optimization/nodes"
        snapshot = nodes / (baseline["code_hash"] + ".py")
        original = snapshot.read_bytes()
        snapshot.write_bytes(original + b"# corrupt\n")
        self.assertEqual(self.run_cli(*stop, status=3)["error"], "restore_failed")
        self.assertEqual(helper.read_text(), "def compute(")
        snapshot.write_bytes(original)
        extra = helper.with_name("unexpected.py")
        extra.touch()
        self.assertEqual(self.run_cli(*stop, status=3)["error"], "restore_failed")
        self.assertFalse((self.root / "optimization/frontier/frontier.json").exists())
        extra.unlink()
        self.assertTrue(self.run_cli(*stop)["stop"])
        self.assertEqual(sha(view), baseline["code_hash"])
        self.assertEqual({str(f): sha(f) for f in helper.parent.glob("*.py")}, baseline["program_files"])
        exported, = (self.root / "optimization/frontier").glob("*.py")
        self.assertEqual(sha(exported), sha(view))
        spec = json.loads(Path(str(exported) + ".runtime.json").read_bytes())
        self.assertEqual({Path(f).name: sha(Path(f)) for f in spec["files"]},
                         {Path(f).name: h for f, h in baseline["program_files"].items()})

    def test_package_block_refuses_host_edit_after_refresh(self):
        self.start_profiled_campaign(tile_search=True, package=True)
        harness = SCRIPTS / "panko_harness.py"
        parent = self.run_cli(harness, "select", "--op-dir", self.root)["selected"]["id"]
        helper = self.root / "package/helper.py"
        helper.write_text(helper.read_text() + "\nHOST_SIDE_CHANGE = 123\n")
        self.run_cli(SCRIPTS / "source_analysis/program_bundle.py", "refresh", "--manifest", "manifest.json")
        state = self.root / "optimization/search_state.json"
        before = sha(state)
        reply = self.run_cli(harness, "block", "--op-dir", self.root, "--parent", parent,
                             "--test-command", "python3 precision.py", status=3)
        self.assertIn("host wrapper", reply["error"])
        self.assertEqual(sha(state), before)
        self.assertFalse((self.root / "optimization/blocks/0").exists())

    def test_measure_rejects_shadowed_package_helper(self):
        view = self.prepare_profiled_package(self.prepare_profiled_kernel(KERNEL))
        measure = [SCRIPTS / "measure.py", "--op-file", view, "--op-dir", self.root, "--device", "0",
                   "--test-command", "python3 precision.py", "--test-files", '["precision.py"]',
                   "--memory-capacity", json.dumps(CAPACITY), "--out"]
        good = self.run_cli(*measure, "good.json")
        self.assertTrue(good["correctness_passed"])
        self.assertEqual(good["p"], 12)
        other = self.root / "other"
        other.mkdir()
        (other / "helper.py").write_text("def compute(x):\n    return x - 1\n")
        precision = self.root / "precision.py"
        precision.write_text(f"import sys\nsys.path.insert(0, {str(other)!r})\nimport helper\n"
                             + precision.read_text() + "\nassert kernel.compute(3) == 2\n"
                             + "import package.helper\nassert package.helper.compute(3) == 4\n")
        bad = self.run_cli(*measure, "wrong.json")
        self.assertTrue(bad.get("impl_module_mismatch"), bad)
        self.assertFalse(bad["correctness_passed"])
        self.assertIsNone(bad["p"])
        # A relative import can coexist with an unrelated top-level module named helper.
        kernel = self.root / "package/kernel.py"
        kernel.write_text(kernel.read_text().replace("from helper import", "from .helper import"))
        self.run_cli(SCRIPTS / "source_analysis/program_bundle.py", "refresh", "--manifest", "manifest.json")
        precision.write_text(f"import sys\nsys.path.insert(0, {str(other)!r})\nimport helper\n"
                             "from package import kernel\nassert kernel.compute(3) == 4 and helper.compute(3) == 2\n")
        relative = self.run_cli(*measure, "relative.json")
        self.assertTrue(relative["correctness_passed"], relative)
        self.assertEqual(relative["p"], 12)

    def test_acquisition_binds_qualified_package_helper(self):
        view = self.prepare_profiled_package(self.prepare_profiled_kernel(KERNEL))
        package = self.root / "package"
        kernel = package / "kernel.py"
        kernel.write_text(kernel.read_text().replace("from helper import", "from package.helper import"))
        self.run_cli(SCRIPTS / "source_analysis/program_bundle.py", "refresh", "--manifest", "manifest.json")
        measure = [SCRIPTS / "measure.py", "--op-file", view, "--op-dir", self.root, "--device", "0",
                   "--test-command", "python3 precision.py", "--test-files", '["precision.py"]',
                   "--memory-capacity", json.dumps(CAPACITY), "--out"]
        receipt = self.run_cli(*measure, "measurement.json")
        self.assertTrue(receipt["correctness_passed"], receipt)
        self.assertEqual(receipt["p"], 12)
        helper = package / "helper.py"
        log = (Path(receipt["msprof_dir"]) / "command.log").read_text()
        check = ("import json; from pathlib import Path; from evaluation.measure_latency import impl_mismatch; "
                 "print(json.dumps(impl_mismatch(Path('report.txt').read_text(), 'view.py')))")
        for old, new in ((str(helper), str(self.root / "elsewhere/helper.py")), (sha(helper), "0" * 64)):
            with self.subTest(report_field=old):
                self.assertIn(old, log)
                (self.root / "report.txt").write_text(log.replace(old, new))
                self.assertIn("staged", self.run_cli("-c", check))
        # Model a collector that completed the launches but omitted this helper's import evidence.
        profiler = self.root / "msprof"
        profiler.write_text(profiler.read_text().replace(
            "subprocess.run([python, *args], check=True)",
            "log = subprocess.check_output([python, *args], text=True)\n"
            "print('\\n'.join(line for line in log.splitlines() if '/helper.py' not in line))"))
        incomplete = self.run_cli(*measure, "incomplete.json")
        self.assertTrue(incomplete["impl_module_mismatch"])
        self.assertFalse(incomplete["correctness_passed"])
        self.assertIsNone(incomplete["p"])

    def test_exported_package_passes_runtime_check(self):
        installer = Path(__file__).resolve().parents[1] / "scripts/install_opencode.py"
        runpy.run_path(str(installer))["install"](self.root)
        scripts = self.root / ".opencode/skills/pypto-pro-op-perf-panko-manual/scripts"
        self.env["PYTHONPATH"] = str(scripts)
        self.prepare_profiled_kernel(KERNEL)
        original = self.prepare_package(scripts)
        measure = [scripts / "measure.py", "--op-dir", self.root, "--test-command", "python3 precision.py",
                   "--device", "0", "--test-files", '["precision.py"]', "--memory-capacity", json.dumps(CAPACITY)]
        before = self.run_cli(*measure, "--op-file", original, "--out", "original.json")
        self.assertTrue(before["correctness_passed"], before)
        self.run_cli(scripts / "panko_harness.py", "init", "--op-dir", self.root, "--op", "kernel",
                     "--op-file", original, "--device", "0", "--chip-envelope", json.dumps(ENVELOPE),
                     "--config", '{"bayesian_optimization_tile_search":false,"eval_budget":1}',
                     "--measurement-file", "original.json")
        stopped = self.run_cli(scripts / "panko_harness.py", "stop", "--op-dir", self.root)
        self.assertTrue(stopped["stop"])
        self.assertEqual(stopped["frontier"]["points"], 1)
        view, = (self.root / "optimization/frontier").glob("*.py")
        self.package.rename(self.root / "old-package")
        check = ("import json, sys; from source_analysis.program_bundle import runtime_spec; "
                 "print(json.dumps(runtime_spec(sys.argv[1])))")
        spec = self.run_cli("-c", check, view)
        receipt = self.run_cli(*measure, "--op-file", view, "--out", "exported.json")
        self.assertTrue(receipt["correctness_passed"], receipt.get("p_error"))
        self.assertEqual(receipt["contract"]["test_files"], before["contract"]["test_files"])
        self.assertEqual(receipt["program_files"], spec["files"])
        self.assertEqual(receipt["p"], 12)
        root = Path(str(view) + ".program") / "mypkg"
        self.assertEqual(spec["root"], str(root))
        self.assertEqual(set(spec["required"]), {str(root / "inner/kernel.py"), str(root / "inner/helper.py")})
        members = ("__init__.py", "inner/__init__.py", "inner/kernel.py", "inner/helper.py")
        self.assertEqual(spec["files"], {str(root / name): sha(self.root / "old-package" / name) for name in members})
