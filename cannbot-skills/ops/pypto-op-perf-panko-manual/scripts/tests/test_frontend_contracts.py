# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Adapter contracts exercised without installing a second frontend."""
import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panko import config, delivery
import panko_harness
from bayesian_optimization import apply, driver, evaluator, space
import _registered_backend
import _support


class FrontendContractTest(unittest.TestCase):
    def test_remeasurement_keeps_one_backend_reading_per_acquisition(self):
        kernel = _support.new_op_dir(self, _support.DEMO_KERNEL)
        with patch.object(_support, "HARNESS", Path(__file__).with_name("_registered_backend.py")):
            def cli(*args):
                reply, rc = _support.run_cli(*args, "--op-dir", str(self.op_dir))
                self.assertEqual(rc, 0, reply)
                return reply

            cli("init", "--op", "demo", "--op-file", str(kernel), "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", self.pre, "--preopt-samples", "99,100,101",
                *_support.INIT_ARGS)
            for _ in range(3):
                result = cli("record", "--remeasure", "--code-hash", self.pre, "--parent", "root",
                             "--s", "1", "--p", "10", "--samples", "9,10,11")
            state = _support.read_state(self.op_dir)
            point, = state["archive"]
            self.assertEqual(result["aicore_e2e_us"], 10)
            self.assertEqual(point["reads"], [100, 10, 10, 10])
            self.assertEqual(state["eval_cache"][self.pre]["reads"], point["reads"])

    def test_registered_backend_restores_and_verifies_the_accepted_candidate(self):
        source = _support.DEMO_KERNEL
        kernel = _support.new_op_dir(self, source)
        baseline_bytes = source.replace("\n", "\r\n").encode("utf-8")
        kernel.write_bytes(baseline_bytes)
        self.pre = _support.file_hash(kernel)
        frontend = Path(__file__).with_name("_registered_backend.py")
        with patch.object(_support, "HARNESS", frontend):
            def cli(cmd, *args):
                reply, rc = _support.run_cli(cmd, "--op-dir", str(self.op_dir), *args)
                self.assertEqual(rc, 0, reply)
                return reply

            cli("init", "--op", "demo", "--op-file", str(kernel), "--preopt-samples", "99,100,101",
                "--preopt-p", "100", "--preopt-hash", self.pre, *_support.INIT_ARGS,
                "--config", json.dumps({"eval_budget": 3, "objectives": ["aicore_e2e", "workspace"]}))
            state = _support.read_state(self.op_dir)
            self.assertEqual(state["backend"], "registered-host")
            self.assertEqual((state["target"]["P_ref"], state["target"]["P_ref_via"]),
                             (1.0, "no_target:1us"))
            kernel.write_bytes(b"# pending edit")
            selected = cli("select")["selected"]
            self.assertTrue(selected["restored"])
            self.assertEqual(_support.file_hash(kernel), selected["restore_code_hash"])
            self.assertEqual(kernel.read_bytes(), baseline_bytes)
            action = selected["id"]

            def record(tile, latency, workspace):
                kernel.write_text(source.replace("64, 128", tile), encoding="utf-8")
                digest = _support.file_hash(kernel)
                reply = cli("record", "--parent", action, "--code-hash", digest,
                            "--s", "1", "--p", str(latency), "--workspace-b", str(workspace),
                            "--samples", f"{latency - 1},{latency},{latency + 1}")
                return digest, reply

            winner, kept = record("128, 128", 80, 1 << 19)
            self.assertTrue(kept["improved"], kept)
            self.assertFalse(cli("stop")["stop"])
            loser, rejected = record("32, 128", 120, 1 << 20)
            self.assertFalse(rejected["improved"], rejected)
            self.assertNotEqual(winner, loser)
            self.assertEqual(_support.file_hash(kernel), winner)
            stop = cli("stop")
            self.assertTrue(stop["stop"])
            self.assertEqual(stop["reason"], "eval_budget")
            state = _support.read_state(self.op_dir)
            self.assertEqual(state["progress"]["evals_used"], 3)
            point, = state["archive"]
            self.assertEqual((point["code_hash"], point["reads"], point["workspace_b"]),
                             (winner, [80], 1 << 19))
            self.assertEqual(state["progress"]["delivered_code_hash"], winner)
            self.assertEqual(state["verification"]["arms"]["delivered"]["code_hash"], winner)
            self.assertIn("warmup=3,", state["verification"]["protocol"])
            for arm in ("preopt", "delivered", "delivered", "preopt"):
                stage = cli("verify")
                self.assertEqual(stage["arm"], arm)
                self.assertIn("--warmup 3 --runs 15 --agg median", stage["measure"])
                baseline = arm == "preopt"
                self.assertEqual(_support.file_hash(kernel), self.pre if baseline else winner)
                samples = ["100" if baseline else "80"] * stage["runs"]
                result = cli("verify", "--samples", ",".join(samples))
            self.assertEqual(result["status"], "confirmed")
            self.assertEqual(_support.file_hash(kernel), winner)

    def test_stop_skips_block_snapshots_that_fail_record_gates(self):
        for error in ("objective_unmeasured", "wrapper_drift"):
            with self.subTest(error=error):
                kernel = _support.new_op_dir(self, _support.DEMO_KERNEL)

                def cli(cmd, *args):
                    argv = [cmd, "--op-dir", str(self.op_dir), *args]
                    return config.call_transition(panko_harness.main, argv)

                cli("init", "--op", "demo", "--op-file", str(kernel), "--p-ref", "50",
                    "--preopt-p", "100", "--allow-unversioned", *_support.INIT_ARGS,
                    "--config", '{"objectives":["aicore_e2e","workspace"],"eval_budget":3,"wall_clock_limit_s":0}')
                src = _support.DEMO_KERNEL.replace("64, 128", "32, 128")
                metrics = {}
                if error == "wrapper_drift":
                    src += "\nHOST_WORK = 1\n"
                    metrics = {"workspace_b": 0, "workspace_state": "zero"}
                kernel.write_text(src)
                with (patch.object(evaluator.RealEvaluator, "_run_test", return_value=0),
                      patch.object(evaluator.RealEvaluator, "_measure",
                                   side_effect=[dict(metrics, p=p) for p in (80, 70)])):
                    block = cli("block", "--parent", "u1", "--test-command", "true", "--max-trials", "2")
                self.assertEqual(block["device_trials"], 2, block)
                reply, rc = _support.run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                                             "--code-hash", block["code_hash"], "--s", "1", "--p", "70",
                                             "--workspace-state", "zero" if metrics else "unavailable")
                self.assertNotEqual(rc, 0, reply)
                self.assertEqual(reply["error"], error)
                reply = cli("stop")
                self.assertEqual(reply["reason"], "wall_clock")
                self.assertEqual(reply["best_code_hash"], self.pre)
                self.assertEqual(kernel.read_bytes(), _support.DEMO_KERNEL.encode())
                state = _support.read_state(self.op_dir)
                self.assertEqual([row["code_hash"] for row in state["archive"]], [self.pre])
                trials = state["blocks"][0]["trials"]
                self.assertEqual([trial["objective_us"] for trial in trials], [80, 70])
                self.assertIn(trials[0]["code_hash"], state["eval_cache"])
                for trial in trials:
                    snap = Path(_support.harness.npath(self.op_dir, trial["code_hash"]))
                    self.assertEqual(_support.file_hash(snap), trial["code_hash"])

    def test_registered_block_preserves_measured_bytes_through_stop(self):
        source = _support.DEMO_KERNEL
        kernel = _support.new_op_dir(self, source)
        staged = []

        def write_crlf(path, text):
            # The extra space makes the staged candidate differ even for a CRLF base.
            text = text.replace("32, 128", "32,  128").replace("\r\n", "\n")
            Path(path).write_bytes(text.replace("\n", "\r\n").encode())

        write_crlf(kernel, source)
        base = kernel.read_bytes()

        def measure():
            staged.append(kernel.read_bytes())
            return {"p": 80}

        def search(filename, objective, hw, options):
            tile = {"vec#0": {"d0": 32, "d1": 128}}
            self.assertEqual(objective(tile)[:2], (1, 80))
            self.assertEqual(kernel.read_bytes(), base)
            options.on_trial({"status": "ok", "objective_us": 80, "config": tile})
            return {"best_config": tile, "best_objective_us": 80, "n_trials": 1, "n_feasible": 1}

        def cli(command, *args):
            argv = [command, "--op-dir", str(self.op_dir), *args]
            return config.call_transition(panko_harness.main, argv)

        with (patch("panko.backend._frontend", _registered_backend),
              patch.object(_registered_backend, "write_source", side_effect=write_crlf) as writer,
              patch.object(evaluator.RealEvaluator, "_run_test", return_value=0),
              patch.object(evaluator.RealEvaluator, "_measure", side_effect=measure),
              patch("bayesian_optimization.block.run", side_effect=search)):
            cli("init", "--op", "demo", "--op-file", str(kernel), "--p-ref", "50", "--preopt-p", "100",
                "--preopt-hash", _support.file_hash(kernel), *_support.INIT_ARGS,
                "--config", '{"bayesian_optimization_tile_search":true,"eval_budget":2,"objectives":["aicore_e2e"]}')
            action = cli("select")["selected"]["id"]
            writer.reset_mock()
            block = cli("block", "--parent", action, "--test-command", "true")
            measured, = staged
            digest = hashlib.sha256(measured).hexdigest()
            self.assertIn(b"\r\n", measured)
            self.assertEqual(measured, base.replace(b"64, 128", b"32,  128"))
            self.assertTrue(writer.called)
            self.assertEqual(block["code_hash"], digest)
            saved_block, = _support.read_state(self.op_dir)["blocks"]
            self.assertEqual(saved_block["trials"][0]["code_hash"], digest)
            self.assertEqual(kernel.read_bytes(), measured)
            self.assertEqual(Path(_support.harness.npath(self.op_dir, digest)).read_bytes(), measured)
            cli("record", "--parent", action, "--code-hash", digest, "--s", "1", "--p", "80")
            kernel.write_bytes(b"# interrupted edit")
            self.assertTrue(cli("stop")["stop"])
            self.assertEqual(kernel.read_bytes(), measured)

    def test_classic_tile_selection_records_latin1_kernel(self):
        kernel = _support.new_op_dir(self, _support.DEMO_KERNEL)
        header = b"# coding: latin-1\r\n# caf\xe9\r\n"
        kernel.write_bytes(header + _support.DEMO_KERNEL.replace("\n", "\r\n").encode())
        catalog = self.op_dir / "catalog.json"
        actions = json.loads(_support.CATALOG.read_text())
        catalog.write_text(json.dumps([a for a in actions if a["id"] == "F-10"]))

        def cli(*args):
            reply, rc = _support.run_cli(*args, "--op-dir", str(self.op_dir))
            self.assertEqual(rc, 0, reply)
            return reply

        cli("init", "--op", "demo", "--op-file", str(kernel), "--catalog", str(catalog),
            "--allow-unversioned", "--p-ref", "50", "--preopt-p", "100", "--preopt-hash",
            _support.file_hash(kernel), *_support.INIT_ARGS,
            "--config", '{"bayesian_optimization_tile_search":true,"eval_budget":2}')
        selection = cli("select")
        proposal = selection["bayesian_optimization"]
        self.assertTrue(proposal["applied"], proposal)
        candidate = kernel.read_bytes()
        self.assertTrue(candidate.startswith(header))
        self.assertNotIn(b"\n", candidate.replace(b"\r\n", b""))
        compile(candidate, str(kernel), "exec")
        digest = _support.file_hash(kernel)
        self.assertEqual(proposal["code_hash"], digest)
        cli("record", "--parent", selection["selected"]["id"], "--s", "1", "--p", "80",
            "--code-hash", digest)
        state = _support.read_state(self.op_dir)
        self.assertEqual(state["progress"]["evals_used"], 2)
        self.assertEqual(state["eval_cache"][digest]["p"], 80)
        study, = state["bayesian_optimization_studies"].values()
        self.assertEqual([t["value"] for t in study["trials"] if t["label"] == "current"], [100])

    def test_registered_tile_selection_records_staged_crlf_bytes(self):
        kernel = _support.new_op_dir(self, _support.DEMO_KERNEL.replace("\n", "\r\n"))
        catalog = self.op_dir / "catalog.json"
        actions = json.loads(_support.CATALOG.read_text())
        catalog.write_text(json.dumps([a for a in actions if a["id"] == "F-10"]))
        frontend = Path(__file__).with_name("_registered_backend.py")
        with patch.object(_support, "HARNESS", frontend):
            def cli(*args, status=0):
                reply, rc = _support.run_cli(*args, "--op-dir", str(self.op_dir))
                self.assertEqual(rc, status, reply)
                return reply

            cli("init", "--op", "demo", "--op-file", str(kernel), "--catalog", str(catalog),
                "--preopt-p", "100", "--preopt-hash", self.pre, *_support.INIT_ARGS,
                "--config", '{"bayesian_optimization_tile_search":true,"eval_budget":2}')
            selection = cli("select")
            proposal = selection["bayesian_optimization"]
            self.assertTrue(proposal["applied"], proposal)
            staged = kernel.read_bytes()
            digest = _support.file_hash(kernel)
            self.assertEqual(proposal["code_hash"], digest)
            self.assertNotEqual(digest, self.pre)
            state = _support.read_state(self.op_dir)
            study, = state["bayesian_optimization_studies"].values()
            self.assertEqual([t["value"] for t in study["trials"] if t["label"] == "current"], [100])
            record = ("record", "--parent", selection["selected"]["id"], "--s", "1", "--p", "80", "--code-hash")
            wrong = hashlib.sha256(staged.replace(b"\r\n", b"\n")).hexdigest()
            self.assertNotEqual(wrong, digest)
            refused = cli(*record, wrong, status=3)
            self.assertEqual(refused["error"], "code_hash_mismatch")
            self.assertEqual(_support.read_state(self.op_dir), state)
            self.assertTrue(cli(*record, digest)["improved"])
            self.assertTrue(cli("stop")["stop"])
            self.assertEqual(kernel.read_bytes(), staged)
            state = _support.read_state(self.op_dir)
            self.assertEqual(state["progress"]["delivered_code_hash"], digest)
            self.assertEqual(state["progress"]["evals_used"], 2)

    def test_namespaced_tile_block_runs_one_trial_with_the_default_policy(self):
        skill = Path(__file__).resolve().parents[2]
        with patch.object(sys, "path", [str(skill), *sys.path]):
            from scripts.bayesian_optimization import block, space as tile_space

            with tempfile.TemporaryDirectory() as directory:
                kernel = Path(directory) / "kernel.py"
                kernel.write_text("def kernel(a):\n    pypto.set_vec_tile_shapes(64, 128)\n", encoding="utf-8")
                result = block.run(kernel, lambda cfg: (1, 10.0), tile_space.HW(calibrated=False),
                                   block.BlockOptions(max_trials=1))

        self.assertEqual(result["n_trials"], 1)
        self.assertEqual(result["best_objective_us"], 10.0)

    def test_device_budget_charges_only_evaluator_calls_after_static_rejection(self):
        source = "def kernel(a):\n    pypto.set_vec_tile_shapes(16, {})\n"
        sites = apply.tunable_sites(source.format(17))
        seeds = [apply.current_config(source.format(width)) for width in (17, 32, 32, 64, 128)]
        opts = driver.SearchOptions(n_trials=2, warm_start=seeds, repair=False)
        policy = replace(driver.classic_search_policy(),
                         optimize=partial(driver.optimize, device_budget=True))
        calls = []

        def measure(cfg):
            calls.append(cfg)
            return 1, 10.0

        result = driver.run_bo(measure, sites, space.HW(calibrated=False), opts, policy=policy)
        self.assertEqual(calls, [seeds[1], seeds[3]])
        self.assertEqual([r["status"] for r in result["history"]],
                         ["infeasible_static", "ok", "duplicate_config", "ok"])

    def test_device_budget_reports_exhaustion_when_all_draws_are_static_refusals(self):
        source = "def kernel(a):\n    pypto.set_vec_tile_shapes(16, 17)\n"
        limit = driver.DEFAULT_FREE_RUN_LIMIT
        opts = driver.SearchOptions(n_trials=2, warm_start=[apply.current_config(source)] * (limit + 1),
                                    repair=False)
        policy = replace(driver.classic_search_policy(),
                         optimize=partial(driver.optimize, device_budget=True),
                         result=lambda study, ctx, memory, replayed: ctx["stag"]["draws_exhausted"])
        measure = Mock(side_effect=AssertionError("static rejection reached evaluator"))
        result = driver.run_bo(measure, apply.tunable_sites(source), space.HW(calibrated=False),
                               opts, policy=policy)
        measure.assert_not_called()
        self.assertEqual((result["draws"], result["device_trials"], result["budget"]), (limit, 0, 2))

    def test_device_budget_stops_on_faults_and_failures_with_stagnation_disabled(self):
        opts = driver.SearchOptions(n_trials=5, fault_limit=2, failure_limit=2,
                                    stagnation_limit=0, startup_min_feasible=1)

        def device_fault(cfg):
            raise evaluator.EnvironmentFault("synthetic device fault", kind="device")

        sites = apply.tunable_sites("def kernel(a):\n    pypto.set_vec_tile_shapes(64, 128)\n")
        hw = space.HW(calibrated=False, dtype_bytes=4)

        def run_device_budget(study, ctx, options):
            measure = ctx["objective"]

            def counted(cfg):
                ctx["state"]["device"] += 1
                return measure(cfg)

            ctx["state"]["device"] = 0
            ctx["objective"] = counted
            driver.optimize(study, ctx, options, device_budget=True, free_run_limit=lambda ctx: 32,
                            exhausted=lambda *args: "draws exhausted")

        policy = replace(driver.classic_search_policy(), optimize=run_device_budget)

        for measure, attempts, reason in (
            (device_fault, 2, "2 consecutive environment faults (device)"),
            (lambda cfg: (0, None), 2, "2 consecutive candidates the device would not run"),
            (lambda cfg: (1, 10.), 5, None),
        ):
            with self.subTest(reason=reason):
                counted = Mock(side_effect=measure, last_detail="")
                result = driver.run_bo(counted, sites, hw, opts, policy=policy)
                self.assertEqual(counted.call_count, attempts)
                self.assertEqual(result["aborted"], reason)
                self.assertFalse(result["stopped_early"])

    def test_search_policy_can_project_the_same_budgeted_trials(self):
        sites = apply.tunable_sites("def kernel(a):\n    pypto.set_vec_tile_shapes(64, 128)\n")
        self.assertTrue(sites)
        hw = space.HW(calibrated=False, dtype_bytes=4)
        opts = driver.SearchOptions(seed=3, n_trials=4)

        def objective(cfg):
            return 1, 10.0

        expected = driver.run_bo(objective, sites, hw, opts)
        policy = replace(driver.classic_search_policy(),
                         result=lambda study, ctx, memory, replayed: ctx["history"])
        actual = driver.run_bo(objective, sites, hw, opts, policy=policy)
        self.assertEqual(len(actual), 4)
        self.assertEqual(actual, expected["history"])

    def test_cli_extension_runs_validation_before_dispatch(self):
        calls = []

        def configure(commands):
            cmd = commands.add_parser("custom")
            cmd.add_argument("--value", type=int, required=True)
            cmd.set_defaults(fn=lambda args: calls.append(("run", args.value)))

        panko_harness.main(["custom", "--value", "7"], configure=configure,
                           before_command=lambda args: calls.append(("validate", args.value)))
        self.assertEqual(calls, [("validate", 7), ("run", 7)])

    def test_refused_transition_preserves_its_reply_and_exit_code(self):
        def refuse(_):
            config.out({"refused": True})
            raise config.Refusal(2)

        with self.assertRaises(config.Refusal) as caught:
            config.call_transition(refuse, None)
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(caught.exception.reply, {"refused": True})
        self.assertEqual(config.call_transition(config.out, {"ok": True}), {"ok": True})

    def test_verification_policy_schedules_the_front_when_delivery_is_baseline(self):
        state = {"nodes": {"closed": [{"code_hash": "base"}]},
                 "progress": {"delivered_code_hash": "base"},
                 "archive": [{"code_hash": h} for h in ("base", "a", "b")],
                 "config": {"end_verify": True, "verify_front": True, "verify_runs": 8}}
        delivery.open_verification(state)
        self.assertEqual(state.pop("verification")["status"], "no_change")
        policy = replace(_registered_backend.verification_policy(), strict_restore=True)
        delivery.open_verification(state, policy=policy)
        record = state.get("verification")
        self.assertIsInstance(record, dict)
        self.assertEqual(record["point"], "a")
        self.assertEqual(state.get("verify_queue"), ["b"])
        self.assertEqual(record["schedule"], ["preopt", "delivered", "delivered", "preopt"])
        self.assertEqual(record["runs_per_step"], 4)
        self.assertEqual(record["warmup"], 3)
        delivery.next_verification(state, ".", policy=replace(policy, warmup=0))
        self.assertEqual(state.get("verification", {}).get("warmup"), 3)

    def test_verify_defaults_to_one_warmup_for_legacy_state(self):
        record = {"status": "pending", "step": 0, "runs_per_step": 4, "schedule": ["preopt"],
                  "arms": {"preopt": {"code_hash": "baseline"}}}
        state = {"op_file": "demo.py", "progress": {"stop_reason": "budget"}, "verification": record}
        with (patch.object(delivery, "load", return_value=state), patch.object(delivery, "restore", return_value=True),
              patch.object(delivery, "out") as emit):
            delivery.cmd_verify(Mock(op_dir=".", skip="", samples=None),
                                policy=_registered_backend.verification_policy())
        self.assertIn("--warmup 1 --runs 4", emit.call_args.args[0]["measure"])
