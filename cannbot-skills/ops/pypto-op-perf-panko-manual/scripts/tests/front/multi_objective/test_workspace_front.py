# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""The second objective is read on every candidate, and the front is reported.

The workspace comes from the runtime's `alloc` callback on the jit wrapper
class, with three states that must never be conflated -- above all, a probe
that failed must not read as a kernel that needs no workspace.
"""

from __future__ import annotations

import inspect
import json
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import (CHIP_ARGS, DEMO_KERNEL as KERNEL, file_hash, harness, INIT_ARGS, new_op_dir,  # noqa: E402
                      read_state, run_cli)
import workspace_probe as WP  # noqa: E402


class FakeTensor:
    def __init__(self, n):
        self.n = n

    @staticmethod
    def element_size():
        return 1

    def numel(self):
        return self.n


def _fake_module(alloc_sizes, by="arg"):
    """A module holding one jit-like kernel whose call allocates `alloc_sizes`."""
    class JitCallableWrapper:
        _cached_signature = None

        def __call__(self):
            for s in alloc_sizes:
                if by == "arg":
                    self.alloc(s)
                elif by == "result":
                    self.alloc(None, count=s)

        @staticmethod
        def compile():
            pass

        @staticmethod
        def tensor(n):
            return FakeTensor(n)

        def alloc(self, size=None, **kw):
            """A BOUND `alloc`, which is the branch the probe has to unwrap.

            `RealSignatureTest` below covers the staticmethod branch; between
            them the probe's two shapes are both exercised.
            """
            return self.tensor(size if size is not None else kw.get("count", 0))

    m = types.ModuleType("fake_impl")
    m.kernel = JitCallableWrapper()
    return m


class ProbeTest(unittest.TestCase):
    def test_measured_from_the_call(self):
        m = _fake_module([401586176])
        p = WP.WorkspaceProbe().install(m)
        for _ in range(3):
            m.kernel()
        r = p.reading(3)
        self.assertEqual(r["workspace_state"], "measured")
        self.assertEqual(r["workspace_b"], 401586176)
        self.assertEqual(r["workspace_calls"], 3)
        self.assertEqual(r["workspace_source"], ["call"])
        p.uninstall()

    def test_measured_from_the_result_when_the_call_carries_no_size(self):
        m = _fake_module([4096], by="result")
        p = WP.WorkspaceProbe().install(m)
        m.kernel()
        r = p.reading(1)
        self.assertEqual((r["workspace_b"], r["workspace_source"]), (4096, ["result"]))

    def test_a_kernel_that_ran_and_asked_for_nothing_is_zero(self):
        """The launcher calls alloc only `if (wsSize)`."""
        m = _fake_module([])
        p = WP.WorkspaceProbe().install(m)
        m.kernel()
        r = p.reading(1)
        self.assertEqual((r["workspace_b"], r["workspace_state"]), (0, "zero"))
        self.assertEqual(r["workspace_launches"], 1)

    def test_a_hook_that_saw_nothing_run_is_unavailable(self):
        """
        A wrapper always invokes its kernel once, so nothing running means
        the hook is on the wrong class object (two installs, a reload). Without
        the launch count that state read as `zero`: a workspace of 0 nothing
        measured.
        """
        m = _fake_module([4096])
        p = WP.WorkspaceProbe().install(m)
        self.addCleanup(p.uninstall)
        r = p.reading(1)
        self.assertEqual(r["workspace_state"], "unavailable")
        self.assertIsNone(r["workspace_b"])
        self.assertEqual(r["workspace_launches"], 0)

    def test_no_wrapper_is_unavailable_never_zero(self):
        p = WP.WorkspaceProbe().install(types.ModuleType("empty"))
        r = p.reading(1)
        self.assertEqual(r["workspace_state"], "unavailable")
        self.assertIsNone(r["workspace_b"])

    def test_several_kernels_per_call_are_summed_and_maxed(self):
        m = _fake_module([100, 300])
        p = WP.WorkspaceProbe().install(m)
        m.kernel()
        m.kernel()
        r = p.reading(2)
        self.assertEqual(r["workspace_b"], 300)
        self.assertEqual(r["workspace_sum_per_call_b"], 400)
        self.assertEqual(r["workspace_distinct"], [100, 300])

    def test_uninstall_restores_alloc(self):
        m = _fake_module([1])
        cls = type(m.kernel)
        orig = cls.alloc
        p = WP.WorkspaceProbe().install(m)
        self.assertIsNot(cls.alloc, orig)
        p.uninstall()
        self.assertIs(cls.alloc, orig)


class RealSignatureTest(unittest.TestCase):
    """
    The installed build carries a STATICMETHOD `alloc(size)` returning
    `data_ptr()`, called by the launcher through a getattr.
    """

    def setUp(self):
        class JitCallableWrapper:
            _cached_signature = None

            @staticmethod
            def alloc(size):
                del size
                return 140737488355328        # a plausible pointer, not a size

            def compile(self):
                pass

        self.cls = JitCallableWrapper
        self.mod = types.ModuleType("fake_impl")
        self.mod.kernel = JitCallableWrapper()

    def test_both_call_shapes_read_the_size(self):
        p = WP.WorkspaceProbe().install(self.mod)
        self.addCleanup(p.uninstall)
        getattr(self.mod.kernel, "alloc")(401586176)      # bound
        getattr(self.cls, "alloc")(401586176)             # unbound / static
        r = p.reading(2)
        self.assertEqual((r["workspace_b"], r["workspace_source"]), (401586176, ["call"]))

    def test_the_pointer_is_never_read_as_a_size(self):
        p = WP.WorkspaceProbe().install(self.mod)
        self.addCleanup(p.uninstall)
        self.assertEqual(self.mod.kernel.alloc(4096), 140737488355328)
        self.assertEqual(p.reading(1)["workspace_b"], 4096)

    def test_uninstall_restores_the_staticmethod(self):
        p = WP.WorkspaceProbe().install(self.mod)
        p.uninstall()
        self.assertIsInstance(inspect.getattr_static(self.cls, "alloc"), staticmethod)
        self.assertEqual(self.cls.alloc(1), 140737488355328)


class FrontTest(unittest.TestCase):
    def test_dominance(self):
        pts = [(10.0, 100, "a"), (12.0, 50, "b"), (12.0, 100, "c"), (9.0, 400, "d")]
        front = [x for _, _, x in harness.pareto_front(pts)]
        self.assertEqual(front, ["b", "a", "d"])


class CliTest(unittest.TestCase):
    def setUp(self):
        self.f = new_op_dir(self, KERNEL)
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", file_hash(self.f),
                "--preopt-workspace-b", str(400 << 20),
                "--op-file", str(self.f), "--allow-unversioned", *INIT_ARGS)

    def test_workspace_rides_on_kept_and_reverted_candidates(self):
        self._record(1, 90, 1600 << 20)             # kept: faster, 4x workspace
        self._record(2, 95, 380 << 20)              # reverted: slower, less workspace
        cache = read_state(self.op_dir)["eval_cache"]
        self.assertEqual(cache[f"{1:064x}"]["m"]["workspace_b"], 1600 << 20)
        self.assertEqual(cache[f"{2:064x}"]["m"]["workspace_state"], "measured")

    def test_unavailable_is_never_recorded_as_zero(self):
        """
        With workspace among the objectives the record is refused outright
        (test_mo_archive covers that); with latency alone the state is still
        recorded as `unavailable`, never as a 0.
        """
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", file_hash(self.f),
                "--op-file", str(self.f), "--allow-unversioned", *CHIP_ARGS,
                "--config", json.dumps({"objectives": ["mix_aic"]}))
        self._record(1, 90, state="unavailable")
        m = read_state(self.op_dir)["eval_cache"][f"{1:064x}"]["m"]
        self.assertEqual(m.get("workspace_state"), "unavailable")
        self.assertNotIn("workspace_b", m, "never a 0 standing in for a reading")

    def test_pareto_finds_the_kept_and_the_reverted_trade_off(self):
        """
        The existence proof, under front acceptance: a trade-off no
        slower than the preopt is kept, and one slower than it is reverted by
        the ratchet and still on the front.
        """
        self._record(1, 90, 1600 << 20)
        self._record(2, 95, 380 << 20)              # kept: on the front, <= preopt
        self._record(3, 99, 1700 << 20)             # dominated by 1
        # The preopt here holds 1 MiB (INIT_ARGS), so only a smaller program is
        # off its shadow: on the front, slower than the preopt.
        self._record(4, 101, 1 << 19)
        res = run_cli("pareto", "--op-dir", str(self.op_dir))[0]
        roles = {r["code_hash"][-1]: r["role"] for r in res["front"]}
        self.assertTrue({"1", "2", "4"} <= set(roles), roles)
        self.assertNotIn("3", roles)
        self.assertEqual(roles["2"], "kept")
        self.assertEqual(roles["4"], "reverted")
        self.assertEqual(res["front_reverted"], 1)
        self.assertTrue(Path(res["written"]).exists())

    def test_pareto_reports_mix_aic_from_its_own_reading(self):
        """The two instruments are not interchangeable.

        `p` IS AICore End-to-End. This readout used to write it under
        `mix_aic_us` as well -- first as a stale label, then kept on purpose as
        a duplicate for older readers -- so a candidate measured at 101 us
        end-to-end and 40 us in MIX_AIC was reported as 101 on both axes, and
        both ratios came out of the same column.
        """
        self._record(1, 90, 1600 << 20, mix_aic=10)        # faster, large
        self._record(2, 101, 1 << 19, mix_aic=40)          # slower, small
        res = run_cli("pareto", "--op-dir", str(self.op_dir))[0]
        rows = {r["code_hash"][-1]: r for r in res["front"]}
        self.assertEqual(res["schema"], 2)
        self.assertEqual((rows["1"]["aicore_e2e_us"], rows["1"]["mix_aic_us"]), (90, 10))
        self.assertEqual((rows["2"]["aicore_e2e_us"], rows["2"]["mix_aic_us"]), (101, 40))
        # Same reference program, each ratio on its own scale.
        self.assertEqual(rows["2"]["aicore_e2e_vs_fastest"], round(101 / 90, 4))
        self.assertEqual(rows["2"]["mix_aic_vs_fastest"], 4.0)

    def test_a_candidate_with_no_mix_aic_reading_says_so(self):
        """Missing is reported as missing, never as the other instrument."""
        self._record(1, 90, 1600 << 20, mix_aic=10)
        # `--measurement` carries the whole measure_latency result, and this one
        # has no MIX_AIC line in it.
        run_cli("record", "--op-dir", str(self.op_dir), "--parent", "u1",
                "--code-hash", f"{2:064x}", "--s", "1", "--p", "101",
                "--workspace-b", str(1 << 19),
                "--measurement", json.dumps({"aicore_e2e": 101}))
        res = run_cli("pareto", "--op-dir", str(self.op_dir))[0]
        rows = {r["code_hash"][-1]: r for r in res["front"]}
        self.assertIsNone(rows["2"]["mix_aic_us"])
        self.assertIsNone(rows["2"]["mix_aic_vs_fastest"])
        self.assertEqual(res["front_without_mix_aic"], 1)

    def _record(self, i, p, ws=None, state=None, mix_aic=None):
        args = ["record", "--op-dir", str(self.op_dir), "--parent", "u1",
                "--code-hash", f"{i:064x}", "--s", "1", "--p", str(p)]
        if ws is not None:
            args += ["--workspace-b", str(ws)]
        if state:
            args += ["--workspace-state", state]
        if mix_aic is not None:
            args += ["--metrics", json.dumps({"mix_aic": mix_aic})]
        return run_cli(*args)[0]


if __name__ == "__main__":
    unittest.main()
