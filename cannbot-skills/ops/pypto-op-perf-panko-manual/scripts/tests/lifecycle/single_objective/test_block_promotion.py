# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Promoting a block winner must not need a measurement the block already took.

The winner is written to the op_file by the block and promoted by an ordinary
`record`, which the optimizer is told to make WITHOUT re-measuring -- so it
carries no reading. Two functions exist to recover one, `block_metrics` and
`block_samples`, and both read `eval_cache`. But `block_apply_winner` POPS
the winner out of `eval_cache`, on purpose: that dict is also the duplicate
index, and a winner left in it would be charged nothing. The pop therefore
removed the very entry the recovery reads, and both paths became dead code.

What that cost, on a device: `workspace` is a default objective, so the
promotion arrived with no value for it and the run HALTED with exit 4,
`objective_unmeasured`, for bytes the block had measured a moment earlier --
and before `workspace` was an objective, the same hole overwrote the archive's
readings with `samples: []`.

The suite could not see any of it. `_support.run_cli` injects
`--workspace-b` into every `record` that does not carry one, so no test in it
ever made the call the optimizer makes. These run the harness directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import CHIP_ARGS, DEMO_KERNEL as KERNEL, file_hash, HARNESS, harness  # noqa: E402

from bayesian_optimization import block  # noqa: E402

WINNER = {"vec#0": {"d0": 32, "d1": 128}}
TRIAL_WS = 512 << 10
TRIAL_SAMPLES = [89.0, 90.0, 91.0]


def run_bare(*args):
    """The harness CLI with NOTHING added -- the call the optimizer makes."""
    r = subprocess.run([sys.executable, str(HARNESS), *args],
                       env=dict(os.environ), capture_output=True, text=True)
    return json.loads(r.stdout), r.returncode


def _trial_entry():
    """The cache entry a measured block trial leaves behind."""
    return {"s": 1, "p": 90.0, "via": "block", "samples": list(TRIAL_SAMPLES),
            "m": {"workspace_b": TRIAL_WS, "workspace_state": "measured",
                  "mix_aic": 90.0}}


class StashTest(unittest.TestCase):
    """The pop keeps what it removes."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.op_file = Path(self._tmp.name) / "demo_impl.py"
        self.op_file.write_text(KERNEL, encoding="utf-8")
        self.new_src, n = block.apply_best(KERNEL, WINNER)
        self.assertTrue(n, "the fixture's winner must actually change the file")
        self.winner = hashlib.sha256(self.new_src.encode("utf-8")).hexdigest()

    def test_the_winner_still_leaves_the_duplicate_index(self):
        """Or `record` would see those bytes as evaluated and charge nothing."""
        st, ctx = self._apply()
        self.assertEqual(ctx["winner_hash"], self.winner)
        self.assertNotIn(self.winner, st["eval_cache"])

    def test_but_its_reading_survives_the_pop(self):
        st, _ = self._apply()
        self.assertEqual(harness.block_metrics(st, self.winner)["workspace_b"],
                         TRIAL_WS)
        self.assertEqual(harness.block_samples(st, self.winner), TRIAL_SAMPLES)

    def test_the_stash_answers_for_its_own_hash_only(self):
        st, _ = self._apply()
        self.assertEqual(harness.block_metrics(st, "a" * 64), {})
        self.assertEqual(harness.block_samples(st, "a" * 64), [])
        self.assertEqual(harness.block_metrics(st, None), {})

    def test_the_cache_wins_over_the_stash(self):
        """A later real measurement of those bytes is the current one."""
        st, _ = self._apply()
        st["eval_cache"][self.winner] = {"m": {"workspace_b": 7}, "samples": [1.0]}
        self.assertEqual(harness.block_metrics(st, self.winner)["workspace_b"], 7)
        self.assertEqual(harness.block_samples(st, self.winner), [1.0])

    def test_an_uncached_winner_stashes_nothing(self):
        """
        A resumed block can return a winner it never drew itself. There is
        no reading to recover and the promotion has to carry one.
        """
        st = {"eval_cache": {}}
        ctx = {"base_src": KERNEL, "op_file": str(self.op_file),
               "winner_hash": None}
        harness.block_apply_winner(st, ctx, {"best_config": WINNER})
        self.assertNotIn("block_winner", st)

    def test_the_payload_reports_the_winner_s_workspace(self):
        """So a caller that prefers to pass it explicitly can."""
        _, ctx = self._apply()
        ctx.update({"winner_mix_aic_us": 90.0, "charged": 1, "remaining": 41,
                    "meter": {"device_trials": 2}, "domain": {}})
        self.assertEqual(ctx["winner_workspace_b"], TRIAL_WS)
        self.assertEqual(ctx["winner_workspace_state"], "measured")

    def _apply(self):
        st = {"eval_cache": {self.winner: _trial_entry()}}
        ctx = {"base_src": KERNEL, "op_file": str(self.op_file),
               "winner_hash": None}
        harness.block_apply_winner(st, ctx, {"best_config": WINNER})
        return st, ctx


class PromotionTest(unittest.TestCase):
    """End to end, through the CLI, with no flags the optimizer would not pass."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.op_dir = Path(self._tmp.name) / "demo"
        (self.op_dir / "optimization").mkdir(parents=True)
        self.op_file = self.op_dir / "demo_impl.py"
        self.op_file.write_text(KERNEL, encoding="utf-8")
        payload, rc = run_bare(
            "init", "--op-dir", str(self.op_dir), "--op", "demo",
            "--p-ref", "50", "--preopt-p", "100", "--preopt-hash",
            file_hash(self.op_file), "--preopt-workspace-b", str(1 << 20),
            "--preopt-metrics", json.dumps({"mix_aic": 100.0}),
            "--op-file", str(self.op_file), "--allow-unversioned", *CHIP_ARGS,
            "--config", json.dumps({"objectives": ["aicore_e2e", "workspace"]}))
        self.assertEqual(rc, 0, payload)
        self.state = self.op_dir / "optimization" / "search_state.json"

    def test_the_promotion_is_not_refused_for_a_reading_the_block_took(self):
        winner = self._block_leaves_a_winner()
        payload, rc = run_bare("record", "--op-dir", str(self.op_dir),
                               "--parent", "u1", "--code-hash", winner,
                               "--s", "1", "--p", "90.0")
        self.assertEqual(rc, 0, payload)
        entry = json.loads(self.state.read_text())["eval_cache"][winner]
        self.assertEqual(entry["m"]["workspace_b"], TRIAL_WS,
                         "the block's bytes, not a zero and not a refusal")
        self.assertEqual(entry["samples"], TRIAL_SAMPLES,
                         "and its readings, which verify and --remeasure read")

    def test_the_promotion_is_still_charged_one_evaluation(self):
        winner = self._block_leaves_a_winner()
        before = json.loads(self.state.read_text())["progress"]["evals_used"]
        payload, rc = run_bare("record", "--op-dir", str(self.op_dir),
                               "--parent", "u1", "--code-hash", winner,
                               "--s", "1", "--p", "90.0")
        self.assertEqual(rc, 0, payload)
        after = json.loads(self.state.read_text())["progress"]["evals_used"]
        self.assertEqual(after, before + 1)
        self.assertNotEqual(payload.get("duplicate"), True)

    def test_a_reading_the_caller_passes_still_wins(self):
        winner = self._block_leaves_a_winner()
        payload, rc = run_bare("record", "--op-dir", str(self.op_dir),
                               "--parent", "u1", "--code-hash", winner,
                               "--s", "1", "--p", "90.0",
                               "--workspace-b", str(256 << 10))
        self.assertEqual(rc, 0, payload)
        entry = json.loads(self.state.read_text())["eval_cache"][winner]
        self.assertEqual(entry["m"]["workspace_b"], 256 << 10)

    def _block_leaves_a_winner(self):
        st = json.loads(self.state.read_text(encoding="utf-8"))
        new_src, _ = block.apply_best(KERNEL, WINNER)
        winner = hashlib.sha256(new_src.encode("utf-8")).hexdigest()
        st["eval_cache"][winner] = _trial_entry()
        ctx = {"base_src": KERNEL, "op_file": str(self.op_file),
               "winner_hash": None}
        harness.block_apply_winner(st, ctx, {"best_config": WINNER})
        self.state.write_text(json.dumps(st), encoding="utf-8")
        self.assertEqual(file_hash(self.op_file), winner)
        return winner


class BlockBaseHashTest(unittest.TestCase):
    """The block record names the source its trials were drawn against.

    `trials` carries every config and, for a refusal, the compiler's own message.
    None of it can be re-judged without the program the config was applied to:
    the tile domain -- the per-site widths, the running extents, the operation
    groups -- is derived from that source. The only path to it was
    `winner_hash` -> node -> parent, and a block whose winner never became a
    node has no such path, so its capacity refusals were unjudgeable offline
    for want of this one field.
    """

    def test_the_record_carries_the_base_source_hash(self):
        rec = self._fold(KERNEL)
        self.assertEqual(rec["base_hash"],
                         hashlib.sha256(KERNEL.encode("utf-8")).hexdigest())

    def test_it_is_the_hash_the_nodes_directory_is_keyed_by(self):
        """
        `file_hash` is what names `nodes/<hash>.py`, so the field resolves to
        a saved source without a tree walk.
        """
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "base_impl.py"
            f.write_text(KERNEL, encoding="utf-8")
            self.assertEqual(self._fold(KERNEL)["base_hash"], file_hash(str(f)))

    def test_a_block_with_no_winner_still_records_it(self):
        """
        The case that needs it: a refusing block whose winner was reverted and
        never became a node.
        """
        rec = self._fold(KERNEL)
        self.assertIsNone(rec["winner_hash"])
        self.assertTrue(rec["base_hash"])

    def test_two_bases_do_not_share_a_hash(self):
        other = KERNEL.replace("set_vec_tile_shapes(64, 128)",
                               "set_vec_tile_shapes(32, 256)")
        self.assertNotEqual(self._fold(KERNEL)["base_hash"],
                            self._fold(other)["base_hash"])

    def _fold(self, base_src):
        st = {"progress": {"evals_used": 0}, "eval_cache": {}}
        ctx = {"base_src": base_src, "memories": {}, "mem_key": "k",
               "refusal_rows": [], "structure": "s", "domain": {},
               "winner_hash": None, "remaining": 4, "charged": 0,
               "meter": {"device_trials": 0}, "trials": [], "faults": [],
               "started_at": "2026-10-02T00:00:00+00:00"}

        class FoldAction:
            parent, reason, live_tiles = "u3", "", 4

        harness.block_fold(FoldAction(), st, ctx, {}, {})
        self.assertIn("blocks", st)
        return st.get("blocks")[-1]


if __name__ == "__main__":
    unittest.main()
