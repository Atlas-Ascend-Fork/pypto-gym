# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Selection is `V_sel = w_stage * g * V`, an argmax, and g does the exploring.

`novelty = 1/(1+tried)` never acted (every live node has tried 0) and, bounded
by one, could only push scores down. Its job moved into g: this run's own
measurements lower it, and reading the posterior's upper quantile (Bayes-UCB)
lifts actions the machine knows little about -- inserts included, whose g now
comes from the insert class's measured record rather than the model's own
self-reported prior.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import DEMO_KERNEL as KERNEL, file_hash, harness, INIT_ARGS, read_state, run_cli  # noqa: E402
import action_posterior as AP  # noqa: E402

TABLE = {"global": {"n_meas": 2000, "n_kept": 200},
         "by_catalog": dict(
             {f"C-{i}": {"n_meas": 100, "n_kept": 4 + 2 * i} for i in range(8)},
             **{"WELL": {"n_meas": 400, "n_kept": 36},
                "insert": {"n_meas": 250, "n_kept": 45}})}


class ScoreTest(unittest.TestCase):
    def test_v_sel_has_no_novelty(self):
        n = {"stage": "frontend", "prior_gain": 0.8, "V": 0.5}
        a = harness.pypto_action_priority(dict(n, tried=0), "frontend", True)
        b = harness.pypto_action_priority(dict(n, tried=5), "frontend", True)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a, 0.4, places=6)


class OptimisticGTest(unittest.TestCase):
    def test_an_unknown_action_reads_higher_than_a_known_ordinary_one(self):
        """Same expected rate; the unknown MIGHT be better, so it scores higher."""
        a0, b0 = AP.empirical_bayes(TABLE)
        mu = a0 / (a0 + b0)
        t = json.loads(json.dumps(TABLE))
        t["by_catalog"]["WELL"] = {"n_meas": 400, "n_kept": round(400 * mu)}
        unknown = AP.learned_prior_gain({"catalog_id": "NEW", "prior_gain": 0.6}, t, {},
                                        quantile=0.8)
        known = AP.learned_prior_gain({"catalog_id": "WELL", "prior_gain": 0.6}, t, {},
                                      quantile=0.8)
        self.assertAlmostEqual(unknown, 0.6, places=3)
        self.assertLess(known, unknown)

    def test_the_mean_is_still_available(self):
        g = AP.learned_prior_gain({"catalog_id": "NEW", "prior_gain": 0.6}, TABLE, {})
        self.assertAlmostEqual(g, 0.6, places=3)

    def test_this_runs_failures_lower_g(self):
        node = {"id": "u1", "catalog_id": "C-3", "prior_gain": 0.6}
        before = AP.learned_prior_gain(node, TABLE, {}, {}, 0.8)
        after = AP.learned_prior_gain(node, TABLE, {}, {"C-3": {"n": 7, "kept": 0}}, 0.8)
        self.assertLess(after, before)

    def test_an_inserts_g_ignores_its_self_reported_prior(self):
        lo = AP.learned_prior_gain({"id": "u9", "prior_gain": 0.1}, TABLE, {}, None, 0.8)
        hi = AP.learned_prior_gain({"id": "u9", "prior_gain": 0.9}, TABLE, {}, None, 0.8)
        self.assertEqual(lo, hi)
        ordinary = AP.learned_prior_gain({"catalog_id": "NEW", "prior_gain": 0.6},
                                         TABLE, {}, None, 0.8)
        self.assertGreater(lo, ordinary, "the insert class keeps 18%, above average")


class CliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.op_dir = Path(self._tmp.name) / "demo"
        self.op_dir.mkdir(parents=True)
        f = self.op_dir / "demo_impl.py"
        f.write_text(KERNEL, encoding="utf-8")
        run_cli("init", "--op-dir", str(self.op_dir), "--op", "demo", "--p-ref", "50",
                "--preopt-p", "100", "--preopt-hash", file_hash(f), "--op-file", str(f),
                "--allow-unversioned", *INIT_ARGS)

    def test_an_insert_gets_the_learned_g_and_keeps_the_models_value(self):
        run_cli("evolve", "--op-dir", str(self.op_dir), "--edits", json.dumps(
            {"insert": [{"parent": "root", "stage": "frontend",
                         "delta": "fold the affine into the GEMM", "V": 0.4,
                         "prior_gain": 0.1}]}))
        node = next(n for n in read_state(self.op_dir)["nodes"]["open"] if not n.get("catalog_id"))
        self.assertEqual(node["prior_gain_model"], 0.1)
        self.assertGreater(node["prior_gain"], 0.6)
        self.assertEqual(node["V_source"], "model")

    def test_a_seeded_v_is_neutral_not_a_copy_of_g(self):
        """A copy of g in V made every catalogue action score g squared."""
        st = read_state(self.op_dir)
        seeded = [n for n in st["nodes"]["open"] if n.get("catalog_id")]
        self.assertTrue(seeded)
        # The init-time symptom index may raise a V (that is the core's
        # symptom boost, not a copy of g), so: neutral or boosted, never g.
        self.assertTrue(all(n["V"] >= harness.SEED_V for n in seeded))
        self.assertGreater(sum(n["V"] == harness.SEED_V for n in seeded), len(seeded) // 2)
        self.assertTrue(all(n["V_source"] == "catalog" for n in seeded))

    def test_select_is_the_argmax_of_v_sel(self):
        res = run_cli("select", "--op-dir", str(self.op_dir))[0]["selected"]
        st = read_state(self.op_dir)
        live = st["nodes"]["open"]
        best = max(live, key=lambda n: harness.pypto_action_priority(n, "frontend", True))
        self.assertEqual(res["id"], best["id"])
        self.assertNotIn("explore", res)


if __name__ == "__main__":
    unittest.main()
