# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Thompson selection: actions chosen from a kept-rate posterior.

The product score never learned anything: an action keeps the rank the
catalogue gave it however its candidates turn out, and an inserted intent,
which has no catalogue star, is rarely reached at all -- though the measured
ones are kept at about the rate of an ordinary catalogue action.
"""

from __future__ import annotations

import collections
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import DEMO_KERNEL as KERNEL, file_hash, INIT_ARGS, new_op_dir, read_state, run_cli  # noqa: E402
import action_posterior as AP  # noqa: E402


def _frontier():
    fr, i = [], 0
    for pg, cnt in ((0.9, 19), (0.6, 28), (0.3, 10)):
        for _ in range(cnt):
            i += 1
            fr.append({"id": f"u{i:02d}", "catalog_id": f"C-{i}", "prior_gain": pg,
                       "V": 0.5, "stage": "frontend"})
    for _ in range(6):
        i += 1
        fr.append({"id": f"u{i:02d}", "prior_gain": 0.5, "V": 0.3, "stage": "frontend"})
    return fr


def _shares(fr, n=3000, stats=None, table=None):
    c = collections.Counter()
    for d in range(n):
        ch, _ = AP.select(fr, lambda _: 1.0, AP.DrawContext(stats or {}, table, {}, 0, d))
        c[ch["id"]] += 1
    return c


def _mean(prior):
    return prior[0] / (prior[0] + prior[1])


class PosteriorTest(unittest.TestCase):
    def test_low_prior_and_inserted_actions_are_selected(self):
        fr = _frontier()
        c = _shares(fr)
        low = sum(c[n["id"]] for n in fr if n.get("prior_gain") == 0.3 and n.get("catalog_id"))
        ins = sum(c[n["id"]] for n in fr if not n.get("catalog_id"))
        self.assertGreater(low / 3000 / 10, 0.004, "the 0.3 class is explored")
        self.assertGreater(ins / 3000 / 6, 0.008, "inserts are explored")

    def test_the_prior_still_orders_the_classes(self):
        fr = _frontier()
        c = _shares(fr)
        hi = sum(c[n["id"]] for n in fr if n.get("prior_gain") == 0.9) / 19
        lo = sum(c[n["id"]] for n in fr if n.get("prior_gain") == 0.3 and n.get("catalog_id")) / 10
        self.assertGreater(hi, lo)
        self.assertLess(hi / lo, 15, "not the fifteenfold starvation of the product")

    def test_it_is_deterministic(self):
        fr = _frontier()
        a = [AP.select(fr, lambda _: 1.0, AP.DrawContext({}, None, {}, 7, d))[0]["id"]
             for d in range(20)]
        b = [AP.select(list(reversed(fr)), lambda _: 1.0, AP.DrawContext({}, None, {}, 7, d))[0]["id"]
             for d in range(20)]
        self.assertEqual(a, b, "frontier order must not matter")

    def test_evidence_moves_the_posterior(self):
        node = {"id": "u1", "catalog_id": "F-1", "prior_gain": 0.9, "V": 0.5}
        a0, b0, _, _ = AP.posterior(node, {}, None, {})
        stats = {}
        for _ in range(10):
            AP.observe(stats, node, False, 0.0)
        a1, b1, _, _ = AP.posterior(node, stats, None, {})
        self.assertLess(a1 / (a1 + b1), a0 / (a0 + b0) / 2)

    TABLE = {"global": {"n_meas": 2000, "n_kept": 200, "sum_log_gain_kept": 10.0},
             "by_catalog": dict(
                 {f"C-{i}": {"n_meas": 100, "n_kept": 4 + 2 * i, "sum_log_gain_kept": 1.0}
                  for i in range(8)},
                 **{"F-1": {"n_meas": 200, "n_kept": 40, "sum_log_gain_kept": 4.0},
                    "S-19": {"n_meas": 2, "n_kept": 2}}),
             "by_catalog_predicate": {
                 "F-1|has_chunked_scan=True": {"n_meas": 40, "n_kept": 2,
                                               "sum_log_gain_kept": 0.1}}}

    def test_the_table_and_its_predicate_split_are_read(self):
        node = {"id": "u1", "catalog_id": "F-1", "prior_gain": 0.9, "V": 0.5}
        plain = AP.base_prior(node, self.TABLE, {"has_chunked_scan": False})
        scan = AP.base_prior(node, self.TABLE, {"has_chunked_scan": True})
        self.assertEqual(plain[3], "table:F-1")
        self.assertLess(_mean(scan), _mean(plain) * 0.9)

    def test_predicate_cells_do_not_recount_the_marginal(self):
        """
        A cell identical to the action's marginal adds nothing, however
        many predicates repeat it (S-10 in the table here).
        """
        t = {"global": self.TABLE["global"], "by_catalog": self.TABLE["by_catalog"],
             "by_catalog_predicate": {f"F-1|p{i}=False": {"n_meas": 200, "n_kept": 40}
                                      for i in range(6)}}
        node = {"id": "u1", "catalog_id": "F-1", "V": 0.5}
        alone = _mean(AP.base_prior(node, t, {}))
        many = _mean(AP.base_prior(node, t, {f"p{i}": False for i in range(6)}))
        self.assertAlmostEqual(alone, many, places=6)

    def test_the_star_orders_unmeasured_actions(self):
        """
        The catalogue star is a hint about which actions work: with no data
        it sets where belief starts.
        """
        hi = _mean(AP.base_prior({"id": "a", "catalog_id": "X-1", "prior_gain": 0.9},
                                 self.TABLE, {}))
        lo = _mean(AP.base_prior({"id": "b", "catalog_id": "X-2", "prior_gain": 0.3},
                                 self.TABLE, {}))
        self.assertAlmostEqual(hi / lo, 3.0, places=6)

    def test_evidence_overrules_the_star(self):
        """A low-star action that keeps beats a high-star one that does not."""
        t = {"global": self.TABLE["global"], "by_catalog": dict(self.TABLE["by_catalog"],
             **{"S-10": {"n_meas": 9, "n_kept": 4}, "S-16": {"n_meas": 34, "n_kept": 0}})}
        s10 = AP.learned_prior_gain({"catalog_id": "S-10", "prior_gain": 0.3}, t, {})
        s16 = AP.learned_prior_gain({"catalog_id": "S-16", "prior_gain": 0.9}, t, {})
        self.assertGreater(s10, s16)

    def test_learned_prior_gain_is_on_the_catalogue_scale(self):
        t = self.TABLE
        mid = AP.learned_prior_gain({"catalog_id": "X-9", "prior_gain": 0.6}, t, {})
        self.assertAlmostEqual(mid, 0.6, places=3)
        for pg in (0.0, 0.3, 0.9, 1.0):
            v = AP.learned_prior_gain({"catalog_id": "X-9", "prior_gain": pg}, t, {})
            self.assertTrue(0.0 <= v <= 1.0)
        # Without a table, the fallback population prior: an ordinary action
        # at the neutral star still reads 0.6.
        self.assertAlmostEqual(AP.learned_prior_gain(
            {"catalog_id": "F-1", "prior_gain": 0.6}, None, {}), 0.6, places=3)

    def test_a_seeded_v_is_not_counted_twice(self):
        """At seeding V is a copy of the star, which the base prior already used."""
        seeded = {"id": "u1", "catalog_id": "X-3", "prior_gain": 0.9, "V": 0.9,
                  "V_source": "catalog"}
        base = AP.base_prior(seeded, self.TABLE, {})
        post = AP.posterior(seeded, {}, self.TABLE, {})
        self.assertEqual(base[:2], post[:2])
        model = dict(seeded, V_source="model")
        self.assertNotEqual(AP.posterior(model, {}, self.TABLE, {})[:2], base[:2])

    def test_a_large_cell_is_down_weighted(self):
        """
        Old code, selection-biased: the table never speaks louder than
        TABLE_CAP candidates on top of the population prior.
        """
        node = {"id": "u1", "catalog_id": "F-1", "V": 0.5}
        a, b, _, _ = AP.base_prior(node, self.TABLE, {})
        a0, b0 = AP.empirical_bayes(self.TABLE)
        self.assertAlmostEqual(a + b, a0 + b0 + AP.TABLE_CAP, places=6)

    def test_an_unmeasured_action_gets_the_population_prior(self):
        """
        Fitted to how measured actions actually spread, not a hand-set
        strength of 2 that let every unknown out-draw a measured winner.
        """
        node = {"id": "u9", "catalog_id": "I-13", "V": 0.5}
        a, b, _, src = AP.base_prior(node, self.TABLE, {})
        self.assertEqual(src, "table:global")
        a0, b0 = AP.empirical_bayes(self.TABLE)
        self.assertAlmostEqual(a, a0, places=9)
        self.assertAlmostEqual(b, b0, places=9)
        self.assertGreater(a + b, 10)

    def test_a_sparse_cell_falls_back(self):
        node = {"id": "u9", "catalog_id": "S-19", "prior_gain": 0.3, "V": 0.5}
        self.assertEqual(AP.base_prior(node, self.TABLE, {})[3], "table:global")

    def test_gain_is_off_by_default(self):
        fr = [{"id": "u1", "catalog_id": "F-1", "V": 0.5, "stage": "frontend"}]
        _, draws = AP.select(fr, lambda _: 1.0, AP.DrawContext({}, self.TABLE, {}, 0, 0))
        self.assertAlmostEqual(draws[0]["score"], draws[0]["p"], places=5)

    def test_a_broken_table_is_the_fallback(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{not json")
        self.assertIsNone(AP.load_table(f.name))
        self.assertIsNone(AP.load_table("/nonexistent/table.json"))


class SelectCliTest(unittest.TestCase):
    def test_seeding_writes_the_learned_prior_and_keeps_the_star(self):
        self._init()
        st = read_state(self.op_dir)
        self.assertEqual(st["prior_gain_source"], "learned")
        nodes = [n for n in st["nodes"]["open"] + st["nodes"]["held"] if n.get("catalog_id")]
        self.assertTrue(all("prior_gain_catalog" in n for n in nodes))
        self.assertTrue(any(n["prior_gain"] != n["prior_gain_catalog"] for n in nodes))
        self.assertTrue(all(n["V_source"] == "catalog" for n in nodes))

    def test_catalog_source_keeps_the_star(self):
        self._init({"prior_gain_source": "catalog"})
        st = read_state(self.op_dir)
        self.assertEqual(st["prior_gain_source"], "catalog")
        nodes = [n for n in st["nodes"]["open"] if n.get("catalog_id")]
        self.assertTrue(all(n["prior_gain"] == n["prior_gain_catalog"] for n in nodes))

    def test_product_is_still_the_default(self):
        self._init()
        sel = run_cli("select", "--op-dir", str(self.op_dir))[0]
        self.assertNotIn("thompson", sel)

    def test_thompson_selects_logs_and_advances_the_draw(self):
        self._init({"selection": "thompson"})
        sel = run_cli("select", "--op-dir", str(self.op_dir))[0]
        th = sel["thompson"]
        self.assertEqual(th["chosen"]["id"], sel["selected"]["id"])
        self.assertEqual(th["draw_no"], 0)
        self.assertEqual(read_state(self.op_dir)["progress"]["select_draws"], 1)
        log = next((self.op_dir / "optimization").glob("*.md")).read_text()
        self.assertIn(f"## {sel['selected']['id']} · ", log)
        self.assertIn("thompson p~", log)

    def test_measurements_are_counted_in_either_mode(self):
        self._init()
        sel = run_cli("select", "--op-dir", str(self.op_dir))[0]["selected"]
        node = next(n for n in read_state(self.op_dir)["nodes"]["open"] if n["id"] == sel["id"])
        run_cli("record", "--op-dir", str(self.op_dir), "--parent", sel["id"],
                "--code-hash", "0" * 64, "--s", "1", "--p", "120")
        key = node.get("catalog_id") or node["id"]
        self.assertEqual(read_state(self.op_dir)["action_stats"][key], {"n": 1, "kept": 0,
                                                                        "sum_log_gain": 0.0})

    def _init(self, config=None):
        self.op_file = new_op_dir(self, KERNEL)
        args = ["init", "--op-dir", str(self.op_dir), "--op", "demo",
                "--p-ref", "50", "--preopt-p", "100",
                "--preopt-hash", file_hash(self.op_file),
                "--op-file", str(self.op_file), "--allow-unversioned", *INIT_ARGS]
        if config:
            args += ["--config", json.dumps(config)]
        run_cli(*args)


if __name__ == "__main__":
    unittest.main()
