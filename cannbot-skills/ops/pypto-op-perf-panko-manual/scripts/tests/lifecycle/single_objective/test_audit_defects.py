# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""One case per defect found in an audit of the whole harness.

Grouped by what fails when the fix is reverted rather than by module, because
that is what a reader of a failure needs: the first two groups are run-ending,
the rest are silent.
"""

from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import harness_source, SCRIPTS_DIR  # noqa: E402,F401

import measure_latency  # noqa: E402
from _support import harness as H  # noqa: E402
import bayesian_optimization as bayesian  # noqa: E402


class RunEndingTest(unittest.TestCase):
    """Crashes with no handler, under the shipped defaults."""

    def test_the_archive_counter_does_not_read_a_second_instrument(self):
        # `p` is aicore_e2e; `mix_aic_us` is msprof's MIX_AIC and is None
        # whenever msprof produced none -- which the default objectives never
        # halt on. The second archived point then evaluated `float > None` and
        # raised out of `cmd_record` before its `save`, leaving the None row in
        # the state so every later `record` raised in the same place.
        src = harness_source()
        body = src[src.index("def archive_record("):src.index("def _frontier_dir(")]
        self.assertIn('e["aicore_e2e_us"]', body)
        self.assertNotIn('e["mix_aic_us"]', body)

    def test_an_unknown_stage_on_an_insert_is_read_as_a_stage(self):
        # `w_stage` indexes STAGE_ORDER with it, so an insert naming a stage
        # outside the vocabulary -- which SKILL.md never lists -- made every
        # later `select` raise ValueError, with the node in `open` and no way
        # for the proposer to guess the prune.
        stage, note = H.clamp_stage("backend")
        self.assertIn(stage, H.STAGE_ORDER)
        self.assertIsNotNone(note)
        self.assertEqual(H.clamp_stage("swimlane"), ("swimlane", None))
        self.assertEqual(H.w_stage(stage, "frontend", True), 1.0)


class PhaseTwoTest(unittest.TestCase):
    """Phase 2 extends a front point. Both halves of that were broken."""

    def test_a_non_improving_candidate_rolls_back_to_the_base(self):
        # It rolled back to the global best instead of to the point `select`
        # restored, so candidate 1 extended the front point and candidates
        # 2..K silently re-ran the greedy search from the global best.
        src = harness_source()
        body = src[src.index("def _apply_ratchet("):src.index("def _halt_on_broken_chain")]
        self.assertIn("_refine_base(r)", body)
        self.assertIn("_base_hash(st, a.op_dir)", body)

    def test_a_close_with_no_candidate_still_advances_the_stage_and_the_base(self):
        src = harness_source()
        body = src[src.index("def _close_without_candidate("):]
        body = body[:body.index("\ndef ")]
        self.assertIn("_advance_stage(st, pr)", body)
        self.assertIn("_advance_front_base(st, a)", body)


class SelectionLivenessTest(unittest.TestCase):

    def test_a_deterministic_release_disproves_the_action(self):
        # `pypto_action_priority` does not read `tried`, so a released action
        # kept its exact score and the deterministic argmax handed it straight
        # back: static rejection, close, select the same action again, for the
        # whole wall clock at zero charged evaluations.
        st = {"config": dict(H.DEFAULT_CONFIG)}
        k = H.DEFAULT_CONFIG["stagnation_K"]
        self.assertEqual(H.disproved_on_structure(st, {"n": 0}, k)[0], False)
        unm = H.DEFAULT_CONFIG["unmeasured_K"]
        for rec in ({"n": k},
                    {"n_undetermined": H.DEFAULT_CONFIG["undetermined_K"]},
                    {"n_static": unm},
                    {"n_noop": unm},
                    {"n_timeout": unm},
                    {"n_dup": unm},
                    # Mixed kinds are the same fact about the action, which is
                    # why they share one bound.
                    {"n_static": 4, "n_dup": 4, "n_noop": unm - 8}):
            ok, why = H.disproved_on_structure(st, rec, k)
            self.assertTrue(ok, rec)
            self.assertTrue(why, rec)

    def test_a_close_aimed_at_another_action_leaves_its_refinement(self):
        # `refuse_early_close` deliberately admits a mismatched id, and the
        # unconditional `pr["refine"] = None` then threw away a DIFFERENT
        # action's local best -- already snapshotted and already beating the
        # global best, since `best_J` only moves at that action's own close.
        src = harness_source()
        body = src[src.index("def _close_without_candidate("):]
        body = body[:body.index("\ndef ")]
        self.assertIn("if mine:", body)


def _measure(run_once, warmup, runs):
    # fd 2 carries this module's progress marker by design; park it so the
    # suite's output stays clean, and run with pytest's own fd capture off.
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".err") as tf:
        saved = os.dup(2)
        try:
            os.dup2(tf.fileno(), 2)
            return measure_latency.measure(run_once, warmup=warmup,
                                           runs=runs, agg="min")
        finally:
            os.dup2(saved, 2)
            os.close(saved)


class MeasurementTest(unittest.TestCase):
    def test_a_percentage_outside_its_range_condemns_the_parse(self):
        # Not a judgement about msprof: a fraction of a part cannot exceed the
        # whole. A recorded run hit exactly this on rows whose COUNT divided
        # cleanly, so the existing refusal passed them through -- and the
        # MIX_AIC duration is read from the same rows by the same cell reader.
        row = {"Task Type": "MIX_AIC", "Op Name": "k", "aiv_time(us)": "5",
               "aic_mac_ratio": "0.5", "aiv_vec_ratio": "0.4",
               "cube_utilization(%)": "59784", "vector_utilization(%)": "50"}
        bad = measure_latency.msprof_metrics([row] * 4, 4, 1)
        self.assertIsNone(bad["p"])
        self.assertIn("out of range", bad["p_error"])
        ok = measure_latency.msprof_metrics(
            [dict(row, **{"cube_utilization(%)": "99.9"})] * 4, 4, 1)
        self.assertEqual(ok["cube_utilization"], 99.9)
        self.assertNotIn("p_error", ok)

    def test_there_is_no_fallback_onto_the_cold_call(self):
        # When the profiler printed `warmup` lines or fewer, the warm slice is
        # empty and the fallback took the WARM-UP reading instead: at the
        # default of one warm-up call, that is the cold call, which this
        # module's own header says is measurably slower. J, which scales
        # `P_ref` against `p`, was inflated by that much, the candidate was
        # reverted, and nothing downstream could tell -- `raw_count` is
        # emitted and has no reader.
        seen = []

        def one_cold_reading():
            if not seen:
                os.write(1, b"AICore End-to-End Time: 400.0\n")
            seen.append(1)

        m = _measure(one_cold_reading, warmup=1, runs=2)
        self.assertIsNone(m["p"], "the cold call is not a warm measurement")
        self.assertIn("p_error", m)
        self.assertEqual(m["raw_count"], 1)

        def always_reads():
            os.write(1, b"AICore End-to-End Time: 100.0\n")

        ok = _measure(always_reads, warmup=1, runs=2)
        self.assertEqual(ok["p"], 100.0)
        self.assertNotIn("p_error", ok)

    def test_an_objective_that_is_not_p_cannot_borrow_p_samples(self):
        with self.assertRaises(ValueError):
            H.parse_objectives({"objectives": [
                {"name": "x", "from": "m.mix_aic",
                 "samples_from": "samples", "direction": "min"}]})
        objs = H.parse_objectives({"objectives": ["aicore_e2e", "mix_aic", "workspace"]})
        self.assertEqual([o["name"] for o in objs],
                         ["aicore_e2e", "mix_aic", "workspace"])

    def test_the_pareto_readout_names_the_instrument_it_reports(self):
        """`p` is AICore End-to-End, and no other field may be labelled with it.

        This readout once wrote `p` under `mix_aic_us` as well: first a stale
        label the rename missed, then a deliberate duplicate for readers that
        predated it. Either way the two instruments were reported as one
        number, and they disagree by a wide factor on real kernels. The
        readout's behaviour is pinned in tests/front/multi_objective/test_workspace_front.py;
        what is pinned here is that no copy of `p` comes back under the other
        instrument's name.
        """
        src = harness_source()
        self.assertNotIn('"mix_aic_us": p', src)
        self.assertIn('"mix_aic_us": m.get("mix_aic")', src)

    def test_remeasuring_the_preopt_refreshes_the_preopt(self):
        # The preopt is an archive point and SKILL.md names `--remeasure` as the
        # way to settle an undetermined pair, so it is a documented thing to
        # remeasure -- and early in a run `ratchet_best_hash` IS the preopt's hash, so
        # the run reported a speedup of the preopt against itself.
        src = harness_source()
        self.assertIn('if a.code_hash == preopt_code_hash(st):', src)
        st = {"nodes": {"closed": [{"id": "root", "code_hash": "h0"},
                                   {"id": "x1", "code_hash": "h1"}]},
              "best_node": "x1"}
        self.assertEqual(H.preopt_code_hash(st), "h0")
        self.assertEqual(H.ratchet_best_hash(st), "h1")


class NormalizeOnAFreshOpDirTest(unittest.TestCase):
    """
    SKILL.md step 0 runs `normalize` BEFORE `init`, on a directory `init` has
    not made yet.
    """

    SRC = ('import pypto\nTILE = 64\n@pypto.frontend.jit()\ndef k(x):\n'
           '    pypto.set_vec_tile_shapes(16, TILE)\n    return pypto.exp(x)\n')

    def test_it_does_not_die_on_the_directory_init_has_not_made(self):
        # `optimization/` was created by `init` and by nothing else, and
        # `append_log` writes into it. The guard checked the PARENT. So the one
        # command the skill mandates before init raised FileNotFoundError on
        # every fresh op dir -- and a re-run on a dir that already had
        # `optimization/` never saw it.
        r, log = self._run(self.SRC.replace("(16, TILE)", "(16, 64)"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn('"ok": true', r.stdout)
        self.assertIn("normalize: ok=True", log or "")

    def test_the_verdict_survives_a_log_that_cannot_be_written(self):
        # The verdict was emitted AFTER the log line, so a log write that raised
        # threw away the answer the command exists to produce and handed the
        # caller a traceback instead. A record of what happened must not be able
        # to destroy what happened.
        import subprocess
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        op = os.path.join(tmp.name, "custom", "myop")
        os.makedirs(op)
        # A FILE where the log directory has to go: makedirs and open both fail.
        with open(os.path.join(op, "optimization"), "w", encoding="utf-8") as f:
            f.write("not a directory")
        before = os.path.join(op, "before.py")
        after = os.path.join(op, "after.py")
        with open(before, "w", encoding="utf-8") as f:
            f.write(self.SRC)
        with open(after, "w", encoding="utf-8") as f:
            f.write(self.SRC.replace("(16, TILE)", "(16, 64)"))
        r = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS_DIR, "panko_harness.py"),
             "normalize", "--before", before, "--after", after,
             "--op-dir", op, "--op", "myop"],
            capture_output=True, text=True, check=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('"ok": true', r.stdout)
        self.assertNotIn("Traceback", r.stderr)

    def test_a_real_refusal_is_still_a_refusal(self):
        r, log = self._run(self.SRC.replace("(16, TILE)", "(16, 8192)"))
        self.assertEqual(r.returncode, 1)
        self.assertIn('"ok": false', r.stdout)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn("normalize: ok=False", log or "")

    def _run(self, after_src):
        import subprocess
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        op = os.path.join(tmp.name, "custom", "myop")
        os.makedirs(op)
        before = os.path.join(op, "before.py")
        after = os.path.join(op, "after.py")
        with open(before, "w", encoding="utf-8") as f:
            f.write(self.SRC)
        with open(after, "w", encoding="utf-8") as f:
            f.write(after_src)
        r = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS_DIR, "panko_harness.py"),
             "normalize", "--before", before, "--after", after,
             "--op-dir", op, "--op", "myop"],
            capture_output=True, text=True, check=False)
        log = os.path.join(op, "optimization", "myop_optimization.md")
        return r, (open(log, encoding="utf-8").read() if os.path.exists(log) else None)


class SamplesPlumbingTest(unittest.TestCase):

    class _Args:
        samples = None
        preopt_samples = None
        measurement = None
        preopt_measurement = None

    def test_the_measurement_carries_the_samples(self):
        # `--measurement` exists so that no reading depends on the caller
        # remembering a flag -- and `samples` was the one that still did, which
        # is the reading the archive's bootstrap and the margin stop run on. A
        # point recorded without it is compared on exact medians while every
        # other pair uses an interval, and it is `undetermined` against
        # EVERYTHING, so the second objective alone decides its dominance.
        import json
        m = json.dumps({"p": 100.0, "samples": [99.0, 101.0, 100.0]})
        self.assertEqual(H.cli_samples(self._args(measurement=m)),
                         [99.0, 101.0, 100.0])
        self.assertEqual(
            H.cli_samples(self._args(preopt_measurement=m), "preopt_"),
            [99.0, 101.0, 100.0])

    def test_the_explicit_flag_still_wins(self):
        import json
        m = json.dumps({"p": 100.0, "samples": [99.0, 101.0]})
        self.assertEqual(
            H.cli_samples(self._args(measurement=m, samples="1,2,3")),
            [1.0, 2.0, 3.0])

    def test_nothing_is_invented_when_the_measurement_has_none(self):
        import json
        for bad in (None, "not json", json.dumps({"p": 100.0}),
                    json.dumps({"samples": "9"}),
                    json.dumps({"samples": [1.0, "x"]}),
                    json.dumps([1, 2, 3])):
            self.assertEqual(H.cli_samples(self._args(measurement=bad)), [],
                             repr(bad))

    def _args(self, **kw):
        a = self._Args()
        for k, v in kw.items():
            setattr(a, k, v)
        return a


class RewriteTest(unittest.TestCase):

    def test_a_trailing_comma_does_not_break_the_rewrite(self):
        # Every insertion point is "just before the closing brace", computed as
        # end-1. A collection written with a trailing comma already has one
        # there, and the result did not parse -- while a SyntaxError carries no
        # pypto ErrCode, so nothing was carried and every trial that set the
        # knob paid for the same broken rewrite.
        src = ('import pypto\n'
               '@pypto.jit(runtime_options={"device_sched_mode": 1,})\n'
               'def k(x):\n'
               '    for i in pypto.loop(0, 4, 1,):\n'
               '        pypto.set_vec_tile_shapes(16, 64)\n'
               '    return x\n')
        cfg = {}
        for s in bayesian.apply.discover_sites(src):
            if s.kind == "global":
                cfg[s.id] = {"value": 4}
            elif s.kind == "unroll":
                cfg[s.id] = {"value": "2, 1"}
        new, n = bayesian.apply.apply(src, cfg)
        self.assertTrue(n)
        ast.parse(new)          # the assertion: it still parses

    def test_a_call_this_reader_cannot_reproduce_is_not_tunable(self):
        # `_site_text` replaces the WHOLE call span from what the six- or
        # four-knob reader parsed out of `node.args`. Anything it did not parse
        # was therefore deleted or mistyped: a fourth cube axis vanished, and
        # `set_vec_tile_shapes(32, 512, buffer_num=2)` came back as
        # `(16, 256, 1)` -- the keyword lost AND a buffer count handed to BO as
        # a third tile dimension.
        for src, want in (
                ('import pypto\npypto.set_cube_tile_shapes('
                 '[16,32],[32,64],[64,128],[8,8])\n', "cube_dyn"),
                ('import pypto\npypto.set_vec_tile_shapes('
                 '32, 512, buffer_num=2)\n', "vec_dyn"),
                ('import pypto\npypto.set_cube_tile_shapes('
                 '[16,32],[32,64],[64,128])\n', "cube"),
                ('import pypto\npypto.set_vec_tile_shapes(32, 512)\n', "vec")):
            kinds = [s.kind for s in bayesian.apply.discover_sites(src)
                     if s.kind.startswith(("vec", "cube"))]
            self.assertEqual(kinds, [want], src)


class StructureSignatureTest(unittest.TestCase):

    BASE = ('import pypto\n'
            '@pypto.frontend.jit(runtime_options={"device_sched_mode": 1})\n'
            'def k(x):\n'
            '    """Doc."""\n'
            '    pypto.set_vec_tile_shapes(16, 64)\n'
            '    a = pypto.exp(x)\n'
            '    for i in pypto.loop(0, 8, 1, unroll_list=[2]):\n'
            '        b = pypto.exp(a)\n'
            '    return b\n')

    def test_prose_does_not_move_the_signature(self):
        # `_live` un-retires an action the moment this moves, and `lever_key`
        # keys the BO study by it -- so a rewritten comment lapsed a prune AND
        # discarded a surrogate whose measurements were bought from the device.
        # Every delta a model writes rewrites comments.
        self.assertFalse(self._moved(
            self.BASE.replace("    return b", "    # a comment\n    return b")))
        self.assertFalse(self._moved(
            self.BASE.replace('"""Doc."""', '"""Different prose entirely."""')))
        self.assertFalse(self._moved(self.BASE.replace(
            "pypto.set_vec_tile_shapes(16, 64)",
            "pypto.set_vec_tile_shapes(\n        16, 64)")))

    def test_a_tunable_value_does_not_move_the_signature(self):
        self.assertFalse(self._moved(self.BASE.replace("(16, 64)", "(32, 128)")))
        self.assertFalse(self._moved(
            self.BASE.replace('"device_sched_mode": 1', '"device_sched_mode": 3')))
        self.assertFalse(self._moved(self.BASE.replace("unroll_list=[2]",
                                                       "unroll_list=[4]")))

    def test_a_structural_edit_still_moves_it(self):
        self.assertTrue(self._moved(
            self.BASE.replace("    return b",
                              "    c = pypto.sigmoid(b)\n    return c")))
        self.assertEqual(bayesian.lever.structural_signature("def ("),
                         "unparseable")

    def _moved(self, new):
        sig_of = bayesian.lever.structural_signature
        return sig_of(new) != sig_of(self.BASE)


class FootprintTest(unittest.TestCase):

    # One operation holding four tiles of vec#0 -- the numbers domain.py's
    # header opens with. `vec_residency` is still read by the search space;
    # `vec_op_groups` is what `verify` and `vec_ladder_cap` gate on.
    DOM = {"ub_bytes": 192 * 1024, "dtype_bytes": 4,
           "vec_residency": {"vec#0": 4},
           "vec_op_groups": [[["vec#0", None]] * 4]}

    def test_a_named_scalar_is_not_a_live_tile(self):
        # `_is_vector_op` keeps a tile set exactly so a tile operand can be told
        # from a loop scalar; the operand COUNT never consulted it. Indices,
        # extents and tile constants are routinely named, so the inflated count
        # was the common case -- and it halves the site's budget and its ladder,
        # the narrowing this module's header says leaves no trace.
        named = ('import pypto\n@pypto.frontend.jit()\n'
                 'def k(x, SCALE, i, n):\n'
                 '    pypto.set_vec_tile_shapes(16, 64)\n'
                 '    a = pypto.sigmoid(x)\n'
                 '    b = pypto.muls(a, SCALE)\n'
                 '    c = pypto.slice(b, i, n)\n'
                 '    return c\n')
        literal = named.replace("pypto.muls(a, SCALE)", "pypto.muls(a, 3)") \
                       .replace("pypto.slice(b, i, n)", "pypto.slice(b, 0, 16)")
        self.assertEqual(bayesian.domain.vec_residency(named),
                         bayesian.domain.vec_residency(literal))

    def test_the_ladder_cap_admits_a_trailing_extent_of_one(self):
        # `space.vec_feasible` admits it explicitly, for the measured
        # `set_vec_tile_shapes(12, 1)` legs. The cap charged the trailing field
        # 16 anyway, so it came out 16x under the bound -- and `verify` kept
        # passing the larger value, so a seed above the cap was enqueued into a
        # study whose categorical domain no longer held it and optuna raised
        # before the block's first measurement.
        cap = bayesian.domain.vec_ladder_cap(self.DOM, "vec#0", "d0", ["d0", "d1"])
        biggest = {"vec#0": {"d0": cap, "d1": 1}}
        self.assertTrue(bayesian.domain.verify(self.DOM, biggest)[0])
        over = {"vec#0": {"d0": cap * 2, "d1": 1}}
        self.assertFalse(bayesian.domain.verify(self.DOM, over)[0])

    def test_the_lever_and_the_block_judge_a_footprint_the_same_way(self):
        # The lever had no residency term at all: at those numbers domain.py
        # opens with, `(32, 512)` passed `lever.gate_draw` at 64 KB and
        # `domain.verify` rejects it at 256 KB -- the tile domain.py's header
        # records as having reached the device and returned s=0.
        cfg = {"vec#0": {"d0": 32, "d1": 512}}
        hw = bayesian.space.HW(calibrated=False)
        gate = bayesian.lever.gate_draw(cfg, hw, 4, None, self.DOM)
        self.assertFalse(gate[0])
        self.assertEqual(gate[2], bayesian.domain.verify(self.DOM, cfg)[1])

    def test_residency_is_part_of_the_fingerprint(self):
        # `capacity_elems` summarises the WORST site only, so a fusion that
        # takes a non-worst site from 2 live tiles to 4 moved that site's budget
        # while every other field stayed put: no retune, and the next block
        # inherited a study taken when that site had twice the room. Unlike
        # `vec_ub_bytes` this does not depend on any tile value, so it cannot
        # loop.
        a = {"views": [], "capacity_elems": 100, "dtype_bytes": 4,
             "vec_residency": {"vec#0": 8, "vec#2": 2},
             "vec_dtype_bytes": {"vec#0": 4}}
        b = dict(a, vec_residency={"vec#0": 8, "vec#2": 4})
        self.assertTrue(bayesian.domain.changed(a, b)[0])

    def test_the_learned_ceiling_is_in_buffer_bytes(self):
        # `domain.verify` charges a site elements x width x residency; the
        # ceiling charged elements x width and took the max over sites, so a
        # refusal caused by a 4-live site was booked against whichever site had
        # the most elements -- retiring the innocent one at exactly its own
        # legal maximum and leaving the culprit under the ceiling.
        class _Opts:
            domain = {"dtype_bytes": 2, "vec_dtype_bytes": {"vec#0": 4},
                      "vec_residency": {"vec#0": 3, "vec#1": 1}}
            dtype_bytes = 2
        w = bayesian.driver.site_widths(_Opts())
        self.assertEqual(w["vec#0"], 12)
        self.assertEqual(w["vec#1"], 2)

    def test_a_covering_seed_is_gated_on_the_derived_domain(self):
        # `block.run` filters the merged warm list through `verify`, and
        # farthest-point selection walks to the extremes -- which is where
        # verify rejects. The k picked were not the k kept.
        src = Path(bayesian.seed.__file__).read_text(encoding="utf-8")
        body = src[src.index("    def _keep(rows):"):]
        body = body[:body.index("\n    # ONE slot")]
        self.assertIn("verify_domain(opts.domain, cfg)", body)


if __name__ == "__main__":
    unittest.main()
