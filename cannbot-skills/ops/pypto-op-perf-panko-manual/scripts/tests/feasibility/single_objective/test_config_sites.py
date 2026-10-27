# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""A per-case config dict is where a FACTORY kernel keeps its tile values.

On such a kernel the core spends its evaluations and runs ZERO tile-BO blocks:
every tile argument is an `ast.Name` bound in the factory's signature, so
`apply` read every call as `cube_dyn` / `vec_dyn` and the model swept `mm_l0` /
`vec_rows` / `n_l1` by hand, one evaluation per value.

The values it was sweeping were integer literals the whole time -- in
`_KERNEL_CONFIG`. These tests pin which of them the core may touch, and the
rule that decides it.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
import json  # noqa: E402
import tempfile  # noqa: E402

from _support import CHIP_ARGS, SCRIPTS_DIR, file_hash, run_cli  # noqa: F401,E402

from bayesian_optimization import apply, config_sites as cs, normalize, space  # noqa: E402

# The config-table idiom, reduced to what the rule has to read:
# module constants for the frozen axes, a factory whose tuning parameters carry
# defaults and whose shape parameters do not, a nested @jit, and a per-case
# dict of integer literals expanded into the factory with `**`.
GSMQ = '''import pypto

MM_L0, MM_L1 = 128, 256
K_L0, K_L1 = 128, 512
N_L0, N_L1 = 256, 512
VEC_ROWS, VEC_WIDE = 16, 512


def _make_static_kernel(M, K, E, N, rows, mm_l0=MM_L0, vec_rows=VEC_ROWS,
                        n_l1=N_L1, stitch=128, max_ws_kb=None, sched=0,
                        unroll_seq=(128, 64, 32, 1)):
    unroll = [v for v in unroll_seq if v <= rows]
    _ro = {"run_mode": pypto.RunMode.NPU, "stitch_function_max_num": stitch}
    if max_ws_kb:
        _ro["max_workspace_kb"] = max_ws_kb
    if sched:
        _ro["device_sched_mode"] = sched

    @pypto.frontend.jit(runtime_options=_ro)
    def kernel(
        x: pypto.Tensor([M, K], pypto.DT_INT8),
        y: pypto.Tensor([M, N], pypto.DT_INT8),
    ):
        for e_idx in pypto.loop(0, E, 1, name="EXPERT_LOOP"):
            for tok, tile_batch in pypto.loop_unroll(0, rows, 1, unroll_list=unroll):
                pypto.set_cube_tile_shapes([mm_l0, MM_L1], [K_L0, K_L1], [N_L0, n_l1])
                mm = pypto.matmul(x, y, pypto.DT_INT32)
                pypto.set_vec_tile_shapes(vec_rows, VEC_WIDE)
                return pypto.cast(mm, pypto.DT_FP32)
    return kernel


_UNIFORM = [(10, 512, 1024, 4, 1024), (13, 256, 512, 4, 512)]

_KERNEL_CONFIG = {
    10: dict(mm_l0=64, vec_rows=32, n_l1=256),
    13: dict(mm_l0=128, stitch=1024, max_ws_kb=524800),
}
_STATIC = {}
for _cid, _M, _K, _E, _N in _UNIFORM:
    _STATIC[(_M, _K, _E, _N)] = _make_static_kernel(
        _M, _K, _E, _N, _M // _E, **_KERNEL_CONFIG.get(_cid, {}))
'''


class AdmissionTest(unittest.TestCase):

    def test_the_tuning_parameters_are_admitted_and_the_shapes_are_not(self):
        knobs = cs.admitted(GSMQ, 10)
        self.assertEqual(
            sorted(knobs),
            ["max_ws_kb", "mm_l0", "n_l1", "sched", "stitch", "vec_rows"])
        for shape in ("M", "K", "E", "N", "rows"):
            self.assertNotIn(shape, knobs,
                             "a parameter with no default is an extent, not a knob")
        self.assertNotIn("unroll_seq", knobs,
                         "it defaults to a tuple; the core tunes integers")

    def test_a_knob_carries_the_tile_slot_it_lands_in(self):
        """The slot is what gives the knob a ladder and a footprint."""
        knobs = cs.admitted(GSMQ, 10)
        self.assertEqual(knobs["mm_l0"].slot, ("cube", "m", "L0"))
        self.assertEqual(knobs["n_l1"].slot, ("cube", "n", "L1"))
        self.assertEqual(knobs["vec_rows"].slot, ("vec", 0))

    def test_a_knob_that_reaches_the_kernel_another_way_has_no_slot(self):
        """
        `stitch` reaches the kernel through the decorator's runtime_options.
        It is tunable; it is not a tile, so it gets no slot.
        """
        knobs = cs.admitted(GSMQ, 10)
        for n in ("stitch", "max_ws_kb", "sched"):
            self.assertIsNone(knobs[n].slot)

    def test_the_value_span_is_the_literal_in_this_case_s_entry(self):
        knobs = cs.admitted(GSMQ, 10)
        self.assertEqual(knobs["mm_l0"].value, 64)
        self.assertEqual(GSMQ[knobs["mm_l0"].start:knobs["mm_l0"].end], "64")
        self.assertTrue(knobs["mm_l0"].present)
        # case 13's entry, not case 10's
        self.assertEqual(cs.admitted(GSMQ, 13)["mm_l0"].value, 128)

    def test_a_knob_the_entry_omits_is_offered_as_an_insertion(self):
        """
        The kernel as written stays reachable, and the core can also try
        setting the knob -- the convention `apply` uses for an absent runtime
        option.
        """
        k = cs.admitted(GSMQ, 10)["stitch"]
        self.assertFalse(k.present)
        self.assertIsNone(k.value)
        self.assertEqual(k.start, k.end, "a zero-width insertion site")
        self.assertEqual(GSMQ[k.start], ")", "just inside the entry's bracket")

    def test_only_the_named_case_is_visible(self):
        self.assertEqual(cs.admitted(GSMQ, 13)["stitch"].value, 1024)
        self.assertFalse(cs.admitted(GSMQ, 10)["stitch"].present)


class DefaultDenyTest(unittest.TestCase):
    """Nothing is admitted unless the whole idiom is there."""

    def test_no_key_admits_nothing(self):
        self.assertEqual(cs.admitted(GSMQ, None), {})

    def test_a_case_the_dict_does_not_mention_admits_nothing(self):
        self.assertEqual(cs.admitted(GSMQ, 99), {},
                         "inserting the entry is normalize's job, not apply's")
        self.assertFalse(cs.entry_exists(GSMQ, 99))
        self.assertTrue(cs.entry_exists(GSMQ, 10))

    def test_a_dict_nothing_expands_is_not_a_config_dict(self):
        src = GSMQ.replace("**_KERNEL_CONFIG.get(_cid, {})", "")
        self.assertEqual(cs.admitted(src, 10), {},
                         "with no factory there is no signature to read")

    def test_an_ordinary_kernel_is_untouched(self):
        src = ("import pypto\n\n\n@pypto.frontend.jit()\ndef k(a, b):\n"
               "    pypto.set_vec_tile_shapes(64, 128)\n    return a + b\n")
        self.assertIsNone(cs.analyse(src))
        self.assertEqual(cs.admitted(src, 10), {})

    def test_one_untunable_entry_does_not_disqualify_the_table(self):
        """The real `_KERNEL_CONFIG` holds `unroll_seq=(128, 64, 16, 4, 1)`.

        Requiring integers everywhere skipped the whole table over that one
        tuple, and the kernel admitted nothing at all. A value the
        core cannot tune freezes THAT KNOB and nothing else.
        """
        src = GSMQ.replace("13: dict(mm_l0=128, stitch=1024, max_ws_kb=524800),",
                           "13: dict(mm_l0=128, stitch=1024, unroll_seq=(64, 8, 1)),")
        knobs = cs.admitted(src, 13)
        self.assertEqual(knobs["mm_l0"].value, 128)
        self.assertNotIn("unroll_seq", knobs, "set, but not as a number")
        self.assertEqual(cs.admitted(src, 10)["mm_l0"].value, 64,
                         "and the other cases are unaffected")

    def test_a_knob_this_case_sets_to_a_non_number_is_left_alone(self):
        src = GSMQ.replace("10: dict(mm_l0=64, vec_rows=32, n_l1=256),",
                           '10: dict(mm_l0="auto", vec_rows=32),')
        knobs = cs.admitted(src, 10)
        self.assertNotIn("mm_l0", knobs,
                         "neither tunable (no ladder fits) nor insertable (already set)")
        self.assertEqual(knobs["vec_rows"].value, 32)


MARKED = GSMQ.replace("10: dict(mm_l0=64, vec_rows=32, n_l1=256),",
                      "10: dict(mm_l0=64, vec_rows=32, n_l1=256),  # panko:tune")


class MarkerTest(unittest.TestCase):
    """Which case is being optimised is carried IN THE KERNEL.

    A comment rather than a parameter threaded through apply, block, lever and
    the evaluator: the campaign's subject already lives in the kernel file and
    travels with every snapshot, so a candidate restored from nodes/<hash>.py
    still says which case it is.
    """

    def test_an_unmarked_kernel_admits_nothing(self):
        self.assertIsNone(cs.marked_key(GSMQ))
        self.assertEqual(cs.admitted(GSMQ), {})

    def test_the_marker_names_the_case(self):
        self.assertEqual(cs.marked_key(MARKED), 10)
        self.assertEqual(cs.admitted(MARKED)["mm_l0"].value, 64)

    def test_two_markers_name_nothing(self):
        two = MARKED.replace("13: dict(mm_l0=128, stitch=1024, max_ws_kb=524800),",
                             "13: dict(mm_l0=128),  # panko:tune")
        self.assertIsNone(cs.marked_key(two), "an ambiguous subject is no subject")
        self.assertEqual(cs.admitted(two), {})


class SiteTest(unittest.TestCase):
    """What `apply` does with them."""

    def test_an_unmarked_kernel_has_no_cfg_site(self):
        self.assertEqual([s.id for s in apply.tunable_sites(GSMQ) if s.kind == "cfg"], [])

    def test_the_marked_case_becomes_tunable_sites(self):
        sites = {s.id: s for s in apply.tunable_sites(MARKED) if s.kind == "cfg"}
        self.assertEqual(sorted(sites),
                         ["cfg:max_ws_kb", "cfg:mm_l0", "cfg:n_l1", "cfg:sched",
                          "cfg:stitch", "cfg:vec_rows"])
        self.assertEqual(sites["cfg:mm_l0"].slot, ("cube", "m", "L0"))
        self.assertEqual(apply.current_config(MARKED)["cfg:mm_l0"], {"value": 64})
        self.assertEqual(apply.current_config(MARKED)["cfg:stitch"],
                         {"value": apply.UNSET}, "not set by this case")

    def test_rewriting_a_present_knob_touches_only_that_number(self):
        out, n = apply.apply(MARKED, {"cfg:mm_l0": {"value": 128}})
        self.assertEqual(n, 1)
        self.assertIn("10: dict(mm_l0=128, vec_rows=32, n_l1=256),", out)
        self.assertIn("13: dict(mm_l0=128, stitch=1024, max_ws_kb=524800),", out,
                      "the other cases are untouched")

    def test_an_absent_knob_is_inserted_into_this_case_s_entry(self):
        out, n = apply.apply(MARKED, {"cfg:stitch": {"value": 1024}})
        self.assertEqual(n, 1)
        self.assertIn("10: dict(mm_l0=64, vec_rows=32, n_l1=256, stitch=1024)", out)

    def test_unset_leaves_the_kernel_as_written(self):
        out, _ = apply.apply(MARKED, {"cfg:stitch": {"value": apply.UNSET}})
        self.assertIn("10: dict(mm_l0=64, vec_rows=32, n_l1=256)", out)
        self.assertNotIn("stitch=", out.split("_KERNEL_CONFIG")[1].split("13:")[0])

    def test_the_rewritten_kernel_still_parses(self):
        out, _ = apply.apply(MARKED, {"cfg:mm_l0": {"value": 256},
                                      "cfg:stitch": {"value": 512},
                                      "cfg:vec_rows": {"value": 64}})
        compile(out, "<candidate>", "exec")
        self.assertIn("mm_l0=256", out)
        self.assertIn("vec_rows=64", out)
        self.assertIn("stitch=512", out)


class DomainTest(unittest.TestCase):
    """The slot is what gives the knob a ladder."""

    def test_a_cube_slot_draws_from_the_cube_ladder(self):
        self.assertEqual(space.cfg_domain(("cube", "m", "L0"), 64), space.CUBE_CHOICES)

    def test_a_leading_vec_slot_draws_from_the_vector_ladder(self):
        self.assertEqual(space.cfg_domain(("vec", 0), 32), space.VEC_CHOICES)

    def test_a_trailing_vec_slot_is_16_aligned(self):
        self.assertEqual(space.cfg_domain(("vec", 1), 512), space.VEC_TRAILING_CHOICES)

    def test_a_knob_with_no_slot_is_scaled_off_its_incumbent(self):
        self.assertEqual(space.cfg_domain(None, 1024), space.const_domain(1024))

    def test_the_incumbent_is_always_reachable(self):
        self.assertIn(40, space.cfg_domain(("cube", "m", "L0"), 40),
                      "the ladder is a sampling grid, not a constraint")


class MarkWriteTest(unittest.TestCase):
    """`mark` is a normalisation: it exposes the knobs, it does not tune them."""

    def test_marking_an_existing_entry_changes_no_value(self):
        out, action = cs.mark(GSMQ, 10)
        self.assertEqual(action, "marked")
        self.assertEqual(cs.marked_key(out), 10)
        compile(out, "<marked>", "exec")
        self.assertEqual({k: v.value for k, v in cs.admitted(out).items() if v.present},
                         {"mm_l0": 64, "vec_rows": 32, "n_l1": 256})

    def test_a_case_with_no_entry_gets_an_empty_one(self):
        """
        Case 10 had no entry, so there was nothing to tune even
        once the core could see the table. An empty entry is a no-op: the
        factory's own defaults apply, exactly as for a case the dict omits.
        """
        out, action = cs.mark(GSMQ, 99)
        self.assertEqual(action, "inserted")
        compile(out, "<marked>", "exec")
        self.assertIn("99: dict(),  # panko:tune", out)
        knobs = cs.admitted(out)
        self.assertEqual(len(knobs), 6)
        self.assertFalse(any(k.present for k in knobs.values()),
                         "every knob is an insertion site")

    def test_marking_is_idempotent_and_re_markable(self):
        once, _ = cs.mark(GSMQ, 10)
        self.assertEqual(cs.mark(once, 10)[0], once)
        self.assertEqual(cs.marked_key(cs.mark(once, 13)[0]), 13,
                         "one subject at a time")

    def test_an_ordinary_kernel_cannot_be_marked(self):
        src = ("import pypto\n\n\n@pypto.frontend.jit()\ndef k(a, b):\n"
               "    pypto.set_vec_tile_shapes(64, 128)\n    return a + b\n")
        self.assertEqual(cs.mark(src, 10), (src, "unavailable"))

    def test_the_lever_is_no_longer_reported_dead(self):
        """
        `tunable_count` is the number `normalize` reports. Counting only tile
        CALLS said 0 on a kernel whose tile values were in the table.
        """
        self.assertEqual(normalize.tunable_count(GSMQ), 0)
        marked, _ = cs.mark(GSMQ, 10)
        self.assertEqual(normalize.tunable_count(marked), 3,
                         "mm_l0, n_l1, vec_rows -- the three that carry a slot")


class InitTest(unittest.TestCase):
    """`init --config-key` writes the marker."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.op_dir = Path(self._tmp.name) / "gsmq"
        self.op_dir.mkdir(parents=True)
        self.f = self.op_dir / "gsmq_impl.py"
        self.f.write_text(GSMQ, encoding="utf-8")

    def test_the_marker_is_written_and_the_baseline_is_the_marked_file(self):
        res, rc = self._init("--config-key", "10")
        self.assertEqual(rc, 0, res)
        self.assertEqual(cs.marked_key(self.f.read_text()), 10)
        st = self._state()
        self.assertEqual(st["config_key"]["key"], 10)
        self.assertEqual(st["config_key"]["action"], "marked")
        self.assertEqual(st["nodes"]["closed"][0]["code_hash"], file_hash(self.f),
                         "the baseline is the program the run starts from")

    def test_a_case_the_table_omits_is_inserted(self):
        self._init("--config-key", "99")
        self.assertEqual(self._state()["config_key"]["action"], "inserted")
        self.assertEqual(cs.marked_key(self.f.read_text()), 99)

    def test_without_the_flag_nothing_is_marked(self):
        self._init()
        self.assertIsNone(cs.marked_key(self.f.read_text()))
        self.assertIsNone(self._state()["config_key"])

    def test_a_kernel_with_no_table_refuses_rather_than_pretending(self):
        self.f.write_text("import pypto\n\n\n@pypto.frontend.jit()\n"
                          "def k(a, b):\n    pypto.set_vec_tile_shapes(64, 128)\n"
                          "    return a + b\n", encoding="utf-8")
        res, rc = self._init("--config-key", "10")
        self.assertEqual(rc, 3)
        self.assertEqual(res["error"], "no_config_table")

    def _init(self, *extra):
        return run_cli("init", "--op-dir", str(self.op_dir), "--op", "gsmq",
                       "--p-ref", "50", "--preopt-p", "100",
                       "--preopt-hash", file_hash(self.f),
                       "--preopt-workspace-b", str(1 << 20),
                       "--op-file", str(self.f), "--allow-unversioned",
                       *CHIP_ARGS, *extra)

    def _state(self):
        return json.loads(
            (self.op_dir / "optimization" / "search_state.json").read_text())


if __name__ == "__main__":
    unittest.main()
