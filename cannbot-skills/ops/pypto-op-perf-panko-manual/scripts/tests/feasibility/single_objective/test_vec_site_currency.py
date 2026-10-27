# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
What a vec site costs: which sites there are, what each one is called, and in
what currency its tile is measured.

`domain.verify` is the gate that decides, for free, whether a candidate can be
held at all. It is also the LAST check -- the module header's "fail wide"
argument is explicitly that a generous ladder is safe because every draw is
re-checked here, so when this check is itself wrong there is nothing downstream
to catch it. A tile it wrongly passes costs a charged evaluation and comes back
a device refusal, and the refusal then lowers `learned_ceiling` in whatever
currency the gate was using.

Three things were wrong with that currency at once, and this file pins all
three. Every fixture here is the shape of a real kernel in the repository, named
in the test.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import SCRIPTS_DIR  # noqa: E402,F401

import bayesian_optimization as bayesian  # noqa: E402

# The shape of every generated node file and of the library
# kernels: the `@jit` def is an entry point that calls a plain helper, and the
# helper is where the tiles are. Most of a node's sites can sit in the helper
# and only a few in the jit body, and a library kernel can be a jit wrapper
# around a helper holding every site.
TILES_IN_A_HELPER = '''
import torch
import pypto


def _kernel_impl(x, y):
    pypto.set_vec_tile_shapes(16, 64)
    a = pypto.exp(x)
    b = pypto.mul(a, y)
    return b


@pypto.frontend.jit()
def kernel(x, y):
    out = _kernel_impl(x, y)
    pypto.set_vec_tile_shapes(32, 32)
    return pypto.cast(out, pypto.DT_BF16)


def wrapper(x, y):
    z = torch.cat([x, y])
    w = z + z
    v = torch.add(w, z)
    return kernel(v, y)
'''

# `set_vec_tile_shapes(v1_tile[0], v1_tile[1])` -- discovered by `apply` but
# never rewritten, so it takes a `vec_dyn#` ordinal and does NOT consume a
# `vec#` one. Fifty-five kernels in the repository have at least one.
DYNAMIC_SITE_IN_THE_MIDDLE = '''
import pypto


@pypto.frontend.jit()
def kernel(x, tile):
    pypto.set_vec_tile_shapes(16, 64)
    a = pypto.exp(x)
    pypto.set_vec_tile_shapes(tile[0], tile[1])
    b = pypto.exp(a)
    pypto.set_vec_tile_shapes(32, 32)
    return pypto.exp(b)
'''

# The antiquant shape: an INT8 cache read once, and a softmax whose every
# operand is the fp32 the matmul above it was asked to produce. The kernel-wide
# minimum over `DT_*` tokens reads 1 byte here; the softmax site is 4.
ANTIQUANT_SOFTMAX = '''
import pypto


@pypto.frontend.jit()
def kernel(q, cache_int8):
    kv = pypto.view(cache_int8, dtype=pypto.DT_INT8)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    sij = pypto.matmul(q, kv, pypto.DT_FP32)
    pypto.set_vec_tile_shapes(16, 512)
    m = pypto.amax(sij, dim=-1)
    p = pypto.exp(pypto.sub(sij, m))
    return pypto.cast(p, pypto.DT_BF16)
'''


def _vec_ids(src):
    return [s.id for s in bayesian.apply.discover_sites(src)
            if s.kind.startswith("vec")]


class SiteDiscoveryTest(unittest.TestCase):
    """Every site the rewriter can reach is a site the gate has an opinion about."""

    def test_a_helper_called_from_the_jit_body_is_kernel_code(self):
        # Restricting the walk to `@jit` defs did not exclude the torch wrapper,
        # it excluded the kernel. On that node it left seven of eleven sites
        # with no residency at all -- `verify` then read 1 for each of them.
        res = bayesian.domain.vec_residency(TILES_IN_A_HELPER)
        self.assertEqual(sorted(res), _vec_ids(TILES_IN_A_HELPER))
        self.assertEqual(len(res), 2)

    def test_the_torch_wrapper_is_still_not_kernel_code(self):
        # The edge runs one way: a kernel calls a helper, and nothing in a
        # kernel calls the wrapper. So widening to reachable helpers does not
        # re-admit the wrapper's `torch.cat` and BinOp, which hold no unified
        # buffer, and neither does a site's reach run past the `def` that ends
        # it -- the last site in the file would otherwise be charged for them.
        before = bayesian.domain.vec_sites(TILES_IN_A_HELPER)
        after = bayesian.domain.vec_sites(
            TILES_IN_A_HELPER.replace("    w = z + z",
                                      "    w = z + z + z + z + z"))
        self.assertEqual(before, after)
        self.assertEqual(before["vec#1"]["live"], 1)

    def test_a_dynamic_site_does_not_shift_the_numbering(self):
        # `apply` counts `vec` and `vec_dyn` separately. A local count in source
        # order does not, and past the first dynamic site the two disagree --
        # mhc_pre's apply `vec#4` was this module's `vec#12`, so `verify` read
        # one site's residency under another site's name.
        ids = _vec_ids(DYNAMIC_SITE_IN_THE_MIDDLE)
        self.assertEqual(ids, ["vec#0", "vec_dyn#0", "vec#1"])
        self.assertEqual(sorted(bayesian.domain.vec_residency(
            DYNAMIC_SITE_IN_THE_MIDDLE)), sorted(ids))


class SiteCurrencyTest(unittest.TestCase):
    """A site is measured in the width of its own operands."""

    def test_an_int8_token_elsewhere_does_not_make_a_softmax_site_int8(self):
        widths = bayesian.domain.vec_dtype_bytes(ANTIQUANT_SOFTMAX)
        self.assertEqual(widths.get("vec#0"), 4)

    def test_the_gate_rejects_what_the_kernel_wide_minimum_passed(self):
        # 65536 elements at the softmax site, held twice by the one operation
        # that reads and writes them: 128 KB at 1 B/element and 512 KB at the
        # 4 B it actually is, against a 192 KB buffer. The old currency passed
        # it, the device refused it, and the refusal taught a ceiling in
        # elements.
        dom = {"ub_bytes": 192 * 1024, "dtype_bytes": 1,
               "vec_residency": {"vec#0": 1},
               "vec_op_groups": [[["vec#0", None], ["vec#0", None]]]}
        cfg = {"vec#0": {"d0": 128, "d1": 512}}
        self.assertTrue(bayesian.domain.verify(dom, cfg)[0])
        dom["vec_dtype_bytes"] = {"vec#0": 4}
        ok, reason = bayesian.domain.verify(dom, cfg)
        self.assertFalse(ok)
        self.assertIn("4B", reason)

    def test_a_site_the_source_does_not_declare_keeps_the_old_currency(self):
        # The per-site map only ever makes the gate more accurate where the
        # source is explicit. A site it cannot read is left exactly as it was,
        # because narrowing on a guess is the one error that leaves no trace.
        self.assertNotIn("vec#0", bayesian.domain.vec_dtype_bytes('''
import pypto


@pypto.frontend.jit()
def kernel(x):
    pypto.set_vec_tile_shapes(16, 64)
    return pypto.exp(x)
'''))

    def test_the_learned_ceiling_is_in_bytes_not_elements(self):
        # `draw_tile_bytes` is what `over_learned_ceiling` compares, and it is
        # compared ACROSS sites. At one width for the whole kernel a refusal at
        # an fp32 site retired int8 tiles of the same element count that would
        # have fitted four times over.
        cfg = {"vec#0": {"d0": 16, "d1": 64}}
        self.assertEqual(bayesian.driver.draw_tile_bytes(cfg, 1), 1024)
        self.assertEqual(
            bayesian.driver.draw_tile_bytes(cfg, 1, {"vec#0": 4}), 4096)


class DerivedDomainTest(unittest.TestCase):

    def test_derive_publishes_the_per_site_widths(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(ANTIQUANT_SOFTMAX)
            path = f.name
        try:
            rep = bayesian.domain.derive(path)
        finally:
            Path(path).unlink()
        self.assertEqual(rep["dtype_bytes"], 1)
        self.assertEqual(rep["vec_dtype_bytes"], {"vec#0": 4})
        # The worst site is the one spending most buffer per ELEMENT -- width
        # times residency, not residency alone. Those two orderings coincided
        # while there was one width for the whole kernel.
        self.assertEqual(rep["capacity_elems"],
                         rep["ub_bytes"] // (4 * rep["live_tiles"]))

    def test_the_running_tile_caps_the_residency_it_disproves(self):
        # `prod(dims) x dtype x residency <= 192 KB` rejects many of this
        # repository's kernels' own DELIVERED tile shapes. Either the buffer is
        # bigger than the model says, or the walk over-counts residency. The
        # first version of this guard bet on the buffer and raised the site's
        # byte budget; the device settled it on one recorded kernel --
        # `F40005 ... 294912 exceeds MEM_UB 196608`, consecutive refusals,
        # block aborted with n_feasible 0. The buffer is exactly 196608. So the
        # budget stands and the RESIDENCY is capped to what the running tile
        # proves.
        import tempfile
        src = ANTIQUANT_SOFTMAX.replace("set_vec_tile_shapes(16, 512)",
                                        "set_vec_tile_shapes(256, 512)")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
            path = f.name
        try:
            rep = bayesian.domain.derive(path)
        finally:
            Path(path).unlink()
        cur = {k: v for k, v in bayesian.apply.current_config(src).items()
               if k.startswith("vec")}
        # One tile of (256, 512) at 4 B is 512 KB, past the whole buffer on its
        # own, so no residency explains it: the site is unmodelled, not capped.
        self.assertEqual(rep["vec_unmodelled"], ["vec#0"])
        self.assertTrue(bayesian.domain.verify(rep, cur)[0])
        self.assertIn("the footprint bound is OFF", " ".join(rep["notes"]))

    def test_the_cap_is_the_device_s_arithmetic(self):
        # The exact numbers from a recorded aborted block. The cap
        # no longer decides what `verify` admits -- that is `vec_op_groups` now,
        # because no per-site quantity can separate a 131072-byte tile that runs
        # from one the device refuses. What the cap still governs is the SEARCH
        # SPACE in `space`, so the arithmetic still has to be the device's.
        ub = 192 * 1024
        dom = {"ub_bytes": ub, "dtype_bytes": 4,
               "vec_residency": {"vec#1": 3, "vec#2": 5},
               "vec_live_cap": {"vec#1": 1, "vec#2": 3}}
        self.assertEqual(bayesian.domain.live_for(dom, "vec#1"), 1)
        self.assertEqual(bayesian.domain.live_for(dom, "vec#2"), 3)
        # Only ever LOWERS, and only where the running tile disproves the walk.
        self.assertLess(bayesian.domain.live_for(dom, "vec#1"),
                        dom["vec_residency"]["vec#1"])
        self.assertLess(bayesian.domain.live_for(dom, "vec#2"),
                        dom["vec_residency"]["vec#2"])
        # A site the cap says nothing about keeps the walk's count.
        self.assertEqual(bayesian.domain.live_for(dom, "vec#9"), 1)

    def test_a_residency_the_running_tile_disproves_is_capped(self):
        # The case a cap CAN explain: the tile fits on its own, the walk's
        # residency is what pushes it over.
        import tempfile
        src = ANTIQUANT_SOFTMAX.replace("set_vec_tile_shapes(16, 512)",
                                        "set_vec_tile_shapes(48, 512)")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
            path = f.name
        try:
            rep = bayesian.domain.derive(path)
        finally:
            Path(path).unlink()
        cur = {k: v for k, v in bayesian.apply.current_config(src).items()
               if k.startswith("vec")}
        self.assertEqual(rep["vec_unmodelled"], [])
        self.assertIn("vec#0", rep["vec_live_cap"])
        self.assertLess(rep["vec_live_cap"]["vec#0"], rep["vec_residency"]["vec#0"])
        # The budget is still the model's -- a larger tile is refused -- but the
        # rule that refuses it is now per OPERATION, and the reason says so. This
        # fixture is a hand-widened directive, not a measured program, so there
        # is no claim here that its own tile has to pass.
        ok, why = bayesian.domain.verify(rep, {"vec#0": {"d0": 256, "d1": 512}})
        self.assertFalse(ok)
        self.assertIn("one operation holds", why)
        self.assertIn(f"{rep['ub_bytes']}B UB", why)
        self.assertTrue(why.startswith("vec#0: "), "space.repair parses this")
        del cur

    def test_an_unmodelled_site_is_not_judged_rather_than_licensed(self):
        # Not the raised budget again: a raised budget invents a number and
        # licenses draws against it. This records that the site is unreadable
        # and leaves it to the device, where `_lower_ceiling` learns a real
        # bound from real refusals. Every site we can model is still gated.
        dom = {"ub_bytes": 192 * 1024, "dtype_bytes": 4,
               "vec_residency": {"vec#0": 2, "vec#1": 2},
               "vec_live_cap": {}, "vec_unmodelled": ["vec#0"],
               "vec_op_groups": [[["vec#0", None], ["vec#0", None]],
                                 [["vec#1", None], ["vec#1", None]]]}
        self.assertTrue(bayesian.domain.verify(
            dom, {"vec#0": {"d0": 100000, "d1": 512}})[0])
        self.assertIsNone(
            bayesian.domain.vec_ladder_cap(dom, "vec#0", "d0", ["d0", "d1"]))
        self.assertFalse(bayesian.domain.verify(
            dom, {"vec#1": {"d0": 100000, "d1": 512}})[0])

    def test_a_site_the_model_already_fits_is_untouched(self):
        dom = {"ub_bytes": 192 * 1024, "dtype_bytes": 4,
               "vec_residency": {"vec#0": 4}, "vec_live_cap": {}}
        self.assertEqual(bayesian.domain.live_for(dom, "vec#0"), 4)

    def test_the_widths_are_part_of_the_fingerprint(self):
        # An edit that changes one site's precision moves that site's budget and
        # nothing else in the fingerprint would show it.
        a = {"views": [], "capacity_elems": 100, "dtype_bytes": 2,
             "vec_residency": {"vec#0": 2}, "vec_dtype_bytes": {"vec#0": 2}}
        b = dict(a, vec_dtype_bytes={"vec#0": 4})
        self.assertNotEqual(bayesian.domain.fingerprint(a),
                            bayesian.domain.fingerprint(b))


if __name__ == "__main__":
    unittest.main()
