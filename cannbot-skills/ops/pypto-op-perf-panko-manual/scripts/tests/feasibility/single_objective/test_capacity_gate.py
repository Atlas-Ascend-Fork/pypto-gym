# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The vec capacity gate, adjudicated against what the device actually did.

`pypto/docs/zh/api/config/pypto-set_vec_tile_shapes.md` states no capacity rule
at all -- "每个维度必须大于0" is the whole of its 约束说明 -- so this bound is
PANKO's own model, answerable only to the device. That is why the table is the
test.

A per-site rule was wrong in both directions at once, and the over-rejection
was the expensive half: it threw away candidates the device would have run,
which is why tile blocks collapsed -- TPE had no feasible observations left to
model and returned `best_config: null`.

What decides is which tiles ONE operation holds, because the buffer is shared.
The same tile at the same width and residency can run on one kernel and be
refused for capacity on another, so no quantity computed from a single site can
separate them. See `domain.vec_op_groups` for the derivation and the three
provenance rules.

Rejecting a refusal before the device is not a claim to have the compiler's
arithmetic. The refusal messages name the operation and its total, and against
those totals the model rarely agrees -- `RefusalArithmeticTest` carries them.
Two readings of that table are the open work, and `domain.vec_op_groups` states
them: a total that does not move when the tile moves is not a function of the
tile, and a copy-to-UB failure of a whole tensor is one no tile gate can
answer.
"""

from __future__ import annotations

import ast
import sys
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import SCRIPTS_DIR  # noqa: E402

sys.path.insert(0, str(SCRIPTS_DIR))
from bayesian_optimization import domain as dom  # noqa: E402

UB = 196608

# A kernel from a recorded campaign. The domain `derive` reads off
# that kernel, and the 16 operation groups its walk finds -- four vec directives,
# `set_vec_tile_shapes(64, 512)` twice then `(128, 128)` twice. A `[320, 1]`
# entry is an online-softmax row statistic: one column, not a tile.
TILED = {
    "ub_bytes": UB, "dtype_bytes": 2,
    "vec_dtype_bytes": {f"vec#{i}": 4 for i in range(4)},
    "vec_residency": {"vec#0": 1, "vec#1": 3, "vec#2": 5, "vec#3": 3},
    "vec_trailing": {"vec#0": 512, "vec#1": 512, "vec#2": 128, "vec#3": 128},
    "vec_op_groups": [
        [["vec#0", [320, 1]]],
        [["vec#0", [320, 1]]],
        [["vec#0", [320, None]]],
        [["vec#1", [320, None]]],
        [["vec#1", [320, None]], ["vec#1", [320, 1]]],
        [["vec#1", [320, 1]], ["vec#1", [320, None]]],
        [["vec#1", [320, None]]],
        [["vec#1", [320, None]], ["vec#1", [320, 1]]],
        [["vec#1", [320, None]]],
        [["vec#2", [320, 1]], ["vec#2", [320, 1]], ["vec#2", [320, 1]]],
        [["vec#2", [320, 1]], ["vec#2", [320, 1]]],
        [["vec#2", [320, 1]], ["vec#2", [320, 1]]],
        [["vec#2", [320, 1]], ["vec#2", [320, 1]], ["vec#2", [320, 1]],
         ["vec#2", [320, 1]]],
        [["vec#2", [320, None]], ["vec#2", [320, None]], ["vec#2", [320, 1]],
         ["vec#2", [320, 1]]],
        [["vec#3", [320, 1]], ["vec#3", [320, None]]],
        [["vec#3", [320, None]]],
    ],
}


def _c(*pairs):
    return {f"vec#{i}": {"d0": a, "d1": b} for i, (a, b) in enumerate(pairs)}


# The tile block as it ran, every trial charged as a device evaluation.
# A trial marked "capacity" is one the compiler refused with a tensor memory
# allocation that exceeded the unified buffer. One marked "other" died at
# startup, which is still unexplained and not this gate's to answer.
TILED_TRIALS = [
    ("b1t1", "capacity", _c((128, 384), (128, 384), (128, 128), (128, 128))),
    ("b1t2", "capacity", _c((96, 512), (96, 512), (96, 128), (96, 128))),
    ("b1t3", "other", _c((48, 96), (8, 16), (8, 2048), (48, 32))),
    ("b1t4", "capacity", _c((128, 384), (128, 384), (128, 128), (128, 128))),
    ("b1t5", "capacity", _c((128, 128), (128, 384), (128, 128), (128, 128))),
    ("b1t6", "missed", _c((128, 256), (128, 256), (128, 128), (128, 128))),
    ("b2t1", "capacity", _c((128, 384), (128, 384), (128, 128), (128, 128))),
    ("b2t2", "capacity", _c((96, 512), (96, 512), (96, 128), (96, 128))),
    ("b2t3", "other", _c((48, 96), (8, 16), (8, 2048), (48, 32))),
    ("b2t4", "capacity", _c((128, 384), (128, 384), (128, 128), (128, 128))),
    ("b2t5", "capacity", _c((128, 128), (128, 384), (128, 128), (128, 128))),
]

# What that kernel is MEASURED running. A program the device ran is proof of
# feasibility and outranks any model of the buffer, so this row is the one the
# gate may never reject: losing it costs the warm start, and with it the block,
# because TPE then has no feasible observation and returns `best_config: null`.
TILED_BASELINE = _c((64, 512), (64, 512), (128, 128), (128, 128))


class RecordedTileBlockTest(unittest.TestCase):
    """The block that spent its evaluations on refusals, re-adjudicated."""

    def test_the_eight_refusals_the_rule_catches_are_free_now(self):
        n = 0
        for tid, verdict, cfg in TILED_TRIALS:
            if verdict != "capacity":
                continue
            n += 1
            ok, why = dom.verify(TILED, cfg)
            self.assertFalse(ok, f"{tid} reached the device and was refused there")
            self.assertIn("one operation holds", why)
            self.assertIn("UB", why)
        self.assertEqual(n, 8, "the recorded refusals this rule accounts for")

    def test_the_one_refusal_the_rule_misses(self):
        """Open case, asserted so it cannot be mistaken for fixed.

        The worst operation under this rule comes to 132096 B of a 196608 B
        buffer, so the gate passes it and the device refused it. When the
        saturation reading in `RefusalArithmeticTest` lands, this test inverts.
        """
        cfg = dict(TILED_TRIALS[5][2])
        self.assertTrue(dom.verify(TILED, cfg)[0])

    def test_the_startup_death_class_is_not_a_capacity_verdict(self):
        for tid, verdict, cfg in TILED_TRIALS:
            if verdict == "other":
                ok, why = dom.verify(TILED, cfg)
                self.assertTrue(ok, f"{tid}: {why}")

    def test_the_measured_baseline_is_never_rejected(self):
        ok, why = dom.verify(TILED, TILED_BASELINE)
        self.assertTrue(ok, f"the kernel's own measured configuration: {why}")

    def test_the_row_statistic_is_charged_one_column(self):
        """
        `mi`/`li` are `full([QT, 1])` under a `(64, 512)` directive. Charging
        them 64x512 is what made the per-site rule reject this kernel's own
        baseline: one column is 64x1, 256 B, not 131072 B.
        """
        self.assertEqual(dom.group_bytes(TILED, TILED_BASELINE, "vec#0", [320, 1]), 256)
        self.assertEqual(dom.group_bytes(TILED, TILED_BASELINE, "vec#0", [320, None]),
                         131072)


class GroupBytesTest(unittest.TestCase):
    """One entry's footprint, factor by factor."""

    def test_the_tensor_extent_bounds_the_tile(self):
        self.assertEqual(dom.group_bytes(TILED, _c((128, 384)), "vec#0", [320, 1]), 512)

    def test_the_running_trailing_extent_bounds_an_unstated_axis(self):
        """
        vec#0 runs a trailing 512 and the tensor's second axis is a kernel
        parameter, so 768 is charged as 512.
        """
        self.assertEqual(
            dom.group_bytes(TILED, _c((128, 768)), "vec#0", [320, None]),
            128 * 512 * 4)

    def test_a_site_absent_from_the_config_costs_nothing(self):
        self.assertEqual(dom.group_bytes(TILED, {}, "vec#0", [320, 1]), 0)

    def test_the_site_s_own_width_is_the_currency(self):
        """
        The kernel-wide `dtype_bytes` is 2 -- the minimum `DT_*` in the file.
        Every vec site here works on fp32 and is charged 4.
        """
        self.assertEqual(TILED["dtype_bytes"], 2)
        self.assertEqual(dom.group_bytes(TILED, _c((16, 16)), "vec#0", None), 16 * 16 * 4)


class VerifyContractTest(unittest.TestCase):
    """What `verify` does when the domain cannot state a bound."""

    def test_no_groups_means_the_bound_is_off(self):
        d = dict(TILED, vec_op_groups=[])
        self.assertTrue(dom.verify(d, _c((4096, 4096)))[0])

    def test_no_budget_means_the_bound_is_off(self):
        d = dict(TILED, ub_bytes=0)
        self.assertTrue(dom.verify(d, _c((4096, 4096)))[0])

    def test_an_unmodelled_site_is_not_judged(self):
        d = dict(TILED, vec_unmodelled=["vec#0", "vec#1", "vec#2", "vec#3"])
        self.assertTrue(dom.verify(d, _c((4096, 4096), (4096, 4096),
                                         (4096, 4096), (4096, 4096)))[0])

    def test_cube_sites_are_not_this_gate_s_business(self):
        ok, why = dom.verify(TILED, {"cube#0": {"mL0": 256, "kL0": 128, "nL0": 128}})
        self.assertTrue(ok, why)

    def test_the_reason_names_every_tile_and_the_total(self):
        ok, why = dom.verify(TILED, _c((128, 384), (128, 384), (128, 128), (128, 128)))
        self.assertFalse(ok)
        self.assertIn("vec#1[320x?]", why)
        self.assertIn(f"> {UB}B UB", why)


# What the compiler SAID, for every capacity refusal whose message was kept:
# (campaign, enum, op, op id, total bytes, how many trials). The buffer is
# 196608 in all of them.
REFUSALS = [
    ("campaign-a", "TENSOR_MEMORY_ALLOCATION", "SUB", 12379, 197120, 1),
    ("campaign-a", "TENSOR_MEMORY_ALLOCATION", "SUB", 12728, 197120, 1),
    ("campaign-a", "TENSOR_MEMORY_ALLOCATION", "SUB", 12867, 197120, 1),
    ("campaign-b", "TENSOR_MEMORY_ALLOCATION", "SUB", 20000, 197120, 3),
    ("campaign-c", "TENSOR_MEMORY_ALLOCATION", "CAST", 10273, 221184, 1),
    ("campaign-c", "TENSOR_MEMORY_ALLOCATION", "CAST", 10291, 294912, 1),
    ("campaign-c", "TENSOR_MEMORY_ALLOCATION", "CAST", 11297, 294912, 1),
    ("campaign-c", "PASS_INNER_ERROR", "COPY", 34, 2097152, 4),
]


class RefusalArithmeticTest(unittest.TestCase):
    """What the compiler's own numbers say, against what this module models.

    Kept as a table because it is the only evidence that distinguishes a bound
    that rejects the right candidates from the compiler's actual rule, and it
    says this module has the first and not the second.
    """

    def test_four_refusals_are_not_a_tile_capacity_failure_at_all(self):
        """
        `ReplaceTensor ... Tensor [34] can not copy to UB, tensor size
        [2097152]` -- one 2 MB tensor the pass must copy whole, the same in
        every trial and independent of the tile drawn. They are recorded as
        `incorrect_capacity`, and no tile gate can answer them.
        """
        copy = [r for r in REFUSALS if r[1] == "PASS_INNER_ERROR"]
        self.assertEqual(sum(r[5] for r in copy), 4)
        self.assertEqual({r[4] for r in copy}, {2097152})
        self.assertEqual({r[4] // UB for r in copy}, {10})

    def test_the_overflow_total_does_not_move_when_the_tile_moves(self):
        """
        Trials across two kernels report 197120 = 196608 + 512 -- the whole
        buffer plus one column -- on draws whose tiles differ. A total that
        does not move when the tile moves is not a function of the tile: what
        the draw decides is whether that operation is reached, not how big it
        is.

        A sibling draw in the same block holding FOUR sites at 98304 B ran. So
        the refused draws are not the big ones; they are the ones asking for a
        trailing extent of 2048 against data whose extent is 128 -- which says
        `set_vec_tile_shapes` does not shrink to the data there the way
        `vec_trailing` assumes, but saturates.
        """
        same = [r for r in REFUSALS if r[4] == 197120]
        self.assertEqual(sum(r[5] for r in same), 6)
        self.assertEqual({r[0] for r in same}, {"campaign-a", "campaign-b"})
        self.assertEqual(197120 - UB, 512)

    def test_the_model_reproduces_one_of_these_totals(self):
        """
        A `CAST` refusal at 294912: vec#1 at 196608 B plus the same site's
        98304 B second operand. That is the only row where this module's
        arithmetic and the compiler's agree, and the docstring says so.
        """
        self.assertEqual(196608 + 98304, 294912)
        distinct = {(r[2], r[4]) for r in REFUSALS}
        self.assertEqual(
            sorted(distinct),
            [("CAST", 221184), ("CAST", 294912), ("COPY", 2097152),
             ("SUB", 197120)],
            "four distinct (op, total) shapes behind thirteen refusals -- "
            "the totals repeat because they do not depend on the tile")


KERNEL = textwrap.dedent('''
    import pypto

    QT = 320

    @pypto.frontend.jit()
    def k(x: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32), width: int):
        pypto.set_vec_tile_shapes(64, 512)
        acc = pypto.full([QT, 1], 0.0, pypto.DT_FP32)
        wide = pypto.full([QT, width], 0.0, pypto.DT_FP32)
        row = pypto.view(x, [QT, width], [0, 0])
        scaled = pypto.mul(row, 2.0)
        hi = pypto.amax(scaled, dim=-1, keepdim=True)
        shifted = pypto.sub(scaled, hi)
        pypto.set_vec_tile_shapes(128, 128)
        out = pypto.add(shifted, acc)
        pypto.assemble(out, [0, 0], x)
''')


class GroupWalkTest(unittest.TestCase):
    """The four readings `vec_op_groups` makes of a source."""

    def setUp(self):
        self.groups = dom.vec_op_groups(KERNEL)

    def test_a_declared_shape_is_read_through_a_module_constant(self):
        """
        `full([QT, 1])` with `QT = 320` is `[320, 1]`; `width` is a kernel
        parameter, so that axis is None -- unstated, never guessed.
        """
        shapes = [sh for g in self.groups for _, sh in g]
        self.assertIn([320, 1], shapes)
        self.assertIn([320, None], shapes)

    def test_a_reduction_closes_the_axis_it_reduces(self):
        """`amax(scaled, dim=-1, keepdim=True)` over `[320, None]` is `[320, 1]`.

        Its group holds the operand it reduces and the one column it writes, and
        the column is what makes the difference: charged as a full tile the
        kernel's own baseline comes out over the buffer.
        """
        def key(g):
            return sorted(repr(sh) for _, sh in g)
        hi = [g for g in self.groups
              if key(g) == sorted([repr([320, 1]), repr([320, None])])]
        self.assertTrue(hi, f"no reduction group in {self.groups}")

    def test_a_negative_axis_survives_the_parser(self):
        """
        `dim=-1` reaches the AST as a UnaryOp, not a negative Constant. Read
        positionally it is None, and then the reduction's column reads as a full
        tile -- this is the bug that cost the gate its capacity catches.
        """
        self.assertEqual(dom.int_literal(ast.parse("-1", mode="eval").body), -1)
        self.assertEqual(dom.int_literal(ast.parse("320", mode="eval").body), 320)
        self.assertIsNone(dom.int_literal(ast.parse("head_dim", mode="eval").body))

    def test_an_unstated_axis_is_not_guessed(self):
        """
        `width` is a kernel parameter. The axis reads None, which means the
        tile is not bounded there -- the running trailing extent still is.
        """
        self.assertEqual(
            dom.literal_shape(ast.parse("[QT, width]", mode="eval").body,
                              {"QT": 320}),
            [320, None])
        self.assertIsNone(
            dom.literal_shape(ast.parse("shape_of(x)", mode="eval").body, {}))

    def test_the_active_directive_governs_every_operand(self):
        """
        `out = add(shifted, acc)` sits under `set_vec_tile_shapes(128, 128)`
        and reads two tensors allocated under `(64, 512)`. The directive in force
        tiles the statements that FOLLOW it, so both operands are charged at
        vec#1, not at the site that first allocated them.
        """
        last = self.groups[-1]
        self.assertTrue(last, "the add under the second directive")
        self.assertEqual({sid for sid, _ in last}, {"vec#1"})

    def test_a_dying_operand_is_the_buffer_the_result_lands_in(self):
        """
        `shifted = sub(scaled, hi)` ends `scaled`'s life and has its extent,
        so it is one buffer, not two. Counting it twice is what put measured
        configurations over the buffer.
        """
        for g in self.groups:
            wide = [sh for _, sh in g if sh == [320, None]]
            self.assertLessEqual(len(wide), 1, f"{g} charges two full tiles")


if __name__ == "__main__":
    unittest.main()
