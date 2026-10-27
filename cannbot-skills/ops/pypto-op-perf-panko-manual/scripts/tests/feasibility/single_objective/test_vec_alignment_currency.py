# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""The trailing-alignment rule, in the currency the hardware uses.

The constraint is 32 BYTES on the contiguous axis. `space.vec_feasible` used to
test `trail % 16 == 0`, which is 32 B at two bytes per element and at nothing
else, and it rejected tiles two kernels were already running:

- `set_vec_tile_shapes(12, 1)` on one of them (fixed by the `trail == 1`
  admission).
- `set_vec_tile_shapes(16, 8)` on the other, against a `[128, 1]`
  FP32 tensor -- 8 x 4 B = 32 B, commented as such in the kernel. The block's
  warm start was rejected before the device, TPE had nothing to model, every
  other draw moved that site off 8 and hit `Tensor [34] ... exceeds the UB`, and
  the block aborted with n_feasible 0.

The admissions only ever ADD to what `trail % 16 == 0` allows.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import SCRIPTS_DIR  # noqa: E402,F401

import bayesian_optimization as bayesian  # noqa: E402

space = bayesian.space

# The shape of the site: an FP32 reduction leg tiled (16, 8).
FP32_LEG_AT_EIGHT = '''
import pypto


@pypto.frontend.jit()
def kernel(x, y):
    pypto.set_vec_tile_shapes(16, 64)
    a = pypto.exp(x)
    pypto.set_vec_tile_shapes(16, 8)
    b = pypto.amax(a, dim=-1, keepdim=True)
    return b
'''


class AlignmentAdmissionTest(unittest.TestCase):

    def test_an_fp32_leg_of_eight_is_32_bytes(self):
        hw = space.HW(calibrated=False, vec_dtype_bytes={"vec#6": 4})
        self.assertTrue(space.vec_feasible({"d0": 16, "d1": 8}, hw, "vec#6")[0])

    def test_the_running_extent_is_admitted_without_a_width(self):
        # vec#6 is a site missing from vec_dtype_bytes, so the width
        # admission cannot fire; the running extent still must.
        hw = space.HW(calibrated=False, vec_dtype_bytes={"vec#1": 4},
                      vec_trailing_ok={"vec#6": 8})
        self.assertTrue(space.vec_feasible({"d0": 16, "d1": 8}, hw, "vec#6")[0])

    def test_nothing_is_admitted_that_neither_width_nor_the_kernel_vouches_for(self):
        hw = space.HW(calibrated=False, vec_dtype_bytes={"vec#6": 2},
                      vec_trailing_ok={"vec#6": 16})
        ok, reason = space.vec_feasible({"d0": 16, "d1": 8}, hw, "vec#6")
        self.assertFalse(ok)
        self.assertIn("not 16-aligned", reason)

    def test_a_caller_that_names_no_site_keeps_the_old_rule(self):
        hw = space.HW(calibrated=False, vec_dtype_bytes={"vec#6": 4},
                      vec_trailing_ok={"vec#6": 8})
        self.assertFalse(space.vec_feasible({"d0": 16, "d1": 8}, hw)[0])
        self.assertTrue(space.vec_feasible({"d0": 16, "d1": 16}, hw)[0])
        self.assertTrue(space.vec_feasible({"d0": 12, "d1": 1}, hw)[0])


class RunningKernelSurvivesTest(unittest.TestCase):

    def test_the_block_gate_admits_the_tile_the_kernel_runs(self):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(FP32_LEG_AT_EIGHT)
            path = f.name
        try:
            hw, dom = bayesian.block.hw_for(path)
        finally:
            Path(path).unlink()
        self.assertEqual(dom["vec_trailing"].get("vec#1"), 8)
        cur = bayesian.apply.current_config(FP32_LEG_AT_EIGHT)
        ok, reason = space.static_feasible(cur, hw)
        self.assertTrue(ok, reason)


if __name__ == "__main__":
    unittest.main()
