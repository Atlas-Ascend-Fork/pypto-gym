# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""The tile study's `current` seed belongs to the program on disk.

`seed_points` hands the study the file's own tile configuration with a value
attached, and `_fill_seeds` records that pair as a COMPLETED observation --
one the device never has to spend a trial on. The value therefore has to be a
measurement OF THAT PROGRAM.

It was `progress.best_aicore_e2e_us`, which is `best_node`'s. With one
objective the two agree. With more than one they do not: `select` restores the
front base, and the base is chosen by hypervolume contribution, so it can be a
slower program that holds far less workspace. The study was then told that the
restored program's tiles had measured the global best's latency -- a sample
nothing had measured, scoring every draw that followed.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from panko import selection  # noqa: E402
from _support import harness as H  # noqa: E402

KERNEL = '''import pypto


@pypto.frontend.jit()
def k(a, b):
    pypto.set_vec_tile_shapes(64, 128)
    return a + b
'''

ENVELOPE = {"ub_kb": 192, "l1_kb": 512, "l0a_kb": 64, "l0b_kb": 64,
            "l0c_kb": 128, "cube_cores": 24, "vector_cores": 48}

GLOBAL_BEST_US = 90.0        # program A: faster, and NOT the one on disk
FRONT_BASE_US = 95.0         # program B: the front base `select` restored


class SeedIncumbentTest(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.op_dir = Path(tmp.name)
        self.f = self.op_dir / "demo_impl.py"
        self.f.write_text(KERNEL, encoding="utf-8")
        self.base = hashlib.sha256(KERNEL.encode("utf-8")).hexdigest()

    def test_the_seed_carries_the_restored_programs_own_latency(self):
        st = self._state({self.base: {"s": 1, "p": FRONT_BASE_US, "m": {}},
                          "a" * 64: {"s": 1, "p": GLOBAL_BEST_US, "m": {}}})
        res = self._propose(st)
        self.assertEqual(self._seed_value(st), FRONT_BASE_US)
        self.assertTrue(res.get("applied"), res)

    def test_an_unmeasured_base_is_measured_rather_than_invented(self):
        """A seed with no value is queued, so the first proposal is to measure
        what the file already holds -- one device trial, not a made-up sample.
        """
        st = self._state({"a" * 64: {"s": 1, "p": GLOBAL_BEST_US, "m": {}}})
        res = self._propose(st)
        self.assertIsNone(self._seed_value(st))
        self.assertEqual(self._pending(st), "current")
        self.assertFalse(res.get("applied"), "the current config is what is on disk")

    def test_an_incorrect_measurement_is_not_an_incumbent(self):
        st = self._state({self.base: {"s": 0, "p": FRONT_BASE_US, "m": {}},
                          "a" * 64: {"s": 1, "p": GLOBAL_BEST_US, "m": {}}})
        self._propose(st)
        self.assertIsNone(self._seed_value(st))
        self.assertEqual(self._pending(st), "current")

    def test_the_helper_reads_the_file_not_progress(self):
        st = self._state({self.base: {"s": 1, "p": FRONT_BASE_US, "m": {}}})
        self.assertEqual(H.incumbent_latency(st, KERNEL), FRONT_BASE_US)
        self.assertIsNone(H.incumbent_latency(st, KERNEL + "\n# other program\n"))

    def _state(self, cache):
        return {"op": "demo", "op_file": str(self.f),
                "config": {"bayesian_optimization_tile_search": True,
                           "objectives": ["aicore_e2e", "workspace"]},
                # `best_node`'s number, and the one this must NOT reach for.
                "progress": {"best_aicore_e2e_us": GLOBAL_BEST_US},
                "eval_cache": dict(cache), "chip_envelope": dict(ENVELOPE)}

    def _propose(self, st):
        res = selection.tile_bayes_propose(st, str(self.op_dir),
                                           {"catalog_id": "F-10", "id": "u1"})
        self.assertIsNotNone(res)
        return res

    def _record(self, st):
        studies = st.get("bayesian_optimization_studies") or {}
        self.assertEqual(len(studies), 1, studies)
        return next(iter(studies.values()))

    def _seed_value(self, st):
        """The value recorded for the `current` seed, or None when it was queued."""
        for t in self._record(st).get("trials") or []:
            if t.get("label") == "current":
                return t.get("value")
        return None

    def _pending(self, st):
        """The seed the study is waiting on a measurement for."""
        return (self._record(st).get("pending") or {}).get("label")


if __name__ == "__main__":
    unittest.main()
