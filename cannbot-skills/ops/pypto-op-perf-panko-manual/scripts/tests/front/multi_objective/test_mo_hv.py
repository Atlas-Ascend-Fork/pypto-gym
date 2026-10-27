# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Hypervolume and exclusive contributions, checked against hand computation."""

from __future__ import annotations

import itertools
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
import _support  # noqa: E402,F401  (puts scripts/ on sys.path)

import mo_hv  # noqa: E402


class HypervolumeTest(unittest.TestCase):
    def test_two_dimensions_by_hand(self):
        # A staircase of (1,3) (2,2) (3,1) under (4,4) encloses three plus two
        # plus one, so six.
        self.assertAlmostEqual(mo_hv.hypervolume([(1, 3), (2, 2), (3, 1)], (4, 4)), 6.0)

    def test_one_dimension_is_the_distance_to_the_best(self):
        self.assertAlmostEqual(mo_hv.hypervolume([(90,), (95,)], (100,)), 10.0)

    def test_a_point_on_the_reference_adds_nothing(self):
        self.assertEqual(mo_hv.hypervolume([(4, 1)], (4, 4)), 0.0)

    def test_three_dimensions_against_a_grid_count(self):
        rng = random.Random(7)
        pts = [tuple(rng.randint(0, 5) for _ in range(3)) for _ in range(6)]
        ref = (6, 6, 6)
        cells = sum(1 for c in itertools.product(range(6), repeat=3)
                    if any(all(p[i] <= c[i] for i in range(3)) for p in pts))
        self.assertAlmostEqual(mo_hv.hypervolume(pts, ref), float(cells))

    def test_dominated_points_change_nothing(self):
        self.assertAlmostEqual(mo_hv.hypervolume([(1, 3), (2, 2), (3, 3)], (4, 4)),
                               mo_hv.hypervolume([(1, 3), (2, 2)], (4, 4)))


class ContributionTest(unittest.TestCase):
    def test_the_staircase(self):
        # (1,3): 1x1, (2,2): 1x1, (3,1): 1x1 under (4,4).
        self.assertEqual(mo_hv.contributions([(1, 3), (2, 2), (3, 1)], (4, 4)),
                         [1.0, 1.0, 1.0])

    def test_one_objective_only_the_fastest_contributes(self):
        self.assertEqual(mo_hv.contributions([(90,), (91,), (95,)], (100,)),
                         [1.0, 0.0, 0.0])

    def test_a_sparse_neighbourhood_contributes_more(self):
        c = mo_hv.contributions([(1, 9), (2, 8), (8, 1)], (10, 10))
        self.assertEqual(max(range(3), key=c.__getitem__), 2)

    def test_the_moving_reference_is_gone(self):
        """
        It was the worst value among the points plus 10%, recomputed every
        time. A point with a worse workspace pushed it outward and inflated the
        volume of everything already on the front, so real latency improvements
        read as gain under the threshold. The reference is fixed for the run
        now (`hv_reference_point`).
        """
        self.assertFalse(hasattr(mo_hv, "nadir_reference"))


if __name__ == "__main__":
    unittest.main()
