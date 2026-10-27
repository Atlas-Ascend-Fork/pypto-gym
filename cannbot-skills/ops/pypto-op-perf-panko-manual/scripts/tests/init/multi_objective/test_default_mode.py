# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The default is the single-objective search; the second objective is opt-in.

A caller who names no objectives gets what PANKO did before the front existed:
`aicore_e2e` alone, no front to trade against, no hypervolume reference to
supply. The multi-objective mode is reached by naming the second axis --
`config.objectives: ["aicore_e2e", "workspace"]` -- and nothing else.

This file pins the boundary, because the default is the whole of the merge
policy: a run that asks for nothing must not acquire a second objective, and
must not acquire the obligation that comes with one (`--chip-memory-mib`, which
`refuse_no_hv_reference` demands of a byte axis and only of a byte axis).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import SCRIPTS_DIR  # noqa: E402,F401

from _support import harness as H  # noqa: E402


class DefaultIsSingleObjective(unittest.TestCase):
    def test_shipped_default_names_one_objective(self):
        self.assertEqual(H.DEFAULT_CONFIG["objectives"], ["aicore_e2e"])

    def test_empty_config_resolves_to_that_one_objective(self):
        for cfg in ({}, None, {"objectives": []}):
            objs = H.parse_objectives(cfg)
            self.assertEqual([o["name"] for o in objs], ["aicore_e2e"], cfg)

    def test_default_objective_reads_the_measured_latency(self):
        o = H.parse_objectives({})[0]
        self.assertEqual(o["from"], "p")
        self.assertEqual(o["direction"], "min")

    def test_default_run_has_no_front(self):
        self.assertFalse(H.is_multi_objective({"config": {}}))
        self.assertTrue(H.is_multi_objective({"config": {"objectives": ["aicore_e2e",
                                                         "workspace"]}}))

    def test_second_objective_is_reached_by_naming_it(self):
        objs = H.parse_objectives({"objectives": ["aicore_e2e", "workspace"]})
        self.assertEqual([o["name"] for o in objs], ["aicore_e2e", "workspace"])


def _state(objectives):
    return {"config": {"objectives": objectives},
            "chip_envelope": {"soc": "Ascend950DT_950x", "memory_mib": None},
            "axis_records": {}}


class OneObjectiveNeedsNoChipMemory(unittest.TestCase):
    """`--chip-memory-mib` is a multi-objective obligation, not a general one."""

    def test_no_refusal_without_a_ceiling_under_the_default(self):
        st = _state(["aicore_e2e"])
        H.refuse_no_hv_reference(st, H.parse_objectives(st["config"]))

    def test_refused_when_a_byte_axis_has_no_ceiling(self):
        st = _state(["aicore_e2e", "workspace"])
        with self.assertRaises(H.Refusal):
            H.refuse_no_hv_reference(st, H.parse_objectives(st["config"]))


if __name__ == "__main__":
    unittest.main()
