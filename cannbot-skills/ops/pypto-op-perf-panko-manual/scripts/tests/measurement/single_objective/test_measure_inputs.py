# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""`eval/test_inputs.py` is the objective's input contract, and it was silent.

`measure_latency` builds the wrapper's arguments from that file before calling
the kernel. When the builder raised, the exception took the whole process down:
msprof still exited 0, no `op_summary_*.csv` was written, and the caller saw a
measurement with no readings -- which is what a device that went away also looks
like. A recorded campaign spent its INIT on exactly that, because
`measure_latency` passes `--device` VERBATIM, so `build_inputs("7")` reached
`torch.randn(device="7")` and raised `Invalid device string: '7'`. Nothing in
the output said the kernel had never run.

Nothing was measured, so it is not a verdict on the candidate and not a failed
measurement. It is now `p: None` with `p_error`, the shape the evaluator
already turns into `EnvironmentFault("objective_unmeasured", ...)`.

The argument's type was also undocumented, which is how the file came to be
written that way. It is the device ID as a string.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import SCRIPTS_DIR  # noqa: E402

sys.path.insert(0, str(SCRIPTS_DIR))
import measure_latency as ml  # noqa: E402


def _mod(**attrs):
    m = types.ModuleType("test_inputs")
    m.__file__ = "/tmp/custom/demo/eval/test_inputs.py"
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class BuilderTest(unittest.TestCase):

    def test_the_device_arrives_verbatim(self):
        """A string, as given on the command line -- not a torch device."""
        seen = []
        got = ml.call_builder(_mod(build_inputs=lambda d: seen.append(d) or ["x"]),
                              "7")
        self.assertEqual(seen, ["7"])
        self.assertEqual(got, ["x"])

    def test_a_builder_that_takes_nothing_is_called_with_nothing(self):
        """
        `torch.npu.set_device` has already run, so ignoring the argument and
        using `.npu()` is a correct builder.
        """
        self.assertEqual(ml.call_builder(_mod(build_inputs=lambda: ["y"]), "7"),
                         ["y"])

    def test_inputs_is_still_accepted(self):
        self.assertEqual(ml.call_builder(_mod(INPUTS=["z"]), "7"), ["z"])

    def test_a_file_exposing_neither_is_named(self):
        with self.assertRaises(ml.InputsError) as e:
            ml.call_builder(_mod(), "7")
        self.assertIn("neither", str(e.exception))
        self.assertIn("test_inputs.py", str(e.exception))

    def test_a_builder_failure_is_reported_with_its_cause(self):
        def build_inputs(device):
            raise RuntimeError(f"Invalid device string: {device!r}")
        with self.assertRaises(ml.InputsError) as e:
            ml.call_builder(_mod(build_inputs=build_inputs), "7")
        msg = str(e.exception)
        self.assertIn("RuntimeError", msg)
        self.assertIn("Invalid device string", msg)
        self.assertIn("npu:", msg, "and says how to write it instead")

    def test_a_type_error_inside_the_builder_is_not_read_as_arity(self):
        """
        Deciding from the signature rather than by catching TypeError: the
        catch would report a bug inside the builder as "takes no device".
        """
        def build_inputs(device):
            raise TypeError("unsupported operand type(s)")
        with self.assertRaises(ml.InputsError) as e:
            ml.call_builder(_mod(build_inputs=build_inputs), "7")
        self.assertIn("unsupported operand", str(e.exception))


class EmissionTest(unittest.TestCase):
    """What the caller sees. `p: None` + `p_error`, never an empty stdout."""

    def setUp(self):
        self._real = ml.build_run_once
        self.addCleanup(lambda: setattr(ml, "build_run_once", self._real))
        self._argv = sys.argv
        self.addCleanup(lambda: setattr(sys, "argv", self._argv))
        # The CHILD path. Without this marker `main` re-execs itself under
        # msprof, which is the parent's job and not what this is about -- the
        # parent copies the child's reading through verbatim, adding only
        # `mix_aic_error` for the op_summary that was never written.
        self._env = os.environ.get(ml.MSPROF_CHILD_ENV)
        os.environ[ml.MSPROF_CHILD_ENV] = "1"
        self.addCleanup(lambda: (os.environ.pop(ml.MSPROF_CHILD_ENV, None)
                                 if self._env is None else
                                 os.environ.__setitem__(ml.MSPROF_CHILD_ENV,
                                                        self._env)))

    def test_a_builder_failure_emits_a_named_measurement(self):
        def boom(*a, **k):
            raise ml.InputsError("build_inputs('7') raised RuntimeError: nope")
        ml.build_run_once = boom
        out = self._run()
        self.assertTrue(out, "an empty stdout is what a dead device looks like")
        d = json.loads(out)
        self.assertIsNone(d["p"])
        self.assertIn("build_inputs", d["p_error"])

    def _run(self):
        sys.argv = ["measure_latency.py", "--op-dir", "/tmp/custom/demo",
                    "--op", "demo", "--op-file", "/tmp/custom/demo/demo_impl.py",
                    "--device", "7"]
        log = logging.getLogger(ml.jsonio.CHANNEL)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        log.addHandler(handler)
        try:
            with redirect_stdout(buf):
                ml.main()
        finally:
            log.removeHandler(handler)
        return buf.getvalue().strip()


if __name__ == "__main__":
    unittest.main()
