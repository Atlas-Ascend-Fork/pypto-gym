# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A failed trial has to record WHY, and under pytest the tail is not why.

A recorded campaign spent device evaluations on tile candidates the compiler
refused, and kept this as the reason for every one of them:

    FAILED test_myop.py::test_... - pypto.error...

The exception is truncated by pytest's own short summary, and `_tail(4)` takes
exactly those four lines. The classification was right -- `_failure_kind` reads
the whole output -- but WHICH buffer overflowed was thrown away, and with it
the only evidence that would have shown the capacity gate was too loose.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from _support import SCRIPTS_DIR  # noqa: E402

sys.path.insert(0, str(SCRIPTS_DIR))
from bayesian_optimization import evaluator as ev  # noqa: E402

# The shape of a real failing run under `python3 -m pytest ... -s`.
PYTEST_OUT = """\
============================= test session starts ==============================
plugins: anyio-4.10.0
collected 1 item

test_myop.py F

=================================== FAILURES ===================================
__________________________ test_myop_l1 _________________________
    def test_myop_l1():
>       out = myop_wrapper(q, k, v)
E       pypto.error.PyptoError: Alloc tensor [471] size [393216] exceeds MEM_UB size [196608]!
=============================== warnings summary ===============================
test_myop.py:1
  DeprecationWarning: pkg_resources is deprecated as an API.
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED test_myop.py::test_myop_l1 - pypto.error...
======================== 1 failed, 18 warnings in 2.27s ========================
"""


# The OTHER pytest layout: the traceback is printed in full, so its last line is
# the frame it raised in -- `<path>:<lineno>: <ExceptionName>` -- and that line
# sits BELOW the message. Taken verbatim from a campaign whose capacity
# refusals recorded this and nothing else.
PYTEST_FULL_TB = """\
============================= test session starts ==============================
collected 1 item

custom/myop2/test_myop2.py F

=================================== FAILURES ===================================
_________________________________ test_e2e_l1 __________________________________
    def _raise_if_exception(self):
>       raise FeError(exc)
E       pypto.error.FeError: ErrCode: F40005! Enum: TensorErr::TENSOR_MEMORY_ALLOCATION. [OoOScheduleState.Operation]:OP CAST[10291] in/output total size [294912] exceeds MEM_UB size [196608]!

/usr/local/lib/python3.10/dist-packages/pypto/_controller.py:400: FeError
=========================== short test summary info ============================
FAILED custom/myop2/test_myop2.py::test_e2e_l1
1 failed in 6.83s
"""


class _Stub(ev.RealEvaluator):
    """The real `_cause` and `_tail` over a canned output, reachable by name.

    A subclass rather than a hand-built stand-in: the methods under test are
    the real ones, and the two public wrappers exist so the cases below do not
    have to reach into the class to call them.
    """

    @classmethod
    def over(cls, out):
        """One instance holding `out`, built without the real constructor.

        `RealEvaluator.__init__` reads the kernel off disk and binds the command
        runner; neither exists here and neither is what these cases are about.
        """
        stub = cls.__new__(cls)
        stub._last_output = out
        return stub

    def cause(self, n=4):
        return self._cause(n)

    def tail(self, n=12):
        return self._tail(n)


class CauseTest(unittest.TestCase):

    def test_the_tail_is_what_threw_the_message_away(self):
        """
        Stated as a fact about the old behaviour, so the regression is
        visible if anyone puts `_tail(4)` back on this path.
        """
        self.assertNotIn("MEM_UB", _Stub.over(PYTEST_OUT).tail(4))

    def test_the_cause_keeps_the_size_and_the_buffer(self):
        d = _Stub.over(PYTEST_OUT).cause()
        self.assertIn("MEM_UB size [196608]", d)
        self.assertIn("393216", d, "the size that did not fit")

    def test_it_still_carries_the_summary(self):
        """Nothing that used to be recorded is lost."""
        self.assertIn("1 failed", _Stub.over(PYTEST_OUT).cause())

    def test_a_plain_script_is_unchanged_in_substance(self):
        """`python3 test_op.py` puts the traceback at the end already."""
        out = ("INFO:torch_npu.env:get env TILE_FWK_DEVICE_ID = 7\n"
               "Traceback (most recent call last):\n"
               "  File \"test_op.py\", line 9, in <module>\n"
               "RuntimeError: Alloc tensor [3] size [524288] exceeds MEM_L1 size [524288]!\n")
        d = _Stub.over(out).cause()
        self.assertIn("MEM_L1", d)

    def test_output_with_no_cause_line_falls_back_to_the_tail(self):
        out = "\n".join(f"line {i}" for i in range(20))
        self.assertEqual(_Stub.over(out).cause(), _Stub.over(out).tail(4))

    def test_empty_output_is_empty(self):
        self.assertEqual(_Stub.over("").cause(), "")

    def test_the_startup_death_case_still_says_what_little_there_was(self):
        """
        The other failure class: one line of startup output and the
        process is gone. There is no cause line, so the tail is all there is --
        and that IS the finding.
        """
        out = "INFO:torch_npu.env:get env TILE_FWK_DEVICE_ID = 7\n"
        self.assertIn("TILE_FWK_DEVICE_ID", _Stub.over(out).cause())


class FullTracebackLayoutTest(unittest.TestCase):
    """The layout the ranked markers exist for.

    Picking the DEEPEST line matching any marker was worse here than the tail it
    replaced: the frame location line below the message matches "Error" through
    the exception's name, so the capacity refusals were lost to it.
    """

    def setUp(self):
        self.why = _Stub.over(PYTEST_FULL_TB).cause()

    def test_the_numbers_survive(self):
        self.assertIn("294912", self.why)
        self.assertIn("196608", self.why)
        self.assertIn("CAST[10291]", self.why)

    def test_the_enum_survives(self):
        self.assertIn("TENSOR_MEMORY_ALLOCATION", self.why)
        self.assertIn("F40005", self.why)

    def test_the_frame_location_line_is_not_the_cause(self):
        """It is what the deepest-match rule chose, and it carries no message."""
        frame = "/usr/local/lib/python3.10/dist-packages/pypto/_controller.py:400: FeError"
        self.assertTrue(ev.is_frame_line(frame))
        self.assertTrue(any(m in frame for m in ev.CAUSE_WEAK),
                        "this is why it was chosen")
        self.assertFalse(any(m in frame for m in ev.CAUSE_STRONG))

    def test_a_frame_line_is_told_from_a_message_line(self):
        self.assertTrue(ev.is_frame_line("a/b/c.py:400: FeError"))
        self.assertTrue(ev.is_frame_line("mod.py:1: ValueError"))
        self.assertFalse(ev.is_frame_line(
            "E   pypto.error.FeError: ErrCode: F40005! Enum: x"))
        self.assertFalse(ev.is_frame_line(
            'FAILED test_x.py::test_y - pypto.error...'))

    def test_a_strong_line_wins_over_a_deeper_weak_one(self):
        out = "E   Alloc tensor [471] size [393216] exceeds MEM_UB size [196608]!\n" \
              "some/mod.py:12: PyptoError\n" \
              "ERROR trailing noise\n"
        self.assertIn("393216", _Stub.over(out).cause())

    def test_a_weak_line_still_answers_when_there_is_no_strong_one(self):
        out = "collected 1 item\n>   assert out.allclose(ref)\nE   AssertionError\n" \
              "1 failed in 3.00s\n"
        why = _Stub.over(out).cause()
        self.assertIn("AssertionError", why)

    def test_output_with_no_marker_falls_back_to_the_tail(self):
        out = "line one\nline two\nline three\n"
        self.assertEqual(_Stub.over(out).cause(), "line one\nline two\nline three")

    def test_empty_output_is_empty(self):
        self.assertEqual(_Stub.over("").cause(), "")


if __name__ == "__main__":
    unittest.main()
