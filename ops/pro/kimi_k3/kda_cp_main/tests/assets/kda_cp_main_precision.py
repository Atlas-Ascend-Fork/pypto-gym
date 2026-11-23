#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Precision criteria for kda_cp_main (frozen tolerance).

The kernel output ``out`` (bf16) and ``final_state`` (fp32) are compared
against the FP32 CPU golden with ``torch.testing.assert_close`` semantics:
``atol=3e-3, rtol=3e-2``; any NaN/Inf mismatch fails implicitly.
"""

import torch

ATOL, RTOL = 3e-3, 3e-2


def check_output(out, final_state, ref_out, ref_state):
    """Compare both outputs of one case against the golden; raise on failure."""
    torch.testing.assert_close(out.cpu(), ref_out.cpu(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(final_state.cpu(), ref_state.cpu(), atol=ATOL, rtol=RTOL)
