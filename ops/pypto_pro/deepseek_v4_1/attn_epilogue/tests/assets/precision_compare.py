# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed upstream acceptance: exact BF16 values and exact quantization bytes."""

import torch


def assert_exact(actual, expected, label="output"):
    """Assert bit-exact bf16 equality elementwise, reporting the first mismatch."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{label}: shape/dtype mismatch")
    a, b = actual.detach().cpu(), expected.detach().cpu()
    if a.is_floating_point() and not bool(torch.isfinite(a.float()).all()):
        raise AssertionError(f"{label}: nonfinite values")
    torch.testing.assert_close(a, b, rtol=0, atol=0, msg=label)
    if a.dtype == torch.bfloat16 and not torch.equal(a.view(torch.int16), b.view(torch.int16)):
        raise AssertionError(f"{label}: BF16 bit patterns differ")


def assert_payload(actual, expected, label="payload"):
    """Assert byte-exact equality of a payload/scale buffer, reporting the first mismatch."""
    assert_exact(actual.view(torch.uint8), expected.view(torch.uint8), label)
