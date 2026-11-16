# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Frozen finite-value and relative RMS/L2 checks for KDA regression tests."""

import torch


def metric(ref, actual):
    ref, actual = ref.float(), actual.float()
    finite = bool(torch.isfinite(ref).all() and torch.isfinite(actual).all())
    if not ref.numel():
        return {"ratio": 0.0, "max_abs": 0.0, "finite": finite, "pass": finite}
    ratio = float((actual - ref).square().mean().sqrt() / (ref.square().mean().sqrt() + 1e-8))
    return {
        "ratio": ratio,
        "max_abs": float((actual - ref).abs().max()),
        "finite": finite,
        "pass": finite and ratio < 8e-3,
    }
