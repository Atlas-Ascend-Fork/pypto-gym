# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fixed per-output tolerances from vLLM PR #56255's correctness test."""
import torch

NAMES = ("new_residual", "new_post_mix", "new_comb_res_mix", "y_bf16", "new_prev_mix")
LIMITS = ((1e-5, 1e-2), (1e-6, 1e-3), (1e-6, 1e-3), (1e-5, 1e-2), (1e-6, 1e-3))


def assert_outputs(actual, expected):
    if len(actual) != 5 or len(expected) != 5:
        raise AssertionError("expected five outputs")
    report = {}
    for name, a, e, (atol, rtol) in zip(NAMES, actual, expected, LIMITS):
        a = a.detach().cpu()
        if a.shape != e.shape or a.dtype != e.dtype:
            raise AssertionError(f"{name}: shape/dtype mismatch")
        if not a.is_contiguous() or not torch.isfinite(a).all():
            raise AssertionError(f"{name}: output is not contiguous/finite")
        torch.testing.assert_close(a, e, atol=atol, rtol=rtol, msg=lambda m: name + ": " + m)
        error = (a.float() - e.float()).abs()
        report[name] = {
            "atol": atol, "rtol": rtol,
            "max_abs": error.max().item() if error.numel() else 0.0,
            "max_tolerance_fraction": (error / (atol + rtol * e.float().abs())).max().item() if error.numel() else 0.0,
        }
    return report
