# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unordered TopK validation with this operator's numerical criteria."""

import torch

SCORE_RTOL = 1 / 128
SCORE_ATOL = 2.5e-5


def _check_topk(indices, values, scores, allowed, offset=0):
    """Check values, completeness and the TopK boundary without ordering ties."""
    valid = indices != -1
    if values is not None:
        if not torch.isneginf(values[~valid]).all():
            raise AssertionError("Invalid TopK values must be -inf")
    selected = indices[valid].long() - offset
    if selected.numel() != min(indices.numel(), int(allowed.sum())):
        raise AssertionError(
            f"selected count {selected.numel()} != min(topk, allowed)"
            f" {min(indices.numel(), int(allowed.sum()))}"
        )
    if selected.unique().numel() != selected.numel():
        raise AssertionError("selected indices must be unique")
    if not ((selected >= 0) & (selected < scores.numel())).all():
        raise AssertionError("selected indices out of range")
    if not allowed[selected].all():
        raise AssertionError("selected indices outside the allowed set")
    if selected.numel() == 0:
        return
    selected_scores = scores[selected]
    if values is not None:
        torch.testing.assert_close(
            values[valid].float(), selected_scores, rtol=SCORE_RTOL, atol=SCORE_ATOL
        )
    boundary = scores[allowed].topk(selected.numel()).values[-1]
    if torch.isposinf(boundary):
        if not torch.isposinf(selected_scores).all():
            raise AssertionError("positive-inf boundary must select +inf scores")
    else:
        floor = boundary - SCORE_ATOL - SCORE_RTOL * boundary.abs()
        if not (selected_scores.min() >= floor).all():
            raise AssertionError(
                f"selected score {float(selected_scores.min())} below TopK boundary {float(boundary)}"
            )
