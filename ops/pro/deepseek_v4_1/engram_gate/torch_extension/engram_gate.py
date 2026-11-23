# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public Torch frontend for the EngramGate operator."""
import torch

_EPS = 1.0e-6
_CLAMP_VALUE = 1.0e-6


def engram_gate(x, key, value, weight, image_mask, eps=_EPS,
                clamp_value=_CLAMP_VALUE):
    """Gated engram attention over one chunk.

    ``x``/``key`` are BF16 ``[T,HC,D]``; ``value`` is BF16 ``[T,D]``; ``weight``
    is FP32 ``[HC,D]``; ``image_mask`` is bool ``[T]``. Returns the BF16
    ``[T,HC,D]`` gated output allocated internally. The kernel bakes
    ``eps``/``clamp_value``; only the contract defaults are accepted.
    """
    import math
    if not math.isclose(eps, _EPS) or not math.isclose(clamp_value, _CLAMP_VALUE):
        raise ValueError(
            f"this kernel bakes eps={_EPS} and clamp_value={_CLAMP_VALUE}, "
            f"got eps={eps} and clamp_value={clamp_value}")
    return torch.ops.pypto_gym.engram_gate(x, key, value, weight, image_mask)
