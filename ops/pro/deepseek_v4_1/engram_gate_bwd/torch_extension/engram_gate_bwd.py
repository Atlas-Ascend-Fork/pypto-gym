# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public Torch frontend for the EngramGate backward operator."""
import torch

_CLAMP_VALUE = 1.0e-6


def engram_gate_bwd(grad_out, x, key, value, weight, rstd_x, rstd_k, raw_dot,
                    gate, image_mask, clamp_value=_CLAMP_VALUE):
    """Backward pass of :func:`engram_gate`.

    ``saved`` forward statistics are packed on the host
    (``[raw_dot | gate | rstd_x | rstd_k]``). Returns
    ``(grad_x, grad_key, grad_value, grad_weight)``; the weight gradient is
    reduced on the same stream. The kernel bakes ``clamp_value``; only the
    contract default is accepted.
    """
    import math
    if not math.isclose(clamp_value, _CLAMP_VALUE):
        raise ValueError(
            f"this kernel bakes clamp_value={_CLAMP_VALUE}, got {clamp_value}")
    return torch.ops.pypto_gym.engram_gate_bwd(
        grad_out, x, key, value, weight, rstd_x, rstd_k, raw_dot, gate,
        image_mask)
