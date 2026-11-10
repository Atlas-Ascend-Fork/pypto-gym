# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public Torch frontend for the precompiled AttenEpilogue operator."""
import torch


def atten_epilogue(o, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos):
    """Return the BF16 ``[T, 5120]`` attention output projection.

    Runs the installed ACLNN binary on the current NPU stream; the output is
    allocated internally. ``o`` is BF16 ``[T,64,512]``; ``woa``/``wob`` are
    FP8 E4M3FN weights with FP8 E8M0FNU block scales ``descale_woa`` /
    ``descale_wob``; ``rope_sin``/``rope_cos`` are FP32 ``[T,64]`` tables.
    The operator provides forward computation only.
    """
    return torch.ops.pypto_gym.atten_epilogue(
        o, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos)
