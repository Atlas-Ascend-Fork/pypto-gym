# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public Torch frontend for the MhcPreFuse operator."""
import torch

_DEFAULTS = {
    "rms_eps": 1.0e-6,
    "mhc_pre_eps": 1.0e-6,
    "mhc_sinkhorn_eps": 1.0e-6,
    "mhc_post_mult_value": 1.0,
    "sinkhorn_repeat": 10,
}


def mhc_pre_fuse(gemm_out_mul, gemm_out_sqrsum, mhc_scale, mhc_base, residual,
                 knobs=None):
    """Fused mHC pre-pass over the GEMM partials.

    ``gemm_out_mul`` is FP32 ``[T, 24]`` (a leading split axis is accepted and
    slot 0 is used); ``gemm_out_sqrsum`` FP32 ``[T]``; ``mhc_scale`` FP32
    ``[3]``; ``mhc_base`` FP32 ``[24]``; ``residual`` BF16 ``[T, 4, hidden]``.
    ``knobs`` carries the five scalar switches (any object exposing the
    attribute names; contract defaults apply when omitted). Returns
    ``(post_mix, comb_mix, layer_input)``: ``[T, 4]`` FP32, ``[T, 16]`` FP32,
    ``[T, hidden]`` BF16.
    """
    values = dict(_DEFAULTS)
    if knobs is not None:
        for key in _DEFAULTS:
            values[key] = getattr(knobs, key)
    if gemm_out_mul.dim() == 3:
        gemm_out_mul = gemm_out_mul[0]
    if gemm_out_sqrsum.dim() == 2:
        gemm_out_sqrsum = gemm_out_sqrsum[0]
    return torch.ops.pypto_gym.mhc_pre_fuse(
        gemm_out_mul, gemm_out_sqrsum, mhc_scale, mhc_base, residual,
        values["rms_eps"], values["mhc_pre_eps"], values["mhc_sinkhorn_eps"],
        values["mhc_post_mult_value"], values["sinkhorn_repeat"])
