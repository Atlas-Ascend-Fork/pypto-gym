# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public Torch frontend for the precompiled AttnPrologue operator."""
import math

import torch


def attn_prologue(x, wqa, wqb, wkv,
                  descale_x, descale_wqa, descale_wqb, descale_wkv,
                  norm_weight_qr, norm_weight_kv,
                  rope_sin, rope_cos, cache_index, kv_cache,
                  qr_norm=None, norm_eps=1e-6):
    """Return (q, qr, descale_qr); kv_cache is updated in place.

    Runs the installed ACLNN binary on the current NPU stream. Inputs follow the
    operator contract: x/weights are E4M3 (weights FRACTAL_NZ), descales E8M0,
    rope/norm weights FP32, cache_index INT64 and kv_cache UINT8 [P,BS,1,544].
    The operator provides forward computation only.
    """
    scalar = isinstance(norm_eps, (int, float)) and not isinstance(norm_eps, bool)
    if not scalar or not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError('norm_eps must be a finite positive scalar')
    q, qr, descale_qr, _ = torch.ops.pypto_gym.attn_prologue.default(
        x, wqa, wqb, wkv, descale_x, descale_wqa, descale_wqb, descale_wkv,
        norm_weight_qr, norm_weight_kv, rope_sin, rope_cos, cache_index,
        kv_cache, qr_norm=qr_norm, norm_eps=float(norm_eps))
    return q, qr, descale_qr.view(torch.int8)
