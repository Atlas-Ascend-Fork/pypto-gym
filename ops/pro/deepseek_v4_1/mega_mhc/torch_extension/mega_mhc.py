# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public Torch frontend for the precompiled MegaMhc operator."""
import math
import torch


def mega_mhc(x, residual, previous_mix, post_mix, comb_res_mix, fn,
             mix_scales, mix_bases, rmsnorm_weight, *, hc_norm_eps=2e-5,
             hc_pre_eps=3e-4, hc_post_scale=1.25, sinkhorn_eps=2e-6,
             num_sinkhorn_iters=10, rmsnorm_eps=7e-6, batch=None):
    """Return (new_residual, new_post_mix, new_comb_res_mix, y, new_previous_mix).

    Runs the installed ACLNN binary on the current NPU stream. batch=None selects
    automatically; batch=16 forces batch16. Outputs are allocated internally.
    The operator provides forward computation only.
    """
    if batch is not None and (type(batch) is not int or batch != 16):
        raise ValueError('public batch must be None or 16')
    values = (hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, rmsnorm_eps)
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or
           not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('eps and post scale must be finite nonnegative scalars')
    if hc_norm_eps <= 0 or rmsnorm_eps <= 0:
        raise ValueError('the two RMS eps values must be positive')
    if type(num_sinkhorn_iters) is not int or not 1 <= num_sinkhorn_iters <= 65535:
        raise ValueError('num_sinkhorn_iters must be an integer in [1,65535]')
    inputs = (x, residual, previous_mix, post_mix, comb_res_mix, fn,
              mix_scales, mix_bases, rmsnorm_weight)
    attrs = dict(hc_norm_eps=hc_norm_eps, hc_pre_eps=hc_pre_eps,
                 hc_post_scale=hc_post_scale, sinkhorn_eps=sinkhorn_eps,
                 num_sinkhorn_iters=num_sinkhorn_iters, rmsnorm_eps=rmsnorm_eps,
                 batch=0 if batch is None else batch)
    return torch.ops.pypto_gym.mega_mhc.default(*inputs, **attrs)
