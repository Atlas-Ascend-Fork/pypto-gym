# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent CPU equations for vLLM PR #56255's shifted BF16 path.

Reference: DeepGEMM 39d8c4c, third-party/tilelang_ops/ref_mhc.py.
No import of the NPU implementation. Both contract-required BF16 boundaries
are explicit.
"""
from typing import NamedTuple

import torch


class MegaMhcInputs(NamedTuple):
    x: torch.Tensor
    residual: torch.Tensor
    previous_mix: torch.Tensor
    post_mix: torch.Tensor
    comb_res_mix: torch.Tensor
    fn: torch.Tensor
    mix_scales: torch.Tensor
    mix_bases: torch.Tensor
    rmsnorm_weight: torch.Tensor


def make_inputs(tokens, hidden, seed=2026, mode="random"):
    gen = torch.Generator().manual_seed(seed)

    def randn(shape):
        return torch.randn(shape, generator=gen, dtype=torch.float32)
    x = randn((tokens, hidden)).bfloat16()
    r = randn((tokens, 4, hidden)).bfloat16()
    prev = randn((tokens, 4)).sigmoid()
    post = randn((tokens, 4, 1)).sigmoid()
    comb = torch.rand((tokens, 4, 4), generator=gen)
    for _ in range(3):
        comb = comb / comb.sum(-1, keepdim=True)
        comb = comb / comb.sum(-2, keepdim=True)
    fn = randn((24, 4 * hidden)) * 0.01
    scales = randn((3,)) * 0.1
    bases = randn((24,)) * 0.1
    gamma = (randn((hidden,)) * 0.1 + 1.0).bfloat16()
    if mode == "zero":
        x.zero_()
        r.zero_()
    elif mode == "tiny":
        x = (x.float() * 1e-7).bfloat16()
        r = (r.float() * 1e-7).bfloat16()
        prev *= 0.125
        post *= 0.125
        fn *= 4
        gamma = (gamma.float() * 0.125).bfloat16()
    elif mode != "random":
        raise ValueError(mode)
    return MegaMhcInputs(x, r, prev, post, comb, fn, scales, bases, gamma)


@torch.no_grad()
def mega_mhc_golden_cpu(x, residual, previous_mix, post_mix, comb_res_mix,
                       fn, mix_scales, mix_bases, rmsnorm_weight, *,
                       hc_norm_eps=2e-5, hc_pre_eps=3e-4, hc_post_scale=1.25,
                       sinkhorn_eps=2e-6, num_sinkhorn_iters=10, rmsnorm_eps=7e-6):
    if not all(t.device.type == "cpu" for t in (x, residual, fn)):
        raise ValueError("golden inputs must be CPU tensors")
    r = (torch.einsum("tij,tih->tjh", comb_res_mix.float(), residual.float())
         + x.float().unsqueeze(1) * post_mix.float()).bfloat16()
    u = (r.float() * previous_mix.float().unsqueeze(-1)).sum(1).bfloat16()
    flat = r.float().flatten(1)
    logits = (flat @ fn.float().T) * torch.rsqrt(flat.square().mean(1) + hc_norm_eps).unsqueeze(1)
    scales = torch.cat((mix_scales[0].expand(4), mix_scales[1].expand(4), mix_scales[2].expand(16)))
    logits = logits * scales + mix_bases
    new_pre = logits[:, :4].sigmoid() + hc_pre_eps
    new_post = (logits[:, 4:8].sigmoid() * hc_post_scale).unsqueeze(-1)
    new_comb = logits[:, 8:].reshape(-1, 4, 4).softmax(-1) + sinkhorn_eps
    new_comb = new_comb / (new_comb.sum(-2, keepdim=True) + sinkhorn_eps)
    for _ in range(num_sinkhorn_iters - 1):
        new_comb = new_comb / (new_comb.sum(-1, keepdim=True) + sinkhorn_eps)
        new_comb = new_comb / (new_comb.sum(-2, keepdim=True) + sinkhorn_eps)
    y = (u.float() * torch.rsqrt(u.float().square().mean(1) + rmsnorm_eps).unsqueeze(1)
         * rmsnorm_weight.float()).bfloat16()
    return r, new_post, new_comb, y, new_pre
