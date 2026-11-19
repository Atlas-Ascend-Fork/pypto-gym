# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Three small Mixture-of-Experts architectures defined locally.

Used when downloading a pretrained MoE is too expensive. These take random
weights and exist only to exercise the conversion pipeline. Each is a tiny
classifier that maps a fixed-length feature vector to logits.
"""
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKMoE(nn.Module):
    """Classic top-k gated MoE on top of an MLP."""
    def __init__(self, dim=64, num_experts=4, top_k=2, num_classes=10):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
            for _ in range(num_experts)
        ])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        gate_logits = self.gate(x)
        topk_val, topk_idx = gate_logits.topk(self.top_k, dim=-1)
        weights = F.softmax(topk_val, dim=-1)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = topk_idx[:, k]
            w = weights[:, k:k + 1]
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    out[mask] = out[mask] + w[mask] * self.experts[e](x[mask])
        return self.head(out)


class SoftMoE(nn.Module):
    """Soft (fully-weighted) MoE: every expert sees every input."""
    def __init__(self, dim=64, num_experts=4, num_classes=10):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            for _ in range(num_experts)
        ])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        weights = F.softmax(self.gate(x), dim=-1)  # (B, E)
        outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, E, dim)
        mixed = (weights.unsqueeze(-1) * outs).sum(dim=1)
        return self.head(mixed)


class SwitchMoE(nn.Module):
    """Switch-Transformer-style top-1 MoE."""
    def __init__(self, dim=64, num_experts=4, num_classes=10):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
            for _ in range(num_experts)
        ])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        gate_logits = self.gate(x)
        idx = gate_logits.argmax(dim=-1)
        gate_val = F.softmax(gate_logits, dim=-1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            mask = (idx == e)
            if mask.any():
                out[mask] = expert(x[mask]) * gate_val[mask].unsqueeze(-1)
        return self.head(out)


TOY_MODELS = {
    "toy_topk_moe": (TopKMoE, {"dim": 64, "num_experts": 4, "top_k": 2}),
    "toy_soft_moe": (SoftMoE, {"dim": 64, "num_experts": 4}),
    "toy_switch_moe": (SwitchMoE, {"dim": 64, "num_experts": 4}),
}


def build_toy(name: str) -> tuple[nn.Module, torch.Tensor]:
    cls, kwargs = TOY_MODELS[name]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        model = cls(**kwargs).eval()
        sample = torch.randn(2, kwargs["dim"])
    return model, sample


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    for name in TOY_MODELS:
        model, sample = build_toy(name)
        out = model(sample)
        n_params = sum(p.numel() for p in model.parameters())
        logging.info(
            "%s: out shape %s, params %s", name, tuple(out.shape), f"{n_params:,}",
        )
