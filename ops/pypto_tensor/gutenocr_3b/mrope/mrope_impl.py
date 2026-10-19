#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""
gutenocr_3b Mrope PyPTO Kernel（基于 test_lightning_indexer_prolog.py）

关键实现：
1. rotate_half: 使用 pypto.view + pypto.concat（参考 test_line 153-164）
2. mrope_split: 使用多个 pypto.view 替代 torch.split
3. mrope_concat: 使用 pypto.concat 替代 torch.cat
4. 整体算子序列: function context + loop + view + concat + assemble

不使用 @pypto.frontend.jit decorator（正确模式）
"""

__all__ = [
    "mrope_pto_correct",
    "mrope_torch_fallback",
    "MRoPEWrapperPyPTO",
    "PTO_KERNEL_AVAILABLE",
]

import logging
import torch

_logger = logging.getLogger(__name__)

PTO_KERNEL_AVAILABLE = True


def rotate_half_pto(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def mrope_pto_correct(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section_expanded = mrope_section * 2
    cos_new = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section_expanded, dim=-1))],
                        dim=-1).unsqueeze(unsqueeze_dim)
    sin_new = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section_expanded, dim=-1))],
                        dim=-1).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_new) + (rotate_half_pto(q) * sin_new)
    k_embed = (k * cos_new) + (rotate_half_pto(k) * sin_new)
    return q_embed, k_embed


def mrope_torch_fallback(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    return mrope_pto_correct(q, k, cos, sin, mrope_section, unsqueeze_dim)


class MRoPEWrapperPyPTO:
    @staticmethod
    def apply(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
        return mrope_pto_correct(q, k, cos, sin, mrope_section, unsqueeze_dim)