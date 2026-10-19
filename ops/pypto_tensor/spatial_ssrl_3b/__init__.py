#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance of the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Spatial-SSRL-3B PyPTO 算子库

使用方式:
    import sys
    sys.path.insert(0, model_path)
    import spatial_ssrl_3b_pto_kernels as pto_kernels
    sys.modules["spatial_ssrl_3b_pto_kernels"] = pto_kernels
    pto_kernels.USE_PTO_RMS_NORM = True
    
    # 之后加载模型，modeling 会自动获取 pto_kernels
"""
from .rms_norm import rms_norm_pto_wrapper
from .rope import (
    apply_rotary_pos_emb_vision_pto_impl,
    apply_multimodal_rotary_pos_emb_pto_impl,
    USE_PTO_ROPE
)

USE_PTO_RMS_NORM = False
USE_PTO_ROPE = False


# Wrapper aliases for modeling code compatibility
def apply_rotary_pos_emb_vision_wrapper(q, k, cos, sin):
    """Wrapper alias for vision RoPE PTO implementation."""
    return apply_rotary_pos_emb_vision_pto_impl(q, k, cos, sin)


def apply_multimodal_rotary_pos_emb_wrapper(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Wrapper alias for multimodal RoPE PTO implementation."""
    return apply_multimodal_rotary_pos_emb_pto_impl(q, k, cos, sin, mrope_section, unsqueeze_dim)


__all__ = [
    "USE_PTO_RMS_NORM",
    "USE_PTO_ROPE",
    "rms_norm_pto_wrapper",
    "apply_rotary_pos_emb_vision_pto_impl",
    "apply_multimodal_rotary_pos_emb_pto_impl",
    "apply_rotary_pos_emb_vision_wrapper",
    "apply_multimodal_rotary_pos_emb_wrapper",
]
