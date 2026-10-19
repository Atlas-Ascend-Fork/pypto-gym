#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance of the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
RoPE PyPTO Implementation 导出

目标文件: core/modeling_spatial_ssrl_3b_vl.py
目标类: GutenOcr_3b_VLVisionAttention, GutenOcr_3b_VLAttention
替换位置: apply_rotary_pos_emb_vision, apply_multimodal_rotary_pos_emb

策略：使用 PyPTO kernel implementation (@pypto.frontend.jit + pypto operations)

归档说明：仅归档 rope_impl.py (核心PyPTO kernel实现)
"""

from .rope_impl import (
    apply_rotary_pos_emb_vision_pto_impl,
    apply_multimodal_rotary_pos_emb_pto_impl
)

USE_PTO_ROPE = False
