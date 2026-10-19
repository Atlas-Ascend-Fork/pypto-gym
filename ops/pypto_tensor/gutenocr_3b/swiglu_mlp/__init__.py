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
SwiGLU MLP PyPTO算子模块
提供融合算子实现（gate_proj + up_proj + down_proj融合）

✅ 所有batch推荐启用（唯一稳定有效算子）
"""

__all__ = ["swiglu_mlp_fused", "swiglu_mlp_fused_static"]

from .swiglu_mlp_impl import swiglu_mlp_fused, swiglu_mlp_fused_static