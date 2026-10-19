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
Qwen3-VL-8B (Instruct Unredacted Max) PyPTO 融合算子库

实际集成的算子：
- RMSNorm (PyPTO 融合版): pypto.rms_norm 融合实现，替代 PyTorch fp32 的 pow+mean+rsqrt 路径

融合范围：
- mean(x^2, dim=-1) + rsqrt(mean+eps) + x * rsqrt * weight

应用场景：
- Qwen3-VL-8B 模型的 text decoder 和 vision merger 中的 Qwen3VLTextRMSNorm 层
- 通过 sys.modules["pto_kernels"] 注入 USE_PTO_RMS_NORM 开关和 rms_norm_wrapper 函数

注入方式：
- sys.modules["pto_kernels"] 提供 USE_PTO_RMS_NORM 开关和 rms_norm_wrapper 函数
- 与 gutenocr_3b、spatial_ssrl_3b、phi_3_mini 共享 "pto_kernels" 模块名，确保每进程仅加载一个模型
"""

USE_PTO_RMS_NORM = False

from .rms_norm.rms_norm_impl import rms_norm_impl

__all__ = [
    'USE_PTO_RMS_NORM',
    'rms_norm_impl',
]
