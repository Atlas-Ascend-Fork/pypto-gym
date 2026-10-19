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
RMSNorm PyPTO Kernel 导出

目标文件: core/modeling_qwen2_5_vl.py
目标类: GutenOcr_3b_VLDecoderLayer
替换位置: input_layernorm, post_attention_layernorm, final norm
"""
from .rms_norm_impl import rms_norm_impl as rms_norm_pto_wrapper

USE_PTO_RMS_NORM = False
