#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Gemma-4-31B-it PyPTO fused kernel adapter.

Switch design: per-operator granularity for incremental validation.
"""

from .attn_softmax.attn_softmax_impl import attn_softmax_wrapper as attn_softmax
from .gqa_decode_attn.gqa_decode_attn_impl import gqa_decode_attn_wrapper as gqa_decode_attn

USE_PTO_SOFTMAX = False
USE_PTO_GQA = False
