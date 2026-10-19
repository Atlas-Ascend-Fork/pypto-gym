# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
kda_flash PyPTO 融合算子库

实际集成的算子：
- fused_recurrent_kda: State-maintaining recurrent Key-Delta-Attention (decode / recurrent mode)
- chunk_kda: Kimi Delta Attention (chunked gated delta-rule linear attention, forward only)

应用场景：
- 目标模型的 Key-Delta-Attention 推理加速
- fused_recurrent_kda 用于 decode 阶段的逐 token 递推
- chunk_kda 用于 prefill 阶段的 chunk 级融合计算
"""

from .chunk_kda.chunk_kda_impl import chunk_kda_wrapper
from .fused_recurrent_kda.fused_recurrent_kda_impl import fused_recurrent_kda_wrapper

__all__ = [
    'chunk_kda_wrapper',
    'fused_recurrent_kda_wrapper',
]
