# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Qwen3.5 PyPTO 融合算子库

集成的算子：
- GDR Forward (gdr_fwd): Gated Delta Rule 前向 chunk-parallel kernel
- GDR Backward (gdr_bwd): Gated Delta Rule 反向 chunk-parallel kernel

融合范围：
- chunk_gated_delta_rule 内部的 l2norm + gate/decay + inverse + WY representation
  + output/state recurrence 等子算子的融合

应用场景：
- Qwen3.5 模型的 prefill / training 推理加速
"""

__all__ = [
    'gdr_fwd',
    'gdr_bwd',
]
