# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""KDA (Kimi Delta Attention) PyPTO kernel package.

    prefill : ``kda_fused_prefill`` implements the fused PyPTO prefill path.
              Unsupported inputs fall back to the reference torch path in the
              modeling layer.
    decode  : ``kda_fused_decode`` (eager, guarded) and
              ``kda_fused_decode_step`` (capture-safe, no guards) run the whole
              per-token KDA path in one operator.

See ``README.md`` for algorithm and dispatch details.
"""
__all__ = [
    # prefill
    "kda_fused_prefill",
    # decode
    "kda_fused_decode", "kda_fused_decode_step",
    "prepare_kda_fused_weights", "KdaFusedWeights",
    "make_fused_buffers", "KdaFusedBuffers", "KdaBufferParams",
    "seed_fused_buffers",
]

from .kda_fused_prefill_impl import kda_fused_prefill
from .kda_fused_decode_impl import (
    kda_fused_decode,
    kda_fused_decode_step,
    prepare_kda_fused_weights,
    KdaFusedWeights,
    make_fused_buffers,
    KdaFusedBuffers,
    KdaBufferParams,
    seed_fused_buffers,
)
