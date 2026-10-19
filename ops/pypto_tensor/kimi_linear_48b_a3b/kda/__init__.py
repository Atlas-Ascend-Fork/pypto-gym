# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""KDA (Kimi Delta Attention) PyPTO kernel package.

    prefill : ``kda_chunk_wrapper`` replaces ``fla.ops.kda.chunk_kda`` in
              ``KimiDeltaAttention.forward``; ``kda_chunk_pypto`` is the
              torch.library-registered capturable entry.
    decode  : ``kda_fused_decode`` (eager, guarded) and
              ``kda_fused_decode_step`` (capture-safe, no guards) run the whole
              per-token KDA path in one operator.

See ``README.md`` for the algorithm, the subchunk=16 stability rationale and the
dispatch toggles.
"""
__all__ = [
    # prefill
    "kda_chunk_wrapper", "kda_chunk_pypto",
    # decode
    "kda_fused_decode", "kda_fused_decode_step",
    "prepare_kda_fused_weights", "KdaFusedWeights",
    "make_fused_buffers", "KdaFusedBuffers", "KdaBufferParams",
    "seed_fused_buffers",
]

from .kda_chunk_impl import kda_chunk_wrapper, kda_chunk_pypto
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
