#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MiniMax MoE PyPTO kernel adapter (shared by MiniMax M2.7 + M3 text backbones).

A single self-contained grouped-GEMM kernel for the MiniMax MoE expert FFN (all
experts in one kernel call), 910B-adapted with a UB-fitting vector tile. The two
text variants differ only in the expert activation, selected by the ``activation``
argument of ``grouped_gemm``:

    "silu"      — MiniMax M2.7 (SiLU-SwiGLU; H=3072, I=1536, E=256, top-8)
    "swigluoai" — MiniMax-M3   (clamped GLU; H=6144, I=3072, E=128, top-4)

The modeling layer reads ``USE_PTO_GROUPED_GEMM`` to decide whether to route the
MoE FFN through the fused kernel (the single switch; routing / gate stay on the
host block). It is opt-in (default False, like llada2_moe / gemma4_31b_it).

The MiniMax-M3 MSA decode kernels (``minimax_m3_msa_indexer`` /
``minimax_m3_msa_sparse_decode``) are imported directly from their ``*_impl``
modules by the modeling layer and tests; they are not re-exported here.
"""

from .minimax_grouped_gemm_impl import (
    minimax_moe_grouped_gemm as grouped_gemm,
    convert_minimax_weights,
    MoeDims,
)

USE_PTO_GROUPED_GEMM = False     # opt-in; the runner / ask script enables it
