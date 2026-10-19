#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""openPangu-Embedded-7B (PanguEmbedded) PyPTO fused-layer kernel adapter.

A single whole-decoder-layer fused kernel, ``@pypto.frontend.jit``-decorated and
registered as a ``torch.ops.pypto.*`` custom (``@allow_in_graph``):

* ``pangu_fused_layer_v2_bsh`` — one decoder layer per kernel instance,
  BSH KV-cache layout, dynamic ``actual_kv_len``.

Each layer fuses: residual-add + RMSNorm -> QKV GEMM (+bias) -> RoPE(Q/K)
-> KV-cache write -> tiled GQA attention with online softmax (QK^T / softmax / PV)
-> O GEMM (+bias) -> residual-add + RMSNorm -> gate/up/down GEMMs + SwiGLU.
The kernel is decode-only (batch=1, q_len=1); prefill stays on the PyTorch
``PanguEmbeddedDecoderLayer``.

The modeling layer (``pypto_gym.transformers.openpangu_v5_7b.modeling_openpangu_dense``)
reads the switch below to decide whether to wire the fused path; it imports the
heavy kernel module lazily and only when the master switch is enabled, so importing
this package alone does not pull in ``pypto``.

Switch design (opt-in, default off — mirrors gemma4_31b_it / minimax / llada2_moe):

    USE_PTO_FUSED_LAYER — master switch for the PyPTO fused decode path.
"""

__all__ = [
    "USE_PTO_FUSED_LAYER",
    "PanguFusedLayerV2BSHModule",
    "DynamicFusedLayerConfigV2BSH",
]

USE_PTO_FUSED_LAYER = False


def __getattr__(name):
    if name in ("PanguFusedLayerV2BSHModule", "DynamicFusedLayerConfigV2BSH"):
        from ops.pypto_tensor.openpangu_v5_7b.pangu_fused_layer_dynamic_v2_bsh import (
            DynamicFusedLayerConfigV2BSH,
            PanguFusedLayerV2BSHModule,
        )
        if name == "PanguFusedLayerV2BSHModule":
            return PanguFusedLayerV2BSHModule
        return DynamicFusedLayerConfigV2BSH
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
