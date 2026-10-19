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
DeepSeek V3.2 实验性 PyPTO 融合算子库

实际集成的算子：
- Lightning Indexer Prolog (量化版): Q/Prolog + RoPE + Hadamard Transform + 量化，为 LightningIndexer 提供 q、weights、q_scale
- Lightning Indexer (量化版): sparse attention 的 decode 阶段 token 选择与索引
- MLA Indexer Prolog (量化版): MLA + Prolog 前处理 + RoPE + 量化
- MLA Prolog (量化版): hidden states 转 Query / Key-Value + RoPE + 量化
- Sparse Attention (反量化版): 离散选取 + 反量化 + 注意力计算
- Sparse Flash Attention (量化版): 量化 + flash attention + 稀疏索引

融合范围：
- 量化 + 反量化 + RoPE + Attention 的端到端融合
- MLA Prolog: hidden states -> q/kv with quantized matmul + RMSNorm + RoPE
- Indexer Prolog: qr * idx_wq_b -> q + Hadamard Transform + RoPE + quant
- Sparse Attention: topk indices -> selective attention with quantized KV cache

应用场景：
- DeepSeek V3.2 模型的推理优化，特别是 MLA (Multi-head Latent Attention) 和量化推理
"""

from .lightning_indexer_prolog_quant_impl import (
    lightning_indexer_prolog_quant,
    IndexerPrologQuantInput,
    IndexerPrologQuantOutput,
    IndexerPrologQuantAttr,
    IndexerPrologQuantConfigs,
)
from .lightning_indexer_quant_impl import (
    lightning_indexer_decode,
    LightningIndexerConfigs,
)
from .mla_indexer_prolog_quant_impl import (
    mla_indexer_prolog_quant_p,
    mla_indexer_prolog_quant_d,
)
from .mla_prolog_quant_impl import (
    mla_prolog_quant_p,
    mla_prolog_quant_d,
    MlaTileConfig,
    MlaQuantInputs,
    RopeTileShapeConfig,
)
from .sparse_attention_antiquant_impl import (
    sparse_attention_antiquant_p,
    sparse_attention_antiquant_d,
    SaTileShapeConfig,
)
from .sparse_flash_attention_quant_impl import (
    sparse_flash_attention_quant_p,
    sparse_flash_attention_quant_d,
    sparse_flash_attention_d_950,
)

__all__ = [
    # Lightning Indexer Prolog
    'lightning_indexer_prolog_quant',
    'IndexerPrologQuantInput',
    'IndexerPrologQuantOutput',
    'IndexerPrologQuantAttr',
    'IndexerPrologQuantConfigs',
    # Lightning Indexer
    'lightning_indexer_decode',
    'LightningIndexerConfigs',
    # MLA Indexer Prolog
    'mla_indexer_prolog_quant_p',
    'mla_indexer_prolog_quant_d',
    # MLA Prolog
    'mla_prolog_quant_p',
    'mla_prolog_quant_d',
    'MlaTileConfig',
    'MlaQuantInputs',
    'RopeTileShapeConfig',
    # Sparse Attention Antiquant
    'sparse_attention_antiquant_p',
    'sparse_attention_antiquant_d',
    'SaTileShapeConfig',
    # Sparse Flash Attention
    'sparse_flash_attention_quant_p',
    'sparse_flash_attention_quant_d',
    'sparse_flash_attention_d_950',
]
