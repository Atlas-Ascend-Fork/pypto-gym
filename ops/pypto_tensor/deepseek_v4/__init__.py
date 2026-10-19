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
DeepSeek V4 PyPTO 融合算子库

实际集成的算子：
- Compress Flash Attention: 支持 PageAttention 的压缩 + flash attention，结合压缩 KV cache 和原始 KV cache 进行注意力计算
- Compressor: 将每 4/128 个 token 的 KV cache 压缩成一个 + RMSNorm + RoPE + Hadamard Transform
- HC Pre (mHC): RMSNorm + 混合 + Sinkhorn-Knopp 算法 + rowsum 后处理
- Lightning Indexer Prolog (V4 量化版): q/weights 前处理 + RoPE + Hadamard + 量化
- MLA Prolog (V4 量化版): MLA hidden states -> Query/Key-Value + RoPE + 量化
- MLA Prolog (V4 非量化版): MLA hidden states -> Query/Key-Value + RoPE (非量化)
- Sparse Compress Flash Attention: 稀疏索引 + 压缩 KV + flash attention
- Sliding Window Attention: 滑动窗口 + PageAttention + 注意力掩码

融合范围：
- 量化/反量化 + RMSNorm + RoPE + Attention 端到端融合
- Compressor: Matmul + 状态更新 + Softmax + ReduceSum + RMSNorm + RoPE + Hadamard
- HC Pre: RMSNorm + Matmul + Sinkhorn-Knopp 迭代 + rowsum
- Compress Flash Attention: PageAttention + 压缩 KV + 原始 KV + flash attention
"""

from .compress_flash_attention_impl import (
    npu_cfa_attention,
    cfa_attention,
    cfa_graph,
)
from .compressor_impl import (
    npu_compressor,
    compressor,
    compressor_pypto,
    Rope2dTileConfig,
)
from .hc_pre_impl import (
    npu_hc_pre,
    hc_pre,
    hc_pre_pypto,
)
from .lightning_indexer_prolog_quant_v4_impl import (
    npu_quant_lightning_indexer_prolog,
    quant_lightning_indexer_prolog,
    IndexerPrologQuantConfig,
)
from .mla_prolog_quant_v4_impl import (
    mla_prolog_v4_in as mla_prolog_quant_v4_in,
    mla_prolog_v4 as mla_prolog_quant_v4,
    MlaPrologV4Output,
    MlaPrologV4Attrs,
    MlaPrologV4Configs,
)
from .mla_prolog_v4_impl import (
    mla_prolog_v4_in,
    mla_prolog_v4,
    mla_prolog,
    mla_prolog_pypto,
    MlaPrologV4Output as MlaPrologV4OutputNonQuant,
    MlaPrologV4Attrs as MlaPrologV4AttrsNonQuant,
    MlaPrologV4Configs as MlaPrologV4ConfigsNonQuant,
)
from .sparse_compress_flash_attention_impl import (
    npu_sparse_compress_flash_attention,
    sparse_compress_flash_attention,
    sparse_compress_flash_attention_graph,
    SCFATileShapeConfig,
)
from .win_attention_impl import (
    deepseekv4_win_atten,
    sliding_window_attention,
    sliding_win_atten_graph,
    get_mask,
)

__all__ = [
    # Compress Flash Attention
    'npu_cfa_attention',
    'cfa_attention',
    'cfa_graph',
    # Compressor
    'npu_compressor',
    'compressor',
    'compressor_pypto',
    'Rope2dTileConfig',
    # HC Pre
    'npu_hc_pre',
    'hc_pre',
    'hc_pre_pypto',
    # Lightning Indexer Prolog
    'npu_quant_lightning_indexer_prolog',
    'quant_lightning_indexer_prolog',
    'IndexerPrologQuantConfig',
    # MLA Prolog (Quant)
    'mla_prolog_quant_v4_in',
    'mla_prolog_quant_v4',
    'MlaPrologV4Output',
    'MlaPrologV4Attrs',
    'MlaPrologV4Configs',
    # MLA Prolog (Non-Quant)
    'mla_prolog_v4_in',
    'mla_prolog_v4',
    'mla_prolog',
    'mla_prolog_pypto',
    'MlaPrologV4OutputNonQuant',
    'MlaPrologV4AttrsNonQuant',
    'MlaPrologV4ConfigsNonQuant',
    # Sparse Compress Flash Attention
    'npu_sparse_compress_flash_attention',
    'sparse_compress_flash_attention',
    'sparse_compress_flash_attention_graph',
    'SCFATileShapeConfig',
    # Sliding Window Attention
    'deepseekv4_win_atten',
    'sliding_window_attention',
    'sliding_win_atten_graph',
    'get_mask',
]
