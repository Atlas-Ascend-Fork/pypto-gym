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
PyPTO 实验性算子目录

本目录收录处于开发阶段的实验性算子实现，按类别组织在以下子目录中：

子类别：
- attention/: 注意力机制实验算子 (BSA/BWD/FWD, Chunked GDR, Incre Flash Attention GQA/MLA, PFA Flash Attention)
- distributed/: 分布式计算实验算子 (config, swimlane analyzer)
- matmul/: 矩阵乘法实验算子 (GMM, MHC, Quant Batch Matmul, Transpose Quant Batch Matmul)
- ops_transformer/: Transformer 结构实验算子 (Flash Attention MHA/Grad/Score,
  Fused SwiGLU, Lightning Indexer, MLA Prolog, Page Attention, Sparse Attention)
- vector/: 向量运算实验算子 (AdamW, RMSProp, AvgPool, BN, RMSNorm, RoPE, Scatter/Gather, MoE)

注意：
- 实验性算子默认不参与 CI 测试 (pytest.ini 中 norecursedirs 自动排除)
- 算子 API 可能变更，不建议在生产环境使用
- 每个子目录下的算子需独立运行测试
"""

__all__ = []
