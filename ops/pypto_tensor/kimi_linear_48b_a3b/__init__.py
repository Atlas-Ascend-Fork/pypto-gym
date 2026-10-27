# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Kimi-Linear-48B-A3B PyPTO 融合算子库

实际集成的算子：
- KDA (Kimi Delta Attention) chunk 融合版: 融合 chunk/subchunk 级别的 KDA kernel，
  用于 prefill 阶段替代 FLA/torch 原始实现。

融合范围：
- chunk_kda 内部的 l2norm + cumulative-gate + pairwise decay + triangular-inverse
  + recurrent state carry 等子算子的融合（subchunk=16 数值稳定）。

应用场景：
- Kimi-Linear-48B-A3B-Instruct 模型的 prefill 推理加速，通过 sys.modules 注入
  kimi_linear_48b_a3b_pto_kernels 模块。

注入方式：
- sys.modules["kimi_linear_48b_a3b_pto_kernels"] 提供 USE_PTO_KDA 开关和 kda_chunk_wrapper 函数。

ACLGraph / torch.compile：
- kda_chunk_pypto 经 torch.library 注册为 pypto::kda_chunk_kimi
  （Meta + NPU），可被 torchair/aclgraph 图捕获；eager 默认路径仍走 *_wrapper，互不影响。
"""

from .kda.kda_chunk_impl import kda_chunk_wrapper, kda_chunk_pypto

# NOTE(multi-NPU coverage): PyPTO binds a JIT kernel to ONE NPU per process. With
# the wrappers enabled on a model sharded across NPUs (single-process device_map),
# only the bound NPU's KDA layers run on PyPTO (e.g. ~6 of 20 with 4-way sharding);
# the rest fall back to torch (the wrapper emits a one-time RuntimeWarning so this
# is not silent). Note (future work): for full 20/20 coverage run one process per
# NPU (pipeline / tensor parallel). See kda/README.md.
USE_PTO_KDA = False

# NPU-graph capture flag (off by default — preserves existing behavior). When True,
# KimiDeltaAttention.forward's chunk branch calls ``kda_chunk_pypto`` (the registered
# torch.ops.pypto.kda_chunk_kimi custom op) DIRECTLY, bypassing the _dispatch_kda
# try/except NotImplementedError fallback. The try/except is NOT capturable by an NPU
# graph (control flow on a host-raised exception), so the graph path must call the
# registered op with no host-side branch. Only meaningful under graph capture, where
# the bench guarantees cache_params=None -> initial_state=None, cu_seqlens=None.
USE_PTO_KDA_GRAPH = False

__all__ = [
    'USE_PTO_KDA',
    'USE_PTO_KDA_GRAPH',
    'kda_chunk_wrapper',
    'kda_chunk_pypto',
]
