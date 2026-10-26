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
- KDA (Kimi Delta Attention) fused prefill: uses `kda_fused_prefill` for the
  PyPTO prefill path.

融合范围：
- kda_fused_prefill covers the whole per-layer KDA prefill attention subgraph;
  see kda/README.md for details.

应用场景：
- Kimi-Linear-48B-A3B-Instruct 模型的 prefill 推理加速，通过 sys.modules 注入
  kimi_linear_48b_a3b_pto_kernels 模块。

注入方式：
- sys.modules["kimi_linear_48b_a3b_pto_kernels"] 提供 USE_PTO_KDA 开关和 kda_fused_prefill 函数。

ACLGraph / torch.compile：
- Graph capture uses the same `kda_fused_prefill` path; there is no separate
  graph-only operator.
"""

from .kda.kda_fused_prefill_impl import kda_fused_prefill

# NOTE(multi-NPU coverage): PyPTO binds a JIT kernel to ONE NPU per process. With
# the wrappers enabled on a model sharded across NPUs (single-process device_map),
# only the bound NPU's KDA layers run on PyPTO (e.g. ~6 of 20 with 4-way sharding);
# the rest fall back to torch (the wrapper emits a one-time RuntimeWarning so this
# is not silent). Note (future work): for full 20/20 coverage run one process per
# NPU (pipeline / tensor parallel). See kda/README.md.
USE_PTO_KDA = False

__all__ = [
    'USE_PTO_KDA',
    'kda_fused_prefill',
]
