# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared scaffold for the KDA op tests.

Holds the common imports, the ``_HAS_NPU`` torch_npu detection, the logging
config, the sys.path inserts for the impl + golden dirs, and the golden op
import. Input builders live in each test (they differ per operator ABI).
"""

__all__ = ["_HAS_NPU", "torch_npu", "_naive_recurrent_kda",
           "HEAD_DIM", "RTOL", "ATOL"]

import logging
import sys
from pathlib import Path

# Single source of truth for the op-test envelope.
HEAD_DIM = 128          # head_k_dim == head_v_dim (real Kimi config)
RTOL = ATOL = 6e-3      # tolerance vs the fp32 naive-recurrent golden

logging.basicConfig(level=logging.INFO, format="%(message)s")

try:
    import torch_npu
    _HAS_NPU = True
except ImportError:
    torch_npu = None
    logging.info("torch_npu not available; this test only runs on Ascend NPU.")
    _HAS_NPU = False

# impl lives under ops/.../kimi_linear_48b_a3b/kda; golden is same-dir.
_IMPL = (Path(__file__).resolve().parents[3]
         / "ops" / "pypto_tensor" / "kimi_linear_48b_a3b" / "kda")
sys.path.insert(0, str(_IMPL))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kda_golden import _naive_recurrent_kda  # noqa: E402

