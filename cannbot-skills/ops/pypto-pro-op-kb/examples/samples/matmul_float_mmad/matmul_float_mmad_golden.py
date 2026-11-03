# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Golden for the matmul_float_mmad transfer (pure torch, no pypto).
#
# SAMPLE PROVENANCE -- describes THIS reference implementation only.
# Do not copy this header into generated code: a generated kernel inherits
# no validation from the sample it was modelled on.
# The oracle and simulator artifacts matched exactly; the reference performance target is 1,483 cycles.
#   RE-STAMP BASIS (this PR): the edit that moved this hash was **text only** -- a
#   docstring/comment change with no executable difference, proved by comparing the
#   parsed AST with docstrings stripped before and after. The recorded result above was
#   NOT re-measured on a board. So this stamp asserts "semantically the same code as the
#   one that produced that record", which is weaker than `--stamp-validation-hashes`'s
#   normal meaning ("this exact code was just re-validated"). Re-run on target before
#   relying on it as a fresh result.
# VALIDATED-CODE-SHA256: c4c115ec3421995800cb8e62e55da51165126e492400f8c157f90e49946038c2
import logging

import torch

LOGGER = logging.getLogger(__name__)


def matmul_float_mmad_golden(x, y):
    """z = x @ y.T with FP32 accumulation.

    x: [M, K] float32,  y: [N, K] float32  ->  z: [M, N] float32.
    Matches the reference implementation's `z_ref = x @ y.t()`.
    The `.float()` is load-bearing: the pypto twin accumulates in FP32 in L0C
    (constraints/precision.md), so the golden must accumulate in FP32 too.
    """
    return x.float() @ y.float().t()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)
    M, N, K = 32, 48, 16
    x = torch.randn(M, K, dtype=torch.float32)
    y = torch.randn(N, K, dtype=torch.float32)
    z = matmul_float_mmad_golden(x, y)
    LOGGER.info("golden z shape: %s", tuple(z.shape))
