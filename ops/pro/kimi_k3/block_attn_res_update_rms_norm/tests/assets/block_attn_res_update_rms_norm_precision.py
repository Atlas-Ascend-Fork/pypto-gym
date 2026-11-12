# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical comparison criteria for block_attn_res_update_rms_norm.

Contract (frozen from the upstream reference operator):
  * relative tolerance rtol=1e-3, absolute atol=0
  * mismatch ratio (fraction of elements failing isclose) must be <= 1e-3
  * every element of candidate and reference must be finite
The ill-conditioned cancellation regression uses the standard "2x reference"
rule inside tests/test_block_attn_res_update_rms_norm.py; this module keeps the contract criteria unchanged.
"""

import numpy as np


def check_output(out, ref, rtol=1e-3, atol=0.0, max_mismatch_ratio=1e-3):
    """Reference check: isclose(rtol, atol) mismatch ratio <= max_mismatch_ratio; non-finite fails."""
    a = out.detach().cpu().float().numpy()
    g = ref.detach().cpu().float().numpy()
    if a.shape != g.shape:
        return False, "shape mismatch"
    finite = bool(np.isfinite(a).all() and np.isfinite(g).all())
    close = np.isclose(a, g, rtol=rtol, atol=atol, equal_nan=False)
    bad = int(np.count_nonzero(~close))
    ratio = bad / a.size if a.size else 0.0
    return (ratio <= max_mismatch_ratio) and finite, f"mismatch ratio {ratio:.3e} bad={bad} finite={finite}"
