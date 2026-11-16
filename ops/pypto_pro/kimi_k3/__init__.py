# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compatibility exports for the operator-local KDA package."""

from .flash_kda.op_kernel import (
    KDACPContext,
    chunk_kda,
    flash_kda_bwd_stage1_wrapper,
    flash_kda_bwd_stage2_wrapper,
    flash_kda_stage1_wrapper,
    flash_kda_stage2_wrapper,
)

__all__ = [
    'chunk_kda', 'KDACPContext',
    'flash_kda_stage1_wrapper', 'flash_kda_stage2_wrapper',
    'flash_kda_bwd_stage1_wrapper', 'flash_kda_bwd_stage2_wrapper',
]
