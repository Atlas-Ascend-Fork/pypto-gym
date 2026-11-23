# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public Torch frontend for the CompressorV2 operator."""
import torch


def compressor_v2(x, wkv, wgate, state_cache, cmp_ratio=4, *,
                  state_block_table=None, cu_seqlens=None, seqused=None,
                  start_pos=None):
    """Return the compressed KV and update ``state_cache`` in place.

    ``state_block_table`` is required. The internal stage/vector routing is
    planned from the platform core counts and the input geometry; unused
    output rows are unspecified.
    """
    if state_block_table is None:
        raise ValueError("state_block_table is required")
    return torch.ops.pypto_gym.compressor_v2(
        x, wkv, wgate, state_cache, state_block_table, cu_seqlens, seqused,
        start_pos, cmp_ratio)
