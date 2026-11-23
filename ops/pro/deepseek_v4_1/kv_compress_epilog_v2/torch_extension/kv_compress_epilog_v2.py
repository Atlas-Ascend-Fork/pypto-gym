# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public Torch frontend for the KvCompressEpilogV2 operator."""
import torch


def kv_compress_epilog_v2(cache, x, slot_mapping, quant_group_size=32,
                          quant_mode=2, round_scale=True, x_scale=1.0,
                          block_stride=0):
    """Quantize BF16 rows and update the supplied cache in place.

    Writes the quantized ``x`` rows into ``cache`` at the slots named by
    ``slot_mapping`` (cast to int64 internally) and returns ``cache``
    unchanged. ``storage_bytes`` for the host tiling check is derived from
    the cache storage behind the (possibly strided) view.
    """
    return torch.ops.pypto_gym.kv_compress_epilog_v2(
        cache, x, slot_mapping, quant_group_size, quant_mode, round_scale,
        x_scale, block_stride)
