# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public Torch frontend for the precompiled MixedQuantSparseFlashMlaMetadata operator."""
import torch

_METADATA_ELEMENTS = 1024


def mixed_quant_sparse_flash_mla_metadata(
        cu_seqlens_q, ori_topk_length, cmp_topk_length=None, *,
        blocks=None, has_cmp=1, k_ori=0, k_cmp=0):
    """Return the int32[1024] MQSMLA schedule on the current NPU stream.

    ``cu_seqlens_q`` is int32 [B+1]; ``ori_topk_length`` / ``cmp_topk_length``
    are int32 [T1, 1]. ``blocks`` defaults to the device AI-Cube core count.
    The schedule is the same one the attention operator's host tiling expects;
    ``k_ori`` / ``k_cmp`` clamp the per-row lengths (0 = keep the value).
    """
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_q.dim() != 1:
        raise ValueError('cu_seqlens_q must be a rank-1 int32 tensor')
    if ori_topk_length.dtype != torch.int32:
        raise ValueError('ori_topk_length must be an int32 tensor')
    if cmp_topk_length is not None and cmp_topk_length.dtype != torch.int32:
        raise ValueError('cmp_topk_length must be an int32 tensor or None')
    if blocks is None:
        blocks = torch.npu.get_device_properties(cu_seqlens_q.device).cube_core_num
    if not 0 < int(blocks) <= 36:
        raise ValueError('blocks must be in 1..36')
    return torch.ops.pypto_gym.mixed_quant_sparse_flash_mla_metadata(
        cu_seqlens_q.contiguous(),
        ori_topk_length.contiguous(),
        None if cmp_topk_length is None else cmp_topk_length.contiguous(),
        int(blocks), int(has_cmp), int(k_ori), int(k_cmp))
