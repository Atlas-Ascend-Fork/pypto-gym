# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public Torch frontend for the precompiled MixedQuantSparseFlashMlaMetadata operator."""
import torch

_METADATA_ELEMENTS = 1024
_TILING_DEVICE_ELEMENTS = 16


def mixed_quant_sparse_flash_mla_metadata(
        cu_seqlens_q, ori_topk_length, cmp_topk_length=None, *,
        blocks=None, has_cmp=1, k_ori=0, k_cmp=0,
        bs_ori=0, bs_cmp=0, bt_ori=0, bt_cmp=0,
        ori_pages=0, cmp_pages=0, ori_kv_stride0=0, cmp_kv_stride0=0):
    """Return ``(metadata, tiling_device)`` produced on the current NPU stream.

    ``cu_seqlens_q`` is int32 [B+1]; ``ori_topk_length`` / ``cmp_topk_length``
    are int32 [T1, 1]. ``blocks`` defaults to the device AI-Cube core count.
    The schedule is the same one the attention operator's host tiling expects;
    ``k_ori`` / ``k_cmp`` clamp the per-row lengths (0 = keep the value).

    ``tiling_device`` (int32[16]) carries the device-side tiling parameters
    consumed by the main operator's kernel; the ``bs_*`` / ``bt_*`` /
    ``*_pages`` / ``*_stride0`` scalars describe the caller's KV/index shapes
    and are passed through unchanged (host-known shapes, no device readback).

    Both outputs stay on device; nothing is copied to the host.
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
    metadata, tiling_device = torch.ops.pypto_gym.mixed_quant_sparse_flash_mla_metadata(
        cu_seqlens_q.contiguous(),
        ori_topk_length.contiguous(),
        None if cmp_topk_length is None else cmp_topk_length.contiguous(),
        int(blocks), int(has_cmp), int(k_ori), int(k_cmp),
        int(bs_ori), int(bs_cmp), int(bt_ori), int(bt_cmp),
        int(ori_pages), int(cmp_pages), int(ori_kv_stride0), int(cmp_kv_stride0))
    return metadata, tiling_device
