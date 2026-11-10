# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public Torch frontend for the precompiled QuantSparseLightningIndexer operator."""
import torch


def quant_sparse_lightning_indexer(
        q, k, w, descale_q, candidate_block_indices, candidate_block_length,
        cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None,
        cmp_residual_k=None, block_table=None, output_idx_offset=None,
        metadata=None, *, descale_k=None, topk, candidate_block_size=8,
        quant_mode=1, max_seqlen_q=-1, mask_mode=0, cmp_ratio=1,
        layout_q="TND", layout_k="TND", return_value=False):
    """Return ``(indices, values)`` selected from packed MXFP4 candidate blocks.

    Runs the installed ACLNN binary on the current NPU stream; both outputs are
    allocated internally with shape ``(T1, 1, topk)`` (int32 / BF16). With
    ``return_value=False`` the scores are omitted and ``values`` is a
    zero-element view. ``metadata`` is the caller-generated int32[1024] LI/LD
    schedule. The operator provides forward computation only.
    """
    if type(topk) is not int or not 1 <= topk <= 8192:
        raise ValueError('topk must be an integer in [1,8192]')
    if type(candidate_block_size) is not int or candidate_block_size != 8:
        raise ValueError('candidate_block_size must be 8')
    if type(quant_mode) is not int or quant_mode != 1:
        raise ValueError('quant_sparse_lightning_indexer supports quant_mode=1 only')
    if type(max_seqlen_q) is not int or max_seqlen_q < -1:
        raise ValueError('max_seqlen_q must be -1 or a nonnegative integer')
    if type(mask_mode) is not int or mask_mode not in (0, 3):
        raise ValueError('mask_mode must be 0 or 3')
    if type(cmp_ratio) is not int or not 1 <= cmp_ratio <= 128:
        raise ValueError('cmp_ratio must be an integer in [1,128]')
    if layout_q != 'TND' or layout_k not in ('PA_BBND', 'TND'):
        raise ValueError('current QSLI layouts must be TND Q with PA_BBND or TND K')
    if metadata is None:
        raise ValueError('metadata must contain the current LI/LD schedule (int32[1024])')
    return torch.ops.pypto_gym.quant_sparse_lightning_indexer(
        q, k, w, descale_q, descale_k, candidate_block_indices,
        candidate_block_length, cu_seqlens_q, cu_seqlens_k, seqused_q,
        seqused_k, cmp_residual_k, block_table, output_idx_offset, metadata,
        topk=topk, candidate_block_size=candidate_block_size,
        quant_mode=quant_mode, max_seqlen_q=max_seqlen_q, mask_mode=mask_mode,
        cmp_ratio=cmp_ratio, layout_q=layout_q, layout_k=layout_k,
        return_value=bool(return_value))
