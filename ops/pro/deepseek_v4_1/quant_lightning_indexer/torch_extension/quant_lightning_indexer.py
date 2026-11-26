# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch frontend for the installed QLI AICPU + AI Core binaries."""
import torch


def quant_lightning_indexer(
    q, k, w, descale_q, descale_k,
    cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None,
    cmp_residual_k=None, block_table=None, output_idx_offset=None, metadata=None,
    *, topk, quant_mode, max_seqlen_q=-1, mask_mode=0, cmp_ratio=1,
    layout_q="TND", layout_k="TND", return_value=False,
    candidate_topk_blocks=-1, candidate_block_size=-1,
):
    """Return token indices, optional scores, candidate blocks and lengths.

    The packaged AICPU planner consumes device sequence arrays directly. This
    frontend only queries static device capacity and forwards the public call.
    """
    properties = torch.npu.get_device_properties(q.device.index)
    return torch.ops.pypto_gym.quant_lightning_indexer(
        q, k, w, descale_q, descale_k, cu_seqlens_q, cu_seqlens_k, seqused_q,
        seqused_k, cmp_residual_k, block_table, output_idx_offset, metadata,
        topk, quant_mode, max_seqlen_q, mask_mode, cmp_ratio,
        candidate_topk_blocks, candidate_block_size, layout_q, layout_k, return_value,
        properties.cube_core_num, properties.vector_core_num)
