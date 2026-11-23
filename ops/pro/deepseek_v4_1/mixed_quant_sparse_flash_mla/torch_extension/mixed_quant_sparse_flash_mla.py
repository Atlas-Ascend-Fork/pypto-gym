# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public Torch frontend for the MixedQuantSparseFlashMla operator."""
import torch


def mixed_quant_sparse_flash_mla(q, ori_kv, ori_sparse_indices,
                                 ori_block_table, cu_seqlens_q, sinks,
                                 metadata=None, quant_mode=1,
                                 softmax_scale=None, cmp_kv=None,
                                 cmp_sparse_indices=None, cmp_block_table=None,
                                 ori_topk_length=None, cmp_topk_length=None,
                                 return_softmax_lse=False, out=None, lse=None,
                                 block_dim=None, fd_enable=None,
                                 seqused_q=None, seqused_ori_kv=None,
                                 seqused_cmp_kv=None,
                                 layout_q="TND", layout_kv="PA_BBND"):
    """Mixed-quantization sparse flash MLA forward.

    Only ``quant_mode=1`` and layouts ``layout_q='TND'`` /
    ``layout_kv='PA_BBND'`` are supported by the delivered kernel;
    ``metadata`` is required. ``cu_seqlens_q`` and the optional
    ``*_topk_length`` tables are consumed as host int64 values. Returns the
    dict ``{"out": out, "lse": softmax_lse}``; outputs are allocated
    internally when ``out``/``lse`` are not supplied. The reserved
    ``block_dim``/``fd_enable``/``seqused_*`` arguments are accepted and
    ignored.
    """
    if quant_mode != 1:
        raise ValueError("only quant_mode=1 is supported")
    if metadata is None:
        raise ValueError("metadata is required")
    scale = 0.0 if softmax_scale is None else float(softmax_scale)
    out_t, lse_t = torch.ops.pypto_gym.mixed_quant_sparse_flash_mla(
        q, ori_kv, ori_sparse_indices, ori_block_table, cu_seqlens_q, sinks,
        metadata, scale, layout_q, layout_kv, cmp_kv, cmp_sparse_indices,
        cmp_block_table, ori_topk_length, cmp_topk_length, out, lse)
    if not return_softmax_lse:
        return {"out": out_t, "lse": lse_t}
    return {"out": out_t, "lse": lse_t}
