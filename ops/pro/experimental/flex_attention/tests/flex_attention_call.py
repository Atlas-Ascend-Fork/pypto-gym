#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""FlexAttention 用例共享调用 helper(非 golden; jit/aclnn 两种入口签名一致)."""

import torch


def flex_attention_from_info(flex_attention, q, k, v, seq_q_list, seq_kv_list,
                             attn_out, softmax_max, softmax_sum, mask_info,
                             layout="varlen"):
    """Unpack a mask_info dict and call the expanded flex_attention() interface."""
    device = q.device

    def _to_dev(x):
        return x.to(device) if x is not None else None

    if layout == "bshd":
        flex_attention(q, k, v, None, None,
                       _to_dev(mask_info["maskr"]),
                       _to_dev(mask_info.get("mask_start")),
                       _to_dev(mask_info.get("mask_len")),
                       _to_dev(mask_info["block_range"]),
                       _to_dev(mask_info["block_compute"]),
                       _to_dev(mask_info["block_mask"]),
                       attn_out, softmax_max, softmax_sum, layout="bshd")
        return

    actual_seq_q = torch.tensor(seq_q_list, dtype=torch.int64, device=device).cumsum(0)
    actual_seq_kv = torch.tensor(seq_kv_list, dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
    flex_attention(q, k, v, actual_seq_q, actual_seq_kv,
                   _to_dev(mask_info["maskr"]),
                   _to_dev(mask_info.get("mask_start")),
                   _to_dev(mask_info.get("mask_len")),
                   _to_dev(mask_info["block_range"]),
                   _to_dev(mask_info["block_compute"]),
                   _to_dev(mask_info["block_mask"]),
                   attn_out, softmax_max, softmax_sum, layout="varlen")
