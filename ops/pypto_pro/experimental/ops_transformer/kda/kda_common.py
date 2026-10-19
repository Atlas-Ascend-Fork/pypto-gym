# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""KDA shared constants and utility functions.

Used by all 5 KDA kernel impl files and re-exported by kda_test_config.py.
"""

import os

import torch

C = 128
K = 128
V = 128
HC = C // 2
K_DIM = K
V_DIM = V
HALF_K = K // 2

_DEVICE_ID = os.environ.get('TILE_FWK_DEVICE_ID', '0')
DEVICE = f"npu:{_DEVICE_ID}"


def make_cu_seqlens_tensor(T, cu_seqlens, device):
    if cu_seqlens is None:
        return torch.tensor([0, T], dtype=torch.int32, device=device)
    return torch.tensor(cu_seqlens, dtype=torch.int32, device=device)


def build_chunk_tables(T, cu_seqlens, chunk_size, device):
    if cu_seqlens is None or len(cu_seqlens) == 2:
        cu_seqlens_t = make_cu_seqlens_tensor(T, cu_seqlens, device)
        num_chunks = (T + chunk_size - 1) // chunk_size
        chunk_tbase = torch.zeros(1, dtype=torch.int32, device=device)
        chunk_valid = torch.zeros(1, dtype=torch.int32, device=device)
    else:
        cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        tbase_list = []
        valid_list = []
        for i in range(len(cu_seqlens) - 1):
            seq_start = cu_seqlens[i]
            seq_end = cu_seqlens[i + 1]
            for j in range((seq_end - seq_start + chunk_size - 1) // chunk_size):
                tb = seq_start + j * chunk_size
                tbase_list.append(tb)
                valid_list.append(min(seq_end - tb, chunk_size))
        num_chunks = len(tbase_list)
        chunk_tbase = torch.tensor(tbase_list, dtype=torch.int32, device=device)
        chunk_valid = torch.tensor(valid_list, dtype=torch.int32, device=device)
    return cu_seqlens_t, chunk_tbase, chunk_valid, num_chunks


def alloc_chunk_h_workspaces(HV_dim, C, K_dim, V_dim, device):
    return (
        torch.empty(HV_dim * K_dim, V_dim, device=device, dtype=torch.float32),
        torch.empty(HV_dim * C, K_dim,     device=device, dtype=torch.float16),
        torch.empty(HV_dim * C, V_dim,     device=device, dtype=torch.float16),
        torch.empty(HV_dim * K_dim, V_dim, device=device, dtype=torch.float16),
        torch.empty(HV_dim * C, K_dim,     device=device, dtype=torch.float16),
    )
