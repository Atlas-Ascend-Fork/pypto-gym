# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto
import numpy as np
import torch


HEADS_PER_GROUP = 4
Q_TILE_SIZE = 256
K_TILE_SIZE = 256
MASK_TEMPLATE_SIZE = 1024
BLOCK_SIZE = 128

NEG_INF_SURROGATE = -1e30
MASK_PENALTY = 40000.0


# ── Single JIT Kernel: Attention ────────────────────────────────────────────
@pypto.frontend.jit(
    runtime_options={
        "device_sched_mode": 1,
        "stitch_function_max_num": 1024,
    },
    pass_options={
        "cube_l1_reuse_setting": {-1: 16},
        "vec_nbuffer_setting": {-2: 1, -1: 16},
        "cube_nbuffer_setting": {-1: 16},
    },
)
def mtgr_ragged_segment_attention(
    q: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    mask0: pypto.Tensor([MASK_TEMPLATE_SIZE, MASK_TEMPLATE_SIZE], pypto.DT_FP32),
    mask1: pypto.Tensor([MASK_TEMPLATE_SIZE, MASK_TEMPLATE_SIZE], pypto.DT_FP32),
    rules: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    segment_starts: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    l_output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    m_output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    cu_seqlens_q: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    cu_seqlens_k: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    batch_size: int,
    seq_num: int,
    num_heads: int,
    head_dim: int,
):
    group_head_dim = head_dim * HEADS_PER_GROUP
    hidden_dim = num_heads * head_dim
    scale = 1.0 / (head_dim ** 0.5)
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128, 256], [128, 256], [128, 128])
    pypto.set_vec_tile_shapes(64, 256)

    for b_idx in pypto.loop(batch_size, name="batch_loop"):
        for h_group_idx in pypto.loop(num_heads // HEADS_PER_GROUP, name="head_loop"):
            h_offset = h_group_idx * group_head_dim

            for seg_q_idx in pypto.loop(seq_num, name="seg_q_loop"):
                q_seg_start = segment_starts[b_idx, seg_q_idx]
                q_seg_end = segment_starts[b_idx, seg_q_idx + 1]
                q_seg_len = q_seg_end - q_seg_start
                q_tile_count = (q_seg_len + Q_TILE_SIZE - 1) // Q_TILE_SIZE
                rule_val = rules[seg_q_idx]

                for q_tile_idx in pypto.loop(q_tile_count, name="q_tile_loop"):
                    q_local = q_tile_idx * Q_TILE_SIZE
                    q_tile_len = pypto.min(q_local + Q_TILE_SIZE, q_seg_len) - q_local
                    q_global_off = cu_seqlens_q[b_idx] + q_seg_start + q_local

                    # ── State tensors inside q_tile_loop, NO pypto.full ──
                    # is_loop_begin handles initialization on first k_tile
                    oi_0 = pypto.tensor([Q_TILE_SIZE, head_dim], pypto.DT_FP32, "oi_state_0")
                    oi_1 = pypto.tensor([Q_TILE_SIZE, head_dim], pypto.DT_FP32, "oi_state_1")
                    oi_2 = pypto.tensor([Q_TILE_SIZE, head_dim], pypto.DT_FP32, "oi_state_2")
                    oi_3 = pypto.tensor([Q_TILE_SIZE, head_dim], pypto.DT_FP32, "oi_state_3")
                    li_0 = pypto.tensor([Q_TILE_SIZE, 1], pypto.DT_FP32, "li_state_0")
                    li_1 = pypto.tensor([Q_TILE_SIZE, 1], pypto.DT_FP32, "li_state_1")
                    li_2 = pypto.tensor([Q_TILE_SIZE, 1], pypto.DT_FP32, "li_state_2")
                    li_3 = pypto.tensor([Q_TILE_SIZE, 1], pypto.DT_FP32, "li_state_3")
                    mi_0 = pypto.tensor([Q_TILE_SIZE, 1], pypto.DT_FP32, "mi_state_0")
                    mi_1 = pypto.tensor([Q_TILE_SIZE, 1], pypto.DT_FP32, "mi_state_1")
                    mi_2 = pypto.tensor([Q_TILE_SIZE, 1], pypto.DT_FP32, "mi_state_2")
                    mi_3 = pypto.tensor([Q_TILE_SIZE, 1], pypto.DT_FP32, "mi_state_3")

                    # ── Off-diagonal: merged pre-diag K segments ──
                    k_pre_diag_start = segment_starts[b_idx, 0]
                    k_pre_diag_end = segment_starts[b_idx, seg_q_idx]
                    k_pre_diag_len = k_pre_diag_end - k_pre_diag_start
                    k_pre_diag_tiles = (k_pre_diag_len + K_TILE_SIZE - 1) // K_TILE_SIZE

                    for k_tile_idx in pypto.loop(k_pre_diag_tiles, name="k_tile_loop_off_merged", \
                                                unroll_list=[4, 2, 1]):
                        k_local = k_tile_idx * K_TILE_SIZE
                        k_tile_len = pypto.min(k_local + K_TILE_SIZE, k_pre_diag_len) - k_local
                        k_global_off = cu_seqlens_k[b_idx] + k_pre_diag_start + k_local

                        q_tile_w = pypto.view(q, [Q_TILE_SIZE, group_head_dim], [q_global_off, h_offset], \
                                                valid_shape=[q_tile_len, group_head_dim])
                        k_tile_w = pypto.view(k, [K_TILE_SIZE, group_head_dim], [k_global_off, h_offset], \
                                                valid_shape=[k_tile_len, group_head_dim])
                        v_tile_w = pypto.view(v, [K_TILE_SIZE, group_head_dim], [k_global_off, h_offset], \
                                                valid_shape=[k_tile_len, group_head_dim])

                        for h_inner_idx in range(HEADS_PER_GROUP):
                            q_sub = pypto.view(q_tile_w, [Q_TILE_SIZE, head_dim], [0, h_inner_idx * head_dim], \
                                                valid_shape=[q_tile_len, head_dim])
                            k_sub = pypto.view(k_tile_w, [K_TILE_SIZE, head_dim], [0, h_inner_idx * head_dim], \
                                                valid_shape=[k_tile_len, head_dim])
                            v_sub = pypto.view(v_tile_w, [K_TILE_SIZE, head_dim], [0, h_inner_idx * head_dim], \
                                                valid_shape=[k_tile_len, head_dim])

                            if h_inner_idx == 0:
                                oi_cur = oi_0
                                li_cur = li_0
                                mi_cur = mi_0
                            elif h_inner_idx == 1:
                                oi_cur = oi_1
                                li_cur = li_1
                                mi_cur = mi_1
                            elif h_inner_idx == 2:
                                oi_cur = oi_2
                                li_cur = li_2
                                mi_cur = mi_2
                            else:
                                oi_cur = oi_3
                                li_cur = li_3
                                mi_cur = mi_3

                            pypto.set_cube_tile_shapes([128, 256], [64, 64], [128, 256])
                            pypto.set_vec_tile_shapes(64, 256)
                            scores = pypto.matmul(q_sub, k_sub, out_dtype=pypto.DT_FP32, b_trans=True)
                            pypto.set_pass_options(sg_set_scope=(1, False, True))
                            scores_used = pypto.mul(scores, scale)

                            mij = pypto.amax(scores_used, dim=-1, keepdim=True)
                            s_shifted = pypto.sub(scores_used, mij)
                            pij = pypto.exp(s_shifted)
                            lij = pypto.sum(pij, dim=-1, keepdim=True)
                            pij_bf16 = pypto.cast(pij, pypto.DT_BF16)
                            pypto.set_pass_options(sg_set_scope=-1)

                            pypto.set_cube_tile_shapes([128, 256], [128, 256], [64, 64])
                            pypto.set_vec_tile_shapes(128, 64)
                            oij_h = pypto.matmul(pij_bf16, v_sub, out_dtype=pypto.DT_FP32)

                            pypto.set_pass_options(sg_set_scope=(2, False, True))
                            if pypto.is_loop_begin(k_tile_idx):
                                oi_cur[:] = oij_h
                                li_cur[:] = lij
                                mi_cur[:] = mij
                            else:
                                mi_new = pypto.maximum(mi_cur, mij)
                                t2 = pypto.exp(pypto.sub(mi_cur, mi_new))
                                t4 = pypto.exp(pypto.sub(mij, mi_new))
                                li_cur[:] = pypto.add(pypto.mul(t2, li_cur), pypto.mul(t4, lij))
                                oi_cur[:] = pypto.add(pypto.mul(oi_cur, t2), pypto.mul(oij_h, t4))
                                mi_cur[:] = mi_new
                            pypto.set_pass_options(sg_set_scope=-1)

                    # ── Diagonal segment ──
                    k_seg_start = segment_starts[b_idx, seg_q_idx]
                    k_seg_end = segment_starts[b_idx, seg_q_idx + 1]
                    k_seg_len = k_seg_end - k_seg_start
                    k_tile_count_in_seg = (k_seg_len + K_TILE_SIZE - 1) // K_TILE_SIZE

                    if rule_val == 1:
                        # ── Full-visibility: no mask ──
                        for k_tile_idx in pypto.loop(k_tile_count_in_seg, name="k_tile_loop_full", \
                                                    unroll_list=[4, 2, 1]):
                            k_local = k_tile_idx * K_TILE_SIZE
                            k_tile_len = pypto.min(k_local + K_TILE_SIZE, k_seg_len) - k_local
                            k_global_off = cu_seqlens_k[b_idx] + k_seg_start + k_local

                            q_tile_w = pypto.view(q, [Q_TILE_SIZE, group_head_dim], [q_global_off, h_offset], \
                                                    valid_shape=[q_tile_len, group_head_dim])
                            k_tile_w = pypto.view(k, [K_TILE_SIZE, group_head_dim], [k_global_off, h_offset], \
                                                    valid_shape=[k_tile_len, group_head_dim])
                            v_tile_w = pypto.view(v, [K_TILE_SIZE, group_head_dim], [k_global_off, h_offset], \
                                                    valid_shape=[k_tile_len, group_head_dim])

                            for h_inner_idx in range(HEADS_PER_GROUP):
                                q_sub = pypto.view(q_tile_w, [Q_TILE_SIZE, head_dim], [0, \
                                                h_inner_idx * head_dim], valid_shape=[q_tile_len, head_dim])
                                k_sub = pypto.view(k_tile_w, [K_TILE_SIZE, head_dim], [0, \
                                                h_inner_idx * head_dim], valid_shape=[k_tile_len, head_dim])
                                v_sub = pypto.view(v_tile_w, [K_TILE_SIZE, head_dim], [0, \
                                                h_inner_idx * head_dim],  valid_shape=[k_tile_len, head_dim])

                                if h_inner_idx == 0:
                                    oi_cur = oi_0
                                    li_cur = li_0
                                    mi_cur = mi_0
                                elif h_inner_idx == 1:
                                    oi_cur = oi_1
                                    li_cur = li_1
                                    mi_cur = mi_1
                                elif h_inner_idx == 2:
                                    oi_cur = oi_2
                                    li_cur = li_2
                                    mi_cur = mi_2
                                else:
                                    oi_cur = oi_3
                                    li_cur = li_3
                                    mi_cur = mi_3

                                pypto.set_cube_tile_shapes([128, 256], [64, 64], [128, 256])
                                pypto.set_vec_tile_shapes(64, 256)
                                scores = pypto.matmul(q_sub, k_sub, out_dtype=pypto.DT_FP32, b_trans=True)
                                pypto.set_pass_options(sg_set_scope=(1, False, True))
                                scores_used = pypto.mul(scores, scale)

                                mij = pypto.amax(scores_used, dim=-1, keepdim=True)
                                s_shifted = pypto.sub(scores_used, mij)
                                pij = pypto.exp(s_shifted)
                                lij = pypto.sum(pij, dim=-1, keepdim=True)
                                pij_bf16 = pypto.cast(pij, pypto.DT_BF16)
                                pypto.set_pass_options(sg_set_scope=-1)

                                pypto.set_cube_tile_shapes([128, 256], [128, 256], [64, 64])
                                pypto.set_vec_tile_shapes(128, 64)
                                oij_h = pypto.matmul(pij_bf16, v_sub, out_dtype=pypto.DT_FP32)

                                pypto.set_pass_options(sg_set_scope=(2, False, True))
                                mi_new = mij
                                li_new = lij
                                oi_new = oij_h
                                if seg_q_idx == 0:
                                    if pypto.is_loop_begin(k_tile_idx):
                                        if pypto.is_loop_end(k_tile_idx):
                                            oi_view = pypto.view(oi_new, [Q_TILE_SIZE, head_dim], [0, 0], \
                                                                valid_shape=[q_tile_len, head_dim])
                                            li_view = pypto.view(li_new, [Q_TILE_SIZE, 1], [0, 0], \
                                                                valid_shape=[q_tile_len, 1])
                                            mi_view = pypto.view(mi_new, [Q_TILE_SIZE, 1], [0, 0], \
                                                                valid_shape=[q_tile_len, 1])
                                            out_fp32 = pypto.div(oi_view, li_view)
                                            out_bf16 = pypto.cast(out_fp32, pypto.DT_BF16)
                                            pypto.assemble(out_bf16, [q_global_off, \
                                                            h_offset + h_inner_idx * head_dim], output)
                                            pypto.assemble(li_view, [q_global_off, 0], l_output)
                                            pypto.assemble(mi_view, [q_global_off, 0], m_output)
                                        else:
                                            oi_cur[:] = oi_new
                                            li_cur[:] = li_new
                                            mi_cur[:] = mi_new
                                    else:
                                        mi_new = pypto.maximum(mi_cur, mij)
                                        t2 = pypto.exp(pypto.sub(mi_cur, mi_new))
                                        t4 = pypto.exp(pypto.sub(mij, mi_new))
                                        li_new = pypto.add(pypto.mul(t2, li_cur), pypto.mul(t4, lij))
                                        oi_new = pypto.add(pypto.mul(oi_cur, t2), pypto.mul(oij_h, t4))
                                        if pypto.is_loop_end(k_tile_idx):
                                            oi_view = pypto.view(oi_new, [Q_TILE_SIZE, head_dim], [0, 0], \
                                                                valid_shape=[q_tile_len, head_dim])
                                            li_view = pypto.view(li_new, [Q_TILE_SIZE, 1], [0, 0], \
                                                                valid_shape=[q_tile_len, 1])
                                            mi_view = pypto.view(mi_new, [Q_TILE_SIZE, 1], [0, 0], \
                                                                valid_shape=[q_tile_len, 1])
                                            out_fp32 = pypto.div(oi_view, li_view)
                                            out_bf16 = pypto.cast(out_fp32, pypto.DT_BF16)
                                            pypto.assemble(out_bf16, [q_global_off, \
                                                            h_offset + h_inner_idx * head_dim], output)
                                            pypto.assemble(li_view, [q_global_off, 0], l_output)
                                            pypto.assemble(mi_view, [q_global_off, 0], m_output)
                                        else:
                                            oi_cur[:] = oi_new
                                            li_cur[:] = li_new
                                            mi_cur[:] = mi_new
                                else:
                                    mi_new = pypto.maximum(mi_cur, mij)
                                    t2 = pypto.exp(pypto.sub(mi_cur, mi_new))
                                    t4 = pypto.exp(pypto.sub(mij, mi_new))
                                    li_new = pypto.add(pypto.mul(t2, li_cur), pypto.mul(t4, lij))
                                    oi_new = pypto.add(pypto.mul(oi_cur, t2), pypto.mul(oij_h, t4))
                                    if pypto.is_loop_end(k_tile_idx):
                                        oi_view = pypto.view(oi_new, [Q_TILE_SIZE, head_dim], [0, 0], \
                                                            valid_shape=[q_tile_len, head_dim])
                                        li_view = pypto.view(li_new, [Q_TILE_SIZE, 1], [0, 0], \
                                                            valid_shape=[q_tile_len, 1])
                                        mi_view = pypto.view(mi_new, [Q_TILE_SIZE, 1], [0, 0], \
                                                            valid_shape=[q_tile_len, 1])
                                        out_fp32 = pypto.div(oi_view, li_view)
                                        out_bf16 = pypto.cast(out_fp32, pypto.DT_BF16)
                                        pypto.assemble(out_bf16, [q_global_off, \
                                                        h_offset + h_inner_idx * head_dim], output)
                                        pypto.assemble(li_view, [q_global_off, 0], l_output)
                                        pypto.assemble(mi_view, [q_global_off, 0], m_output)
                                    else:
                                        oi_cur[:] = oi_new
                                        li_cur[:] = li_new
                                        mi_cur[:] = mi_new
                                pypto.set_pass_options(sg_set_scope=-1)
                    else:
                        # ── Diagonal with mask: below-diag loop + on-diag inline ──
                        if rule_val == 0:
                            for k_tile_idx in pypto.loop(q_tile_idx, name="k_tile_loop_diag_below", unroll_list=[1]):
                                k_local = k_tile_idx * K_TILE_SIZE
                                k_tile_len = pypto.min(k_local + K_TILE_SIZE, k_seg_len) - k_local
                                k_global_off = cu_seqlens_k[b_idx] + k_seg_start + k_local

                                q_tile_w = pypto.view(q, [Q_TILE_SIZE, group_head_dim], [q_global_off, h_offset], \
                                                        valid_shape=[q_tile_len, group_head_dim])
                                k_tile_w = pypto.view(k, [K_TILE_SIZE, group_head_dim], [k_global_off, h_offset], \
                                                        valid_shape=[k_tile_len, group_head_dim])
                                v_tile_w = pypto.view(v, [K_TILE_SIZE, group_head_dim], [k_global_off, h_offset], \
                                                        valid_shape=[k_tile_len, group_head_dim])

                                for h_inner_idx in range(HEADS_PER_GROUP):
                                    q_sub = pypto.view(q_tile_w, [Q_TILE_SIZE, head_dim], [0, h_inner_idx * head_dim], \
                                                        valid_shape=[q_tile_len, head_dim])
                                    k_sub = pypto.view(k_tile_w, [K_TILE_SIZE, head_dim], [0, h_inner_idx * head_dim], \
                                                        valid_shape=[k_tile_len, head_dim])
                                    v_sub = pypto.view(v_tile_w, [K_TILE_SIZE, head_dim], [0, h_inner_idx * head_dim], \
                                                        valid_shape=[k_tile_len, head_dim])

                                    if h_inner_idx == 0:
                                        oi_cur = oi_0
                                        li_cur = li_0
                                        mi_cur = mi_0
                                    elif h_inner_idx == 1:
                                        oi_cur = oi_1
                                        li_cur = li_1
                                        mi_cur = mi_1
                                    elif h_inner_idx == 2:
                                        oi_cur = oi_2
                                        li_cur = li_2
                                        mi_cur = mi_2
                                    else:
                                        oi_cur = oi_3
                                        li_cur = li_3
                                        mi_cur = mi_3

                                    pypto.set_cube_tile_shapes([128, 256], [64, 64], [128, 256])
                                    pypto.set_vec_tile_shapes(64, 256)
                                    scores = pypto.matmul(q_sub, k_sub, out_dtype=pypto.DT_FP32, b_trans=True)
                                    pypto.set_pass_options(sg_set_scope=(1, False, True))
                                    scores_used = pypto.mul(scores, scale)

                                    mij = pypto.amax(scores_used, dim=-1, keepdim=True)
                                    s_shifted = pypto.sub(scores_used, mij)
                                    pij = pypto.exp(s_shifted)
                                    lij = pypto.sum(pij, dim=-1, keepdim=True)
                                    pij_bf16 = pypto.cast(pij, pypto.DT_BF16)
                                    pypto.set_pass_options(sg_set_scope=-1)

                                    pypto.set_cube_tile_shapes([128, 256], [128, 256], [64, 64])
                                    pypto.set_vec_tile_shapes(128, 64)
                                    oij_h = pypto.matmul(pij_bf16, v_sub, out_dtype=pypto.DT_FP32)

                                    pypto.set_pass_options(sg_set_scope=(2, False, True))
                                    if seg_q_idx == 0:
                                        if pypto.is_loop_begin(k_tile_idx):
                                            oi_cur[:] = oij_h
                                            li_cur[:] = lij
                                            mi_cur[:] = mij
                                        else:
                                            mi_new = pypto.maximum(mi_cur, mij)
                                            t2 = pypto.exp(pypto.sub(mi_cur, mi_new))
                                            t4 = pypto.exp(pypto.sub(mij, mi_new))
                                            li_cur[:] = pypto.add(pypto.mul(t2, li_cur), pypto.mul(t4, lij))
                                            oi_cur[:] = pypto.add(pypto.mul(oi_cur, t2), pypto.mul(oij_h, t4))
                                            mi_cur[:] = mi_new
                                    else:
                                        mi_new = pypto.maximum(mi_cur, mij)
                                        t2 = pypto.exp(pypto.sub(mi_cur, mi_new))
                                        t4 = pypto.exp(pypto.sub(mij, mi_new))
                                        li_cur[:] = pypto.add(pypto.mul(t2, li_cur), pypto.mul(t4, lij))
                                        oi_cur[:] = pypto.add(pypto.mul(oi_cur, t2), pypto.mul(oij_h, t4))
                                        mi_cur[:] = mi_new
                                    pypto.set_pass_options(sg_set_scope=-1)

                        # On-diagonal tile: always execute (rule_val==0 or rule_val==2)
                        k_local_diag = q_tile_idx * K_TILE_SIZE
                        k_tile_len_diag = pypto.min(k_local_diag + K_TILE_SIZE, k_seg_len) - k_local_diag
                        k_global_off_diag = cu_seqlens_k[b_idx] + k_seg_start + k_local_diag

                        q_tile_w_diag = pypto.view(q, [Q_TILE_SIZE, group_head_dim], [q_global_off, h_offset], \
                                                    valid_shape=[q_tile_len, group_head_dim])
                        k_tile_w_diag = pypto.view(k, [K_TILE_SIZE, group_head_dim], [k_global_off_diag, h_offset], \
                                                    valid_shape=[k_tile_len_diag, group_head_dim])
                        v_tile_w_diag = pypto.view(v, [K_TILE_SIZE, group_head_dim], [k_global_off_diag, h_offset], \
                                                    valid_shape=[k_tile_len_diag, group_head_dim])

                        for h_inner_idx in range(HEADS_PER_GROUP):
                            q_sub_diag = pypto.view(q_tile_w_diag, [Q_TILE_SIZE, head_dim], [0, h_inner_idx * head_dim], \
                                                    valid_shape=[q_tile_len, head_dim])
                            k_sub_diag = pypto.view(k_tile_w_diag, [K_TILE_SIZE, head_dim], [0, h_inner_idx * head_dim], \
                                                    valid_shape=[k_tile_len_diag, head_dim])
                            v_sub_diag = pypto.view(v_tile_w_diag, [K_TILE_SIZE, head_dim], [0, h_inner_idx * head_dim], \
                                                    valid_shape=[k_tile_len_diag, head_dim])

                            if h_inner_idx == 0:
                                oi_cur = oi_0
                                li_cur = li_0
                                mi_cur = mi_0
                            elif h_inner_idx == 1:
                                oi_cur = oi_1
                                li_cur = li_1
                                mi_cur = mi_1
                            elif h_inner_idx == 2:
                                oi_cur = oi_2
                                li_cur = li_2
                                mi_cur = mi_2
                            else:
                                oi_cur = oi_3
                                li_cur = li_3
                                mi_cur = mi_3

                            pypto.set_cube_tile_shapes([128, 256], [64, 64], [128, 256])
                            pypto.set_vec_tile_shapes(64, 256)
                            scores = pypto.matmul(q_sub_diag, k_sub_diag, out_dtype=pypto.DT_FP32, b_trans=True)
                            scores_scaled = pypto.mul(scores, scale)

                            if rule_val == 0:
                                mask_tile = pypto.view(mask0, [Q_TILE_SIZE, K_TILE_SIZE], [0, 0], \
                                                        valid_shape=[q_tile_len, k_tile_len_diag])
                            else:
                                mask_tile = pypto.view(mask1, [Q_TILE_SIZE, K_TILE_SIZE], [0, 0], \
                                                        valid_shape=[q_tile_len, k_tile_len_diag])
                            mask_penalty = pypto.mul(mask_tile, MASK_PENALTY)
                            scores_used = pypto.sub(scores_scaled, mask_penalty)

                            pypto.set_pass_options(sg_set_scope=(1, False, True))
                            mij = pypto.amax(scores_used, dim=-1, keepdim=True)
                            s_shifted = pypto.sub(scores_used, mij)
                            pij = pypto.exp(s_shifted)
                            lij = pypto.sum(pij, dim=-1, keepdim=True)
                            pij_bf16 = pypto.cast(pij, pypto.DT_BF16)
                            pypto.set_pass_options(sg_set_scope=-1)

                            pypto.set_cube_tile_shapes([128, 256], [128, 256], [64, 64])
                            pypto.set_vec_tile_shapes(128, 64)
                            oij_h = pypto.matmul(pij_bf16, v_sub_diag, out_dtype=pypto.DT_FP32)

                            pypto.set_pass_options(sg_set_scope=(2, False, True))
                            mi_new = mij
                            li_new = lij
                            oi_new = oij_h
                            if seg_q_idx == 0:
                                if rule_val == 0:
                                    if q_tile_idx > 0:
                                        mi_new = pypto.maximum(mi_cur, mij)
                                        t2 = pypto.exp(pypto.sub(mi_cur, mi_new))
                                        t4 = pypto.exp(pypto.sub(mij, mi_new))
                                        li_new = pypto.add(pypto.mul(t2, li_cur), pypto.mul(t4, lij))
                                        oi_new = pypto.add(pypto.mul(oi_cur, t2), pypto.mul(oij_h, t4))
                                elif rule_val == 2:
                                    pass
                            else:
                                mi_new = pypto.maximum(mi_cur, mij)
                                t2 = pypto.exp(pypto.sub(mi_cur, mi_new))
                                t4 = pypto.exp(pypto.sub(mij, mi_new))
                                li_new = pypto.add(pypto.mul(t2, li_cur), pypto.mul(t4, lij))
                                oi_new = pypto.add(pypto.mul(oi_cur, t2), pypto.mul(oij_h, t4))
                            oi_view = pypto.view(oi_new, [Q_TILE_SIZE, head_dim], [0, 0], \
                                                valid_shape=[q_tile_len, head_dim])
                            li_view = pypto.view(li_new, [Q_TILE_SIZE, 1], [0, 0], valid_shape=[q_tile_len, 1])
                            mi_view = pypto.view(mi_new, [Q_TILE_SIZE, 1], [0, 0], valid_shape=[q_tile_len, 1])
                            out_fp32 = pypto.div(oi_view, li_view)
                            out_bf16 = pypto.cast(out_fp32, pypto.DT_BF16)
                            pypto.assemble(out_bf16, [q_global_off, h_offset + h_inner_idx * head_dim], output)
                            pypto.assemble(li_view, [q_global_off, 0], l_output)
                            pypto.assemble(mi_view, [q_global_off, 0], m_output)
                            pypto.set_pass_options(sg_set_scope=-1)

                    # ── NO post-loop assemble ──
