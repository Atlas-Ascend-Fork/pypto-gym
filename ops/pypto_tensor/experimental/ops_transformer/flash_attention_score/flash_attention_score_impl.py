# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto

BLOCK_Q = 320
BLOCK_KV = 320


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 1024,
    },
    pass_options={
        "cube_l1_reuse_setting": {-1: 8},
        "vec_nbuffer_setting": {-1: 8},
        "cube_nbuffer_setting": {-1: 8},
    }
)
def flash_attention_score_kernel_npu(
    query: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    key: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    value: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    atten_mask: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    pse: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    drop_mask: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    softmax_max: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    softmax_sum: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    pse_type: int,
    keep_prob: float,
    scale_value: float,
):
    B = query.shape[0]
    N = query.shape[1]
    Sq = query.shape[2]
    D = query.shape[3]
    N_kv = key.shape[1]
    Skv = key.shape[2]
    group = N // N_kv
    num_blocks_q = (Sq + BLOCK_Q - 1) // BLOCK_Q
    num_blocks_kv = (Skv + BLOCK_KV - 1) // BLOCK_KV

    total_q = B * N * Sq
    total_kv = B * N_kv * Skv
    total_pse = B * N * Sq

    query_2d = pypto.reshape(query, [total_q, D], inplace=True)
    key_2d = pypto.reshape(key, [total_kv, D], inplace=True)
    value_2d = pypto.reshape(value, [total_kv, D], inplace=True)
    pse_2d = pypto.reshape(pse, [total_pse, Skv], inplace=True)

    v1_tile = [64, 512]

    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128, 128], [128, 256], [128, 128])
    pypto.set_vec_tile_shapes(64, 256)

    for b_idx in pypto.loop(B, name="batch_idx"):
        for kv_h in pypto.loop(N_kv, name="kv_head_idx"):
            for grp in pypto.loop(group, name="group_idx"):
                n_idx = kv_h * group + grp

                for qb in pypto.loop(num_blocks_q, name="q_block_idx"):
                    q_start = qb * BLOCK_Q
                    q_tile_len = (Sq - q_start).min(BLOCK_Q)
                    q_row = b_idx * N * Sq + n_idx * Sq + q_start
                    q_tile_view = pypto.view(query_2d, [BLOCK_Q, D],
                                             [q_row, 0],
                                             valid_shape=[q_tile_len, D])

                    mi_update = pypto.tensor([BLOCK_Q, 1], pypto.DT_FP32, "mi")
                    li_update = pypto.tensor([BLOCK_Q, 1], pypto.DT_FP32, "li")
                    oi_update = pypto.tensor([BLOCK_Q, D], pypto.DT_FP32, "oi")

                    for kvb in pypto.loop(num_blocks_kv, name="kv_block_idx"):
                        kv_start = kvb * BLOCK_KV
                        k_tile_len = (Skv - kv_start).min(BLOCK_KV)

                        k_row = b_idx * N_kv * Skv + kv_h * Skv + kv_start
                        k_tile_view = pypto.view(key_2d, [BLOCK_KV, D],
                                                 [k_row, 0],
                                                 valid_shape=[k_tile_len, D])
                        v_row = b_idx * N_kv * Skv + kv_h * Skv + kv_start
                        v_tile_view = pypto.view(value_2d, [BLOCK_KV, D],
                                                 [v_row, 0],
                                                 valid_shape=[k_tile_len, D])

                        pse_row = b_idx * N * Sq + n_idx * Sq + q_start
                        pse_tile_view = pypto.view(pse_2d, [BLOCK_Q, BLOCK_KV],
                                                   [pse_row, kv_start],
                                                   valid_shape=[q_tile_len, k_tile_len])

                        pse_fp32 = pypto.cast(pse_tile_view, pypto.DT_FP32)

                        pypto.set_cube_tile_shapes([64, 512], [64, 64], [512, 512])
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])

                        scores = pypto.matmul(q_tile_view, k_tile_view,
                                              out_dtype=pypto.DT_FP32,
                                              b_trans=True)

                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])

                        if pse_type == 1:
                            scores_scaled = pypto.mul(
                                pypto.add(scores, pse_fp32), scale_value)
                        else:
                            scores_scaled = pypto.add(
                                pypto.mul(scores, scale_value), pse_fp32)

                        mask_tile = pypto.view(atten_mask, [BLOCK_Q, BLOCK_KV],
                                               [q_start, kv_start],
                                               valid_shape=[q_tile_len, k_tile_len])
                        mask_fp32 = pypto.cast(mask_tile, pypto.DT_FP32)
                        valid_mask = pypto.mul(pypto.add(mask_fp32, -1.0), -1.0)

                        drop_tile = pypto.view(drop_mask, [BLOCK_Q, BLOCK_KV],
                                               [q_start, kv_start],
                                               valid_shape=[q_tile_len, k_tile_len])
                        drop_fp32 = pypto.cast(drop_tile, pypto.DT_FP32)

                        m_ij = pypto.amax(scores_scaled, dim=-1, keepdim=True)
                        s_shifted = pypto.sub(scores_scaled, m_ij)
                        p_ij = pypto.exp(s_shifted)
                        p_ij = pypto.mul(p_ij, valid_mask)
                        p_ij = pypto.mul(p_ij, drop_fp32)
                        l_ij = pypto.sum(p_ij, dim=-1, keepdim=True)

                        pypto.set_cube_tile_shapes([128, 512], [256, 512], [64, 64])

                        p_ij_bf16 = pypto.cast(p_ij, pypto.DT_BF16)
                        o_ij = pypto.matmul(p_ij_bf16, v_tile_view,
                                            out_dtype=pypto.DT_FP32)

                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])

                        if pypto.is_loop_begin(kvb):
                            if pypto.is_loop_end(kvb):
                                pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                p_ij_norm = pypto.div(p_ij, l_ij,
                                    precision_type=pypto.PrecisionType.INTRINSIC)
                                p_ij_bf16_final = pypto.cast(p_ij_norm, pypto.DT_BF16)

                                pypto.set_cube_tile_shapes([128, 512], [256, 512], [64, 64])
                                o_final = pypto.matmul(p_ij_bf16_final, v_tile_view,
                                                       out_dtype=pypto.DT_BF16)

                                o_v = pypto.view(o_final, [BLOCK_Q, D], [0, 0],
                                                 valid_shape=[q_tile_len, D])
                                m_v = pypto.view(m_ij, [BLOCK_Q, 1], [0, 0],
                                                 valid_shape=[q_tile_len, 1])

                                out_row = b_idx * N * Sq + n_idx * Sq + q_start
                                pypto.assemble(o_v, [out_row, 0], output)
                                pypto.assemble(m_v, [out_row, 0], softmax_max)

                                if keep_prob < 1.0:
                                    l_out = pypto.mul(l_ij, 1.0 / keep_prob)
                                else:
                                    l_out = l_ij
                                l_v = pypto.view(l_out, [BLOCK_Q, 1], [0, 0],
                                                 valid_shape=[q_tile_len, 1])
                                pypto.assemble(l_v, [out_row, 0], softmax_sum)
                            else:
                                oi_update[:] = o_ij
                                li_update[:] = l_ij
                                mi_update[:] = m_ij
                        else:
                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])

                            mi = pypto.view(mi_update, [BLOCK_Q, 1], [0, 0],
                                            valid_shape=[q_tile_len, 1])
                            li = pypto.view(li_update, [BLOCK_Q, 1], [0, 0],
                                            valid_shape=[q_tile_len, 1])
                            oi = pypto.view(oi_update, [BLOCK_Q, D], [0, 0],
                                            valid_shape=[q_tile_len, D])

                            mi_new = pypto.maximum(mi, m_ij)
                            t1 = pypto.sub(mi, mi_new)
                            t2 = pypto.exp(t1)
                            t3 = pypto.sub(m_ij, mi_new)
                            t4 = pypto.exp(t3)

                            li_new = pypto.add(pypto.mul(t2, li),
                                               pypto.mul(t4, l_ij))
                            oi_tmp = pypto.add(pypto.mul(oi, t2),
                                               pypto.mul(o_ij, t4))

                            if pypto.is_loop_end(kvb):
                                out_fp32 = pypto.div(oi_tmp, li_new,
                                    precision_type=pypto.PrecisionType.INTRINSIC)
                                out_bf16 = pypto.cast(out_fp32, pypto.DT_BF16)

                                o_v = pypto.view(out_bf16, [BLOCK_Q, D], [0, 0],
                                                 valid_shape=[q_tile_len, D])
                                m_v = pypto.view(mi_new, [BLOCK_Q, 1], [0, 0],
                                                 valid_shape=[q_tile_len, 1])

                                out_row = b_idx * N * Sq + n_idx * Sq + q_start
                                pypto.assemble(o_v, [out_row, 0], output)
                                pypto.assemble(m_v, [out_row, 0], softmax_max)

                                if keep_prob < 1.0:
                                    l_out = pypto.mul(li_new, 1.0 / keep_prob)
                                else:
                                    l_out = li_new
                                l_v = pypto.view(l_out, [BLOCK_Q, 1], [0, 0],
                                                 valid_shape=[q_tile_len, 1])
                                pypto.assemble(l_v, [out_row, 0], softmax_sum)
                            else:
                                oi_update[:] = oi_tmp
                                li_update[:] = li_new
                                mi_update[:] = mi_new
