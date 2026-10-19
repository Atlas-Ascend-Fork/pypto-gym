#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Gated Delta Rule Implementation Module

This module implements the core computation functions for the Chunk Gated Delta Rule
attention mechanism used in Qwen3Next model. It provides efficient linear attention
computation with O(n) complexity instead of O(n²) for traditional attention.

Main Functions:
    - l2norm: L2 normalization for query and key
    - pre_attn: Pre-attention computation including gate cumsum and decay mask
    - inverse_pto: Block-wise matrix inversion
    - cal_value_and_key_cumdecay: Value and key cumulative decay computation
    - recurrent_state_attn_all: Recurrent state attention computation
    - chunk_gated_delta_rule: Main fused operator entry point

Example:
    See tests/ops/qwen3_next/test_gated_delta_rule.py for usage examples.
"""

import pypto


def l2norm(
    query: pypto.Tensor, key: pypto.Tensor, eps: float = 1e-6
) -> tuple[pypto.Tensor, pypto.Tensor]:
    """
    L2 normalization.

    Parameters
    ---------
    query: [L, D]
    key: [L, D]
    eps=1e-6

    Return
    ---------
    query_after_l2norm: [L, D]
    key_after_l2norm: [L, D]
    """

    pypto.set_vec_tile_shapes(128, 128)
    # L2
    query_after_l2norm = query / pypto.sqrt((query * query).sum(-1, keepdim=True) + eps)
    key_after_l2norm = key / pypto.sqrt((key * key).sum(-1, keepdim=True) + eps)

    return query_after_l2norm, key_after_l2norm


def pre_attn(
    gate_view: pypto.Tensor,
    key_view_2d: pypto.Tensor,
    beta_view: pypto.Tensor,
    tril: pypto.Tensor,
    mask: pypto.Tensor,
) -> tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor]:
    """
    Calculate gate_cumsum, decay_mask, beta_k and kkt.

    Parameters
    ---------
    gate: [L, 1]
    key: [L, D]
    beta: [L, 1]
    tril: [L, L]
    mask: [L, L]

    ---------
    gate_cum: [L, 1]
    decay_mask: [L, L]
    A: [L, L]
    key_beta: [L, D]
    """

    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    # cal_cumsum
    gate_cum = pypto.matmul(tril, gate_view, pypto.DT_FP32)  # [L,1]
    # cal_decay_mask
    decay_mask = ((gate_cum - gate_cum.transpose(0, 1)) * tril).exp()  # [L,L]
    # beta_k
    key_beta = key_view_2d * beta_view  # [L,D]
    # kkt
    kkt = pypto.matmul(key_beta, key_view_2d, pypto.DT_FP32, b_trans=True)  # [L,L]
    a = kkt * decay_mask * mask  # [L,L]

    return gate_cum, decay_mask, a, key_beta


def inverse_pto(**kwargs) -> pypto.Tensor:
    """
    Calculate inverse of big matrix.

    Parameters
    ---------
    attn: [L, L]
    eye: [L // 8, L]
    size: matrix size
    zeros_16: 16 * 16 zero matrix
    zeros_32: 32 * 32 zero matrix
    zeros_64: 64 * 64 zero matrix

    Return
    ---------
    attn_inv: [L, L]
    """
    attn = kwargs.get("attn")
    eye = kwargs.get("eye")
    size = kwargs.get("size")
    zeros_16 = kwargs.get("zeros_16")
    zeros_32 = kwargs.get("zeros_32")
    zeros_64 = kwargs.get("zeros_64")

    min_length = size // 8
    pypto.set_vec_tile_shapes(128, 128)

    attn_8_8_list = []
    for i in range(8):
        attn_8_8_list.append(attn.view([min_length, min_length], [min_length * i, min_length * i]) + 0.0)
    attn_tmp_dim1 = pypto.concat(attn_8_8_list, dim=1)

    attn_tmp_dim1_inv = inverse_pto_min_length(attn_tmp_dim1, eye, min_length, min_length * 8)

    attn_8_8_inv_list = []
    for i in range(8):
        attn_8_8_inv_list.append(attn_tmp_dim1_inv[:, min_length * i:min_length * (i + 1)] + 0.0)

    attn_4_inv_list = []
    for i in range(4):
        attn_4_inv_list.append(inverse_matmul(attn=attn, attn_1_1_inv=attn_8_8_inv_list[i * 2],
            attn_2_2_inv=attn_8_8_inv_list[i * 2 + 1], x_ofs=min_length * i * 2, y_ofs=min_length * i * 2,
            m_len=min_length, zero_tensor=zeros_16))

    attn_2_inv_list = []
    for i in range(2):
        attn_2_inv_list.append(inverse_matmul(attn=attn, attn_1_1_inv=attn_4_inv_list[i * 2],
            attn_2_2_inv=attn_4_inv_list[i * 2 + 1], x_ofs=min_length * i * 4, y_ofs=min_length * i * 4,
            m_len=min_length * 2, zero_tensor=zeros_32))
    attn_inv = inverse_matmul(attn=attn, attn_1_1_inv=attn_2_inv_list[0],
        attn_2_2_inv=attn_2_inv_list[1], x_ofs=0, y_ofs=0, m_len=min_length * 4, zero_tensor=zeros_64)
    return attn_inv


def inverse_pto_min_length(
    attn_dim1: pypto.Tensor,
    eye: pypto.Tensor,
    row_num: int,
    col_num: int,
) -> pypto.Tensor:
    """
    Calculate inverse of matrix with tail concat optimization.

    Parameters
    ---------
    attn_dim0: [L, L // 8]
    attn_dim1: [L // 8, L]
    eye: [L // 8, L]
    row_num: L // 8
    col_num: L

    Return
    ---------
    res: [L, L]
    """
    size = col_num // row_num

    attn_inv_list = {}
    attn_inv_list[1] = attn_dim1[:2, :]
    pypto.set_vec_tile_shapes(128, 128)

    for i in range(2, row_num, 1):
        # Add 0.0 to enable attn_inv_cur to enter the UB in advance
        attn_inv_cur = attn_inv_list.get(i - 1) + 0.0
        row = attn_dim1.view([1, col_num], [i, 0])
        row_expand = row.reshape([size, row_num]).view([size, i], [0, 0]).transpose(1, 0).reshape([size * i, 1])
        attn_inv_cur_reshape = attn_inv_cur.reshape([size * i, row_num])
        prod_mul = (row_expand * attn_inv_cur_reshape).reshape([i, col_num])

        prod = prod_mul.sum(0, keepdim=True)
        attn_update = row + prod

        attn_inv_list[i] = pypto.concat([attn_inv_cur, attn_update], dim=0)

    res = attn_inv_list.get(row_num - 1) + eye

    return res


def inverse_matmul(**kwargs) -> pypto.Tensor:
    """
    Calculate inverse of small matrix.

    Parameters
    ---------
    attn: [L, L]
    attn_1_1_inv: attn upper left matrix
    attn_2_2_inv: attn bottom right matrix
    x_ofs: row offset
    y_ofs: column offset
    m_len: matrix length
    zero_tensor: zero matrix

    Return
    ---------
    attn_inv: [len * 2, len * 2]
    """
    attn = kwargs.get("attn")
    attn_1_1_inv = kwargs.get("attn_1_1_inv")
    attn_2_2_inv = kwargs.get("attn_2_2_inv")
    x_ofs = kwargs.get("x_ofs")
    y_ofs = kwargs.get("y_ofs")
    m_len = kwargs.get("m_len")
    zero_tensor = kwargs.get("zero_tensor")

    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

    attn_2_1 = attn.view([m_len, m_len], [x_ofs + m_len, y_ofs])

    attn_2_1_inv = (attn_2_2_inv @ attn_2_1) @ attn_1_1_inv

    attn_inv = pypto.tensor([m_len * 2, m_len * 2], dtype=attn_1_1_inv.dtype)
    attn_inv[0:m_len, 0:m_len] = attn_1_1_inv
    attn_inv[0:m_len, m_len:m_len * 2] = zero_tensor
    attn_inv[m_len:m_len * 2, 0:m_len] = attn_2_1_inv
    attn_inv[m_len:m_len * 2, m_len:m_len * 2] = attn_2_2_inv

    return attn_inv


def cal_value_and_key_cumdecay(
    attn: pypto.Tensor,
    value_view: pypto.Tensor,
    beta_view: pypto.Tensor,
    key_beta: pypto.Tensor,
    gate_cum: pypto.Tensor,
) -> tuple[pypto.Tensor, pypto.Tensor]:
    """
    Calculate value and k cumdecay.

    Parameters
    ---------
    attn: [L, L]
    value_view: [L, D]
    beta_view: [L, D]
    key_beta: [L, D]
    gate_cum: [L, 1]

    Return
    ---------
    value_out: [L, D]
    key_cum_out: [L, D]
    """

    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    # value_out
    value_beta_view = value_view * beta_view  # [L, D]
    value_out = pypto.matmul(attn, value_beta_view, pypto.DT_FP32)  # [L, D]
    # k_cumdecay_out
    g_exp = pypto.exp(gate_cum)  # [L, 1]
    weighted_k_beta_view = key_beta * g_exp  # [L, D]
    key_cum_out = pypto.matmul(attn, weighted_k_beta_view, pypto.DT_FP32)  # [L, D]

    return value_out, key_cum_out


def recurrent_state_attn_all(**kwargs) -> tuple[pypto.Tensor, pypto.Tensor]:
    """
    Calculate attention.

    Parameters
    ---------
    query: [L, D]
    key: [L, D]
    value:[L, Dv]
    k_cumdecay:[L, Dk]
    gate: [L, 1]
    state: [D, D]
    decay_mask: [L, L]
    tril: [L, L]

    Return
    ---------
    chunk_attn_out: [L, D]
    state_new:[Dv, Dk]
    """
    query = kwargs.get("query")
    key = kwargs.get("key")
    value = kwargs.get("value")
    k_cumdecay = kwargs.get("k_cumdecay")
    gate = kwargs.get("gate")
    state = kwargs.get("state")
    decay_mask = kwargs.get("decay_mask")
    tril = kwargs.get("tril")

    dv = value.shape[-1]
    l = gate.valid_shape[0]
    gate_exp = gate.exp()
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    pypto.set_vec_tile_shapes(64, 128)
    _last_gate_1 = gate[l - 1:l, :]
    kgexp = key * (_last_gate_1 - gate).exp()  # [L, Dk]
    qgexp = query * gate_exp
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
    v_prime = pypto.matmul(k_cumdecay, state, pypto.DT_FP32, b_trans=True)  # [L, Dk] @ [Dk, Dv] = [L, Dv]
    attn_inter = pypto.matmul(qgexp, state, pypto.DT_FP32, b_trans=True)  # [L, Dk] @ [Dk, Dv] = [L, Dv]
    pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])
    temp_matmul_vprime = pypto.matmul(v_prime, kgexp, pypto.DT_FP32, a_trans=True)  # [Dv, L] @ [L, Dk] = [Dv, Dk]
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    temp_matmul_value = pypto.matmul(value, kgexp, pypto.DT_FP32, a_trans=True)  # [Dv, L] @ [L, Dk] = [L, Dk]
    attn = pypto.matmul(query, key, pypto.DT_FP32, b_trans=True)  # [L, Dk] @ [Dk, L] = [L, L]
    _last_gate_2 = pypto.expand_clone(gate_exp[l - 1:l, :], (dv, 1))  # [Dv, 1]
    final_state_1 = state * _last_gate_2
    state_new = final_state_1 + temp_matmul_value - temp_matmul_vprime
    attn_tmp = attn * decay_mask * tril  # [L, L]
    chunk_attn_value = pypto.matmul(attn_tmp, value, pypto.DT_FP32)  # [L, L] @ [L, Dv] = [L, Dv]
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
    chunk_attn_vprime = pypto.matmul(attn_tmp, v_prime, pypto.DT_FP32)  # [L, L] @ [L, Dv] = [L, Dv]
    chunk_attn_out = attn_inter + chunk_attn_value - chunk_attn_vprime
    return chunk_attn_out, state_new


def chunk_gated_delta_rule(b, nqk, nv, d, l):

    t = pypto.DYNAMIC
    b1 = b + 1
    b1 = pypto.DYNAMIC
    b = pypto.DYNAMIC
    query_shape = [t, nqk, d]
    key_shape = [t, nqk, d]
    value_shape = [t, nv, d]
    beta_shape = [t, nv]
    gate_shape = [t, nv]
    states_shape = [b, nv, d, d]
    mask_shape = [l, l]
    tril_mask_shape = [l, l]
    eye_shape = [16, l]
    act_seq_len_shape = [b1]
    core_attn_out_shape = [t, nv, d]
    last_state_data_shape = [b, nv, d, d]

    @pypto.frontend.jit(
        runtime_options={
            "stitch_function_max_num": 1,
        },
    )
    def kernel(
            query: pypto.Tensor(query_shape, pypto.DT_FP32),
            key: pypto.Tensor(key_shape, pypto.DT_FP32),
            value: pypto.Tensor(value_shape, pypto.DT_FP32),
            beta: pypto.Tensor(beta_shape, pypto.DT_FP32),
            gate: pypto.Tensor(gate_shape, pypto.DT_FP32),
            states: pypto.Tensor(states_shape, pypto.DT_FP32),
            mask: pypto.Tensor(mask_shape, pypto.DT_FP32),
            tril_mask: pypto.Tensor(tril_mask_shape, pypto.DT_FP32),
            eye: pypto.Tensor(eye_shape, pypto.DT_FP32),
            act_seq_len: pypto.Tensor(act_seq_len_shape, pypto.DT_INT32),
            core_attn_out: pypto.Tensor(core_attn_out_shape, pypto.DT_FP32),
            last_state_data: pypto.Tensor(last_state_data_shape, pypto.DT_FP32)
        ):
        """
        Chunk Gated Delta Rule fused operator.

        This is the main entry point for the Gated Delta Rule attention computation
        in the scenario where Sequence length is divisible by L.
        It processes input sequences in chunks of size L=128, maintaining recurrent
        state across chunks for efficient long sequence modeling.

        Parameters
        ----------
        query : Input query tensor, shape [T, Nqk, D], dtype float32
        key : Input key tensor, shape [T, Nqk, D], dtype float32
        value : Input value tensor, shape [T, Nv, D], dtype float32
        beta : Beta scaling factor, shape [T, Nv], dtype float32
        gate : Gate signal, shape [T, Nv], dtype float32
        states : Initial recurrent states, shape [B, Nv, D, D], dtype float32
        mask : Attention mask (lower triangular negative), shape [L, L], dtype float32
        tril_mask : Lower triangular mask, shape [L, L], dtype float32
        eye : Identity matrix (specially processed), shape [16, 128], dtype float32
        act_seq_len : Cumulative sequence length indices, shape [B+1], dtype int32
        core_attn_out : Output attention tensor, shape [T, Nv, D], dtype float32
        last_state_data : Output updated states, shape [B, Nv, D, D], dtype float32
        """
        _, nqk, d = query.shape
        _, nv, d = value.shape
        b = states.shape[0]
        l, l = mask.shape
        group = nv // nqk
        last_state = pypto.tensor([d, d], pypto.DT_FP32)
        pypto.experimental.set_operation_options(combine_axis=True)

        for b_idx in pypto.loop(b, name="LOOP_B_TND", idx_name="b_idx"):
            s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
            b_ofs = act_seq_len[b_idx]
            for nv_idx in pypto.loop(nv, name="LOOP_Nv_TND", idx_name="nv_idx"):
                nqk_idx = nv_idx // group
                pypto.set_vec_tile_shapes(16, 16, 128, 128)
                last_state = states[b_idx, nv_idx]
                for s_idx in pypto.loop(0, s, l, name="LOOP_S_TND", idx_name="s_idx", unroll_list=[16, 1]):
                    bs_ofs = b_ofs + s_idx
                    actual_l = (s - s_idx).min(l)
                    ## view
                    query_view = pypto.view(query, [l, 1, d], [bs_ofs, nqk_idx, 0], valid_shape=[actual_l, 1, d])
                    key_view = pypto.view(key, [l, 1, d], [bs_ofs, nqk_idx, 0], valid_shape=[actual_l, 1, d])
                    value_view = pypto.view(value, [l, 1, d], [bs_ofs, nv_idx, 0], valid_shape=[actual_l, 1, d])
                    beta_view = pypto.view(beta, [l, 1], [bs_ofs, nv_idx], valid_shape=[actual_l, 1])
                    gate_view = pypto.view(gate, [l, 1], [bs_ofs, nv_idx], valid_shape=[actual_l, 1])

                    pypto.set_vec_tile_shapes(128, 128, 128)
                    query_view_2d = pypto.reshape(query_view, [l, d], valid_shape=[actual_l, d])
                    key_view_2d = pypto.reshape(key_view, [l, d], valid_shape=[actual_l, d])
                    value_view_2d = pypto.reshape(value_view, [l, d], valid_shape=[actual_l, d])
                    pypto.set_pass_options(sg_set_scope=1)
                    zeros_16 = pypto.full(size=[16, 16], fill_value=0.0, dtype=pypto.DT_FP32)
                    zeros_32 = pypto.full(size=[32, 32], fill_value=0.0, dtype=pypto.DT_FP32)
                    zeros_64 = pypto.full(size=[64, 64], fill_value=0.0, dtype=pypto.DT_FP32)
                    pypto.set_pass_options(sg_set_scope=-1)

                    # compute
                    # qk_l2norm
                    query_norm, key_norm = l2norm(query_view_2d, key_view_2d)
                    scale = 1 / d ** 0.5
                    query_scale = query_norm * scale

                    # kv_beta & g_cumsum & decay_mask & pre_attn
                    gate_cum, decay_mask, a_block, key_beta = pre_attn(gate_view, key_norm, beta_view, tril_mask, mask)

                    # inverse
                    a_block_inverse = inverse_pto(attn=a_block, eye=eye, size=128, zeros_16=zeros_16,
                        zeros_32=zeros_32, zeros_64=zeros_64)

                    # cal_value_and_keycumdecay
                    value_out, key_cum_out = cal_value_and_key_cumdecay(a_block_inverse, value_view_2d,
                        beta_view, key_beta, gate_cum)

                    chunk_attn_out, cur_state = recurrent_state_attn_all(query=query_scale, key=key_norm,
                        value=value_out, k_cumdecay=key_cum_out, gate=gate_cum, state=last_state,
                        decay_mask=decay_mask, tril=tril_mask)

                    # assemble
                    pypto.set_vec_tile_shapes(16, 16, 128, 128)
                    last_state[:] = cur_state
                    core_attn_out[bs_ofs:bs_ofs + l, nv_idx] = chunk_attn_out
                    last_state_data[b_idx, nv_idx] = last_state

    return kernel


def chunk_gated_delta_rule_unaligned(b, nqk, nv, d, l):

    t_unaligned = pypto.DYNAMIC
    b1 = b + 1
    b1 = pypto.DYNAMIC
    b = pypto.DYNAMIC
    query_shape = [t_unaligned, nqk, d]
    key_shape = [t_unaligned, nqk, d]
    value_shape = [t_unaligned, nv, d]
    beta_shape = [t_unaligned, nv]
    gate_shape = [t_unaligned, nv]
    states_shape = [b, nv, d, d]
    mask_shape = [l, l]
    tril_mask_shape = [l, l]
    eye_shape = [16, l]
    act_seq_len_shape = [b1]
    core_attn_out_shape = [t_unaligned, nv, d]
    last_state_data_shape = [b, nv, d, d]

    @pypto.frontend.jit(
        runtime_options={
            "stitch_function_max_num": 1,
        },
    )
    def kernel(
            query: pypto.Tensor(query_shape, pypto.DT_FP32),
            key: pypto.Tensor(key_shape, pypto.DT_FP32),
            value: pypto.Tensor(value_shape, pypto.DT_FP32),
            beta: pypto.Tensor(beta_shape, pypto.DT_FP32),
            gate: pypto.Tensor(gate_shape, pypto.DT_FP32),
            states: pypto.Tensor(states_shape, pypto.DT_FP32),
            mask: pypto.Tensor(mask_shape, pypto.DT_FP32),
            tril_mask: pypto.Tensor(tril_mask_shape, pypto.DT_FP32),
            eye: pypto.Tensor(eye_shape, pypto.DT_FP32),
            act_seq_len: pypto.Tensor(act_seq_len_shape, pypto.DT_INT32),
            core_attn_out: pypto.Tensor(core_attn_out_shape, pypto.DT_FP32),
            last_state_data: pypto.Tensor(last_state_data_shape, pypto.DT_FP32)
        ):
        """
        Chunk Gated Delta Rule fused operator.

        This is the main entry point for the Gated Delta Rule attention computation
        in the scenario where Sequence length is not divisible by L.
        It processes input sequences in chunks of size L=128, maintaining recurrent
        state across chunks for efficient long sequence modeling.


        Parameters
        ----------
        query : Input query tensor, shape [T, Nqk, D], dtype float32
        key : Input key tensor, shape [T, Nqk, D], dtype float32
        value : Input value tensor, shape [T, Nv, D], dtype float32
        beta : Beta scaling factor, shape [T, Nv], dtype float32
        gate : Gate signal, shape [T, Nv], dtype float32
        states : Initial recurrent states, shape [B, Nv, D, D], dtype float32
        mask : Attention mask (lower triangular negative), shape [L, L], dtype float32
        tril_mask : Lower triangular mask, shape [L, L], dtype float32
        eye : Identity matrix (specially processed), shape [16, 16], dtype float32
        act_seq_len : Cumulative sequence length indices, shape [B+1], dtype int32
        core_attn_out : Output attention tensor, shape [T, Nv, D], dtype float32
        last_state_data : Output updated states, shape [B, Nv, D, D], dtype float32
        """
        _, nqk, d = query.shape
        _, nv, d = value.shape
        b = states.shape[0]
        l, l = mask.shape
        group = nv // nqk
        pypto.experimental.set_operation_options(combine_axis=True)

        for b_idx in pypto.loop(b, name="LOOP_B_TND", idx_name="b_idx"):
            s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
            b_ofs = act_seq_len[b_idx]
            for nv_idx in pypto.loop(nv, name="LOOP_Nv_TND", idx_name="nv_idx"):
                nqk_idx = nv_idx // group
                pypto.set_vec_tile_shapes(16, 16, 128, 128)
                last_state = states[b_idx, nv_idx]
                for s_idx in pypto.loop(0, s, l, name="LOOP_S_TND", idx_name="s_idx", unroll_list=[16, 1]): # 2
                    bs_ofs = b_ofs + s_idx
                    actual_l = (s - s_idx).min(l)

                    zeros_16 = pypto.full(size=[16, 16], fill_value=0.0, dtype=pypto.DT_FP32)
                    zeros_32 = pypto.full(size=[32, 32], fill_value=0.0, dtype=pypto.DT_FP32)
                    zeros_64 = pypto.full(size=[64, 64], fill_value=0.0, dtype=pypto.DT_FP32)

                    # view
                    query_view = pypto.view(query, [l, 1, d], [bs_ofs, nqk_idx, 0], valid_shape=[actual_l, 1, d])
                    key_view = pypto.view(key, [l, 1, d], [bs_ofs, nqk_idx, 0], valid_shape=[actual_l, 1, d])
                    value_view = pypto.view(value, [l, 1, d], [bs_ofs, nv_idx, 0], valid_shape=[actual_l, 1, d])
                    beta_view = pypto.view(beta, [l, 1], [bs_ofs, nv_idx], valid_shape=[actual_l, 1])
                    gate_view = pypto.view(gate, [l, 1], [bs_ofs, nv_idx], valid_shape=[actual_l, 1])

                    pypto.set_vec_tile_shapes(128, 128, 128)
                    query_view_2d = pypto.reshape(query_view, [l, d], valid_shape=[actual_l, d])
                    key_view_2d = pypto.reshape(key_view, [l, d], valid_shape=[actual_l, d])
                    value_view_2d = pypto.reshape(value_view, [l, d], valid_shape=[actual_l, d])

                    if pypto.is_loop_end(s_idx):
                        pad_q = pypto.fillpad(query_view_2d, "constant", 0.0)
                        pad_k = pypto.fillpad(key_view_2d, "constant", 0.0)
                        pad_v = pypto.fillpad(value_view_2d, "constant", 0.0)
                        pad_b = pypto.fillpad(beta_view, "constant", 0.0)
                        pad_g = pypto.fillpad(gate_view, "constant", 0.0)

                        # compute
                        # qk_l2norm
                        query_norm, key_norm = l2norm(pad_q, pad_k)
                        scale = 1 / d ** 0.5
                        query_scale = query_norm * scale

                        # kv_beta & g_cumsum & decay_mask & pre_attn
                        gate_cum, decay_mask, a_block, key_beta = pre_attn(pad_g, key_norm, pad_b, tril_mask, mask)

                        # inverse
                        a_block_inverse = inverse_pto(attn=a_block, eye=eye, size=128, zeros_16=zeros_16,
                                zeros_32=zeros_32, zeros_64=zeros_64)

                        # cal_value_and_keycumdecay
                        value_out, key_cum_out = cal_value_and_key_cumdecay(
                            a_block_inverse, pad_v, pad_b, key_beta, gate_cum)
                        chunk_attn_out, cur_state = recurrent_state_attn_all(query=query_scale, key=key_norm,
                            value=value_out, k_cumdecay=key_cum_out, gate=gate_cum, state=last_state,
                            decay_mask=decay_mask, tril=tril_mask)
                        # assemble
                        last_state[:] = cur_state
                        last_state_data[b_idx, nv_idx] = last_state
                        pypto.set_vec_tile_shapes(128, 16, 128)
                        chunk_attn_out_reshaped = chunk_attn_out.reshape([l, 1, d], valid_shape=[actual_l, 1, d])
                        pypto.assemble(chunk_attn_out_reshaped, [bs_ofs, nv_idx, 0], core_attn_out)

                    else:
                        query_norm, key_norm = l2norm(query_view_2d, key_view_2d)
                        scale = 1 / d ** 0.5
                        query_scale = query_norm * scale

                        gate_cum, decay_mask, a_block, key_beta = pre_attn(
                            gate_view, key_norm, beta_view, tril_mask, mask)
                        # inverse
                        a_block_inverse = inverse_pto(attn=a_block, eye=eye, size=128, zeros_16=zeros_16,
                            zeros_32=zeros_32, zeros_64=zeros_64)

                        # cal_value_and_keycumdecay
                        value_out, key_cum_out = cal_value_and_key_cumdecay(
                            a_block_inverse, value_view_2d, beta_view, key_beta, gate_cum)
                        chunk_attn_out, cur_state = recurrent_state_attn_all(query=query_scale, key=key_norm,
                            value=value_out, k_cumdecay=key_cum_out, gate=gate_cum, state=last_state,
                            decay_mask=decay_mask, tril=tril_mask)
                        # assemble
                        last_state[:] = cur_state
                        last_state_data[b_idx, nv_idx] = last_state
                        core_attn_out[bs_ofs:bs_ofs + l, nv_idx] = chunk_attn_out

    return kernel
