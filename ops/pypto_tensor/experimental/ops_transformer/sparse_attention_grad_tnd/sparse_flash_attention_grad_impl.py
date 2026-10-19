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
Sparse Flash Attention Grad - PyPTO Implementation (TND format, nope/rope split, dynamic shapes)

实现SFA反向算子：根据前向中间结果和输出梯度，计算dQ, dK, dV。

**TND 格式**
  - 输入: q_nope(t_1, n_1, d), q_pe(t_1, n_1, dr), k_nope(t_2, n_2, d), k_pe(t_2, n_2, dr), value(t_2, n_2, d)
  - t_1 动态 (total_q tokens), t_2 动态 (total_kv tokens), batch 静态
  - actual_seq_qlen(batch,) 和 actual_seq_kvlen(batch,) 为前缀和格式
  - 主循环: for b in batch -> for s in s_count，s_count = qlen[b] - qlen[b-1]*(b>0)

**静态shape规则**
  - 所有输入tensor shape必须是静态的，只有valid_shape可以用动态符号
  - KV-side tensor 使用 MAX_TOTAL_KV 作为静态上界
  - Q-side 的 nope/rope 在循环内 view 出静态 [group, d]/[group, dr]，然后 concat
  - k-side 的 nope/rope 分别 index_select 出静态 [k, d]/[k, dr]，然后 concat

**动态shape设计**
  - t_1, t_2 通过 pypto.frontend.dynamic 标识为动态轴
  - KV tensor 传入时 shape 为 (t_2, n_2, d)，内部 reshape 为 2D 后
    view 到 [MAX_TOTAL_KV*n_2, d] with valid_shape=[t_2*n_2, d]
  - dk_nope/dv 用 MAX_TOTAL_KV 静态shape + valid_shape
  - dk_pe 暂保持动态 shape (如有报错再改为 MAX_TOTAL_KV)

**valid_shape on indices**
  - cur_kv_len = min(k, actual_kv_len_for_batch)
  - sparse_idx view带valid_shape=[1, cur_kv_len]

**关键设计：批量处理G个head**
  - 对每个(b, s)位置，同时处理所有G个query heads (M=group)
  - G个heads共享相同的sparse indices (因为N2=1)

计算公式（对每个(b, s)位置）：
  1. Gather: sel_k_nope, sel_k_pe, sel_v 分别 index_select  -> 各 (k, d)/(k, dr)/(k, d)
  2. sel_k = concat(sel_k_nope, sel_k_pe)                     -> (k, d+dr)
  3. Q_group = concat(q_nope_view, q_pe_view)                  -> (group, d+dr)
  4. S = Q_group @ sel_k^T * scale                             -> (group, k)
  5. P = exp(S - mi) / li                                      -> (group, k)
  6. dP = dO_group @ sel_v^T                                   -> (group, k)
  7. dV_local = P^T @ dO_group                                 -> (k, d)
  8. D_val = rowsum(dO * out)                                   -> (group, 1)
  9. dS = P * (dP - D_val)                                     -> (group, k)
  10. dQ_local = dS @ sel_k * scale                             -> (group, d+dr)
  11. dK_local = dS^T @ Q_group * scale                         -> (k, d+dr)
  12. view dQ_local/dK_local 的 nope/rope 部分，分别写回输出
"""
import os
import math
import logging
from dataclasses import dataclass
import collections
from typing import Any
import numpy as np
import pypto
import torch
from torch._dynamo import allow_in_graph
from torch._subclasses.fake_tensor import FakeTensor

MAX_TOTAL_KV = 128 * 1024

SfaGradComputeOutput = collections.namedtuple(
    "SfaGradComputeOutput",
    ["dq_nope_2d", "dq_pe_2d", "dk_nope_2d", "dk_pe_2d", "dv_2d", "dk_out"],
)


def sparse_flash_attention_grad_compute(
    q_nope, q_pe, k_nope, k_pe, value,
    sparse_idx,
    d_out, out, sm_max, sm_sum,
    actual_seq_qlen, actual_seq_kvlen,
    dq_nope_2d, dq_pe_2d, dk_nope_2d, dk_pe_2d, dv_2d, dk_nope_out, dk_pe_out, dk_2d, dk_out,
    n_1, n_2, d, dr, k, group, scale_value,
):
    """
    PyPTO实现SFA反向计算核心逻辑 - TND格式, nope/rope split, 动态shape。

    输入 (TND 3D):
        q_nope:          (t_1, n_1, d) BF16        t_1 动态
        q_pe:            (t_1, n_1, dr) BF16
        k_nope:          (t_2, n_2, d) BF16        t_2 动态
        k_pe:            (t_2, n_2, dr) BF16
        value:           (t_2, n_2, d) BF16
        sparse_idx:      (t_1, n_2, k) INT32       indices into t_2
        d_out:           (t_1, n_1, d) BF16
        out:             (t_1, n_1, d) BF16
        sm_max:          (n_2, t_1, group) FP32
        sm_sum:          (n_2, t_1, group) FP32
        actual_seq_qlen: (batch,) INT32              前缀和
        actual_seq_kvlen:(batch,) INT32              前缀和

    Returns:
        dq_nope, dq_pe, dk_nope, dk_pe, dv (all TND 3D)
    """
    dtype = q_nope.dtype  # BF16
    d_full = d + dr

    t1_sym = q_nope.shape[0]       # dynamic
    t2_sym = k_nope.shape[0]       # dynamic (actual total_kv)
    b_sym = actual_seq_qlen.shape[0]  # dynamic

    # ======== Reshape 3D -> 2D in init loop ========
    for _ in pypto.loop(0, 1, 1, name="LOOP_reshape"):
        pypto.set_vec_tile_shapes(128, 576)
        # Q-side: dynamic t_1
        q_nope_2d = pypto.reshape(q_nope, [t1_sym * n_1, d], inplace=True)      # (t_1*n_1, d)
        q_pe_2d = pypto.reshape(q_pe, [t1_sym * n_1, dr], inplace=True)         # (t_1*n_1, dr)
        d_out_2d = pypto.reshape(d_out, [t1_sym * n_1, d], inplace=True)        # (t_1*n_1, d)
        out_2d = pypto.reshape(out, [t1_sym * n_1, d], inplace=True)            # (t_1*n_1, d)
        sparse_idx_2d = pypto.reshape(sparse_idx, [t1_sym * n_2, k], inplace=True)  # (t_1*n_2, k)
        sparse_idx_1d = pypto.reshape(sparse_idx, [t1_sym * n_2 * k], inplace=True)  # (t_1*n_2*k)

        # KV-side: t_2 dynamic, reshape to 2D then view to static shape with valid_shape
        k_nope_2d = pypto.reshape(k_nope, [t2_sym * n_2, d], inplace=True)  # dynamic (t_2*n_2, d)
        k_pe_2d = pypto.reshape(k_pe, [t2_sym * n_2, dr], inplace=True)     # dynamic (t_2*n_2, dr)
        value_2d = pypto.reshape(value, [t2_sym * n_2, d], inplace=True)    # dynamic (t_2*n_2, d)

        # sm_max/sm_sum: (n_2, t_1, group) -> reshape to (n_2*t_1, group)
        sm_max_2d = pypto.reshape(sm_max, [n_2 * t1_sym, group], inplace=True)      # (n_2*t_1, group)
        sm_sum_2d = pypto.reshape(sm_sum, [n_2 * t1_sym, group], inplace=True)      # (n_2*t_1, group)

    # ======== Main compute: for batch -> for S ========
    for b_idx in pypto.loop(0, b_sym, 1, name="LOOP_B", idx_name="bIdx"):
        # 计算当前batch的q/kv offset和s_count
        # actual_seq_qlen 是前缀和: q_offset = qlen[b-1] if b>0 else 0, s_count = qlen[b] - q_offset
        cur_qlen = actual_seq_qlen[b_idx]         # qlen prefix sum at b      # 0 when b=0, prev_qlen when b>0
        if pypto.cond(b_idx == 0):
            q_offset = 0
            kv_offset = 0
        else:
            q_offset = actual_seq_qlen[b_idx - 1]
            kv_offset = actual_seq_kvlen[b_idx - 1]
        s_count = cur_qlen - q_offset

        cur_kvlen = actual_seq_kvlen[b_idx]
        actual_kv_len = cur_kvlen - kv_offset       # actual kv length for this batch

        for s_idx in pypto.loop(0, s_count, 1, name="LOOP_S", idx_name="sIdx"):
            t_idx = q_offset + s_idx  # global token index in t_1
            # casual
            cur_kv_valid = (actual_kv_len - s_count + 1 + s_idx).max(0).min(k)

            for kv_head_idx in range(1):
                # ======== Step 1: Get sparse indices with valid_shape ========
                topk_row = t_idx
                pypto.set_vec_tile_shapes(1, k)
                cur_indices_2d = pypto.view(sparse_idx_2d, [1, k], [topk_row, 0],
                                            valid_shape=[1, cur_kv_valid])
                cur_indices_1d = pypto.view(sparse_idx_1d, [k], [topk_row * k],
                                               valid_shape=[cur_kv_valid])

                # ======== Step 2: Gather sel_k_nope, sel_k_pe, sel_v (static shapes) ========
                # index_select produces static shapes: (k, d), (k, dr), (k, d)
                pypto.set_vec_tile_shapes(128, d)
                k_nope_2d_view = pypto.view(k_nope_2d, [MAX_TOTAL_KV, d], [0, 0], valid_shape=[t1_sym * n_1, d])
                sel_k_nope = pypto.index_select(k_nope_2d_view, 0, cur_indices_1d)  # (k, d) BF16
                pypto.set_vec_tile_shapes(128, dr)
                k_pe_2d_view = pypto.view(k_pe_2d, [MAX_TOTAL_KV, dr], [0, 0], valid_shape=[t1_sym * n_1, dr])
                sel_k_pe = pypto.index_select(k_pe_2d_view, 0, cur_indices_1d)      # (k, dr) BF16

                # Concat sel_k_nope + sel_k_pe -> sel_k (static [k, d+dr])
                pypto.set_vec_tile_shapes(16, d)
                sel_k = pypto.tensor([k, d + dr], dtype=pypto.DT_BF16)
                pypto.assemble(sel_k_nope, [0, 0], sel_k)
                pypto.assemble(sel_k_pe, [0, d], sel_k)

                # ======== Step 3: Q_group (static shape concat), dO_group, Out_group ========
                q_head_start = kv_head_idx * group
                q_offset_2d = t_idx * n_1 + q_head_start
                # View static-shape slices from q_nope_2d/q_pe_2d, then concat
                qn_slice = pypto.view(q_nope_2d, [group, d], [q_offset_2d, 0])    # static (group, d)
                qr_slice = pypto.view(q_pe_2d, [group, dr], [q_offset_2d, 0])     # static (group, dr)
                pypto.set_vec_tile_shapes(16, d)
                q_group = pypto.tensor([group, d + dr], dtype=pypto.DT_BF16)
                pypto.assemble(qn_slice, [0, 0], q_group)
                pypto.assemble(qr_slice, [0, d], q_group)

                sm_row = kv_head_idx * t1_sym + t_idx

                # ======== Step 4: S = Q_full @ sel_k^T * scale ========
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [256, 256])
                sel_k_view = pypto.view(sel_k, [k, d_full], [0, 0], valid_shape=[cur_kv_valid, d_full])
                s_scores = pypto.matmul(q_group, sel_k_view, pypto.DT_FP32,
                                        a_trans=False, b_trans=True)  # (group, k)

                # ======== Step 5: 恢复softmax P ========
                pypto.set_vec_tile_shapes(1, 16)
                cur_mi_row = pypto.view(sm_max_2d, [1, group], [sm_row, 0])
                cur_li_row = pypto.view(sm_sum_2d, [1, group], [sm_row, 0])
                mi_col = pypto.reshape(cur_mi_row, [group, 1])
                li_col = pypto.reshape(cur_li_row, [group, 1])

                pypto.set_vec_tile_shapes(16, k)
                s_scaled = pypto.mul(s_scores, scale_value)
                s_shifted = pypto.sub(s_scaled, mi_col)
                exp_s = pypto.exp(s_shifted)
                p_mat = pypto.div(exp_s, li_col)
                p_bf16 = pypto.cast(p_mat, dtype)

                # ======== Step 6: dP = dO_group @ sel_v^T ========
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [256, 256])
                do_group = pypto.view(d_out_2d, [group, d], [q_offset_2d, 0])     # (group, d)
                sel_v_view = pypto.view(sel_k_nope, [k, d], [0, 0], valid_shape=[cur_kv_valid, d])
                dp_mat = pypto.matmul(do_group, sel_v_view, pypto.DT_FP32,
                                      a_trans=False, b_trans=True)  # (group, k)

                # ======== Step 7: dV_local = P^T @ dO_group ========
                pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128])
                dv_local = pypto.matmul(p_bf16, do_group, pypto.DT_FP32,
                                        a_trans=True, b_trans=False)  # (k, d)
                dv_local_valid = pypto.view(dv_local, [k, d], [0, 0], valid_shape=[cur_kv_valid, d])

                # ======== Step 8: D_val, dS ========
                pypto.set_vec_tile_shapes(16, d)
                do_f32 = pypto.cast(do_group, pypto.DT_FP32)
                out_group = pypto.view(out_2d, [group, d], [q_offset_2d, 0])      # (group, d)
                out_f32 = pypto.cast(out_group, pypto.DT_FP32)
                do_out_prod = pypto.mul(do_f32, out_f32)
                d_val = pypto.sum(do_out_prod, dim=-1, keepdim=True)  # (group, 1)

                pypto.set_vec_tile_shapes(8, k)
                dp_minus_d = pypto.sub(dp_mat, d_val)
                ds_mat = pypto.mul(p_mat, dp_minus_d)
                ds_bf16 = pypto.cast(ds_mat, dtype)

                # ======== Step 9: dQ_local = dS @ sel_k * scale -> (group, d+dr) ========
                pypto.set_cube_tile_shapes([128, 128], [256, 256], [128, 128])
                sel_k_view_2 = pypto.view(sel_k, [k, d_full], [0, 0], valid_shape=[cur_kv_valid, d_full])
                dq_local = pypto.matmul(ds_bf16, sel_k_view_2, pypto.DT_FP32,
                                        a_trans=False, b_trans=False)  # (group, d+dr)
                pypto.set_vec_tile_shapes(16, d_full)
                dq_scaled = pypto.mul(dq_local, scale_value)
                dq_cast = pypto.cast(dq_scaled, dtype)

                # assemble dQ nope/rope
                dq_nope_part = pypto.view(dq_cast, [group, d], [0, 0])
                dq_pe_part = pypto.view(dq_cast, [group, dr], [0, d])
                pypto.set_vec_tile_shapes(16, d)
                pypto.assemble(dq_nope_part, [q_offset_2d, 0], dq_nope_2d)
                pypto.set_vec_tile_shapes(16, dr)
                pypto.assemble(dq_pe_part, [q_offset_2d, 0], dq_pe_2d)

                # ======== Step 10: dK_local = dS^T @ Q_group * scale -> (k, d+dr) ========
                pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128])
                dk_local = pypto.matmul(ds_bf16, q_group, pypto.DT_FP32,
                                        a_trans=True, b_trans=False)  # (k, d+dr)
                pypto.set_vec_tile_shapes(16, d)
                dk_scaled = pypto.mul(dk_local, scale_value)

                # ======== Step 11: scatter-add dK (nope+rope), dV ========
                # dk_scaled: (k, d+dr) FP32, dv_local: (k, d) FP32
                dk_nope_part = pypto.view(dk_scaled, [k, d], [0, 0], valid_shape=[cur_kv_valid, d])     # (k, d) FP32
                dk_v = pypto.add(dk_nope_part, dv_local_valid)
                dk_pe_part = pypto.view(dk_scaled, [k, dr], [0, d], valid_shape=[cur_kv_valid, dr])       # (k, dr) FP32
                dk_all_tmp = pypto.concat([dk_v, dk_pe_part], dim=-1)
                dk_all = pypto.view(dk_all_tmp, [k, d + dr], [0, 0], valid_shape=[cur_kv_valid, d + dr])

                pypto.set_vec_tile_shapes(16, d_full)
                dk_2d_view = pypto.view(dk_2d, [MAX_TOTAL_KV, d + dr], [0, 0], valid_shape=[t2_sym, d + dr])
                dk_out[:] = pypto.index_add_(dk_2d_view, -2, cur_indices_1d, dk_all)

    return SfaGradComputeOutput(dq_nope_2d, dq_pe_2d, dk_nope_2d, dk_pe_2d, dv_2d, dk_out)


@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 4, -2: 1},
        "cube_l1_reuse_setting": {-1: 8},
    },
    runtime_options={
        "device_sched_mode": 3,
        "stitch_function_max_num": 256
    },
)
def sparse_flash_attention_grad(
    q_nope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16), # t_1, n_1, d
    q_pe: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16), # t_1, n_1, dr
    k_nope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16), # t_2, n_2, d
    k_pe: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16), # t_2, n_2, dr
    value: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16), # t_2, n_2, d
    sparse_idx: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT32), # t_1, n_2, k
    d_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16), # t_1, n_1, d
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16), # t_1, n_1, d
    sm_max: pypto.Tensor([pypto.STATIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # n_2, t_1, group
    sm_sum: pypto.Tensor([pypto.STATIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # n_2, t_1, group
    actual_seq_qlen: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32), # batch
    actual_seq_kvlen: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32), # batch
    dq_nope_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16), # t_1 * n_1, d
    dq_pe_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16), # t_1 * n_1, dr

    dk_nope_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # t_2 * n_2, d
    dk_pe_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # t_2 * n_2, dr
    dv_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # t_2 * n_2, d

    dk_nope_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # t_2 * n_2, d
    dk_pe_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # t_1 * n_1, dr

    dk_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # t_2 * n_2, d
    dk_2d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # t_2 * n_2, d
    n_1, n_2, d, dr, k, group, scale_value
):
    """JIT-compiled SFA backward kernel, TND format, nope/rope split, dynamic shapes."""
    pypto.experimental.set_operation_options(combine_axis=True)

    sparse_flash_attention_grad_compute(
        q_nope, q_pe, k_nope, k_pe, value,
        sparse_idx,
        d_out, out, sm_max, sm_sum,
        actual_seq_qlen, actual_seq_kvlen,
        dq_nope_out, dq_pe_out, dk_nope_out, dk_pe_out, dv_out, dk_nope_in, dk_pe_in, dk_out, dk_2d,
        n_1, n_2, d, dr, k, group, scale_value
    )


@dataclass
class CheckInputOutputShapeDtypeInputs:
    q_nope: Any
    q_pe: Any
    k_nope: Any
    k_pe: Any
    value: Any
    sparse_idx: Any
    d_out: Any
    out: Any
    sm_max: Any
    sm_sum: Any
    actual_seq_qlen: Any
    actual_seq_kvlen: Any


def check_input_output_shape_dtype(inputs: CheckInputOutputShapeDtypeInputs):
    assert inputs.actual_seq_kvlen is not None and inputs.actual_seq_kvlen.dim() == 1, \
        f"inputs.actual_seq_kvlen dim num is {inputs.actual_seq_kvlen.dim()}, expected 1"
    assert inputs.actual_seq_qlen is not None and inputs.actual_seq_qlen.dim() == 1, \
        f"inputs.actual_seq_qlen dim num is {inputs.actual_seq_qlen.dim()}, expected 1"
    assert inputs.sparse_idx is not None and inputs.sparse_idx.dim() == 3, \
        f"topk_indices_npu dim num is {inputs.sparse_idx.dim()}, expected 3"

    assert inputs.q_nope.dim() == 3 and inputs.q_nope.size(2) == 512 \
        and inputs.q_nope.dtype == torch.bfloat16, \
        (f"q_nope: dim={inputs.q_nope.dim()} size1={inputs.q_nope.size(1)} "
         f"dtype={inputs.q_nope.dtype} (expect 3,512,bf16)")
    assert inputs.q_pe.dim() == 3 and inputs.q_pe.size(2) == 64 \
        and inputs.q_pe.dtype == torch.bfloat16, \
        (f"q_pe: dim={inputs.q_pe.dim()} size1={inputs.q_pe.size(1)} "
         f"dtype={inputs.q_pe.dtype} (expect 3,64,bf16)")

    assert inputs.k_nope.dim() == 3 and inputs.k_nope.size(2) == 512 \
        and inputs.k_nope.dtype == torch.bfloat16, \
        (f"k_nope: dim={inputs.k_nope.dim()} size2={inputs.k_nope.size(2)} "
         f"dtype={inputs.k_nope.dtype} (expect 3,512,bf16)")
    assert inputs.k_pe.dim() == 3 and inputs.k_pe.size(2) == 64 and inputs.k_pe.dtype == torch.bfloat16, \
        f"k_pe: dim={inputs.k_pe.dim()} size2={inputs.k_pe.size(2)} dtype={inputs.k_pe.dtype} (expect 3,64,bf16)"

    assert inputs.value.dim() == 3 and inputs.value.size(2) == 512 and inputs.value.dtype == torch.bfloat16, \
        f"value: dim={inputs.value.dim()} size2={inputs.value.size(2)} dtype={inputs.value.dtype} (expect 3,512,bf16)"

    assert inputs.d_out.dim() == 3 and inputs.d_out.size(2) == 512 and inputs.d_out.dtype == torch.bfloat16, \
        f"d_out: dim={inputs.d_out.dim()} size2={inputs.d_out.size(2)} dtype={inputs.d_out.dtype} (expect 3,512,bf16)"
    assert inputs.out.dim() == 3 and inputs.out.size(2) == 512 and inputs.out.dtype == torch.bfloat16, \
        f"out: dim={inputs.out.dim()} size2={inputs.out.size(2)} dtype={inputs.out.dtype} (expect 3,512,bf16)"

    assert inputs.sm_max.dim() == 3 and inputs.sm_max.size(0) == 1 and inputs.sm_max.dtype == torch.float32, \
        f"sm_max: dim={inputs.sm_max.dim()} size0={inputs.sm_max.size(0)} dtype={inputs.sm_max.dtype} (expect 3,1,f32)"
    assert inputs.sm_sum.dim() == 3 and inputs.sm_sum.size(0) == 1 and inputs.sm_sum.dtype == torch.float32, \
        f"sm_sum: dim={inputs.sm_sum.dim()} size0={inputs.sm_sum.size(0)} dtype={inputs.sm_sum.dtype} (expect 3,1,f32)"


@allow_in_graph
def npu_sfa_sparse_attention_grad(q_nope, q_pe, k_nope, k_pe, value, sparse_idx, d_out, out, sm_max, sm_sum,
        actual_seq_qlen, actual_seq_kvlen, scale_value):
    assert not isinstance(q_nope, FakeTensor), f"q_nope is FakeTensor"

    # check_input_output_shape_dtype
    check_input_output_shape_dtype(CheckInputOutputShapeDtypeInputs(
        q_nope=q_nope, q_pe=q_pe, k_nope=k_nope, k_pe=k_pe, value=value, sparse_idx=sparse_idx,
        d_out=d_out, out=out, sm_max=sm_max, sm_sum=sm_sum,
        actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen))

    logging.info("****************** pypto npu_sfa_sparse_attention_grad *************************")

    t_1, n_1, d = q_nope.size(0), q_nope.size(1), q_nope.size(2)
    dr = q_pe.size(2)
    t_2, n_2 = k_nope.size(0), k_nope.size(1)
    dtype = q_nope.dtype
    k = sparse_idx.size(2)
    group = n_1 // n_2

    # 创建输出tensor
    dq_nope_out_shape = (t_1 * n_1, d)
    dq_pe_out_shape = (t_1 * n_1, dr)
    dk_nope_out_shape = (t_2 * n_2, d)
    dk_pe_out_shape = (t_2 * n_2, dr)
    dv_out_shape = (t_2 * n_2, d)
    dq_nope_out = torch.empty(dq_nope_out_shape, dtype=dtype, device=q_nope.device)
    dq_pe_out = torch.empty(dq_pe_out_shape, dtype=dtype, device=q_nope.device)
    dk_nope_out = torch.zeros(dk_nope_out_shape, dtype=torch.float32, device=q_nope.device)

    dk_pe_out = torch.zeros(dk_pe_out_shape, dtype=torch.float32, device=q_nope.device)
    dv_out = torch.zeros(dv_out_shape, dtype=torch.float32, device=q_nope.device)

    dk_out = torch.zeros((t_2 * n_2, d + dr), dtype=torch.float32, device=q_nope.device)

    pto_inputs = [q_nope, q_pe, k_nope, k_pe, value, sparse_idx, d_out, out, sm_max, sm_sum,
        actual_seq_qlen, actual_seq_kvlen, dq_nope_out, dq_pe_out, dk_nope_out, dk_pe_out, dv_out, 
        dk_nope_out, dk_pe_out, dk_out, dk_out, n_1, n_2, d, dr, k, group, scale_value]

    sparse_flash_attention_grad(*pto_inputs)

    dq_nope_out = torch.reshape(dq_nope_out, (t_1, n_1, d))
    dq_pe_out = torch.reshape(dq_pe_out, (t_1, n_1, dr))

    dk_nope_out = torch.reshape(dk_out[:, :d], (t_2, n_2, d)).to(dtype)
    dk_pe_out = torch.reshape(dk_out[:, d:], (t_2, n_2, dr)).to(dtype)
    dv_out = torch.reshape(dv_out, (t_2, n_2, d)).to(dtype)

    return dq_nope_out, dq_pe_out, dk_nope_out, dk_pe_out, dv_out
