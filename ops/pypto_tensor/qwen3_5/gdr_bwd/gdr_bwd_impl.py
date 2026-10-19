# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import math
import logging
import torch
import pypto

# NOTE: Single-letter variables (a_mat, batch, head_dim, n_heads, d_head, n_seqs, seq_len, d_value) are PyPTO kernel
# parameters. Functions with many arguments are inherent to the kernel API.

_LN2 = math.log(2.0)   # exp2(x) == exp(_LN2 * x): Triton's base-2 gate units


# ====================================================================================================
# SECTION 1 - PyPTO NPU kernel (device backward)
#
# Ported from custom/GDR/GDR_impl.py: `_gdr_chunk_forward` (GDR_impl.py:91-146) and
# `_gdr_backward_body` (GDR_impl.py:238-442).  The math is byte-for-byte the verified
# GDR_impl code apart from the four deliberate deltas marked "CHANGED vs GDR_impl":
#   (a) the gate arrives already chunk-cumsum'd (no in-kernel `tril_mask @ g_col`);
#   (b) MIXED PRECISION mirroring gdr_fwd: q/k/v/beta/do arrive as bf16; the elementwise/
#       vector math upcasts them to fp32, but each matmul runs on bf16 operands (fp32
#       accumulate), rounding fp32 intermediates back to bf16 right before the matmul.
#       gcum/rstd/a_mat, the sin_ws cache, the state carry and all gradient outputs stay
#       fp32; the reverse-cumsum `tril^seq_len @ d_gcum` (drives dg) is the one matmul kept fp32;
#   (c) ds is seeded from the caller's `dht` (per (b,h)) instead of a hardcoded zero;
#   (d) the post-loop ds is assembled out as `dh0`.
# ====================================================================================================


def _fwd_chunk_body(q_f, k_f, v_f, b_f, gcum_col, a_inv_tile,
                    last_state, sin_ws, ws_off, head_dim, bt):
    """One PASS-1 chunk: snapshot the ENTERING state to GM, advance forward state carry.

    w / v_new are computed here only to drive the state update -- they are NOT cached: PASS-2
    recomputes them locally from A_inv + s_i (see `_bwd_chunk_body`), which is bit-identical and
    removes the ~2/3-of-workspace `wv_ws` GM cache (and its per-chunk write + read of HBM)."""
    # q/k/v/beta arrive bf16 (mirrors gdr_fwd): upcast for the vector math, round each
    # matmul operand back to bf16 right before the call; caches/state stay fp32.
    k_f32 = pypto.cast(k_f, pypto.DT_FP32)                                   # [bt,head_dim]
    v_f32 = pypto.cast(v_f, pypto.DT_FP32)                                   # [bt,head_dim]
    b_f32 = pypto.cast(b_f, pypto.DT_FP32)                                   # [bt,1]
    v_beta = pypto.mul(v_f32, b_f32)                                         # [bt,head_dim]
    k_beta = pypto.mul(k_f32, b_f32)                                         # [bt,head_dim]
    gcum_exp = pypto.exp(gcum_col)                                           # [bt,1]
    k_beta_g = pypto.mul(k_beta, gcum_exp)                                   # [bt,head_dim]

    a_inv_bf = pypto.cast(a_inv_tile, pypto.DT_BF16)                # matmul operand (reused u/w)
    v_beta_bf = pypto.cast(v_beta, pypto.DT_BF16)
    k_beta_g_bf = pypto.cast(k_beta_g, pypto.DT_BF16)
    u = pypto.matmul(a_inv_bf, v_beta_bf, pypto.DT_FP32)                          # [bt,head_dim]
    w = pypto.matmul(a_inv_bf, k_beta_g_bf, pypto.DT_FP32)                        # [bt,head_dim]
    w_bf = pypto.cast(w, pypto.DT_BF16)
    ls_bf = pypto.cast(last_state, pypto.DT_BF16)
    v_prime = pypto.matmul(w_bf, ls_bf, pypto.DT_FP32)                       # [bt,head_dim] w @ S
    v_new = pypto.sub(u, v_prime)                                            # [bt,head_dim] NS#1 residual

    # snapshot ENTERING state (before update) -> sin_ws cache for pass 2 (the ONE quantity
    # PASS-2 cannot recompute: reversing the state recurrence would divide by an underflowing
    # decay).  w / v_new are recomputed in PASS-2 instead of cached.
    sin = last_state + 0.0                                           # [head_dim,head_dim]
    pypto.assemble(sin, [ws_off, 0], sin_ws)

    # state update (NS#4 single exp(g_last - g_cum) <= 0)
    g_last = gcum_col[bt - 1:bt, :]                                  # [1,1]
    decay_s = pypto.expand_clone(gcum_exp[bt - 1:bt, :], (head_dim, 1))     # [head_dim,1] e^{g_last}
    k_dec = pypto.mul(k_f32, pypto.exp(pypto.sub(g_last, gcum_col))) # [bt,head_dim]
    k_dec_bf = pypto.cast(k_dec, pypto.DT_BF16)
    v_new_bf = pypto.cast(v_new, pypto.DT_BF16)
    kdv = pypto.matmul(k_dec_bf, v_new_bf, pypto.DT_FP32, a_trans=True)  # [head_dim,head_dim]
    cur_state = pypto.add(pypto.mul(last_state, decay_s), kdv)       # [head_dim,head_dim]

    last_state[:] = cur_state                                        # in-place SERIAL carry


def _bwd_chunk_body(q_f, k_f, v_f, b_f, gcum_col, do_i, a_inv_tile, qr_f, kr_f,
                    off, act, tail, hc, h1, ws_off,
                    sin_ws, ds, trilc, maskc, elastc,
                    dq_out, dk_out, dv_out, dg_out, dbeta_out,
                    head_dim, bt, scale_val, use_rstd):
    s_i = pypto.view(sin_ws, [head_dim, head_dim], [ws_off, 0]) + 0.0                  # entering state (pass-1 cache)

    # ---- _chunk_forward_light inlined: cheap forward quantities ----
    gcum_row = pypto.transpose(gcum_col, 0, 1)                               # [1,bt]
    pypto.set_pass_options(sg_set_scope=1)
    # q/k/v/beta/do arrive bf16 (mirrors gdr_fwd): upcast to fp32 for the vector math,
    # then round each matmul operand back to bf16 right before the matmul (the舍入点).
    q_f32 = pypto.cast(q_f, pypto.DT_FP32)                                   # [bt,head_dim]
    k_f32 = pypto.cast(k_f, pypto.DT_FP32)                                   # [bt,head_dim]
    v_f32 = pypto.cast(v_f, pypto.DT_FP32)                                   # [bt,head_dim]
    b_f32 = pypto.cast(b_f, pypto.DT_FP32)                                   # [bt,1]
    v_beta = pypto.mul(v_f32, b_f32)                                         # [bt,head_dim]
    k_beta = pypto.mul(k_f32, b_f32)                                         # [bt,head_dim]
    g_diff = pypto.sub(gcum_col, gcum_row)                                   # [bt,bt]
    g_diff_l = pypto.mul(g_diff, trilc)
    l_mask = pypto.mul(pypto.exp(g_diff_l), trilc)                           # [bt,bt] mask-before-exp
    gcum_exp = pypto.exp(gcum_col)                                           # [bt,1]
    k_beta_g = pypto.mul(k_beta, gcum_exp)                                   # [bt,head_dim]
    q_s = pypto.mul(q_f32, scale_val)                                        # [bt,head_dim]
    pypto.set_pass_options(sg_set_scope=-1)

    # ---- recompute w / v_new (was the PASS-1 wv_ws GM cache) from A_inv + s_i; bit-for-bit
    #      identical to _fwd_chunk_body's u/w/v_new -- same operands, same bf16 rounding, and
    #      s_i == the fwd `last_state`.  a_inv_bf/v_beta_bf/k_beta_g_bf/w_bf/s_i_bf are the SAME
    #      operands the gradient matmuls below reuse, so they are cast ONCE here. ----
    a_inv_bf = pypto.cast(a_inv_tile, pypto.DT_BF16)                # matmul operand (reused 4x below)
    v_beta_bf = pypto.cast(v_beta, pypto.DT_BF16)                   # matmul operand (reused d_a_inv_1)
    k_beta_g_bf = pypto.cast(k_beta_g, pypto.DT_BF16)              # matmul operand (reused d_a_inv_2)
    s_i_bf = pypto.cast(s_i, pypto.DT_BF16)                         # matmul operand (reused d_qg/d_w); == fwd ls_bf
    u = pypto.matmul(a_inv_bf, v_beta_bf, pypto.DT_FP32)                          # [bt,head_dim]
    w = pypto.matmul(a_inv_bf, k_beta_g_bf, pypto.DT_FP32)                        # [bt,head_dim]
    w_bf = pypto.cast(w, pypto.DT_BF16)                             # matmul operand (reused ds_from_vprime)
    v_prime = pypto.matmul(w_bf, s_i_bf, pypto.DT_FP32)                     # [bt,head_dim] w @ s_i
    v_new = pypto.sub(u, v_prime)                                            # [bt,head_dim] NS#1 residual
    k_beta_bf = pypto.cast(k_beta, pypto.DT_BF16)                    # matmul operand (reused kkt/d_k_3)
    q_s_bf = pypto.cast(q_s, pypto.DT_BF16)                          # matmul operand (reused qkt/d_k_1)
    kkt = pypto.matmul(k_beta_bf, k_f, pypto.DT_FP32, b_trans=True)          # [bt,bt]
    qkt = pypto.matmul(q_s_bf, k_f, pypto.DT_FP32, b_trans=True)     # [bt,bt]

    pypto.set_pass_options(sg_set_scope=3)
    attn = pypto.mul(qkt, l_mask)                                    # [bt,bt]
    qg = pypto.mul(q_s, gcum_exp)                                    # [bt,head_dim]
    g_last = gcum_col[bt - 1:bt, :]                                  # [1,1]
    decay_k = pypto.exp(pypto.sub(g_last, gcum_col))                  # [bt,1] in (0,1]  (NS#4)
    k_dec = pypto.mul(k_f32, decay_k)                               # [bt,head_dim]
    decay_s_d1 = pypto.expand_clone(gcum_exp[bt - 1:bt, :], (head_dim, 1))  # [head_dim,1] e^{g_last}
    pypto.set_pass_options(sg_set_scope=-1)

    # ---- o backward ----  (do_i already bf16; round the fp32 operands to bf16)
    v_new_bf = pypto.cast(v_new, pypto.DT_BF16)                     # matmul operand (reused o/state)
    attn_bf = pypto.cast(attn, pypto.DT_BF16)
    qg_bf = pypto.cast(qg, pypto.DT_BF16)
    d_attn = pypto.matmul(do_i, v_new_bf, pypto.DT_FP32, b_trans=True)   # [bt,bt]
    d_v_new_1 = pypto.matmul(attn_bf, do_i, pypto.DT_FP32, a_trans=True) # [bt,head_dim]
    d_qg = pypto.matmul(do_i, s_i_bf, pypto.DT_FP32, b_trans=True)       # [bt,head_dim]
    ds_from_ointer = pypto.matmul(qg_bf, do_i, pypto.DT_FP32, a_trans=True) # [head_dim,head_dim]

    # ---- attn backward ----
    pypto.set_pass_options(sg_set_scope=4)
    d_qkt = pypto.mul(d_attn, l_mask)                              # [bt,bt]
    d_lmask_1 = pypto.mul(d_attn, qkt)                             # [bt,bt]
    pypto.set_pass_options(sg_set_scope=-1)
    d_qkt_bf = pypto.cast(d_qkt, pypto.DT_BF16)                    # matmul operand (reused d_q_s_1/d_k_1)
    d_q_s_1 = pypto.matmul(d_qkt_bf, k_f, pypto.DT_FP32)           # [bt,head_dim]
    d_k_1 = pypto.matmul(d_qkt_bf, q_s_bf, pypto.DT_FP32, a_trans=True)  # [bt,head_dim]

    # ---- qg backward ----
    pypto.set_pass_options(sg_set_scope=5)
    d_q_s_2 = pypto.mul(d_qg, gcum_exp)                            # [bt,head_dim]
    d_gcum_exp_1 = pypto.sum(pypto.mul(d_qg, q_s), -1, keepdim=True)   # [bt,1]
    pypto.set_pass_options(sg_set_scope=-1)

    # ---- state-update backward: vec (depends on ds only) then 2 matmuls ----
    pypto.set_pass_options(sg_set_scope=6)
    ds_from_decay = pypto.mul(ds, decay_s_d1)                      # [head_dim,head_dim]
    d_decay_s = pypto.sum(pypto.sum(pypto.mul(ds, s_i), -1, keepdim=True), 0, keepdim=True)  # [1,1]
    pypto.set_pass_options(sg_set_scope=-1)
    ds_bf = pypto.cast(ds, pypto.DT_BF16)                          # matmul operand (reused; entering ds)
    k_dec_bf = pypto.cast(k_dec, pypto.DT_BF16)
    d_k_dec = pypto.matmul(v_new_bf, ds_bf, pypto.DT_FP32, b_trans=True) # [bt,head_dim]
    d_v_new_2 = pypto.matmul(k_dec_bf, ds_bf, pypto.DT_FP32)             # [bt,head_dim]

    # ---- k_dec/v_new backward (vec, depends on d_k_dec/d_v_new results) ----
    pypto.set_pass_options(sg_set_scope=6)
    d_k_2 = pypto.mul(d_k_dec, decay_k)                            # [bt,head_dim]
    d_decay_k = pypto.sum(pypto.mul(d_k_dec, k_f32), -1, keepdim=True)   # [bt,1]
    d_arg = pypto.mul(d_decay_k, decay_k)                          # [bt,1] arg = g_last - g_cum
    d_gcum_col_1 = pypto.mul(d_arg, -1.0)                          # [bt,1]
    d_glast_1 = pypto.sum(d_arg, 0, keepdim=True)                  # [1,1]
    d_v_new = pypto.add(d_v_new_1, d_v_new_2)                      # [bt,head_dim]
    d_u = d_v_new                                                  # [bt,head_dim]
    d_v_prime = pypto.mul(d_v_new, -1.0)                           # [bt,head_dim]
    pypto.set_pass_options(sg_set_scope=-1)

    d_v_prime_bf = pypto.cast(d_v_prime, pypto.DT_BF16)             # matmul operand (reused d_w/dS_vprime)
    d_w = pypto.matmul(d_v_prime_bf, s_i_bf, pypto.DT_FP32, b_trans=True) + 0.0    # [bt,head_dim]
    ds_from_vprime = pypto.matmul(w_bf, d_v_prime_bf, pypto.DT_FP32, a_trans=True)  # [head_dim,head_dim]

    # ---- carry: ds_out = dL/dS_i ----
    ds_out = pypto.add(pypto.add(ds_from_decay, ds_from_ointer), ds_from_vprime)  # [head_dim,head_dim]

    # ---- u, w backward -> A_inv, v_beta, k_beta_g  (a_inv_bf/v_beta_bf/k_beta_g_bf cast above) ----
    d_u_bf = pypto.cast(d_u, pypto.DT_BF16)                       # matmul operand (reused)
    d_w_bf = pypto.cast(d_w, pypto.DT_BF16)                       # matmul operand (reused)
    d_a_inv_1 = pypto.matmul(d_u_bf, v_beta_bf, pypto.DT_FP32, b_trans=True)   # [bt,bt]
    d_v_beta = pypto.matmul(a_inv_bf, d_u_bf, pypto.DT_FP32, a_trans=True)     # [bt,head_dim]
    d_a_inv_2 = pypto.matmul(d_w_bf, k_beta_g_bf, pypto.DT_FP32, b_trans=True) # [bt,bt]
    d_k_beta_g = pypto.matmul(a_inv_bf, d_w_bf, pypto.DT_FP32, a_trans=True)   # [bt,head_dim]
    d_a_inv = pypto.add(d_a_inv_1, d_a_inv_2)                             # [bt,bt]

    # ---- A_inv backward -> a_mat (matrix-inverse rule); tmp -> next matmul, so keep it bf16 ----
    d_a_inv_bf = pypto.cast(d_a_inv, pypto.DT_BF16)
    tmp = pypto.matmul(a_inv_bf, d_a_inv_bf, pypto.DT_BF16, a_trans=True)
    d_a_full = pypto.matmul(tmp, a_inv_bf, pypto.DT_FP32, b_trans=True) + 0.0

    # ---- a_mat backward -> kkt, l_mask (mask = -1/0: strict-lower gate AND negation) ----
    pypto.set_pass_options(sg_set_scope=7)
    d_kl = pypto.mul(d_a_full, maskc)                              # [bt,bt]
    d_kkt = pypto.mul(d_kl, l_mask)                                # [bt,bt]
    d_lmask_2 = pypto.mul(d_kl, kkt)                               # [bt,bt]
    pypto.set_pass_options(sg_set_scope=-1)

    # ---- kkt backward -> k_beta, k ----
    d_kkt_bf = pypto.cast(d_kkt, pypto.DT_BF16)                    # matmul operand (reused d_k_beta_1/d_k_3)
    d_k_beta_1 = pypto.matmul(d_kkt_bf, k_f, pypto.DT_FP32)        # [bt,head_dim]
    d_k_3 = pypto.matmul(d_kkt_bf, k_beta_bf, pypto.DT_FP32, a_trans=True)  # [bt,head_dim]

    # ---- l_mask backward -> g_cum ----
    pypto.set_pass_options(sg_set_scope=8)
    d_lmask = pypto.add(d_lmask_1, d_lmask_2)                      # [bt,bt]
    d_e = pypto.mul(d_lmask, trilc)                                # outer tril
    d_gdiff_l = pypto.mul(d_e, l_mask)                             # * exp(g_diff_l)
    d_gdiff = pypto.mul(d_gdiff_l, trilc)                          # inner tril
    rowsum = pypto.sum(d_gdiff, -1, keepdim=True)                  # [bt,1]
    colsum = pypto.sum(pypto.transpose(d_gdiff, 0, 1), -1, keepdim=True)  # [bt,1]
    d_gcum_l = pypto.sub(rowsum, colsum)                           # [bt,1]

    # ---- k_beta_g backward -> k_beta, gcum_exp ----
    d_k_beta_2 = pypto.mul(d_k_beta_g, gcum_exp)                   # [bt,head_dim]
    d_gcum_exp_2 = pypto.sum(pypto.mul(d_k_beta_g, k_beta), -1, keepdim=True)  # [bt,1]
    # ---- gcum_exp total -> g_cum (fold decay_s at last row via e_last) ----
    d_gcum_exp = pypto.add(d_gcum_exp_1, d_gcum_exp_2)             # [bt,1]
    d_gcum_exp = pypto.add(d_gcum_exp, pypto.mul(elastc, d_decay_s))
    d_gcum_from_exp = pypto.mul(d_gcum_exp, gcum_exp)              # [bt,1]
    # ---- g_cum total ----
    d_gcum_col = pypto.add(d_gcum_l, d_gcum_col_1)
    d_gcum_col = pypto.add(d_gcum_col, d_gcum_from_exp)
    d_gcum_col = pypto.add(d_gcum_col, pypto.mul(elastc, d_glast_1))
    pypto.set_pass_options(sg_set_scope=-1)

    d_g_col = pypto.matmul(trilc, d_gcum_col, pypto.DT_FP32, a_trans=True)  # [bt,1]

    # ---- v_beta backward -> v, beta + k_beta backward -> k, beta + totals ----
    pypto.set_pass_options(sg_set_scope=9)
    d_v_f = pypto.mul(d_v_beta, b_f32)                            # [bt,head_dim]
    d_beta_1 = pypto.sum(pypto.mul(d_v_beta, v_f32), -1, keepdim=True)   # [bt,1]
    d_k_beta = pypto.add(d_k_beta_1, d_k_beta_2)                   # [bt,head_dim]
    d_k_4 = pypto.mul(d_k_beta, b_f32)                            # [bt,head_dim]
    d_beta_2 = pypto.sum(pypto.mul(d_k_beta, k_f32), -1, keepdim=True)   # [bt,1]
    d_q_s = pypto.add(d_q_s_1, d_q_s_2)                            # [bt,head_dim]
    d_q_f = pypto.mul(d_q_s, scale_val)                            # [bt,head_dim]
    d_k_f = pypto.add(pypto.add(d_k_1, d_k_2), pypto.add(d_k_3, d_k_4))  # [bt,head_dim]
    d_beta = pypto.add(d_beta_1, d_beta_2)                         # [bt,1]

    # ---- OPTIONAL: fold the L2norm VJP so dq/dk come back w.r.t. the RAW (pre-norm) q/k ----
    if use_rstd:
        dot_q = pypto.sum(pypto.mul(d_q_f, q_f32), -1, keepdim=True)        # [bt,1] dy.y
        d_q_f = pypto.mul(pypto.sub(d_q_f, pypto.mul(q_f32, dot_q)), qr_f)  # [bt,head_dim]
        dot_k = pypto.sum(pypto.mul(d_k_f, k_f32), -1, keepdim=True)        # [bt,1]
        d_k_f = pypto.mul(pypto.sub(d_k_f, pypto.mul(k_f32, dot_k)), kr_f)  # [bt,head_dim]

    # ---- carry update (in-place REVERSE serial carry) ----
    ds[:] = ds_out
    pypto.set_pass_options(sg_set_scope=-1)

    # ---- store 5 grads (tail: valid_shape-narrowed write prevents cross-seq corruption) ----
    # dq/dk/dv/db leave the kernel already in bf16 -- the SAME dtype the kernel takes q/k/v/beta
    # in (DT_BF16), which is exactly what `_aligned_cast_outputs` used to produce on the host as
    # a full fp32->bf16 HBM re-pass.  The on-chip cast folds that pass away (mirrors gdr_fwd's
    # bf16 `o_out`).  dg/dh0 stay fp32 (the fla contract for use_gate_in_kernel=False).
    d_q_o = pypto.cast(d_q_f, pypto.DT_BF16)
    d_k_o = pypto.cast(d_k_f, pypto.DT_BF16)
    d_v_o = pypto.cast(d_v_f, pypto.DT_BF16)
    d_beta_o = pypto.cast(d_beta, pypto.DT_BF16)
    if tail:
        pypto.assemble(pypto.view(d_q_o, [bt, head_dim], [0, 0], valid_shape=[act, head_dim]), [off, hc], dq_out)
        pypto.assemble(pypto.view(d_k_o, [bt, head_dim], [0, 0], valid_shape=[act, head_dim]), [off, hc], dk_out)
        pypto.assemble(pypto.view(d_v_o, [bt, head_dim], [0, 0], valid_shape=[act, head_dim]), [off, hc], dv_out)
        pypto.assemble(pypto.view(d_g_col, [bt, 1], [0, 0], valid_shape=[act, 1]), [off, h1], dg_out)
        pypto.assemble(pypto.view(d_beta_o, [bt, 1], [0, 0], valid_shape=[act, 1]), [off, h1], dbeta_out)
    else:
        pypto.assemble(d_q_o, [off, hc], dq_out)
        pypto.assemble(d_k_o, [off, hc], dk_out)
        pypto.assemble(d_v_o, [off, hc], dv_out)
        pypto.assemble(d_g_col, [off, h1], dg_out)
        pypto.assemble(d_beta_o, [off, h1], dbeta_out)


def _fused_backward_body(
    q, k, v, beta, gcum,
    states,
    do,
    q_rstd, k_rstd,
    a_inv_2d,
    mask, tril_mask, eye, dmask, e_last, ones_bt,
    dht,
    sin_ws,
    seqlens,
    dq_out, dk_out, dv_out, dg_out, dbeta_out, dh0_out,
    head_dim, bt, n_heads, nt_max, scale_val,
    use_rstd,
):
    n_seq = seqlens.shape[0] - 1

    last_state = pypto.tensor([head_dim, head_dim], pypto.DT_FP32)
    ds = pypto.tensor([head_dim, head_dim], pypto.DT_FP32)

    pypto.set_vec_tile_shapes(_VT0, _VT1)

    # ===== PASS 1: forward serial scan; snapshot entering state S_in[i] into sin_ws =====
    for nh in pypto.loop(n_seq * n_heads, name="fwd_seq", idx_name="nh"):
        n_idx = nh // n_heads
        h_idx = nh - n_idx * n_heads
        s0 = seqlens[n_idx]
        slen = seqlens[n_idx + 1] - s0
        hbase = s0
        hc = h_idx * head_dim
        h1 = h_idx
        hac = h_idx * bt

        pypto.set_cube_tile_shapes([64, _CT], [_CT, _CT], [_CT, _CT])
        for c in pypto.loop(0, slen, bt, name="fwd_chunk", idx_name="c", unroll_list=[4, 2, 1]):
            if pypto.cond(pypto.is_loop_begin(c)):
                last_state[:] = pypto.view(states, [head_dim, head_dim], [nh * head_dim, 0]) + 0.0
            off = hbase + c
            act = (slen - c).min(bt)
            slot = nh * nt_max + c // bt
            ws_off = slot * head_dim

            if pypto.cond(pypto.is_loop_end(c)):
                q_f = pypto.fillpad(
                    pypto.view(q, [bt, head_dim], [off, hc], valid_shape=[act, head_dim]),
                    "constant", 0.0)
                k_f = pypto.fillpad(
                    pypto.view(k, [bt, head_dim], [off, hc], valid_shape=[act, head_dim]),
                    "constant", 0.0)
                v_f = pypto.fillpad(
                    pypto.view(v, [bt, head_dim], [off, hc], valid_shape=[act, head_dim]),
                    "constant", 0.0)
                b_f = pypto.fillpad(pypto.view(beta, [bt, 1], [off, h1], valid_shape=[act, 1]), "constant", 0.0)
                one_col = pypto.view(ones_bt, [bt, 1], [0, 0])
                valid_m = pypto.fillpad(pypto.view(ones_bt, [bt, 1], [0, 0], valid_shape=[act, 1]), "constant", 0.0)
                pad_m = pypto.sub(one_col, valid_m)
                g_zero = pypto.fillpad(pypto.view(gcum, [bt, 1], [off, h1], valid_shape=[act, 1]), "constant", 0.0)
                g_last_pad = pypto.expand_clone(pypto.view(gcum, [1, 1], [off + act - 1, h1]), (bt, 1))
                gcum_col = pypto.add(g_zero, pypto.mul(g_last_pad, pad_m))
                a_inv_tile = pypto.fillpad(
                    pypto.view(a_inv_2d, [bt, bt], [off, hac], valid_shape=[act, bt]), "constant", 0.0)
                _fwd_chunk_body(q_f, k_f, v_f, b_f, gcum_col, a_inv_tile,
                                last_state, sin_ws, ws_off, head_dim, bt)
            else:
                q_f = pypto.view(q, [bt, head_dim], [off, hc])
                k_f = pypto.view(k, [bt, head_dim], [off, hc])
                v_f = pypto.view(v, [bt, head_dim], [off, hc])
                b_f = pypto.view(beta, [bt, 1], [off, h1])
                gcum_col = pypto.view(gcum, [bt, 1], [off, h1]) + 0.0
                a_inv_tile = pypto.view(a_inv_2d, [bt, bt], [off, hac])
                _fwd_chunk_body(q_f, k_f, v_f, b_f, gcum_col, a_inv_tile,
                                last_state, sin_ws, ws_off, head_dim, bt)

    # ===== PASS 2: backward reverse scan; carry ds; write the 5 grads + dh0 =====
    for nh in pypto.loop(n_seq * n_heads, name="bwd_seq", idx_name="nh"):
        n_idx = nh // n_heads
        h_idx = nh - n_idx * n_heads
        s0 = seqlens[n_idx]
        slen = seqlens[n_idx + 1] - s0
        hbase = s0
        hc = h_idx * head_dim
        h1 = h_idx
        hac = h_idx * bt
        nt_b = (slen + bt - 1) // bt

        pypto.set_cube_tile_shapes([64, _CT], [_CT, _CT], [_CT, _CT])
        for cj in pypto.loop(0, slen, bt, name="bwd_chunk", idx_name="cj", unroll_list=[4, 2, 1]):
            trilc = pypto.view(tril_mask, [bt, bt], [0, 0])
            maskc = pypto.view(mask, [bt, bt], [0, 0])
            elastc = pypto.view(e_last, [bt, 1], [0, 0])

            # ---- reverse chunk index: rev in {(nt_b-1)*bt, ..., 0} (always aligned) ----
            rev = (nt_b - 1) * bt - cj
            off = hbase + rev
            act = (slen - rev).min(bt)                                       # valid rows L ≤ bt
            slot = nh * nt_max + (nt_b - 1) - cj // bt
            ws_off = slot * head_dim

            # gate pad rows REPLICATE gcum[act-1] (not zero); q/k/v/beta/do pad rows are 0.
            if pypto.cond(pypto.is_loop_begin(cj)):
                ds[:] = pypto.view(dht, [head_dim, head_dim], [nh * head_dim, 0]) + 0.0
                q_f = pypto.fillpad(
                    pypto.view(q, [bt, head_dim], [off, hc], valid_shape=[act, head_dim]),
                    "constant", 0.0)
                k_f = pypto.fillpad(
                    pypto.view(k, [bt, head_dim], [off, hc], valid_shape=[act, head_dim]),
                    "constant", 0.0)
                v_f = pypto.fillpad(
                    pypto.view(v, [bt, head_dim], [off, hc], valid_shape=[act, head_dim]),
                    "constant", 0.0)
                b_f = pypto.fillpad(pypto.view(beta, [bt, 1], [off, h1], valid_shape=[act, 1]), "constant", 0.0)
                one_col = pypto.view(ones_bt, [bt, 1], [0, 0])
                valid_m = pypto.fillpad(pypto.view(ones_bt, [bt, 1], [0, 0], valid_shape=[act, 1]), "constant", 0.0)
                pad_m = pypto.sub(one_col, valid_m)
                g_zero = pypto.fillpad(pypto.view(gcum, [bt, 1], [off, h1], valid_shape=[act, 1]), "constant", 0.0)
                g_last_pad = pypto.expand_clone(pypto.view(gcum, [1, 1], [off + act - 1, h1]), (bt, 1))
                gcum_col = pypto.add(g_zero, pypto.mul(g_last_pad, pad_m))
                do_i = pypto.fillpad(
                    pypto.view(do, [bt, head_dim], [off, hc], valid_shape=[act, head_dim]),
                    "constant", 0.0)
                a_inv_tile = pypto.fillpad(
                    pypto.view(a_inv_2d, [bt, bt], [off, hac], valid_shape=[act, bt]), "constant", 0.0)
                if use_rstd:
                    qr_f = pypto.fillpad(pypto.view(q_rstd, [bt, 1], [off, h1], valid_shape=[act, 1]), "constant", 0.0)
                    kr_f = pypto.fillpad(pypto.view(k_rstd, [bt, 1], [off, h1], valid_shape=[act, 1]), "constant", 0.0)
                _bwd_chunk_body(q_f, k_f, v_f, b_f, gcum_col, do_i, a_inv_tile,
                                qr_f if use_rstd else None, kr_f if use_rstd else None,
                                off, act, True, hc, h1, ws_off,
                                sin_ws, ds, trilc, maskc, elastc,
                                dq_out, dk_out, dv_out, dg_out, dbeta_out,
                                head_dim, bt, scale_val, use_rstd)
            else:
                q_f = pypto.view(q, [bt, head_dim], [off, hc])
                k_f = pypto.view(k, [bt, head_dim], [off, hc])
                v_f = pypto.view(v, [bt, head_dim], [off, hc])
                b_f = pypto.view(beta, [bt, 1], [off, h1])
                gcum_col = pypto.view(gcum, [bt, 1], [off, h1]) + 0.0
                do_i = pypto.view(do, [bt, head_dim], [off, hc])
                a_inv_tile = pypto.view(a_inv_2d, [bt, bt], [off, hac])
                if use_rstd:
                    qr_f = pypto.view(q_rstd, [bt, 1], [off, h1]) + 0.0
                    kr_f = pypto.view(k_rstd, [bt, 1], [off, h1]) + 0.0
                _bwd_chunk_body(q_f, k_f, v_f, b_f, gcum_col, do_i, a_inv_tile,
                                qr_f if use_rstd else None, kr_f if use_rstd else None,
                                off, act, False, hc, h1, ws_off,
                                sin_ws, ds, trilc, maskc, elastc,
                                dq_out, dk_out, dv_out, dg_out, dbeta_out,
                                head_dim, bt, scale_val, use_rstd)

        # ---- after reverse chunk scan, ds = dL/d(initial_state) for this (n,h) ----
        pypto.assemble(ds, [nh * head_dim, 0], dh0_out)


# `_VT0 x _VT1` is the VECTOR tile.  The original (16, 64) split every [bt,head_dim]=[64,128]
# tensor into 8 tiles, so the whole backward ran at ~29% core utilisation with a 60%
# bubble rate -- the single largest loss in the op.  (64, 512) covers a full chunk row
# block in one tile; measured -38% wall on the varlen case, the biggest single win of the
# tuning pass.  Values above 64 on the ROW axis regress (a [bt,*] tile cannot exceed
# bt=64 rows without padding waste); 1024 on the column axis is within noise of 512.
_VT0, _VT1 = 128, 128
# device_sched_mode 0 beats 1 here: with the tiles above, each task is large enough that
# the more aggressive mode-1 reordering only adds scheduling latency.
_SCHED = 0
_CT = 128


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU,
                     "device_sched_mode": _SCHED,
                     "stitch_function_max_num": 128,
                     "launch_sched_aicpu_num": 3},
    pass_options={"vec_nbuffer_setting": {-2: 1, -1: 8},
                  "cube_nbuffer_setting": {-1: 8},
                  "cube_l1_reuse_setting": {-1: 8}},
    )


def gdr_bwd_fused_kernel_npu(
    q: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),          # [tt, n_heads*head_dim] bf16 token-major
    k: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),          # [tt, n_heads*head_dim] bf16
    v: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),          # [tt, n_heads*head_dim] bf16
    beta: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),       # [tt, n_heads] bf16
    gcum: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),       # [tt, n_heads] chunk-cumsum, natural
    states: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),   # [N*H*D, D] initial_state
    do: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),       # [tt, n_heads*head_dim] dL/do bf16
    q_rstd: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),     # [tt, n_heads] L2norm rstd(q)
    k_rstd: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),     # [tt, n_heads] L2norm rstd(k)
    a_inv_2d: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32), # A_inv from forward
    mask: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),       # [bt,bt] strict-lower -1/0
    tril_mask: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),  # [bt,bt] lower incl-diag 1/0
    eye: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),        # [bt,bt] identity
    dmask: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),      # [nlev*bt,bt] doubling masks
    e_last: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_FP32),                 # [bt,1] last-row indicator
    ones_bt: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_FP32),                # [bt,1] all-ones (pad mask)
    dht: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),      # dL/d(final_state)
    sin_ws: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),     # S_in cache
    seqlens: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),                  # [n_seqs+1] segment table
    # dq/dk/dv/dbeta come out bf16 -- the SAME dtype the kernel takes q/k/v/beta in (DT_BF16),
    # which is exactly what `_aligned_cast_outputs` used to produce.  Fixed here, no runtime knob.
    dq_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),     # [tt, n_heads*head_dim] bf16
    dk_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),     # [tt, n_heads*head_dim] bf16
    dv_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),     # [tt, n_heads*head_dim] bf16
    dg_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),     # [tt, n_heads]   fp32 (D14)
    dbeta_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),  # [tt, n_heads]   bf16
    dh0_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),    # [n_seqs*n_heads*head_dim, head_dim] fp32
    head_dim: int, bt: int, n_heads: int, nt_max: int, scale_val: float,                  # specialized non-tensor knobs
    use_rstd: bool,
):
    pypto.experimental.set_operation_options(combine_axis=True)
    _fused_backward_body(
        q, k, v, beta, gcum,
        states,
        do,
        q_rstd, k_rstd,
        a_inv_2d,
        mask, tril_mask, eye, dmask, e_last, ones_bt,
        dht,
        sin_ws,
        seqlens,
        dq_out, dk_out, dv_out, dg_out, dbeta_out, dh0_out,
        head_dim, bt, n_heads, nt_max, scale_val,
        use_rstd,
    )


# ---------------------------------------------------------------------------------------
# Host launcher -- layout / alloc / cast / reshape ONLY (no arithmetic beyond the exact
# `g * ln2` gate-unit rescale).  Adapted from GDR_impl.py:510-683.
# ---------------------------------------------------------------------------------------

_FUSED_CONST_CACHE = {}


def _fused_host_constants(bt, device):
    """Host-built 0/+-1 constant matrices.  Verbatim GDR_impl.py:565-585 + :650-651.

    MEMOIZED on ``(bt, device)``: every tensor returned here is a pure function of those
    two, is read-only on the device side (the kernel only ever ``pypto.view``s them), and
    is rebuilt identically on each call.  Reconstructing them per launch cost 2163us of a
    4816us backward -- 45% of the whole op -- because the `dmask` build alone issues
    ~6 levels x 4 elementwise NPU kernels plus two `arange`s.  Caching is bit-exact.
    """
    key = (int(bt), str(device))
    hit = _FUSED_CONST_CACHE.get(key)
    if hit is not None:
        return hit
    mask = torch.tril(-torch.ones(bt, bt, dtype=torch.float32, device=device), diagonal=-1)
    tril_mask = torch.ones(bt, bt, dtype=torch.float32, device=device).tril()
    eye = torch.eye(bt, dtype=torch.float32, device=device)

    n_lev = int(round(math.log2(bt)))
    r_idx = torch.arange(bt, device=device).view(bt, 1)
    c_idx = torch.arange(bt, device=device).view(1, bt)
    dm_levels = []
    for lvl in range(1, n_lev + 1):
        s = 1 << (lvl - 1)
        two_s = s << 1
        sel = (((r_idx // two_s) == (c_idx // two_s))
               & ((r_idx % two_s) >= s)
               & ((c_idx % two_s) < s)).to(torch.float32)
        dm_levels.append(sel)
    dmask = torch.cat(dm_levels, dim=0).contiguous()

    e_last = torch.zeros(bt, 1, dtype=torch.float32, device=device)
    e_last[bt - 1, 0] = 1.0
    # all-ones column: `fillpad` on a valid_shape-narrowed view of it yields the tail
    # chunk's 1/0 validity mask without an `arange` + boolean `where` (see `_pad_chunk_gate`).
    ones_bt = torch.ones(bt, 1, dtype=torch.float32, device=device)
    out = (mask, tril_mask, eye, dmask, e_last, ones_bt)
    _FUSED_CONST_CACHE[key] = out
    return out


def _varlen_seq_lens(cu_seqlens, seq_len):
    """Validate `cu_seqlens` (SPEC.md:160) and return the python list of segment lengths.

    Kept on the host: `cu_seqlens` is a small [n_seqs+1] int64 tensor, so the `.tolist()` sync
    is one tiny D2H copy per call and buys a fully static kernel launch shape.
    """
    cu = cu_seqlens.to(torch.int64).reshape(-1).tolist()                       # [n_seqs+1]
    if len(cu) < 2:
        raise ValueError(f"cu_seqlens must have at least 2 entries, got {cu}.")
    if cu[0] != 0 or cu[-1] != seq_len:
        raise ValueError(
            f"cu_seqlens must be a prefix sum starting at 0 and ending at seq_len={seq_len}, "
            f"got [{cu[0]}, ..., {cu[-1]}].")
    lens = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    for i, s in enumerate(lens):
        if s <= 0:
            raise ValueError(f"cu_seqlens must be strictly increasing; segment {i} has length {s}.")
    return cu, lens


# ====================================================================================================
# SECTION 2 - Triton-aligned drop-in wrapper (exact chunk_gated_delta_rule_bwd interface)
# ====================================================================================================


def _aligned_check_unsupported(state_v_first, cu_seqlens, cp_context, use_gate_in_kernel):
    if state_v_first:
        raise NotImplementedError("aligned wrapper: state_v_first=True is not supported.")
    # D6 (DESIGN.md:272,311): `cu_seqlens` (varlen) IS supported -- it is the sole driver of
    # the packed path; `chunk_indices` is accepted for signature parity and ignored.
    if cp_context is not None:
        raise NotImplementedError("aligned wrapper: cp_context (context-parallel) is not supported.")
    if use_gate_in_kernel:
        raise NotImplementedError("aligned wrapper: use_gate_in_kernel=True is not supported.")


def _fwd_chunk_body_950(k_f, v_f, b_f, gcum_col, a_inv_tile,
                    last_state, head_dim, bt):
    """One PASS-1 chunk: snapshot the ENTERING state to GM, advance forward state carry.

    w / v_new are computed here only to drive the state update -- they are NOT cached: PASS-2
    recomputes them locally from A_inv + s_i (see `_bwd_chunk_bodys`), which is bit-identical and
    removes the ~2/3-of-workspace `wv_ws` GM cache (and its per-chunk write + read of HBM)."""
    # q/k/v/beta arrive bf16 (mirrors gdr_fwd): upcast for the vector math, round each
    # matmul operand back to bf16 right before the call; caches/state stay fp32.

    pypto.set_pass_options(sg_set_scope=(20001, True, False))
    k_f32 = pypto.cast(k_f, pypto.DT_FP32)
    v_f32 = pypto.cast(v_f, pypto.DT_FP32)
    b_f32 = pypto.cast(b_f, pypto.DT_FP32)
    v_beta = pypto.mul(v_f32, b_f32)
    k_beta = pypto.mul(k_f32, b_f32)
    gcum_exp = pypto.exp(gcum_col)
    k_beta_g = pypto.mul(k_beta, gcum_exp)
    a_inv_bf = pypto.cast(a_inv_tile, pypto.DT_BF16)
    v_beta_bf = pypto.cast(v_beta, pypto.DT_BF16)
    k_beta_g_bf = pypto.cast(k_beta_g, pypto.DT_BF16)
    u = pypto.matmul(a_inv_bf, v_beta_bf, pypto.DT_FP32)
    w = pypto.matmul(a_inv_bf, k_beta_g_bf, pypto.DT_FP32)
    w_bf = pypto.cast(w, pypto.DT_BF16)
    g_last = gcum_col[bt - 1:bt, :]

    decay_s = pypto.expand_clone(gcum_exp[bt - 1:bt, :], (head_dim, 1))     # [head_dim,1] e^{g_last}
    k_dec = pypto.mul(k_f32, pypto.exp(pypto.sub(g_last, gcum_col))) # [bt,head_dim]
    k_dec_bf = pypto.cast(k_dec, pypto.DT_BF16)
    pypto.set_pass_options(sg_set_scope=-1)

    pypto.set_pass_options(sg_set_scope=20002)
    ls_bf = pypto.cast(last_state, pypto.DT_BF16)
    v_prime = pypto.matmul(w_bf, ls_bf, pypto.DT_FP32)                       # [bt,head_dim] w @ S
    v_new = pypto.sub(u, v_prime)
    v_new_bf = pypto.cast(v_new, pypto.DT_BF16)
    kdv = pypto.matmul(k_dec_bf, v_new_bf, pypto.DT_FP32, a_trans=True)  # [head_dim,head_dim]
    last_state_mul = pypto.mul(last_state, decay_s)
    cur_state = pypto.add(last_state_mul, kdv)       # [head_dim,head_dim]
    pypto.set_pass_options(sg_set_scope=-1)
    return cur_state


def _bwd_chunk_body_950(q_f, k_f, v_f, b_f, gcum_col, do_i, a_inv_tile, qr_f, kr_f,
                    ws_off, sin_ws, ds, trilc, maskc, elastc,
                    head_dim, bt, scale_val, use_rstd):
    # ---- _chunk_forward_light inlined: cheap forward quantities ----
    gcum_row = pypto.transpose(gcum_col, 0, 1)                               # [1,bt]
    # q/k/v/beta/do arrive bf16 (mirrors gdr_fwd): upcast to fp32 for the vector math,
    # then round each matmul operand back to bf16 right before the matmul (the舍入点).
    q_f32 = pypto.cast(q_f, pypto.DT_FP32)                                   # [bt,head_dim]
    k_f32 = pypto.cast(k_f, pypto.DT_FP32)                                   # [bt,head_dim]
    v_f32 = pypto.cast(v_f, pypto.DT_FP32)                                   # [bt,head_dim]
    b_f32 = pypto.cast(b_f, pypto.DT_FP32)                                   # [bt,1]
    v_beta = pypto.mul(v_f32, b_f32)                                         # [bt,head_dim]
    k_beta = pypto.mul(k_f32, b_f32)                                         # [bt,head_dim]
    g_diff = pypto.sub(gcum_col, gcum_row)                                   # [bt,bt]
    g_diff_l = pypto.mul(g_diff, trilc)
    l_mask = pypto.mul(pypto.exp(g_diff_l), trilc)                           # [bt,bt] mask-before-exp
    gcum_exp = pypto.exp(gcum_col)                                           # [bt,1]
    k_beta_g = pypto.mul(k_beta, gcum_exp)                                   # [bt,head_dim]
    q_s = pypto.mul(q_f32, scale_val)                                        # [bt,head_dim]

    # ---- recompute w / v_new (was the PASS-1 wv_ws GM cache) from A_inv + s_i; bit-for-bit
    a_inv_bf = pypto.cast(a_inv_tile, pypto.DT_BF16)                # matmul operand (reused 4x below)
    v_beta_bf = pypto.cast(v_beta, pypto.DT_BF16)                   # matmul operand (reused d_a_inv_1)
    k_beta_g_bf = pypto.cast(k_beta_g, pypto.DT_BF16)              # matmul operand (reused d_a_inv_2)
    u = pypto.matmul(a_inv_bf, v_beta_bf, pypto.DT_FP32)                          # [bt,head_dim]
    w = pypto.matmul(a_inv_bf, k_beta_g_bf, pypto.DT_FP32)                        # [bt,head_dim]
    w_bf = pypto.cast(w, pypto.DT_BF16)                             # matmul operand (reused ds_from_vprime)

    s_i = pypto.view(sin_ws, [head_dim, head_dim], [ws_off, 0])
    s_i_bf = pypto.cast(s_i, pypto.DT_BF16)                         # matmul operand (reused d_qg/d_w); == fwd ls_bf
    v_prime = pypto.matmul(w_bf, s_i_bf, pypto.DT_FP32)                     # [bt,head_dim] w @ s_i
    v_new = pypto.sub(u, v_prime)                                            # [bt,head_dim] NS#1 residual
    k_beta_bf = pypto.cast(k_beta, pypto.DT_BF16)                    # matmul operand (reused kkt/d_k_3)
    q_s_bf = pypto.cast(q_s, pypto.DT_BF16)                          # matmul operand (reused qkt/d_k_1)
    kkt = pypto.matmul(k_beta_bf, k_f, pypto.DT_FP32, b_trans=True)          # [bt,bt]
    qkt = pypto.matmul(q_s_bf, k_f, pypto.DT_FP32, b_trans=True)     # [bt,bt]

    attn = pypto.mul(qkt, l_mask)                                    # [bt,bt]
    qg = pypto.mul(q_s, gcum_exp)                                    # [bt,head_dim]
    g_last = gcum_col[bt - 1:bt, :]                                  # [1,1]
    decay_k = pypto.exp(pypto.sub(g_last, gcum_col))                  # [bt,1] in (0,1]  (NS#4)
    k_dec = pypto.mul(k_f32, decay_k)                               # [bt,head_dim]
    decay_s_d1 = pypto.expand_clone(gcum_exp[bt - 1:bt, :], (head_dim, 1))  # [head_dim,1] e^{g_last}

    # ---- o backward ----  (do_i already bf16; round the fp32 operands to bf16)
    v_new_bf = pypto.cast(v_new, pypto.DT_BF16)                     # matmul operand (reused o/state)
    attn_bf = pypto.cast(attn, pypto.DT_BF16)
    qg_bf = pypto.cast(qg, pypto.DT_BF16)
    d_attn = pypto.matmul(do_i, v_new_bf, pypto.DT_FP32, b_trans=True)   # [bt,bt]
    d_v_new_1 = pypto.matmul(attn_bf, do_i, pypto.DT_FP32, a_trans=True) # [bt,head_dim]
    d_qg = pypto.matmul(do_i, s_i_bf, pypto.DT_FP32, b_trans=True)       # [bt,head_dim]
    ds_from_ointer = pypto.matmul(qg_bf, do_i, pypto.DT_FP32, a_trans=True) # [head_dim,head_dim]

    # ---- attn backward ----
    d_qkt = pypto.mul(d_attn, l_mask)                              # [bt,bt]
    d_lmask_1 = pypto.mul(d_attn, qkt)                             # [bt,bt]
    d_qkt_bf = pypto.cast(d_qkt, pypto.DT_BF16)                    # matmul operand (reused d_q_s_1/d_k_1)
    d_q_s_1 = pypto.matmul(d_qkt_bf, k_f, pypto.DT_FP32)           # [bt,head_dim]
    d_k_1 = pypto.matmul(d_qkt_bf, q_s_bf, pypto.DT_FP32, a_trans=True)  # [bt,head_dim]

    # ---- qg backward ----
    d_q_s_2 = pypto.mul(d_qg, gcum_exp)                            # [bt,head_dim]
    d_gcum_exp_1 = pypto.sum(pypto.mul(d_qg, q_s), -1, keepdim=True)   # [bt,1]

    k_dec_bf = pypto.cast(k_dec, pypto.DT_BF16)
    d_q_s = pypto.add(d_q_s_1, d_q_s_2)                            # [bt,head_dim]
    d_q_f = pypto.mul(d_q_s, scale_val)                            # [bt,head_dim]
    if use_rstd:
        dot_q = pypto.sum(pypto.mul(d_q_f, q_f32), -1, keepdim=True)        # [bt,1] dy.y
        d_q_f = pypto.mul(pypto.sub(d_q_f, pypto.mul(q_f32, dot_q)), qr_f)  # [bt,head_dim]
    d_q_o = pypto.cast(d_q_f, pypto.DT_BF16)

    # ======================================================== all depend on ds:
    # ---- state-update backward: vec (depends on ds only) then 2 matmuls ----
    d_decay_s = pypto.sum(pypto.sum(pypto.mul(ds, s_i), -1, keepdim=True), 0, keepdim=True)  # [1,1]
    ds_bf = pypto.cast(ds, pypto.DT_BF16)                          # matmul operand (reused; entering ds)
    d_k_dec = pypto.matmul(v_new_bf, ds_bf, pypto.DT_FP32, b_trans=True) # [bt,head_dim]
    d_v_new_2 = pypto.matmul(k_dec_bf, ds_bf, pypto.DT_FP32)             # [bt,head_dim]

    # ---- k_dec/v_new backward (vec, depends on d_k_dec/d_v_new results) ----
    d_k_2 = pypto.mul(d_k_dec, decay_k)                            # [bt,head_dim]
    d_decay_k = pypto.sum(pypto.mul(d_k_dec, k_f32), -1, keepdim=True)   # [bt,1]
    d_arg = pypto.mul(d_decay_k, decay_k)                          # [bt,1] arg = g_last - g_cum
    d_gcum_col_1 = pypto.mul(d_arg, -1.0)                          # [bt,1]
    d_glast_1 = pypto.sum(d_arg, 0, keepdim=True)                  # [1,1]
    d_v_new = pypto.add(d_v_new_1, d_v_new_2)                      # [bt,head_dim]
    d_u = d_v_new                                                  # [bt,head_dim]
    d_v_prime = pypto.mul(d_v_new, -1.0)                           # [bt,head_dim]

    d_v_prime_bf = pypto.cast(d_v_prime, pypto.DT_BF16)             # matmul operand (reused d_w/dS_vprime)
    d_w = pypto.matmul(d_v_prime_bf, s_i_bf, pypto.DT_FP32, b_trans=True)    # [bt,head_dim]
    ds_from_vprime = pypto.matmul(w_bf, d_v_prime_bf, pypto.DT_FP32, a_trans=True)  # [head_dim,head_dim]

    # ---- carry: ds_out = dL/dS_i ----
    ds_from_decay = pypto.mul(ds, decay_s_d1)                      # [head_dim,head_dim]
    ds_from_add = pypto.add(ds_from_decay, ds_from_ointer)
    ds[:] = pypto.add(ds_from_add, ds_from_vprime)  # [head_dim,head_dim]

    # ---- u, w backward -> A_inv, v_beta, k_beta_g  (a_inv_bf/v_beta_bf/k_beta_g_bf cast above) ----
    d_u_bf = pypto.cast(d_u, pypto.DT_BF16)                       # matmul operand (reused)
    d_w_bf = pypto.cast(d_w, pypto.DT_BF16)                       # matmul operand (reused)
    d_a_inv_1 = pypto.matmul(d_u_bf, v_beta_bf, pypto.DT_FP32, b_trans=True)   # [bt,bt]
    d_v_beta = pypto.matmul(a_inv_bf, d_u_bf, pypto.DT_FP32, a_trans=True)     # [bt,head_dim]
    d_v_f = pypto.mul(d_v_beta, b_f32)                            # [bt,head_dim]
    d_v_o = pypto.cast(d_v_f, pypto.DT_BF16)
    d_beta_1 = pypto.sum(pypto.mul(d_v_beta, v_f32), -1, keepdim=True)   # [bt,1]

    d_a_inv_2 = pypto.matmul(d_w_bf, k_beta_g_bf, pypto.DT_FP32, b_trans=True) # [bt,bt]
    d_k_beta_g = pypto.matmul(a_inv_bf, d_w_bf, pypto.DT_FP32, a_trans=True)   # [bt,head_dim]
    d_k_beta_2 = pypto.mul(d_k_beta_g, gcum_exp)                   # [bt,head_dim]
    d_a_inv = pypto.add(d_a_inv_1, d_a_inv_2)                             # [bt,bt]

    # ---- A_inv backward -> a_mat (matrix-inverse rule); tmp -> next matmul, so keep it bf16 ----
    d_a_inv_bf = pypto.cast(d_a_inv, pypto.DT_BF16)
    tmp = pypto.matmul(a_inv_bf, d_a_inv_bf, pypto.DT_BF16, a_trans=True)
    d_a_full = pypto.matmul(tmp, a_inv_bf, pypto.DT_FP32, b_trans=True)

    # ---- a_mat backward -> kkt, l_mask (mask = -1/0: strict-lower gate AND negation) ----
    d_kl = pypto.mul(d_a_full, maskc)                              # [bt,bt]
    d_kkt = pypto.mul(d_kl, l_mask)                                # [bt,bt]
    d_lmask_2 = pypto.mul(d_kl, kkt)                               # [bt,bt]

    # ---- kkt backward -> k_beta, k ----
    d_kkt_bf = pypto.cast(d_kkt, pypto.DT_BF16)                    # matmul operand (reused d_k_beta_1/d_k_3)
    d_k_beta_1 = pypto.matmul(d_kkt_bf, k_f, pypto.DT_FP32)        # [bt,head_dim]
    d_k_beta = pypto.add(d_k_beta_1, d_k_beta_2)                   # [bt,head_dim]
    d_k_4 = pypto.mul(d_k_beta, b_f32)                            # [bt,head_dim]
    d_beta_2 = pypto.sum(pypto.mul(d_k_beta, k_f32), -1, keepdim=True)   # [bt,1]
    d_beta = pypto.add(d_beta_1, d_beta_2)                         # [bt,1]
    d_beta_o = pypto.cast(d_beta, pypto.DT_BF16)
    d_k_3 = pypto.matmul(d_kkt_bf, k_beta_bf, pypto.DT_FP32, a_trans=True)  # [bt,head_dim]
    d_k_f = pypto.add(pypto.add(d_k_1, d_k_2), pypto.add(d_k_3, d_k_4))  # [bt,head_dim]

    # ---- OPTIONAL: fold the L2norm VJP so dk comes back w.r.t. the RAW (pre-norm) k ----
    if use_rstd:
        dot_k = pypto.sum(pypto.mul(d_k_f, k_f32), -1, keepdim=True)        # [bt,1]
        d_k_f = pypto.mul(pypto.sub(d_k_f, pypto.mul(k_f32, dot_k)), kr_f)  # [bt,head_dim]
    d_k_o = pypto.cast(d_k_f, pypto.DT_BF16)

    # ---- l_mask backward -> g_cum ----
    d_lmask = pypto.add(d_lmask_1, d_lmask_2)                      # [bt,bt]
    d_e = pypto.mul(d_lmask, trilc)                                # outer tril
    d_gdiff_l = pypto.mul(d_e, l_mask)                             # * exp(g_diff_l)
    d_gdiff = pypto.mul(d_gdiff_l, trilc)                          # inner tril
    rowsum = pypto.sum(d_gdiff, -1, keepdim=True)                  # [bt,1]
    colsum = pypto.sum(pypto.transpose(d_gdiff, 0, 1), -1, keepdim=True)  # [bt,1]
    d_gcum_l = pypto.sub(rowsum, colsum)                           # [bt,1]

    # ---- k_beta_g backward -> k_beta, gcum_exp ----
    d_gcum_exp_2 = pypto.sum(pypto.mul(d_k_beta_g, k_beta), -1, keepdim=True)  # [bt,1]
    # ---- gcum_exp total -> g_cum (fold decay_s at last row via e_last) ----
    d_gcum_exp = pypto.add(d_gcum_exp_1, d_gcum_exp_2)             # [bt,1]
    d_gcum_exp = pypto.add(d_gcum_exp, pypto.mul(elastc, d_decay_s))
    d_gcum_from_exp = pypto.mul(d_gcum_exp, gcum_exp)              # [bt,1]
    # ---- g_cum total ----
    d_gcum_col = pypto.add(d_gcum_l, d_gcum_col_1)
    d_gcum_col = pypto.add(d_gcum_col, d_gcum_from_exp)
    d_gcum_col = pypto.add(d_gcum_col, pypto.mul(elastc, d_glast_1))
    d_g_col = pypto.matmul(trilc, d_gcum_col, pypto.DT_FP32, a_trans=True)  # [bt,1]
    return d_q_o, d_k_o, d_v_o, d_g_col, d_beta_o


def _fused_backward_body_950(
    q, k, v, beta, gcum,
    states,
    do,
    q_rstd, k_rstd,
    a_inv_2d,
    mask, tril_mask, eye, dmask, e_last, ones_bt,
    dht,
    sin_ws,
    seqlens,
    dq_out, dk_out, dv_out, dg_out, dbeta_out, dh0_out,
    head_dim, bt, n_heads, nt_max, scale_val,
    use_rstd,
):
    n_seq = seqlens.shape[0] - 1
    last_state = pypto.tensor([head_dim, head_dim], pypto.DT_FP32)
    ds = pypto.tensor([head_dim, head_dim], pypto.DT_FP32)
    pypto.set_vec_tile_shapes(_FVT0, _FVT1)
    pypto.set_cube_tile_shapes([_FC_MT, _FC_MT], [_FC_KNT, _FC_KNT], [_FC_KNT, _FC_KNT])

    # ===== FUSED: forward + backward per (seq, head) =====
    for nh in pypto.loop(n_seq * n_heads, name="fused_seq", idx_name="nh"):
        n_idx = nh // n_heads
        h_idx = nh - n_idx * n_heads
        s0 = seqlens[n_idx]
        slen = seqlens[n_idx + 1] - s0
        hbase = s0
        hc = h_idx * head_dim
        h1 = h_idx
        hac = h_idx * bt
        nt_b = (slen + bt - 1) // bt

        # ----- PASS 1: forward serial scan; snapshot entering state S_in[i] into sin_ws -----
        last_state[:] = pypto.view(states, [head_dim, head_dim], [nh * head_dim, 0])
        for c in pypto.loop(0, slen, bt, name="fwd_chunk", idx_name="cf", unroll_list=[16]):
            off = hbase + c
            act = (slen - c).min(bt)
            slot = nh * nt_max + c // bt
            ws_off = slot * head_dim

            if pypto.cond(pypto.is_loop_end(c)):
                pypto.set_pass_options(sg_set_scope=(20001, True, False))
                k_f = pypto.fillpad(
                    pypto.view(k, [bt, head_dim], [off, hc], valid_shape=[act, head_dim]),
                    "constant", 0.0)
                v_f = pypto.fillpad(
                    pypto.view(v, [bt, head_dim], [off, hc], valid_shape=[act, head_dim]),
                    "constant", 0.0)
                b_f = pypto.fillpad(pypto.view(beta, [bt, 1], [off, h1], valid_shape=[act, 1]), "constant", 0.0)
                one_col = pypto.view(ones_bt, [bt, 1], [0, 0])
                valid_m = pypto.fillpad(pypto.view(ones_bt, [bt, 1], [0, 0], valid_shape=[act, 1]), "constant", 0.0)
                pad_m = pypto.sub(one_col, valid_m)
                g_zero = pypto.fillpad(pypto.view(gcum, [bt, 1], [off, h1], valid_shape=[act, 1]), "constant", 0.0)
                g_last_pad = pypto.expand_clone(pypto.view(gcum, [1, 1], [off + act - 1, h1]), (bt, 1))
                gcum_col = pypto.add(g_zero, pypto.mul(g_last_pad, pad_m))
                a_inv_tile = pypto.fillpad(
                    pypto.view(a_inv_2d, [bt, bt], [off, hac], valid_shape=[act, bt]), "constant", 0.0)

                k_f32 = pypto.cast(k_f, pypto.DT_FP32)
                v_f32 = pypto.cast(v_f, pypto.DT_FP32)
                b_f32 = pypto.cast(b_f, pypto.DT_FP32)
                v_beta = pypto.mul(v_f32, b_f32)
                k_beta = pypto.mul(k_f32, b_f32)
                gcum_exp = pypto.exp(gcum_col)
                k_beta_g = pypto.mul(k_beta, gcum_exp)
                a_inv_bf = pypto.cast(a_inv_tile, pypto.DT_BF16)
                v_beta_bf = pypto.cast(v_beta, pypto.DT_BF16)
                k_beta_g_bf = pypto.cast(k_beta_g, pypto.DT_BF16)
                u = pypto.matmul(a_inv_bf, v_beta_bf, pypto.DT_FP32)
                w = pypto.matmul(a_inv_bf, k_beta_g_bf, pypto.DT_FP32)
                w_bf = pypto.cast(w, pypto.DT_BF16)
                g_last = gcum_col[bt - 1:bt, :]
                decay_s = pypto.expand_clone(gcum_exp[bt - 1:bt, :], (head_dim, 1))     # [head_dim,1] e^{g_last}
                k_dec = pypto.mul(k_f32, pypto.exp(pypto.sub(g_last, gcum_col))) # [bt,head_dim]
                k_dec_bf = pypto.cast(k_dec, pypto.DT_BF16)
                pypto.set_pass_options(sg_set_scope=-1)

                pypto.set_pass_options(sg_set_scope=20002)
                ls_bf = pypto.cast(last_state, pypto.DT_BF16)
                v_prime = pypto.matmul(w_bf, ls_bf, pypto.DT_FP32)                       # [bt,head_dim] w @ S
                v_new = pypto.sub(u, v_prime)
                v_new_bf = pypto.cast(v_new, pypto.DT_BF16)
                kdv = pypto.matmul(k_dec_bf, v_new_bf, pypto.DT_FP32, a_trans=True)  # [head_dim,head_dim]
                last_state_mul = pypto.mul(last_state, decay_s)
                cur_state = pypto.add(last_state_mul, kdv)       # [head_dim,head_dim]
                pypto.set_pass_options(sg_set_scope=-1)
                pypto.assemble(last_state, [ws_off, 0], sin_ws)
                last_state[:] = cur_state
            else:
                k_f = pypto.view(k, [bt, head_dim], [off, hc])
                v_f = pypto.view(v, [bt, head_dim], [off, hc])
                b_f = pypto.view(beta, [bt, 1], [off, h1])
                gcum_col = pypto.view(gcum, [bt, 1], [off, h1])
                a_inv_tile = pypto.view(a_inv_2d, [bt, bt], [off, hac])
                cur_state = _fwd_chunk_body_950(k_f, v_f, b_f, gcum_col, a_inv_tile,
                                last_state, head_dim, bt)
                pypto.assemble(last_state, [ws_off, 0], sin_ws)
                last_state[:] = cur_state

        # ----- PASS 2: backward reverse scan; carry ds; write the 5 grads + dh0 -----
        pypto.set_vec_tile_shapes(_BVT0, _BVT1)
        pypto.set_cube_tile_shapes([_BC_MT, _BC_MT], [_BC_KNT, _BC_KNT], [_BC_KNT, _BC_KNT])
        ds[:] = pypto.view(dht, [head_dim, head_dim], [nh * head_dim, 0])
        for c in pypto.loop(0, slen, bt, name="bwd_chunk", idx_name="cb", unroll_list=[16]):
            # ---- reverse chunk index: rev in {(nt_b-1)*bt, ..., 0} (always aligned) ----
            rev = (nt_b - 1) * bt - c
            off_b = hbase + rev
            act_b = (slen - rev).min(bt)                                       # valid rows L ≤ bt
            slot_b = nh * nt_max + (nt_b - 1) - c // bt
            ws_off_b = slot_b * head_dim

            trilc = pypto.view(tril_mask, [bt, bt], [0, 0])
            maskc = pypto.view(mask, [bt, bt], [0, 0])
            elastc = pypto.view(e_last, [bt, 1], [0, 0])

            # gate pad rows REPLICATE gcum[act_b-1] (not zero); q/k/v/beta/do pad rows are 0.
            if pypto.cond(pypto.is_loop_begin(c)):
                q_f = pypto.fillpad(
                    pypto.view(q, [bt, head_dim], [off_b, hc], valid_shape=[act_b, head_dim]),
                    "constant", 0.0)
                k_f = pypto.fillpad(
                    pypto.view(k, [bt, head_dim], [off_b, hc], valid_shape=[act_b, head_dim]),
                    "constant", 0.0)
                v_f = pypto.fillpad(
                    pypto.view(v, [bt, head_dim], [off_b, hc], valid_shape=[act_b, head_dim]),
                    "constant", 0.0)
                b_f = pypto.fillpad(pypto.view(beta, [bt, 1], [off_b, h1], valid_shape=[act_b, 1]), "constant", 0.0)
                one_col = pypto.view(ones_bt, [bt, 1], [0, 0])
                valid_m = pypto.fillpad(pypto.view(ones_bt, [bt, 1], [0, 0], valid_shape=[act_b, 1]), "constant", 0.0)
                pad_m = pypto.sub(one_col, valid_m)
                g_zero = pypto.fillpad(pypto.view(gcum, [bt, 1], [off_b, h1], valid_shape=[act_b, 1]), "constant", 0.0)
                g_last_pad = pypto.expand_clone(pypto.view(gcum, [1, 1], [off_b + act_b - 1, h1]), (bt, 1))
                gcum_col = pypto.add(g_zero, pypto.mul(g_last_pad, pad_m))
                do_i = pypto.fillpad(
                    pypto.view(do, [bt, head_dim], [off_b, hc], valid_shape=[act_b, head_dim]),
                    "constant", 0.0)
                a_inv_tile = pypto.fillpad(
                    pypto.view(a_inv_2d, [bt, bt], [off_b, hac], valid_shape=[act_b, bt]), "constant", 0.0)
                if use_rstd:
                    qr_f = pypto.fillpad(pypto.view(q_rstd, [bt, 1], [off_b, h1],
                                                    valid_shape=[act_b, 1]), "constant", 0.0)
                    kr_f = pypto.fillpad(pypto.view(k_rstd, [bt, 1], [off_b, h1],
                                                    valid_shape=[act_b, 1]), "constant", 0.0)
                d_q_o, d_k_o, d_v_o, d_g_col, d_beta_o = _bwd_chunk_body_950(q_f, k_f, v_f, b_f, gcum_col, do_i,
                                a_inv_tile, qr_f if use_rstd else None, kr_f if use_rstd else None,
                                ws_off_b, sin_ws, ds, trilc, maskc, elastc,
                                head_dim, bt, scale_val, use_rstd)

                pypto.assemble(pypto.view(d_q_o, [bt, head_dim], [0, 0],
                                          valid_shape=[act_b, head_dim]), [off_b, hc], dq_out)
                pypto.assemble(pypto.view(d_k_o, [bt, head_dim], [0, 0],
                                          valid_shape=[act_b, head_dim]), [off_b, hc], dk_out)
                pypto.assemble(pypto.view(d_v_o, [bt, head_dim], [0, 0],
                                          valid_shape=[act_b, head_dim]), [off_b, hc], dv_out)
                pypto.assemble(pypto.view(d_g_col, [bt, 1], [0, 0],
                                          valid_shape=[act_b, 1]), [off_b, h1], dg_out)
                pypto.assemble(pypto.view(d_beta_o, [bt, 1], [0, 0],
                                          valid_shape=[act_b, 1]), [off_b, h1], dbeta_out)
            else:
                q_f = pypto.view(q, [bt, head_dim], [off_b, hc])
                k_f = pypto.view(k, [bt, head_dim], [off_b, hc])
                v_f = pypto.view(v, [bt, head_dim], [off_b, hc])
                b_f = pypto.view(beta, [bt, 1], [off_b, h1])
                gcum_col = pypto.view(gcum, [bt, 1], [off_b, h1])
                do_i = pypto.view(do, [bt, head_dim], [off_b, hc])
                a_inv_tile = pypto.view(a_inv_2d, [bt, bt], [off_b, hac])
                if use_rstd:
                    qr_f = pypto.view(q_rstd, [bt, 1], [off_b, h1])
                    kr_f = pypto.view(k_rstd, [bt, 1], [off_b, h1])
                d_q_o, d_k_o, d_v_o, d_g_col, d_beta_o = _bwd_chunk_body_950(q_f, k_f, v_f, b_f, gcum_col, do_i,
                                a_inv_tile, qr_f if use_rstd else None, kr_f if use_rstd else None,
                                ws_off_b, sin_ws, ds, trilc, maskc, elastc,
                                head_dim, bt, scale_val, use_rstd)
                pypto.assemble(d_q_o, [off_b, hc], dq_out)
                pypto.assemble(d_k_o, [off_b, hc], dk_out)
                pypto.assemble(d_v_o, [off_b, hc], dv_out)
                pypto.assemble(d_g_col, [off_b, h1], dg_out)
                pypto.assemble(d_beta_o, [off_b, h1], dbeta_out)

        # ---- after reverse chunk scan, ds = dL/d(initial_state) for this (n,h) ----
        pypto.assemble(ds, [nh * head_dim, 0], dh0_out)


_FVT0, _FVT1 = 64, 128
_FC_MT, _FC_KNT = 64, 128
_BVT0, _BVT1 = 128, 128
_BC_MT, _BC_KNT = 128, 128
_SCHED_950 = 1


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU,
                     "device_sched_mode": _SCHED_950,
                     "stitch_function_max_num": 284,
                     "launch_sched_aicpu_num": 3,
                     "max_workspace_kb": 1003024
                     },
    pass_options={"vec_nbuffer_setting": {-2: 1, -1: 4},
                  "cube_nbuffer_setting": {-1: 4},
                  "cube_l1_reuse_setting": {-1: 8},
                  },
    host_options={"compile_monitor_enable": 0}
    )
def gdr_bwd_fused_kernel_npu_950(
    q: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),          # [tt, n_heads*head_dim] bf16 token-major
    k: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),          # [tt, n_heads*head_dim] bf16
    v: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),          # [tt, n_heads*head_dim] bf16
    beta: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),       # [tt, n_heads] bf16
    gcum: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),       # [tt, n_heads] chunk-cumsum, natural
    states: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),   # [N*H*D, D] initial_state
    do: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),       # [tt, n_heads*head_dim] dL/do bf16
    q_rstd: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),     # [tt, n_heads] L2norm rstd(q)
    k_rstd: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),     # [tt, n_heads] L2norm rstd(k)
    a_inv_2d: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32), # A_inv from forward
    mask: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),       # [bt,bt] strict-lower -1/0
    tril_mask: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),  # [bt,bt] lower incl-diag 1/0
    eye: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),        # [bt,bt] identity
    dmask: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),      # [nlev*bt,bt] doubling masks
    e_last: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_FP32),                 # [bt,1] last-row indicator
    ones_bt: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_FP32),                # [bt,1] all-ones (pad mask)
    dht: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),      # dL/d(final_state)
    sin_ws: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),     # S_in cache
    seqlens: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),                  # [n_seqs+1] segment table
    dq_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),     # [tt, n_heads*head_dim] bf16
    dk_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),     # [tt, n_heads*head_dim] bf16
    dv_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),     # [tt, n_heads*head_dim] bf16
    dg_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),     # [tt, n_heads]   fp32 (D14)
    dbeta_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),  # [tt, n_heads]   bf16
    dh0_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),    # [n_seqs*n_heads*head_dim, head_dim] fp32
    head_dim: int, bt: int, n_heads: int, nt_max: int, scale_val: float,                  # specialized non-tensor knobs
    use_rstd: bool,
):
    pypto.experimental.auto_mix_partition(1)
    pypto.experimental.set_operation_options(combine_axis=True)
    _fused_backward_body_950(
        q, k, v, beta, gcum,
        states,
        do,
        q_rstd, k_rstd,
        a_inv_2d,
        mask, tril_mask, eye, dmask, e_last, ones_bt,
        dht,
        sin_ws,
        seqlens,
        dq_out, dk_out, dv_out, dg_out, dbeta_out, dh0_out,
        head_dim, bt, n_heads, nt_max, scale_val,
        use_rstd,
    )


# NOTE: the former `_aligned_cast_outputs` host re-pass is GONE -- the kernel now writes the
# final chunk_gated_delta_rule_bwd dtypes directly (dq/dk/dv/db bf16 via the on-chip cast --
# bf16 IS the kernel's q/k/v/beta dtype; dg/dh0 fp32).  The fla contract it encoded, for the
# record:
#   dq, dk, dv -> q/k/v dtype (== bf16, the kernel's input dtype)   (fla chunk_o.py:729-730)
#   db         -> beta.dtype  (== bf16)  (torch.empty_like(beta), wy_fast.py:301)
#   dg         -> fp32 here, None when g is None           (use_gate_in_kernel=False; D14)
#   dh0        -> fp32 when initial_state is given, else None (chunk_delta_h.py:700)
# Only the two None-semantics survive, applied inline in the wrapper below.


def chunk_gated_delta_rule_backward_wrapper(
    q, k, v, g, beta, a_mat, scale, initial_state, do, dht,
    state_v_first=False, cu_seqlens=None, cp_context=None, chunk_indices=None,
    use_gate_in_kernel=False, g_input=None, a_log=None, dt_bias=None, chunk_size=64,
    *, g_is_natural_cumsum=False, q_rstd=None, k_rstd=None,
):
    """PyPTO NPU-kernel path, presenting the exact chunk_gated_delta_rule_bwd interface.

    PRODUCTION path: everything -- input preprocessing (layout ONLY), workspace / output
    allocation, the kernel launch and the output reshape -- lives right here in the wrapper;
    there is no `_fused_backward_launch` indirection.  Input preprocessing does NOT convert
    dtype: it VALIDATES that each input already has the dtype the kernel declares (q/k/v/beta/do
    bf16; g/a_mat/initial_state/dht/q_rstd/k_rstd fp32) and only reshapes the token axis, so a
    mismatched-dtype input raises rather than being silently re-cast.  The lone value transform
    kept is the exact `g * ln2` gate-unit rescale (fp32->fp32, not a dtype change).  A_inv is
    passed from the forward pass (``a_mat``, no longer recomputed on device).  `chunk_indices`,
    `g_input`, `a_log`, `dt_bias` are accepted for signature parity and ignored.  `dht` is
    NOT modified.

    varlen (`cu_seqlens is not None`, D6): `q.shape[0]` must be 1 and the token axis is the
    packed `sum(T_i)`; `initial_state` / `dht` are indexed by sequence (first dim == n_seqs ==
    len(cu_seqlens)-1) and `dh0` comes back as [n_seqs, hv, d_head, d_value].

    Keyword-only extensions (appended AFTER the 19 positional inputs, so
    `eval/test_inputs.py::PRIMARY_INPUT_ORDER` and upstream fla call sites are unaffected):

    `g_is_natural_cumsum=False`
        `g` is the fla gate, i.e. `RCP_LN2 * chunk_local_cumsum(g_nat)`; the host does one
        exact `* ln2` to recover natural-log units.  Set **True** when the caller already
        holds the natural-log chunk-local inclusive cumsum -- then that multiply is skipped
        and `g` is consumed as-is, removing the host unit round trip.  `dg` is the gradient
        w.r.t. the PRE-cumsum per-token gate in BOTH cases (unchanged semantics).

    `q_rstd` / `k_rstd` : [batch,seq_len,n_heads] fp32, or None
        The `rstd` of the L2norm that produced the `q`/`k` being passed in (so `q` is the
        normalized `y`).  When BOTH are given the kernel folds the L2norm VJP
        `dx = rstd * (dy - (dy.y) * y)` into `dq`/`dk` before storing, so the caller gets
        gradients w.r.t. the RAW pre-normalization q/k -- no host torch elementwise pass.
        When both are None the result is byte-for-byte what it was before this feature.
        Passing exactly one is a ValueError.
    """
    _aligned_check_unsupported(state_v_first, cu_seqlens, cp_context, use_gate_in_kernel)
    if a_mat is None:
        raise ValueError("aligned wrapper: a_mat (from forward pass) must be provided; "
                         "the kernel no longer recomputes A_inv on device.")
    if (q_rstd is None) != (k_rstd is None):
        raise ValueError(
            "aligned wrapper: q_rstd and k_rstd must be given together; "
            f"got q_rstd={'None' if q_rstd is None else 'tensor'}, "
            f"k_rstd={'None' if k_rstd is None else 'tensor'}.")
    batch, seq_len, n_heads, d_head = q.shape
    hv = v.shape[2]
    d_value = v.shape[-1]
    if hv != n_heads:
        raise NotImplementedError(f"aligned wrapper: GVA (hv={hv} != n_heads={n_heads}) is not supported.")
    if d_value != d_head:
        raise NotImplementedError(
            f"aligned wrapper: d_head == d_value required "
            f"(got d_head={d_head}, d_value={d_value}).")
    head_dim = d_head

    bt = int(chunk_size)
    if bt & (bt - 1):
        raise NotImplementedError(f"chunk_size must be a power of two (got {bt}).")
    device = k.device
    f32 = torch.float32
    bf16 = torch.bfloat16
    scale_val = float(scale) if scale is not None else (head_dim ** -0.5)
    tt = batch * seq_len                                                    # total tokens on the flat axis

    # -- segment table (validated) + NT_max, host-side python ints.  ONE code path: equal length
    #    is just the table [0, seq_len, 2T, ..., batch*seq_len]; varlen is the cu_seqlens prefix sum (D6).  The
    #    kernel's inner `pypto.loop(0, slen, bt)` derives each sequence's chunking on device, so
    #    there is no host packing / padding on either path. --
    if cu_seqlens is None:
        seqlens_l = [i * seq_len for i in range(batch + 1)]
        seq_lens = [seq_len] * batch
    else:
        if batch != 1:
            raise ValueError(f"varlen requires q.shape[0] == 1 (packed token axis), got batch={batch}.")
        seqlens_l, seq_lens = _varlen_seq_lens(cu_seqlens, seq_len)     # shape/monotonicity/prefix-sum
    n_seqs = len(seq_lens)
    if cu_seqlens is not None:
        for _name, _t in (("initial_state", initial_state), ("dht", dht)):
            if _t is not None and _t.shape[0] != n_seqs:
                raise ValueError(
                    f"varlen: {_name}.shape[0] must be n_seqs={n_seqs} (= len(cu_seqlens)-1), "
                    f"got {_t.shape[0]}.")
        # `chunk_indices` is deliberately NOT consumed (D6, DESIGN.md:311): the chunking is
        # derived from `cu_seqlens` alone, so passing None must give identical results.
    num_heads = n_seqs * n_heads
    nt_max = max((s + bt - 1) // bt for s in seq_lens)

    # ============ input preprocessing: VALIDATE dtype (no conversion), reshape ONLY ============
    # The kernel declares q/k/v/beta/do as bf16 and gate/state/a_mat/rstd as fp32; we assert the
    # caller already passed those dtypes and only lay out the token axis.  `.contiguous()
    # .reshape(...)` is a free view for the already-contiguous production inputs.  Layout:
    # [batch,seq_len,n_heads,head_dim] -> [tt, n_heads*head_dim] and
# [batch,seq_len,n_heads] -> [tt, n_heads] are pure reshapes (TOKEN-major, no transpose).
    def _val(x, name, want):
        if x.dtype != want:
            raise ValueError(
                f"aligned wrapper: `{name}` must be {want}, got {x.dtype}.  Preprocessing "
                "validates dtype and does NOT convert -- pass the tensor already in that dtype.")
        return x

    def _flat4(x, name, want):                          # [B,T,H,D] -> [tt, n_heads*head_dim]
        return _val(x, name, want).contiguous().reshape(tt, n_heads * head_dim)

    def _flat3(x, name, want):                                    # [batch,seq_len,n_heads]   -> [tt, n_heads]
        return _val(x, name, want).contiguous().reshape(tt, n_heads)

    with torch.no_grad():
        # q/k/v/beta/do enter AS bf16 (kernel DT_BF16); each tile is upcast to fp32 on-chip
        # before any arithmetic, so the compute stays fp32.
        q2d = _flat4(q, "q", bf16)
        k2d = _flat4(k, "k", bf16)
        v2d = _flat4(v, "v", bf16)
        do2d = _flat4(do, "do", bf16)
        beta2d = _flat3(beta, "beta", bf16)
        # natural-log cumulative gate the device math wants (skipped when the caller already
        # holds natural-log units).  This is an fp32->fp32 value rescale, NOT a dtype change.
        gcum2d = _flat3(g, "g", f32)
        if not g_is_natural_cumsum:
            gcum2d = gcum2d * _LN2
        # L2NORM VJP rstd: both-or-neither (checked above).  The unused placeholder aliases the
        # fp32 `gcum2d` (read-guarded by `use_rstd`, never read) rather than allocating -- it
        # must be fp32 to match the kernel's DT_FP32 q_rstd/k_rstd params.
        use_rstd = q_rstd is not None
        qr2d = _flat3(q_rstd, "q_rstd", f32) if use_rstd else gcum2d
        kr2d = _flat3(k_rstd, "k_rstd", f32) if use_rstd else gcum2d

        if initial_state is None:
            states2d = torch.zeros(num_heads * head_dim, head_dim, dtype=f32, device=device)
        else:
            states2d = _val(initial_state, "initial_state", f32).contiguous().reshape(num_heads * head_dim, head_dim)
        if dht is None:
            dht2d = torch.zeros(num_heads * head_dim, head_dim, dtype=f32, device=device)
        else:
            # read-only on the device (ds is copied with `+ 0.0`), no defensive clone needed.
            dht2d = _val(dht, "dht", f32).contiguous().reshape(num_heads * head_dim, head_dim)

        # External A_inv from the forward pass: [batch,seq_len,n_heads,bt] -> [tt, n_heads*bt] (col offset h*bt ==
        # the kernel's `hac = h_idx * bt`).
        a_inv_2d = _val(a_mat, "a_mat", f32).contiguous().reshape(tt, n_heads * bt)

        seqlens = torch.tensor(seqlens_l, dtype=torch.int32, device=device).contiguous()
        mask, tril_mask, eye, dmask, e_last, ones_bt = _fused_host_constants(bt, device)

        # GM workspace, NT_max-strided per (n,h) so a slot address never depends on the
        # sequence's own length (short sequences leave their tail slots unused -- workspace
        # ~2/3 of the backward workspace) is GONE -- PASS-2 recomputes w/v_new from A_inv + s_i,
        # bit-identically, saving its per-chunk HBM write + read as well as the allocation.
        sin_ws = torch.empty(num_heads * nt_max * head_dim, head_dim, dtype=f32, device=device)

        # OUTPUT tensors, allocated HERE.  dq/dk/dv/db bf16 -- the kernel casts on-chip to its
        # q/k/v/beta boundary dtype, so the result is already the final chunk_gated_delta_rule_bwd
        # dtype (no `_aligned_cast_outputs` re-pass).  dg/dh0 fp32 (fla contract, use_gate_in_kernel
        # =False).  Every flat row is written by exactly one chunk, so `empty` leaves no reachable
        # uninitialised element.
        dq2d = torch.empty(tt, n_heads * head_dim, dtype=bf16, device=device)
        dk2d = torch.empty(tt, n_heads * head_dim, dtype=bf16, device=device)
        dv2d = torch.empty(tt, n_heads * head_dim, dtype=bf16, device=device)
        dg2d = torch.empty(tt, n_heads, dtype=f32, device=device)
        dbeta2d = torch.empty(tt, n_heads, dtype=bf16, device=device)
        dh02d = torch.empty(num_heads * head_dim, head_dim, dtype=f32, device=device)

        logging.info(f"q2d : {q2d.shape} {q2d.dtype}")
        logging.info(f"k2d : {k2d.shape} {k2d.dtype}")
        logging.info(f"v2d : {v2d.shape} {v2d.dtype}")
        logging.info(f"beta2d : {beta2d.shape} {beta2d.dtype}")
        logging.info(f"gcum2d : {gcum2d.shape} {gcum2d.dtype}")
        logging.info(f"states2d : {states2d.shape} {states2d.dtype}")
        logging.info(f"do2d : {do2d.shape} {do2d.dtype}")
        logging.info(f"qr2d : {qr2d.shape} {qr2d.dtype}")
        logging.info(f"kr2d : {kr2d.shape} {kr2d.dtype}")
        logging.info(f"a_inv_2d : {a_inv_2d.shape} {a_inv_2d.dtype}")
        logging.info(f"mask : {mask.shape} {mask.dtype}")
        logging.info(f"tril_mask : {tril_mask.shape} {tril_mask.dtype}")
        logging.info(f"eye : {eye.shape} {eye.dtype}")
        logging.info(f"dmask : {dmask.shape} {dmask.dtype}")
        logging.info(f"e_last : {e_last.shape} {e_last.dtype}")
        logging.info(f"ones_bt : {ones_bt.shape} {ones_bt.dtype}")
        logging.info(f"dht2d : {dht2d.shape} {dht2d.dtype}")
        logging.info(f"sin_ws : {sin_ws.shape} {sin_ws.dtype}")
        logging.info(f"seqlens : {seqlens.shape} {seqlens.dtype} {seqlens}")
        logging.info(f"dq2d : {dq2d.shape} {dq2d.dtype}")
        logging.info(f"dk2d : {dk2d.shape} {dk2d.dtype}")
        logging.info(f"dv2d : {dv2d.shape} {dv2d.dtype}")
        logging.info(f"dg2d : {dg2d.shape} {dg2d.dtype}")
        logging.info(f"dbeta2d : {dbeta2d.shape} {dbeta2d.dtype}")
        logging.info(f"dh02d : {dh02d.shape} {dh02d.dtype}")
        logging.info(f"head_dim, bt, n_heads: {head_dim} {bt} {n_heads}")
        logging.info(f"nt_max, scale_val, use_rstd: {nt_max} {scale_val} {use_rstd}")

        if pypto.platform.npuarch == 'DAV_3510':
            gdr_bwd_fused_kernel_npu_950(
                q2d, k2d, v2d, beta2d, gcum2d,
                states2d, do2d,
                qr2d, kr2d,
                a_inv_2d,
                mask, tril_mask, eye, dmask, e_last, ones_bt,
                dht2d, sin_ws, seqlens,
                dq2d, dk2d, dv2d, dg2d, dbeta2d, dh02d,
                head_dim, bt, n_heads, nt_max, scale_val,
                use_rstd,
            )
        else:
            gdr_bwd_fused_kernel_npu(
                q2d, k2d, v2d, beta2d, gcum2d,
                states2d, do2d,
                qr2d, kr2d,
                a_inv_2d,
                mask, tril_mask, eye, dmask, e_last, ones_bt,
                dht2d, sin_ws, seqlens,
                dq2d, dk2d, dv2d, dg2d, dbeta2d, dh02d,
                head_dim, bt, n_heads, nt_max, scale_val,
                use_rstd,
            )

    # unflatten [tt,*] -> [batch,seq_len,n_heads,*] (varlen: batch==1, seq_len==sum T_i); free views.
    # dh0 -> [n_seqs,n_heads,head_dim,head_dim].
    dq = dq2d.reshape(batch, seq_len, n_heads, head_dim)
    dk = dk2d.reshape(batch, seq_len, n_heads, head_dim)
    dv = dv2d.reshape(batch, seq_len, n_heads, head_dim)
    db = dbeta2d.reshape(batch, seq_len, n_heads)
    dg = dg2d.reshape(batch, seq_len, n_heads)
    dh0 = dh02d.reshape(n_seqs, n_heads, head_dim, head_dim)

    # Outputs are already in the final fla dtypes; only the None-semantics of the two optional
    # grads remain:
    #   dg  -> None when g is None            (else fp32; use_gate_in_kernel=False, D14)
    #   dh0 -> None when initial_state is None (else the fp32 dL/d(initial_state))
    return (
        dq, dk, dv, db,
        dg if g is not None else None,
        dh0 if initial_state is not None else None,
        None, None,   # dA_log, ddt_bias (use_gate_in_kernel=False)
    )
# ====================================================================================================
# SECTION 3 - eval/adversarial_runner.py naming contract
#
# `adversarial_runner.py::_resolve_impl_wrapper` looks up `gdr_bwd_module<suffix>_wrapper`,
# where suffix = _phase_suffix(--up-to-module).  At the full boundary (--up-to-module 10,
# suffix "12345678910") the whole op is under test, so the wrapper is the production
# drop-in itself.  Positional-only: the runner calls it with the 19 primary inputs in
# `eval/test_inputs.py::PRIMARY_INPUT_ORDER` order.
# ====================================================================================================

gdr_bwd_module12345678910_wrapper = chunk_gated_delta_rule_backward_wrapper
