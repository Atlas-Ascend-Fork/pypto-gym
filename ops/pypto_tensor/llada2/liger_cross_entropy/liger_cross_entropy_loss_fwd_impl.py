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
"""PyPTO LigerCrossEntropyLoss Forward — static-v（pypto.STATIC）适配版。

v（词表大小）用 **``pypto.STATIC``** 标注：编译期静态轴、运行时值可变
（每次调用的 v 由输入 shape 决定，PyPTO 按该值编译对应变体）。bt 为 DYNAMIC。
因此 kernel 内 ``x.shape[1]`` 是 Python int，可直接做全部 tile 划分算术
（n_v_tiles / last_v_size）、``for range`` 展开与尾块判断，wrapper 不传任何
shape 元数、不对数据 padding：

  - v 非对齐（如 157184 % 2048 = 1536 ≠ 0）：``for t in range(n_v_tiles)`` 展开，
    末尾 tile 用实际长度 [bt, last_v_size] view —— 不越界。
  - bt 动态：``pypto.loop`` + valid_shape 尾块。

与动态版（custom/liger_cross_entropy_loss）的区别仅在于 v 轴的 STATIC 标注与
kernel 内部直接算元数；计算逻辑、golden 对齐、col_idx 索引匹配完全一致。
"""

import pypto
import torch
import torch_npu  # noqa: F401

TILE_BT = 8
TILE_V = 1024


@pypto.frontend.jit(
    runtime_options={
        "device_sched_mode": 1,
        "stitch_function_max_num": 128,
        "launch_sched_aicpu_num": 3,
    },
    debug_options={
        "runtime_debug_mode": 0
    }
)
def _ce_fwd_kernel_bf16(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    col_idx: pypto.Tensor([1, pypto.STATIC], pypto.DT_FP32),
    target: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_FP32),
    ori_x_y: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_FP32),
    weight_y: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_FP32),
    weight_vec: pypto.Tensor([1, pypto.STATIC], pypto.DT_FP32),
    loss_out: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_BF16),
    z_loss_out: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_BF16),
    saved_input: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    pred_tokens_out: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_FP32),
    token_acc_out: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_FP32),
    inv_n_loss: float,
    inv_n_z: float,
    inv_n_grad: float,
    lse_square_scale: float,
    label_smoothing: float,
    weight_sum: float,
    ignore_index: float,
    has_weight: bool,
    has_softcap: bool,
    has_gradients: bool,
    softcap: float,
    compute_token_acc: bool,
    compute_pred_tokens: bool,
):
    """Fused forward kernel — v=STATIC、bt=DYNAMIC；kernel 内算全部 tile 元数。

    ``v = x.shape[1]`` 为 Python int（STATIC 轴）：n_v_tiles / last_v_size、
    哨兵等全部 kernel 内派生；bt 用 SymbolicScalar + valid_shape 尾块。
    """
    bt = x.shape[0]  # SymbolicScalar（DYNAMIC）
    v = x.shape[1]   # Python int（STATIC）
    n_bt_tiles = (bt + TILE_BT - 1) // TILE_BT
    n_v_tiles = (v + TILE_V - 1) // TILE_V
    last_v_size = v - (n_v_tiles - 1) * TILE_V

    # kernel 内算派生标量（label_smoothing 参数 + v 常量，Python 侧求值）
    eps_f = label_smoothing / v
    one_minus_ls = 1.0 - label_smoothing

    pypto.experimental.set_operation_options(combine_axis=True)

    pypto.set_vec_tile_shapes(TILE_BT, TILE_V)

    for bt_idx in pypto.loop(n_bt_tiles, name="bt_loop", unroll_list=[4, 1]):
        bt_off = bt_idx * TILE_BT
        actual_bt = (bt - bt_off).min(TILE_BT)

        # ---- ignore mask (golden L138-141) ----
        pypto.set_vec_tile_shapes(TILE_BT, 1)
        tgt_tile = pypto.view(target, [TILE_BT, 1], [bt_off, 0],
                              valid_shape=[actual_bt, 1])
        ig_elem = pypto.Element(pypto.DT_FP32, ignore_index)
        is_ignore = pypto.eq(tgt_tile, ig_elem)
        mask_tile = pypto.where(is_ignore, pypto.Element(pypto.DT_FP32, 0.0),
                                pypto.Element(pypto.DT_FP32, 1.0))

        # ============================================================
        # Pass 1: online softmax → gmax, tsum → lse
        # ============================================================
        gmax = pypto.tensor([TILE_BT, 1], pypto.DT_FP32, "gmax")
        tsum = pypto.tensor([TILE_BT, 1], pypto.DT_FP32, "tsum")
        scaled_x_sum = pypto.tensor([TILE_BT, 1], pypto.DT_FP32, "scaled_x_sum")

        for t in range(n_v_tiles):
            v_off = t * TILE_V
            cur_v = last_v_size if t == n_v_tiles - 1 else TILE_V

            pypto.set_pass_options(sg_set_scope=1)
            pypto.set_vec_tile_shapes(TILE_BT, cur_v)
            row = pypto.view(x, [TILE_BT, cur_v], [bt_off, v_off],
                             valid_shape=[actual_bt, cur_v])
            r = pypto.cast(row, pypto.DT_FP32)

            if has_softcap:
                r_div = pypto.div(r, softcap)
                tt = pypto.tanh(r_div)
                r = pypto.mul(tt, softcap)

            local_max = pypto.amax(r, dim=-1, keepdim=True)
            local_exp = pypto.exp(pypto.sub(r, local_max))
            local_sum = pypto.sum(local_exp, dim=-1, keepdim=True)

            if label_smoothing > 0.0:
                if has_weight:
                    w_tile = pypto.view(weight_vec, [1, cur_v], [0, v_off],
                                        valid_shape=[1, cur_v])
                    wx = pypto.mul(r, w_tile)
                    local_sxs = pypto.mul(pypto.sum(wx, dim=-1, keepdim=True), -eps_f)
                else:
                    local_sxs = pypto.mul(pypto.sum(r, dim=-1, keepdim=True), -eps_f)
            else:
                local_sxs = pypto.mul(local_sum, 0.0)

            if t == 0:
                gmax[:] = local_max
                tsum[:] = local_sum
                scaled_x_sum[:] = local_sxs
            else:
                new_max = pypto.maximum(gmax, local_max)
                rescale_old = pypto.exp(pypto.sub(gmax, new_max))
                rescale_new = pypto.exp(pypto.sub(local_max, new_max))
                tsum[:] = pypto.add(
                    pypto.mul(tsum, rescale_old),
                    pypto.mul(local_sum, rescale_new),
                )
                gmax[:] = new_max
                scaled_x_sum[:] = pypto.add(scaled_x_sum, local_sxs)
            pypto.set_pass_options(sg_set_scope=-1)

        lse = pypto.add(gmax, pypto.log(tsum))

        # ============================================================
        # Loss computation (golden L184-214)
        # ============================================================
        pypto.set_vec_tile_shapes(TILE_BT, 1)
        oy = pypto.view(ori_x_y, [TILE_BT, 1], [bt_off, 0], valid_shape=[actual_bt, 1])
        wy = pypto.view(weight_y, [TILE_BT, 1], [bt_off, 0], valid_shape=[actual_bt, 1])

        loss = pypto.sub(lse, oy)
        if has_weight:
            loss = pypto.mul(loss, wy)
        if label_smoothing > 0.0:
            if has_weight:
                smooth = pypto.add(scaled_x_sum, pypto.mul(pypto.mul(lse, eps_f), weight_sum))
            else:
                smooth = pypto.add(scaled_x_sum, pypto.mul(lse, label_smoothing))
            loss = pypto.add(pypto.mul(loss, one_minus_ls), smooth)

        lse_sq = pypto.mul(lse, lse)
        z_loss = pypto.mul(lse_sq, lse_square_scale)

        loss = pypto.mul(loss, inv_n_loss)
        z_loss = pypto.mul(z_loss, inv_n_z)
        loss = pypto.add(loss, z_loss)
        loss = pypto.mul(loss, mask_tile)
        z_loss = pypto.mul(z_loss, mask_tile)

        loss_bf16 = pypto.cast(loss, pypto.DT_BF16)
        z_loss_bf16 = pypto.cast(z_loss, pypto.DT_BF16)
        pypto.assemble(loss_bf16, [bt_off, 0], loss_out)
        pypto.assemble(z_loss_bf16, [bt_off, 0], z_loss_out)

        # ============================================================
        # First-occurrence argmax (golden L236-256)
        # ============================================================
        if compute_token_acc or compute_pred_tokens:
            pypto.set_vec_tile_shapes(1, 1)
            pred_idx = pypto.full([TILE_BT, 1], float(v), pypto.DT_FP32)
            pypto.set_vec_tile_shapes(TILE_BT, TILE_V)

            for t in range(n_v_tiles):
                v_off = t * TILE_V
                cur_v = last_v_size if t == n_v_tiles - 1 else TILE_V

                pypto.set_vec_tile_shapes(TILE_BT, cur_v)
                row = pypto.view(x, [TILE_BT, cur_v], [bt_off, v_off],
                                 valid_shape=[actual_bt, cur_v])
                r = pypto.cast(row, pypto.DT_FP32)

                is_max = pypto.eq(r, gmax)
                col_tile = pypto.view(col_idx, [1, cur_v], [0, v_off],
                                      valid_shape=[1, cur_v])
                candidate = pypto.where(is_max, col_tile,
                                        pypto.Element(pypto.DT_FP32, float(v)))
                tile_min = pypto.amin(candidate, dim=-1, keepdim=True)
                pypto.set_vec_tile_shapes(1, 1)
                pred_idx[:] = pypto.minimum(pred_idx, tile_min)
                pypto.set_vec_tile_shapes(TILE_BT, cur_v)

            pypto.set_vec_tile_shapes(TILE_BT, 1)
            pred_tokens_fp32 = pypto.where(
                is_ignore, pypto.Element(pypto.DT_FP32, -1.0), pred_idx)
            pypto.assemble(pred_tokens_fp32, [bt_off, 0], pred_tokens_out)

            if compute_token_acc:
                is_correct = pypto.eq(pred_idx, tgt_tile)
                acc_fp32 = pypto.where(is_correct, pypto.Element(pypto.DT_FP32, 1.0),
                                       pypto.Element(pypto.DT_FP32, 0.0))
                acc_masked = pypto.mul(acc_fp32, mask_tile)
                pypto.assemble(acc_masked, [bt_off, 0], token_acc_out)

        # ============================================================
        # Pass 2: gradient (golden L263-305)
        # ============================================================
        if has_gradients:
            for t in range(n_v_tiles):
                v_off = t * TILE_V
                cur_v = last_v_size if t == n_v_tiles - 1 else TILE_V

                pypto.set_pass_options(sg_set_scope=1)
                pypto.set_vec_tile_shapes(TILE_BT, cur_v)
                row = pypto.view(x, [TILE_BT, cur_v], [bt_off, v_off],
                                 valid_shape=[actual_bt, cur_v])
                r = pypto.cast(row, pypto.DT_FP32)

                if has_softcap:
                    r_div = pypto.div(r, softcap)
                    tt = pypto.tanh(r_div)
                    r = pypto.mul(tt, softcap)

                p = pypto.exp(pypto.sub(r, lse))

                if has_weight:
                    pw = pypto.mul(p, wy)
                    dloss_ori = pypto.mul(pw, one_minus_ls)
                    w_tile = pypto.view(weight_vec, [1, cur_v], [0, v_off],
                                        valid_shape=[1, cur_v])
                    neg_w = pypto.mul(w_tile, -1.0)
                    pws = pypto.mul(p, weight_sum)
                    smooth_base = pypto.add(neg_w, pws)
                    dloss_smooth = pypto.mul(smooth_base, eps_f)
                    lse_p = pypto.mul(lse, p)
                    dz_loss = pypto.mul(lse_p, 2.0 * lse_square_scale)
                    nonz = pypto.add(dloss_ori, dloss_smooth)
                    dx = pypto.add(pypto.mul(nonz, inv_n_grad),
                                   pypto.mul(dz_loss, inv_n_z))
                else:
                    if lse_square_scale != 0.0:
                        lse_p = pypto.mul(lse, p)
                        dz = pypto.mul(lse_p, 2.0 * lse_square_scale)
                        dx = pypto.add(pypto.sub(p, eps_f), dz)
                    else:
                        if eps_f != 0.0:
                            dx = pypto.sub(p, eps_f)
                        else:
                            dx = p
                    dx = pypto.mul(dx, inv_n_grad)

                # target class correction (golden L274-276 / L291-294)
                col_tile = pypto.view(col_idx, [1, cur_v], [0, v_off],
                                      valid_shape=[1, cur_v])
                is_target = pypto.eq(col_tile, tgt_tile)
                if has_weight:
                    corr_effect = pypto.mul(pypto.mul(wy, one_minus_ls), inv_n_grad)
                else:
                    corr_effect = pypto.mul(mask_tile, one_minus_ls * inv_n_grad)
                dx_corrected = pypto.sub(dx, corr_effect)
                dx = pypto.where(is_target, dx_corrected, dx)

                if has_softcap:
                    t_ratio = pypto.div(r, softcap)
                    t_sq = pypto.mul(t_ratio, t_ratio)
                    neg_t_sq = pypto.mul(t_sq, -1.0)
                    chain = pypto.add(neg_t_sq, 1.0)
                    dx = pypto.mul(dx, chain)

                dx = pypto.mul(dx, mask_tile)

                dx_bf16 = pypto.cast(dx, pypto.DT_BF16)
                pypto.assemble(dx_bf16, [bt_off, v_off], saved_input)
                pypto.set_pass_options(sg_set_scope=-1)
        else:
            for t in range(n_v_tiles):
                v_off = t * TILE_V
                cur_v = last_v_size if t == n_v_tiles - 1 else TILE_V

                pypto.set_vec_tile_shapes(TILE_BT, cur_v)
                row = pypto.view(x, [TILE_BT, cur_v], [bt_off, v_off],
                                 valid_shape=[actual_bt, cur_v])
                r = pypto.cast(row, pypto.DT_FP32)
                r_masked = pypto.mul(r, mask_tile)
                r_bf16 = pypto.cast(r_masked, pypto.DT_BF16)
                pypto.assemble(r_bf16, [bt_off, v_off], saved_input)


# ============================================================
# Host wrapper — 不传任何 shape 元数，不做任何 padding
# ============================================================

def liger_cross_entropy_loss_fwd_wrapper(
    _input: torch.Tensor,
    target: torch.Tensor,
    weight=None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap=None,
    return_z_loss: bool = False,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
):
    """Forward wrapper — v=pypto.STATIC（运行时值可变）、bt 动态。

    仅支持 bf16（kernel annotation 为 bf16）。所有 tile 划分在 kernel 内。
    """
    device = _input.device
    bt, v = _input.shape
    input_dtype = _input.dtype

    # ---- 输入校验 ----
    if input_dtype != torch.bfloat16:
        raise TypeError(
            f"liger_cross_entropy_loss (PyPTO static-v kernel) 仅支持 bf16，got {input_dtype}")
    if _input.ndim != 2:
        raise ValueError(f"_input must be 2D [bt, v]. Got: {_input.ndim}D")
    if target.ndim != 1 or target.shape[0] != bt:
        raise ValueError(
            f"target must be 1D with length bt={bt}. Got: shape={target.shape}")

    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    # ---- OOB assertion ----
    target_mask = target != ignore_index
    n_non_ignore = int(target_mask.sum().item())
    target_masked = target * target_mask

    # ---- weight setup ----
    sum_non_ignore_weight = float(n_non_ignore)
    weight_sum = 0.0
    if weight is not None:
        sum_non_ignore_weight = float(weight[target.masked_select(target_mask)].sum().item())
        weight_sum = float(weight.sum().item())

    has_weight = weight is not None
    has_softcap = softcap is not None
    has_gradients = _input.requires_grad

    # ---- gather ori_x_y (reuse target_masked from OOB, gather replaces advanced indexing) ----
    ori_x_y = _input.gather(1, target_masked.unsqueeze(1)).float().squeeze(1)
    if has_softcap:
        ori_x_y = softcap * torch.tanh(ori_x_y / softcap)
    ori_x_y = ori_x_y.unsqueeze(1)

    weight_y = torch.empty(bt, 1, dtype=torch.float32, device=device)
    if has_weight:
        weight_y[:, 0] = weight.float().gather(0, target_masked)

    weight_vec = (weight.float().unsqueeze(0).to(device) if has_weight
                  else torch.empty(1, v, dtype=torch.float32, device=device))

    col_idx = torch.arange(v, dtype=torch.float32, device=device).unsqueeze(0)

    target_fp32 = target.float().unsqueeze(1).to(device)

    # ---- inv_n ----
    if reduction == "mean":
        inv_n_loss = 1.0 / sum_non_ignore_weight if sum_non_ignore_weight > 0 else 0.0
        inv_n_z = 1.0 / n_non_ignore if n_non_ignore > 0 else 0.0
        inv_n_grad = inv_n_loss
    else:
        inv_n_loss = 1.0
        inv_n_z = 1.0
        inv_n_grad = 1.0

    # ---- output buffers（kernel 各分支逐 tile 全覆盖写回 → empty）----
    loss_1d = torch.empty(bt, 1, dtype=torch.bfloat16, device=device)
    z_loss_1d = torch.empty(bt, 1, dtype=torch.bfloat16, device=device)
    saved_input = torch.empty(bt, v, dtype=torch.bfloat16, device=device)
    pred_tokens = torch.empty(bt, 1, dtype=torch.float32, device=device)
    token_acc_sum = torch.empty(bt, 1, dtype=torch.float32, device=device)

    _ce_fwd_kernel_bf16(
        _input, col_idx, target_fp32, ori_x_y, weight_y, weight_vec,
        loss_1d, z_loss_1d, saved_input,
        pred_tokens, token_acc_sum,
        inv_n_loss, inv_n_z, inv_n_grad,
        lse_square_scale, label_smoothing, weight_sum,
        float(ignore_index),
        has_weight, has_softcap, has_gradients,
        softcap if softcap is not None else 0.0,
        return_token_accuracy, return_predicted_tokens,
    )

    # ---- extract ----
    if return_token_accuracy:
        acc_1d = token_acc_sum.squeeze(1)
        if reduction == "none":
            token_accuracy = acc_1d
        else:
            token_accuracy = torch.sum(acc_1d) / float(n_non_ignore)
    else:
        token_accuracy = None

    if return_predicted_tokens:
        predicted_tokens = pred_tokens.squeeze(1).long()
    else:
        predicted_tokens = None

    if reduction == "none":
        loss = loss_1d.squeeze(1)
    else:
        # 规避 pypto_SAS 9.1.0 runtime [1,1] 输出写回 bug：不使用 loss_sum，
        # 改用 [bt,1] 的 loss_1d 在 device 侧归约（bf16 行值 → fp32 求和 → bf16）
        loss = loss_1d.squeeze(1).float().sum().to(torch.bfloat16)

    if return_z_loss:
        if reduction == "none":
            z_loss = z_loss_1d.squeeze(1)
        else:
            z_loss = z_loss_1d.squeeze(1).float().sum().to(torch.bfloat16)
    else:
        z_loss = None

    return loss, z_loss, token_accuracy, predicted_tokens, saved_input