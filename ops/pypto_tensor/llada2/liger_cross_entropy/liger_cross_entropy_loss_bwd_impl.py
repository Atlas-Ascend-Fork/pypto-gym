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
"""PyPTO LigerCrossEntropyLoss Backward — static-V（pypto.STATIC）适配版。

V 用 ``pypto.STATIC``：编译期静态轴、运行时值可变。kernel 内 ``x.shape[1]``
为 Python int，直接算 N_V_TILES / LAST_V_SIZE，尾块用实际长度 view。
wrapper 不传 shape 元数、不对数据 padding。
"""

import pypto
import torch
import torch_npu  # noqa: F401

TILE_BT = 4
TILE_V = 8192


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
def _ce_bwd_kernel_bf16(
    saved_input: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    grad_vec: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_BF16),
    grad_input: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
):
    """Fused backward kernel — V=STATIC、BT=DYNAMIC；尾块用 last_v_size。

    逆路径统一为「逐行乘 grad_vec[bt]」（[BT,1] 乘子）：
      - golden path2（reduction='none'）：grad_vec[bt] = grad_output[bt]（逐行不同）
      - golden path3（mean/sum）：grad_vec[bt] = 标量广播（每行同值）
    乘子是 host 构造（.item() 同步 + 0-dim 无法参与 2D mul），kernel 内无分支。
    """
    bt = saved_input.shape[0]
    v = saved_input.shape[1]
    n_bt_tiles = (bt + TILE_BT - 1) // TILE_BT
    n_v_tiles = (v + TILE_V - 1) // TILE_V
    last_v_size = v - (n_v_tiles - 1) * TILE_V

    for bt_idx in pypto.loop(n_bt_tiles, name="bt_loop", unroll_list=[4, 1]):
        bt_off = bt_idx * TILE_BT
        actual_bt = (bt - bt_off).min(TILE_BT)

        pypto.set_vec_tile_shapes(TILE_BT, 1)
        gv = pypto.view(grad_vec, [TILE_BT, 1], [bt_off, 0],
                        valid_shape=[actual_bt, 1])

        for t in range(n_v_tiles):
            v_off = t * TILE_V
            cur_v = last_v_size if t == n_v_tiles - 1 else TILE_V

            pypto.set_pass_options(sg_set_scope=1)
            pypto.set_vec_tile_shapes(TILE_BT, cur_v)
            row = pypto.view(saved_input, [TILE_BT, cur_v], [bt_off, v_off],
                             valid_shape=[actual_bt, cur_v])

            result = pypto.mul(row, gv)

            pypto.assemble(result, [bt_off, v_off], grad_input)
            pypto.set_pass_options(sg_set_scope=-1)


def liger_cross_entropy_loss_bwd_wrapper(
    saved_input: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """Backward wrapper — 三分支显式对应 golden。

    kernel 已统一「逐行乘 [BT,1] 乘子」，此处仅剩 host 必须的判断/构造：
      - path1: ``torch.equal`` 判定、跳 kernel（.item() 同步控制流）；
      - path3: ``float(.item())`` 取标量并广播为 [BT,1]（0-dim 无法参与 2D mul）；
      - 返回 buffer：path2 新 tensor / path3 原地（golden 契约）。
    """
    device = saved_input.device
    saved_input = saved_input.to(device)
    grad_output = grad_output.to(device)
    input_dtype = saved_input.dtype

    # ---- 前置校验（先于一切路径分支）----
    if input_dtype != torch.bfloat16:
        raise TypeError(
            f"liger_cross_entropy_loss (PyPTO static-V kernel) 仅支持 bf16，got {input_dtype}")

    # ---- path1: last-layer 直通（golden L92）----
    # torch.equal 比较「值 且 dtype」：grad_output 为 fp32 标量 1.0 而 saved 为
    # bf16 时二者不等 → 落到 path3（与 golden 的 dtype 语义完全一致）。
    # 该判断是「是否跳过 kernel」的 host 控制流（.item() 同步），无法进 kernel。
    if torch.equal(grad_output, torch.tensor(1.0, device=device)):
        return saved_input

    bt = saved_input.shape[0]
    v = saved_input.shape[1]

    # ---- path2 vs path3 分支（仅决定乘子构造与返回 buffer）----
    #   grad_output.ndim > 0（reduction='none'）→ 逐行加权，新 tensor；
    #   grad_output.ndim == 0（mean/sum）→ 标量广播为 [bt,1]，原地写回。
    if grad_output.ndim > 0:
        # path2: reduction='none' —— grad_output [bt] 逐行加权
        grad_vec = grad_output.to(torch.bfloat16).unsqueeze(1).contiguous()
        grad_input = torch.empty(bt, v, dtype=torch.bfloat16, device=device)
        _ce_bwd_kernel_bf16(saved_input, grad_vec, grad_input)
        return grad_input
    else:
        # path3: mean/sum —— 0-dim 标量广播为 [bt,1]（每行同值，golden mul_ 语义原地写回）
        grad_vec = torch.full((bt, 1), float(grad_output.item()),
                              dtype=torch.bfloat16, device=device)
        _ce_bwd_kernel_bf16(saved_input, grad_vec, saved_input)
        return saved_input