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

"""mhc_post_fwd：手写 PyPTO-Pro（``@pl.jit`` + ``@pl.vector_function``）向量算子。

算子目的
--------
``mhc``（manifold hyper-connection）为每个 token 保留 ``mhc=4`` 份隐藏状态。本算子把本层
新算出的 ``x`` 与 4 份历史残差 ``residual``，用按 token 变化的两组系数重新混合：

.. code-block:: text

    out[n, mo, h] = post_layer_mix[n, mo] * x[n, h]
                    + sum_{mi=0}^{3} comb_res_mix[n, mi, mo] * residual[n, mi, h]

``x``/``residual``/``out`` 为 BF16，``post_layer_mix``/``comb_res_mix`` 为 FP32；乘积与 4 项
求和在 FP32 累加，落盘前一次性舍入到 BF16。求和只跨 ``mi``，不跨 token、不跨 hidden。

分块
----------------------------
hidden 只按 ``h_blk`` 分块，``h_blk`` 由 **wrapper** 按参考公式现算：

.. code-block:: text

    h_blk = align(h if h <= 4096 else min(ceil_div(h, 2), 4096), 64)

``h_blk`` 是编译期常量（决定 UB 布局与 tile 形状），因此对每个不同的 ``h_blk`` 生成一个
特化 kernel，由 ``_kernel_for(h_blk)`` 惰性构建并缓存。四个 P0 hidden 得到 3 个不同
``h_blk``（1280 / 2560 / 3584）。

数据流与写法规约
----------------
每个向量核按 persistent grid-stride 领取若干 (token, hidden-block) 工作项：先把 16 个
``comb`` 与 4 个 ``post`` 系数一次载入 UB，再在 VF 内广播进寄存器（``BRC_B32``）并沿
hidden 分块流式处理——``x``/``residual`` 各 chunk 提升到 FP32，做 1 次 ``mul`` + 4 次
``mul_add_dst``，收窄回 BF16 后写回 GM。

实现是寄存器级的，各步与底层指令语义一一对应：

.. code-block:: text

    系数广播（comb / post）                      -> vf.load_align(dist=LoadDist.BRC_B32)
    bf16 unpack + fp32 提升                     -> load_align(UNPK_B16) + astype(CastLayout.ZERO)
    mul / fma(out_reg, res, comb)               -> vf.mul / vf.mul_add_dst
    fp32→bf16 收窄 + 偶数 lane 连续打包          -> vf.astype(DT_BF16) + store_align(StoreDist.PACK_B32)

FMA 顺序固定为「先 ``post*x``，再按 ``mi = 0,1,2,3`` 累加」，不依赖浮点重结合。

建议阅读顺序
------------
公开入口 ``mhc_post_fwd`` → kernel 工厂 ``_make_kernel``（内含寄存器级 ``mhc_post_fwd_vf``）
→ 公开 kernel ``mhc_post_fwd_kernel`` → host 侧 ``h_blk_of`` / ``_validate_inputs``。

掩码与尾块
----------
全部 lane 掩码都是**循环外 2 条常量** ``PAT_ALL``（bf16 / fp32 各一条），VF 内没有
``vf.update_mask``。这是可证的，不是"只覆盖 P0"的巧合：

* ``_validate_inputs`` 强制 ``hidden % VL == 0``，且 ``h_blk`` 也按 64 对齐，
  因此每个工作项的 ``valid = min(h_blk, hidden - h0)`` **恒为 64 的倍数**；
* 于是 ``n_chunks = valid // VL`` 的每一拍都是满 64 lane，逐拍 mask 恒等于 ``PAT_ALL``。

所以 per-chunk 的 ``min``/``update_mask`` 与 ``full_tile`` 双变体都是冗余的；``h % h_blk != 0``
（如 ``h = 4160``、``h_blk = 2112``）只是最后一块变窄，**仍然 64 对齐**，常量掩码照样成立。

本工具链实测约束（均由上板探针确定）
------------------------------------

1. 4 行 FP32 输出必须用 4 个独立累加寄存器；单个累加 tile 在 ``pl.range`` 里复用时，一旦
   单核执行多于一个工作项就会算错（确定性错误，曾据此定位，回归见 ``test_multi_item_per_core``）。
2. ``[4,4]`` / ``[4,1]`` FP32 tile 非法（``Cols * sizeof(dtype)`` 需 32B 对齐）；系数承载为
   ``[4,8]``（valid ``[4,4]``）与 ``[8,1]`` DN（valid ``[4,1]``）。
3. ``pl.getval(tile, k)`` 每次调用生成一对 ``get_buf(PIPE_S)/rls_buf(PIPE_S)``，是 tile 级
   写法慢约 4 倍的主因。
4. ``vf.axpy`` 的 multiplicand 必须是标量；``vmula`` 对应 ``vf.mul_add_dst``。
5. ``load_align(dist=UNPK_B16)`` + ``astype(layout=CastLayout.ZERO, DT_FP32)`` 位精确还原
   bf16→fp32；fp32→bf16 收窄把结果放在**偶数** lane，只有 ``StoreDist.PACK_B32`` 能连续打包。
6. ``vf.load_align`` 不能作为调用实参嵌套，必须用赋值形式。
7. UB 编译期容量是 ``0x40000``（256 KB）；``h_blk`` 最大 4096（公式上限），三级缓冲占用
   227328 B，仍落在预算内（``_make_kernel`` 内有 UB 预算闸门，超限 ``raise ValueError``）。
8. ``wrapper-boundary``（mandatory）禁止 host 侧任何形状处理——**包括纯 view**。wrapper 只做
   校验 / 整数推导 / ``torch.empty`` / 一次启动；token 轴展平由核内 ``pl.make_tensor`` 完成。
9. 掩码常量外提的前提是上面「掩码与尾块」的不变量；若将来放宽 ``hidden % 64 == 0``，必须
   同时恢复逐拍 mask。
"""

from __future__ import annotations

import torch

import torch_npu  # noqa: F401

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

# ---------------------------------------------------------------------------
# Frozen geometry / tiling constants
# ---------------------------------------------------------------------------
MHC = 4                     # hyper-connection multiplicity, fixed
VL = 64                     # fp32 lanes per 256B register
UB_LIMIT = 256 * 1024       # compiler-reported Vec capacity (0x40000)
STAGES = 3                  # rotating UB slots per tile group (3-deep, see DESIGN.md)


def h_blk_of(hidden: int) -> int:
    """Hidden tile width: 64-aligned, capped at 4096.

    ``align(h if h <= 4096 else min(ceil_div(h, 2), 4096), 64)``
    """
    blk = hidden if hidden <= 4096 else min((hidden + 1) // 2, 4096)
    return ((blk + 63) // 64) * 64


def _align(v: int, a: int) -> int:
    return (v + a - 1) // a * a


def _slot(nbytes: int) -> int:
    """1KB-aligned slot size (A5 UB tiles are kept 1KB-aligned for stability)."""
    return _align(nbytes, 1024)


def _load_tile(tile, src, offset, valid_shape, full_shape):
    """把一次搬入写成「夹有效形状 → 搬运 → 恢复物理形状」三段（尾块在核内夹紧）。

    这三段在 kernel 里对 4 个 tile 各出现一次，抽成宏可让主循环只读一行；调用发生在
    被 trace 的 kernel 体内，因此其中的 ``pl.*`` 照常被记录。
    """
    pl.set_validshape(tile, valid_shape)
    pl.load(tile, src, offset)
    pl.set_validshape(tile, full_shape)


# ---------------------------------------------------------------------------
# Kernel factory: one @pl.jit kernel per compile-time h_blk
# ---------------------------------------------------------------------------
def _make_kernel(h_blk: int):
    """Build the kernel specialised on the compile-time hidden tile width.

    The UB layout of the five rotating tile groups is derived from ``h_blk`` and frozen
    here; the returned ``mhc_post_fwd_kernel`` is a ``@pl.jit`` callable whose sole
    compute body is the register-level ``mhc_post_fwd_vf``.
    """
    slots = (("cf", MHC * 8 * 4), ("pc", 8 * 4), ("xbf", 2 * h_blk),
             ("rbf", MHC * h_blk * 2), ("obf", MHC * h_blk * 2))
    addrs, mutex = {}, {}
    off = mid = 0
    for name, size in slots:
        s = _slot(size)
        addrs[name] = [off + i * s for i in range(STAGES)]
        mutex[name] = [mid + i for i in range(STAGES)]
        mid += STAGES
        off += STAGES * s
    # UB 预算闸门：同样用 raise 而非 assert（``-O`` 下 assert 会被剥掉）。
    if off > UB_LIMIT:
        raise ValueError(
            f"h_blk={h_blk} does not fit the UB budget: {off} B > {UB_LIMIT} B (0x40000)"
        )

    @pl.vector_function
    def mhc_post_fwd_vf(xbf, rbf, cf, pc, obf, n_chunks: pl.DT_INT64):
        """逐 64 lane 的寄存器级实现：``obf[mo] = post[mo]*x + sum_mi comb[mi,mo]*res[mi]``。

        ``h_blk`` 是外层工厂的编译期常量，直接以闭包引用（不占参数）；``n_chunks`` 是本
        工作项的有效 chunk 数。20 次系数广播只依赖 token，故都在 chunk 循环之外；每个
        chunk 把 5 行 BF16 提升为 FP32，按 ``mi = 0,1,2,3`` 固定顺序做一次乘 + 四次乘加，
        最后一次性收窄回 BF16 写回。

        注：本体的展开写法是刻意的——``@pl.vector_function`` 内不允许调用非 vector 内联
        函数，也不允许普通 ``for``/推导式（详见 DESIGN.md「接口选择」），因此无法进一步
        拆成宏或循环。
        """
        preg_b = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
        preg_f = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)

        c00 = vf.load_align(cf, 0, dist=pl.LoadDist.BRC_B32)
        c01 = vf.load_align(cf, 1, dist=pl.LoadDist.BRC_B32)
        c02 = vf.load_align(cf, 2, dist=pl.LoadDist.BRC_B32)
        c03 = vf.load_align(cf, 3, dist=pl.LoadDist.BRC_B32)
        c10 = vf.load_align(cf, 8, dist=pl.LoadDist.BRC_B32)
        c11 = vf.load_align(cf, 9, dist=pl.LoadDist.BRC_B32)
        c12 = vf.load_align(cf, 10, dist=pl.LoadDist.BRC_B32)
        c13 = vf.load_align(cf, 11, dist=pl.LoadDist.BRC_B32)
        c20 = vf.load_align(cf, 16, dist=pl.LoadDist.BRC_B32)
        c21 = vf.load_align(cf, 17, dist=pl.LoadDist.BRC_B32)
        c22 = vf.load_align(cf, 18, dist=pl.LoadDist.BRC_B32)
        c23 = vf.load_align(cf, 19, dist=pl.LoadDist.BRC_B32)
        c30 = vf.load_align(cf, 24, dist=pl.LoadDist.BRC_B32)
        c31 = vf.load_align(cf, 25, dist=pl.LoadDist.BRC_B32)
        c32 = vf.load_align(cf, 26, dist=pl.LoadDist.BRC_B32)
        c33 = vf.load_align(cf, 27, dist=pl.LoadDist.BRC_B32)
        p0 = vf.load_align(pc, 0, dist=pl.LoadDist.BRC_B32)
        p1 = vf.load_align(pc, 1, dist=pl.LoadDist.BRC_B32)
        p2 = vf.load_align(pc, 2, dist=pl.LoadDist.BRC_B32)
        p3 = vf.load_align(pc, 3, dist=pl.LoadDist.BRC_B32)

        for ch in pl.range(0, n_chunks):
            col = ch * VL
            xrb = vf.load_align(xbf, col, dist=pl.LoadDist.UNPK_B16)
            x_r = vf.astype(xrb, preg_b, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
            rb = vf.load_align(rbf, col, dist=pl.LoadDist.UNPK_B16)
            r0 = vf.astype(rb, preg_b, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
            rb = vf.load_align(rbf, h_blk + col, dist=pl.LoadDist.UNPK_B16)
            r1 = vf.astype(rb, preg_b, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
            rb = vf.load_align(rbf, 2 * h_blk + col, dist=pl.LoadDist.UNPK_B16)
            r2 = vf.astype(rb, preg_b, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
            rb = vf.load_align(rbf, 3 * h_blk + col, dist=pl.LoadDist.UNPK_B16)
            r3 = vf.astype(rb, preg_b, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)

            # 四个输出份复用同一个输出寄存器；各行 live range 不重叠。
            o = vf.mul(x_r, p0, preg_f)
            o = vf.mul_add_dst(r0, c00, preg_f)
            o = vf.mul_add_dst(r1, c10, preg_f)
            o = vf.mul_add_dst(r2, c20, preg_f)
            o = vf.mul_add_dst(r3, c30, preg_f)
            b = vf.astype(o, preg_f, dtype=pl.DT_BF16)
            vf.store_align(obf, b, preg_b, col, dist=pl.StoreDist.PACK_B32)

            o = vf.mul(x_r, p1, preg_f)
            o = vf.mul_add_dst(r0, c01, preg_f)
            o = vf.mul_add_dst(r1, c11, preg_f)
            o = vf.mul_add_dst(r2, c21, preg_f)
            o = vf.mul_add_dst(r3, c31, preg_f)
            b = vf.astype(o, preg_f, dtype=pl.DT_BF16)
            vf.store_align(obf, b, preg_b, h_blk + col, dist=pl.StoreDist.PACK_B32)

            o = vf.mul(x_r, p2, preg_f)
            o = vf.mul_add_dst(r0, c02, preg_f)
            o = vf.mul_add_dst(r1, c12, preg_f)
            o = vf.mul_add_dst(r2, c22, preg_f)
            o = vf.mul_add_dst(r3, c32, preg_f)
            b = vf.astype(o, preg_f, dtype=pl.DT_BF16)
            vf.store_align(obf, b, preg_b, 2 * h_blk + col, dist=pl.StoreDist.PACK_B32)

            o = vf.mul(x_r, p3, preg_f)
            o = vf.mul_add_dst(r0, c03, preg_f)
            o = vf.mul_add_dst(r1, c13, preg_f)
            o = vf.mul_add_dst(r2, c23, preg_f)
            o = vf.mul_add_dst(r3, c33, preg_f)
            b = vf.astype(o, preg_f, dtype=pl.DT_BF16)
            vf.store_align(obf, b, preg_b, 3 * h_blk + col, dist=pl.StoreDist.PACK_B32)

    @pl.jit(auto_mutex=True, name=f"mhc_post_fwd_h{h_blk}")
    def mhc_post_fwd_kernel(
        x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
        residual: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, MHC, pl.DYNAMIC], pl.DT_BF16],
        post: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, MHC, 1], pl.DT_FP32],
        comb: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, MHC, MHC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, MHC, pl.DYNAMIC], pl.DT_BF16],
    ):
        """沿 hidden 轴按编译期 ``h_blk`` 切块；尾块用 ``set_validshape`` 夹有效长度。

        参数按**原生形状**接收（``wrapper-boundary`` 强制：host 不得做任何形状处理，
        包括纯 view）。token 轴的展平在核内用 ``pl.make_tensor`` 完成，只是重算
        偏移，无数据搬运：

            x        (BF16[n0,n1,h])        residual (BF16[n0,n1,MHC,h])
            post     (FP32[n0,n1,MHC,1])    comb     (FP32[n0,n1,MHC,MHC])
            out      (BF16[n0,n1,MHC,h])

        不变量：``h`` 为 ``VL``(64) 的倍数；grid 由 host 决定，work 按
        ``get_block_num()`` 步长划分。
        """
        t_cf = pl.TileType(shape=[MHC, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                           valid_shape=[-1, -1])
        t_pc = pl.TileType(shape=[8, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                           layout=pl.DN, valid_shape=[-1, -1])
        t_xbf = pl.TileType(shape=[1, h_blk], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
        t_rbf = pl.TileType(shape=[MHC, h_blk], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
        t_obf = pl.TileType(shape=[MHC, h_blk], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)

        g_cf = pl.make_tile_group(type=t_cf, addrs=addrs["cf"], mutex_ids=mutex["cf"])
        g_pc = pl.make_tile_group(type=t_pc, addrs=addrs["pc"], mutex_ids=mutex["pc"])
        g_xbf = pl.make_tile_group(type=t_xbf, addrs=addrs["xbf"], mutex_ids=mutex["xbf"])
        g_rbf = pl.make_tile_group(type=t_rbf, addrs=addrs["rbf"], mutex_ids=mutex["rbf"])
        g_obf = pl.make_tile_group(type=t_obf, addrs=addrs["obf"], mutex_ids=mutex["obf"])

        with pl.section_vector():
            num_cores = pl.get_block_num()
            core_id = pl.get_block_idx()
            n0 = x.shape[0]
            n1 = x.shape[1]
            h = x.shape[2]
            n_tok = n0 * n1

            # 核内展平：合并连续的 token 轴只重算偏移，无数据搬运。
            x2 = pl.make_tensor(x, [n_tok, h], [h, 1])
            res2 = pl.make_tensor(residual, [n_tok * MHC, h], [h, 1])
            out2 = pl.make_tensor(out, [n_tok * MHC, h], [h, 1])
            post2 = pl.make_tensor(post, [n_tok * MHC, 1], [1, 1])
            comb2 = pl.make_tensor(comb, [n_tok * MHC, MHC], [MHC, 1])

            n_ht = (h + h_blk - 1) // h_blk
            total = n_tok * n_ht
            for work in pl.range(core_id, total, num_cores):
                tok = work // n_ht
                h0 = (work % n_ht) * h_blk
                base = tok * MHC
                valid = pl.min(h_blk, h - h0)

                cf = g_cf.next()
                pc = g_pc.next()
                xbf = g_xbf.next()
                rbf = g_rbf.next()
                obf = g_obf.next()

                _load_tile(cf, comb2, [base, 0], [MHC, MHC], [MHC, 8])
                _load_tile(pc, post2, [base, 0], [MHC, 1], [8, 1])
                _load_tile(xbf, x2, [tok, h0], [1, valid], [1, h_blk])
                _load_tile(rbf, res2, [base, h0], [MHC, valid], [MHC, h_blk])

                mhc_post_fwd_vf(xbf, rbf, cf, pc, obf, valid // VL)

                pl.set_validshape(obf, [MHC, valid])
                pl.store(out2, obf, [base, h0])
                pl.set_validshape(obf, [MHC, h_blk])

    return mhc_post_fwd_kernel


# Kernel cache keyed by the compile-time hidden tile width (lazy: compiled on first call).
_KERNELS: dict[int, object] = {}


def _kernel_for(h_blk: int):
    """Return the (cached) kernel specialised on ``h_blk``, building it on first use."""
    kern = _KERNELS.get(h_blk)
    if kern is None:
        kern = _KERNELS[h_blk] = _make_kernel(h_blk)
    return kern


# ---------------------------------------------------------------------------
# Host-side helpers
# ---------------------------------------------------------------------------
def _validate_inputs(x, residual, post_layer_mix, comb_res_mix, out):
    """校验 dtype/shape/device/连续性，返回 ``(token 数, hidden)``。

    全部用 ``raise ValueError`` 而非 ``assert``：``assert`` 会在 ``python -O`` 下被剥掉，
    那时非法输入会静默进入 kernel 而不是报错。连续性同样是**硬前置条件**，不用
    ``.contiguous()`` 兜底——kernel 的扁平视图按显式连续 stride 构造，wrapper 侧补一次
    设备拷贝既越界又有代价（见 ``wrapper-boundary``）。
    """
    if x.dim() != 3 or residual.dim() != 4:
        raise ValueError(f"unexpected ranks: x.dim()={x.dim()}, residual.dim()={residual.dim()}")
    n0, n1, mhc, h = residual.shape
    if mhc != MHC:
        raise ValueError(f"mhc must be {MHC}, got {mhc}")
    if tuple(x.shape) != (n0, n1, h):
        raise ValueError(f"x shape {tuple(x.shape)} != {(n0, n1, h)}")
    if tuple(post_layer_mix.shape) != (n0, n1, MHC, 1):
        raise ValueError(f"post_layer_mix shape {tuple(post_layer_mix.shape)} != {(n0, n1, MHC, 1)}")
    if tuple(comb_res_mix.shape) != (n0, n1, MHC, MHC):
        raise ValueError(f"comb_res_mix shape {tuple(comb_res_mix.shape)} != {(n0, n1, MHC, MHC)}")
    if x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise ValueError(f"x/residual must be bfloat16, got {x.dtype}/{residual.dtype}")
    if post_layer_mix.dtype != torch.float32 or comb_res_mix.dtype != torch.float32:
        raise ValueError(
            f"post_layer_mix/comb_res_mix must be float32, "
            f"got {post_layer_mix.dtype}/{comb_res_mix.dtype}"
        )
    if h % VL != 0:
        raise ValueError(f"hidden must be a multiple of {VL}, got {h}")
    if out is not None and (out.dtype != residual.dtype or tuple(out.shape) != (n0, n1, MHC, h)):
        raise ValueError(
            f"out must be {(n0, n1, MHC, h)}/{residual.dtype}, got {tuple(out.shape)}/{out.dtype}"
        )
    for name, t in (("x", x), ("residual", residual), ("post_layer_mix", post_layer_mix),
                    ("comb_res_mix", comb_res_mix), ("out", out)):
        if t is None:
            continue
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous (the kernel builds contiguous token views)")
        if t.device != residual.device:
            raise ValueError(f"{name} must live on the same device as residual, got {t.device}")
    return n0 * n1, h


def _num_cores(total_work: int) -> int:
    """Launch width: device vector cores, clamped by the task count (no per-call query)."""
    from pypto_pro.runtime.platform import get_platform_info

    return max(1, min(get_platform_info().vector_core_num, total_work))


# ---------------------------------------------------------------------------
# Public wrapper（校验 / 选 h_blk / 分配输出 / 一次启动；不含计算）
# ---------------------------------------------------------------------------
def mhc_post_fwd(x, residual, post_layer_mix, comb_res_mix, out=None):
    """``out[n,mo,h] = post[n,mo]*x[n,h] + sum_mi comb[n,mi,mo]*residual[n,mi,h]``。

    x               [n0,n1,h]        BF16
    residual        [n0,n1,mhc,h]    BF16
    post_layer_mix  [n0,n1,mhc,1]    FP32
    comb_res_mix    [n0,n1,mhc,mhc]  FP32
    out             [n0,n1,mhc,h]    BF16（``out=None`` 时按 residual 形状新建）
    """
    if out is None:
        out = torch.empty(residual.shape, dtype=residual.dtype, device=residual.device)
    n, h = _validate_inputs(x, residual, post_layer_mix, comb_res_mix, out)
    h_blk = h_blk_of(h)
    kern = _kernel_for(h_blk)
    # 直接按原生形状传入：展平在核内完成（wrapper-boundary 禁止 host 侧 view/reshape）。
    kern[None, _num_cores(n * ((h + h_blk - 1) // h_blk))](
        x, residual, post_layer_mix, comb_res_mix, out)
    return out
