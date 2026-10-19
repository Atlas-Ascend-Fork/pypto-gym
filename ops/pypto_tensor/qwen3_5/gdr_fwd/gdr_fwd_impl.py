# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import warnings
from typing import Optional, Tuple

import pypto
import torch
import torch_npu  # noqa: F401  NPU 设备初始化需要

_EPS_L2 = 1e-6      # L2norm 的加性 eps，加在平方和上（MATH.md §1）

# 常量矩阵缓存：key = (chunk_size, device)。这些矩阵与输入数据无关，
# 每次 wrapper 调用重建一遍要 ~148us（占 E2E 3%）。见 _build_masks。
_MASK_CACHE = {}
_SEQLEN_CACHE = {}     # 非 varlen 段表，key = (B, T, device)
_DUMMY_CACHE = {}      # res_flag/l2_flag 关闭时的 1×1 哑张量，key = (dtype, device)
_RMS_GAMMA_CACHE = {}  # res+l2 host 特化的 fp32 gamma，key = (K, device)

# chunk 计算路径的 vec tile。实测 wall ≈ (AIV任务数/40)×(1.63us忙 + 4.12us固定间隙)，
# 即**任务数**就是时钟。tile 越小任务越多：[64,64] 在 16×64 下要 4 个任务、
# [128,128] 要 16 个，而每个任务不论内容都要付 ~6us 的核槽位。
# ⚠️ [128,128] 实测反而更慢（本路径的 [BT,BT] 矩阵只有 64 行，放大 tile 只是补零），
# 开了 vec_nbuffer 合图之后重扫结论不变，故保持 [64,128]。
_VEC_TR = 64
_VEC_TC = 128

# L2 归一化前置 pass 的分块：行块 512 行、tile [128,128]。
# 归一化按行独立，行块可以开大；512 是上限——1024 行时中间量单个就 262144B，
# 超过 UB 的 196608B，直接 TENSOR_MEMORY_ALLOCATION。
_L2_ROWS = 512
_L2_TR = 128
_L2_TC = 128

# =============================================================================
# Layer H — PyPTO 子内核（每个只做一件事，各自设置所需 tile）
# =============================================================================


def pypto_l2norm(x_bf, tile_r=16, tile_c=64):
    """L2 归一化（MATH.md §1）：``x / sqrt(Σx² + 1e-6)``，fp32 计算、输出回 bf16。

    ``rst`` 是反向需要的残差（``l2norm`` 的 VJP 要 ``rstd`` 与归一化后的 ``y``），
    它本来就是正向的中间量，**一并返回不产生任何额外 FLOP**。

    Args:
        x_bf: ``[R, DK]`` bf16。
        tile_r, tile_c: vec tile 形状。前置 pass 用大行块（见 ``_gdr_fwd_body``），
            故做成参数而非写死 ``(16, 64)``。
    Returns:
        ``(y[R, DK] bf16, rst[R, 1] fp32)``，``rst = rsqrt(Σx² + 1e-6)``。
    """
    pypto.set_vec_tile_shapes(tile_r, tile_c)
    xf = pypto.cast(x_bf, pypto.DT_FP32)                      # [BT,DK] fp32
    ss = pypto.sum(pypto.mul(xf, xf), -1, keepdim=True)       # [BT,1]  fp32
    rst = pypto.rsqrt(pypto.add(ss, _EPS_L2))                 # [BT,1]  eps 在根号内
    return pypto.cast(pypto.mul(xf, rst), pypto.DT_BF16), rst


def pypto_gate_and_a(gp, kp, bp, trilc, strictc):
    """步骤 2/3（MATH.md §3、§4）：γ、两张 decay 矩阵、以及 ``a_neg = -Araw``。

    Args:
        gp:      ``[BT, 1]``  fp32，log 域 gate（尾块已 fillpad 清零）。
        kp:      ``[BT, DK]`` bf16。
        bp:      ``[BT, 1]``  bf16，beta。
        trilc:   ``[BT, BT]`` fp32，下三角**含对角** 1/0。
        strictc: ``[BT, BT]`` fp32，**严格**下三角 1/0。
    Returns:
        ``(gcum[BT,1] fp32, expg[BT,1] fp32, dec_in[BT,BT] fp32,
           a_neg[BT,BT] fp32, bf32[BT,1] fp32)``
    """
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    # chunk-local inclusive 前缀和；tril @ g 规避末轴为 1 的 fp32 transpose 32B 陷阱。
    gcum = pypto.matmul(trilc, gp, pypto.DT_FP32)                     # [BT,1] fp32
    # kkt 的 matmul 操作数是**裸 k**（bf16），beta 稍后在 fp32 上按行乘（MATH.md §4）。
    kkt = pypto.matmul(kp, kp, pypto.DT_FP32, b_trans=True)           # [BT,BT] fp32

    pypto.set_vec_tile_shapes(_VEC_TR, _VEC_TC)
    gdiff = pypto.sub(gcum, pypto.transpose(gcum, 0, 1))              # [BT,BT] γ_t - γ_s
    # 先乘掩码再 exp 再乘掩码（MATH.md 坑 3）：掩码外的指数参数恒为 0，杜绝 0×inf → NaN。
    dec_in = pypto.mul(pypto.exp(pypto.mul(gdiff, trilc)), trilc)      # 含对角（t >= s，坑 2）
    expg = pypto.exp(gcum)                                             # [BT,1] ∈ (0,1]

    bf32 = pypto.cast(bp, pypto.DT_FP32)                               # [BT,1]
    # Araw[t,s] = beta[t]·exp(γt-γs)·(k_t·k_s)；beta 按**行**缩放（坑 2）。
    #
    # 两处省算子（wall ≈ 任务数 × ~6us，算子数就是时钟）：
    #   ① dec_lo 不再独立算一遍 exp —— 严格下三角 ⊂ 含对角，且两者在严格区取值相同，
    #      故 dec_lo = dec_in ⊙ strict，省掉一次 mul+exp+mul；
    #   ② -1 折进掩码常量（strict_neg = -strict），省掉最后那次 mul(·, -1.0)。
    # 合计每 chunk 少 3 个 vector 任务，且数值上逐位等价（都是同一批乘法的重排）。
    a_neg = pypto.mul(pypto.mul(kkt, pypto.mul(dec_in, strictc)), bf32)  # = -Araw
    return gcum, expg, dec_in, a_neg, bf32


_KDA_MIN = 16   # BT=128 分支的 16×16 叶子块边长（= 128/8）

# BT=128 求逆走哪条路：True = §14.11「层内 Batch」层次求逆（batch matmul + batch concat，
# 减少 Cube↔Vector 核切换），False = 原逐块 _kda_inverse8。翻此开关即可 A/B / 回退。
_BATCH_INVERSE = True


# =============================================================================
# BT=128 专用求逆：移植自 chunk_kda（8 个 16×16 叶子前向替换 + 3 级层次 merge）。
# 输入 neg_a = -A（严格下三角），返回 (I+A)^{-1}，与本模块 a_neg / T_inv 约定一致。
# =============================================================================


def _kda_inv_min_length(attn_dim1, eye, row_num, col_num):
    """8 个堆叠的 16×16 对角块 [16,128] → 各自的逆 [16,128]（单位下三角前向替换）。"""
    size = col_num // row_num                                    # 8
    pypto.set_vec_tile_shapes(128, 128)
    cur = attn_dim1[:2, :]                                       # [2,128] 行 0,1
    for i in range(2, row_num, 1):                              # 2..15
        cur = cur + 0.0                                         # 落 UB
        row = attn_dim1.view([1, col_num], [i, 0])             # [1,128] 第 i 行跨 8 块
        row_exp = row.reshape([size, row_num]).view([size, i], [0, 0]).transpose(1, 0).reshape([size * i, 1])
        cur_r = cur.reshape([size * i, row_num])               # [8i,16]
        prod_mul = (row_exp * cur_r).reshape([i, col_num])     # [i,128]
        prod = pypto.sum(prod_mul, dim=0, keepdim=True)        # [1,128]
        cur = pypto.concat([cur, row + prod], dim=0)          # 增长到 [i+1,128]
    return cur + eye                                           # [16,128] 加对角 I


def _stack_batch(tiles):
    """把若干 [m,m] tile 沿新 batch 维堆成 [B,m,m]（reshape+concat，SSA，无就地写）。

    先 ``+0.0`` 落地每个 tile 再 reshape：``neg_a`` 的**跨步 view**（非连续切片）直接
    reshape 到 [1,m,m] 在本 build 上会取到错误数据（full-kernel 下表现为该 batch 结果为 0）。
    """
    m = tiles[0].shape[0]
    pypto.set_vec_tile_shapes(128, 128)
    mats = [t + 0.0 for t in tiles]
    pypto.set_vec_tile_shapes(1, 128, 128)
    return pypto.concat([t.reshape([1, m, m]) for t in mats], dim=0)


def _merge_batch(inv11, inv22, neg_a21):
    """层内批量块 2×2 求逆（设计文档 §14.2）。输入 inv11/inv22/neg_a21 均为 ``[B,m,m]``。

    与逐 merge 数值等价——同样的结合序 / 符号（``tmp = inv22 @ neg_a21``；
    ``x10 = tmp @ inv11``；符号由 ``a_neg = -A`` 承载）与装配布局 ``[[inv11,0],[x10,inv22]]``，
    只是把同一层 B 个独立子问题合到 batch 维、一次 3D Batch Matmul + 一次整 batch Concat
    完成，Cube/Vector 核切换从每块一次降到每层一次。返回 ``[B,2m,2m]``。
    """
    b = inv11.shape[0]
    m = inv11.shape[1]
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    tmp = pypto.matmul(inv22, neg_a21, pypto.DT_FP32)                        # [B,m,m]
    x10 = pypto.matmul(tmp, inv11, pypto.DT_FP32)                            # [B,m,m]
    pypto.set_vec_tile_shapes(1, 128, 128)
    zero = pypto.full(size=[b, m, m], fill_value=0.0, dtype=pypto.DT_FP32)   # [B,m,m] 右上零块
    top = pypto.concat([inv11, zero], dim=2)                                 # [B,m,2m]
    bot = pypto.concat([x10, inv22], dim=2)                                  # [B,m,2m]
    return pypto.concat([top, bot], dim=1)                                   # [B,2m,2m]


def _kda_inverse8_batch(neg_a, eye_stack):
    """**层内 Batch** 层次求逆（设计文档 §14.11 推荐方案），BT=128 唯一路径。返回 [128,128] fp32。

    块求逆公式与 matmul 结合序与逐块递归求逆完全一致，只把每级 4/2/1 个独立 merge
    合并成 3D Batch Matmul + 层末 Batch Concat 连续执行。保留
    ``pypto.concat``（可靠的 SSA 拼接），用 PyPTO 已支持的 3D Batch Matmul，不使用
    「就地 assemble 组装局部 Tensor」（那条路在本 build 上多缓冲并存会调度竞争）。
    收益来自任务合并 / 核切换减少，而非 FLOPs 或 concat 字节量下降（§14.9）。

    第一层用 pair_order ``[(0,1),(4,5),(2,3),(6,7)]``，使第二层的 inv11/inv22 可由
    ``blocks32`` 的连续 batch 段直接 View 取出，避免奇偶 batch 重排（§14.8.4）。
    """
    block_dim = _KDA_MIN                                                             # 16
    pypto.set_vec_tile_shapes(128, 128)
    diag = [neg_a.view([block_dim, block_dim], [block_dim * i, block_dim * i]) + 0.0 for i in range(8)]
    diag_inv = _kda_inv_min_length(pypto.concat(diag, dim=1), eye_stack, block_dim, 128)   # 叶子保留
    li = [diag_inv[:, block_dim * i:block_dim * (i + 1)] + 0.0 for i in range(8)]            # X0..X7 [16,16]

    # ---- 第一层 16→32，B=4，pair_order [(0,1),(4,5),(2,3),(6,7)]（§14.3）----
    inv11_16 = _stack_batch([li[0], li[4], li[2], li[6]])
    inv22_16 = _stack_batch([li[1], li[5], li[3], li[7]])
    neg_a21_16 = _stack_batch([
        neg_a.view([block_dim, block_dim], [block_dim, 0]),
        neg_a.view([block_dim, block_dim], [5 * block_dim, 4 * block_dim]),
        neg_a.view([block_dim, block_dim], [3 * block_dim, 2 * block_dim]),
        neg_a.view([block_dim, block_dim], [7 * block_dim, 6 * block_dim]),
    ])
    blocks32 = _merge_batch(inv11_16, inv22_16, neg_a21_16)                  # [4,32,32]=[B01,B45,B23,B67]

    # ---- 第二层 32→64，B=2（连续 View 取上下两组）----
    inv11_32 = pypto.view(blocks32, [2, 2 * block_dim, 2 * block_dim],
                          [0, 0, 0])
    inv22_32 = pypto.view(blocks32, [2, 2 * block_dim, 2 * block_dim],
                          [2, 0, 0])
    neg_a21_32 = _stack_batch([
        neg_a.view([2 * block_dim, 2 * block_dim], [2 * block_dim, 0]),
        neg_a.view([2 * block_dim, 2 * block_dim], [6 * block_dim, 4 * block_dim]),
    ])
    blocks64 = _merge_batch(inv11_32, inv22_32, neg_a21_32)                  # [2,64,64]=[B0123,B4567]

    # ---- 第三层 64→128，B=1（2D）----
    pypto.set_vec_tile_shapes(1, 128, 128)
    inv11_64 = pypto.view(blocks64, [1, 4 * block_dim, 4 * block_dim],
                          [0, 0, 0]).reshape([4 * block_dim, 4 * block_dim])
    inv22_64 = pypto.view(blocks64, [1, 4 * block_dim, 4 * block_dim],
                          [1, 0, 0]).reshape([4 * block_dim, 4 * block_dim])
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    tmp128 = pypto.matmul(inv22_64, neg_a.view([4 * block_dim, 4 * block_dim], [4 * block_dim, 0]), pypto.DT_FP32)
    x10_128 = pypto.matmul(tmp128, inv11_64, pypto.DT_FP32)
    pypto.set_vec_tile_shapes(128, 128)
    zero64 = pypto.full(size=[4 * block_dim, 4 * block_dim], fill_value=0.0, dtype=pypto.DT_FP32)   # [64,64] 右上零块
    top128 = pypto.concat([inv11_64, zero64], dim=1)                        # [64,128]
    bot128 = pypto.concat([x10_128, inv22_64], dim=1)                       # [64,128]
    return pypto.concat([top128, bot128], dim=0)                            # [128,128]


def pypto_inverse(a_neg, eye_stack):
    """``T_inv = (I + Araw)^{-1}``（MATH.md §5）—— BT=128 的 8×16 层次求逆。

    chunk_size 恒为 128：走移植自 chunk_kda 的 8×16 层次求逆——8 个 16×16 叶子并行
    前向替换（深度仅 ~14 个小 vector op，但 8 块并排一次算完）+ 3 级块 2×2 merge
    （每级 2 个整块 matmul）。z32 零块在 kernel 内 ``pypto.full`` 建，z8/z16 由其
    view 出。

    ``eye_stack [16,128]`` block-diag I = ``torch.eye(16).repeat(1,8)``——常量只依赖
    BT=128，与输入/chunk 无关，故 P0-A 起改由 host 侧一次性构造并缓存（``_build_masks``），
    kernel 直接消费，省掉每 chunk 8 view + 8 ``+0.0`` + 1 concat 的重建。

    全程 fp32，仅最终结果落 bf16。

    Args:
        a_neg: ``[BT, BT]`` fp32，``-Araw``（严格下三角）。
        eye_stack: ``[16, 128]`` fp32，block-diag 单位阵（host 侧提供）。
    Returns:
        ``[BT, BT]`` fp32。
    """
    return _kda_inverse8_batch(a_neg, eye_stack)


def pypto_wy(tinv_bf, vp, kp, bf32, expg):
    """步骤 5（MATH.md §6）：WY 表示 ``u`` / ``w``，bf16 舍入点 3 与 4。

    ``u`` **不带** gate 因子，``w`` **带** ``exp(γ_s)``；三项在 fp32 里乘完再一次性落 bf16。
    舍入点 3/4 由 matmul 的 bf16 输出（cube epilogue）直接完成，不再单独发 vector cast。

    Args:
        tinv_bf: ``[BT, BT]`` bf16。
        vp:      ``[BT, DK]`` bf16。
        kp:      ``[BT, DK]`` bf16。
        bf32:    ``[BT, 1]``  fp32，beta。
        expg:    ``[BT, 1]``  fp32，``exp(γ)``。
    Returns:
        ``(u[BT,DK] bf16, w[BT,DK] bf16)``
    """
    pypto.set_vec_tile_shapes(_VEC_TR, _VEC_TC)
    vb = pypto.cast(pypto.mul(pypto.cast(vp, pypto.DT_FP32), bf32), pypto.DT_BF16)
    kbg = pypto.cast(pypto.mul(pypto.mul(pypto.cast(kp, pypto.DT_FP32), bf32), expg),
                     pypto.DT_BF16)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    # matmul 直接输出 bf16：Cube 内部仍 fp32 累加，落 bf16 在 cube epilogue 完成
    # （round-to-nearest-even，与独立 vector cast 逐 bit 等价 → 保留舍入点 3/4，且与
    #  golden 的 u=round(tinv@vb) / w=round(tinv@kbg) 完全一致）。由此消掉每 chunk 两个
    # 「只含一个 cast」的 vector 子图，并让 w_bf → ws(=w@S_i) 变成 cube→cube 直连，
    # 去掉 WY→输出相之间被迫的 C→V→C 搬运（见 CV_FRAGMENTATION_ANALYSIS.md §2 WY 行）。
    u_bf = pypto.matmul(tinv_bf, vb, pypto.DT_BF16)                   # [BT,DK] bf16（舍入点 3）
    w_bf = pypto.matmul(tinv_bf, kbg, pypto.DT_BF16)                  # [BT,DK] bf16（舍入点 4）
    return u_bf, w_bf


def pypto_out_and_state(qp, kp, u_bf, w_bf, h_bf, state, dec_in, expg, gcum,
                        dk, bt, scale_val):
    """步骤 6/7（MATH.md §7、§8）：``v_new`` → ``o`` → state 递推。

    ``h_bf`` 是**进入本 chunk 之前**的状态（坑 9），且 ``w@S`` 与 ``q@S`` 复用同一份
    bf16 快照（舍入点 5）。``scale`` 施加在**最末端**，``P`` 在乘 scale **前**落 bf16（坑 8）。

    Args:
        qp, kp:  ``[BT, DK]`` bf16。
        u_bf, w_bf: ``[BT, DK]`` bf16。
        h_bf:    ``[DK, DK]`` bf16，``S_i`` 的 bf16 快照。
        state:   ``[DK, DK]`` fp32，``S_i``。
        dec_in:  ``[BT, BT]`` fp32，含对角 decay。
        expg:    ``[BT, 1]``  fp32。
        gcum:    ``[BT, 1]``  fp32。
        dk, bt:  Python int。
        scale_val: Python float。
    Returns:
        ``(o[BT,DK] bf16, S_next[DK,DK] fp32)``
    """
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    ws = pypto.matmul(w_bf, h_bf, pypto.DT_FP32)                      # [BT,DK] w @ S_i
    qkt = pypto.matmul(qp, kp, pypto.DT_FP32, b_trans=True)           # [BT,BT] q @ kᵀ
    o_int = pypto.matmul(qp, h_bf, pypto.DT_FP32)                     # [BT,DK] q @ S_i

    pypto.set_vec_tile_shapes(_VEC_TR, _VEC_TC)
    v_new = pypto.cast(pypto.sub(pypto.cast(u_bf, pypto.DT_FP32), ws),
                       pypto.DT_BF16)                                 # 舍入点 6
    p_bf = pypto.cast(pypto.mul(qkt, dec_in), pypto.DT_BF16)          # 舍入点 8（scale 之前）
    o_int_g = pypto.mul(o_int, expg)                                  # [BT,DK] fp32

    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    pv = pypto.matmul(p_bf, v_new, pypto.DT_FP32)                     # [BT,DK] P @ v_new

    pypto.set_vec_tile_shapes(_VEC_TR, _VEC_TC)
    # o = scale·o_inter + scale·(P @ v_new)：两项**分别**乘 scale，与 B2 逐位一致。
    o_f = pypto.add(pypto.mul(o_int_g, scale_val), pypto.mul(pv, scale_val))
    o_bf = pypto.cast(o_f, pypto.DT_BF16)                             # 舍入点 9

    # ---- state 递推：尾块 γ 平坦，故 γ_last 直接取静态行 BT-1（坑 4）----
    g_last = gcum[bt - 1:bt, :]                                       # [1,1] fp32
    dec_v = pypto.exp(pypto.sub(g_last, gcum))                        # [BT,1] ≤ 1
    # decay 乘在 v_new 上而不是 k 上（坑 7）；v_new 用已落 bf16 的值（舍入点 7）。
    vd = pypto.cast(pypto.mul(pypto.cast(v_new, pypto.DT_FP32), dec_v), pypto.DT_BF16)
    dec_s = pypto.expand_clone(expg[bt - 1:bt, :], (dk, 1))           # [DK,1] e^{γ_last}

    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    kv = pypto.matmul(kp, vd, pypto.DT_FP32, a_trans=True)            # [DK,DK] kᵀ @ vd

    pypto.set_vec_tile_shapes(_VEC_TR, _VEC_TC)
    s_next = pypto.add(pypto.mul(state, dec_s), kv)                   # [DK,DK] fp32
    return o_bf, s_next


def pypto_chunk_core(qc, kc, vc, bc, gc, state, trilc, strictc, eye_stack,
                     ainv_out, off, hv_idx, act,
                     dk, bt, scale_val, res_flag, is_tail):
    """单个 chunk 的完整计算链：γ/A → 求逆 → WY → 输出 + state 递推。

    **满块分支与尾块分支共用本函数**，二者唯一的差别是入参是否先经过 ``fillpad``
    （见 ``_gdr_fwd_body``）—— 计算体只此一份，不随分流复制。

    ⚠️ ``qc/kc`` 必须是**已归一化**的 q/k：``l2_flag == 1`` 时由 ``_gdr_fwd_body``
    的前置 L2 pass 预先算好并从 ``qhat/khat`` 缓冲读入，本函数内不再做归一化。

    Args:
        qc, kc, vc: ``[BT, DK]`` bf16，chunk 切片。满块传裸 view，尾块传 fillpad 后的副本。
        bc:      ``[BT, 1]`` bf16，beta。
        gc:      ``[BT, 1]`` fp32，log 域 gate。
        state:   ``[DK, DK]`` fp32，loop-carry 状态；读入 ``S_i``，**就地更新**为 ``S_i+1``。
        trilc:   ``[BT, BT]`` fp32，下三角含对角 1/0。
        strictc: ``[BT, BT]`` fp32，严格下三角 1/0。
        eye_stack: ``[16, 128]`` fp32，block-diag 单位阵（求逆叶子用）。
        dk, bt: Python int。
        scale_val: Python float。
    Returns:
        ``(o[BT,DK] bf16, gcum[BT,1] fp32)``。``o`` 由调用方按满块/尾块选择写回；
        ``gcum`` 是反向残差。A_inv 在 cast 后立即写到 ``ainv_out``，缩短其存活区间。
    """
    # ---- γ / decay / A ----
    gcum, expg, dec_in, a_neg, bf32 = pypto_gate_and_a(gc, kc, bc, trilc, strictc)

    # ---- 求逆：全程 fp32，只有最终结果落 bf16（舍入点 2）----
    ainv_fp32 = pypto_inverse(a_neg, eye_stack)
    tinv_bf = pypto.cast(ainv_fp32, pypto.DT_BF16)
    if res_flag == 1:
        pypto.set_vec_tile_shapes(_VEC_TR, _VEC_TC)
        if is_tail:
            ainv_v = pypto.view(
                ainv_fp32, [bt, bt], [0, 0], valid_shape=[act, bt])
            pypto.assemble(ainv_v, [off, hv_idx * bt], ainv_out)
        else:
            pypto.assemble(ainv_fp32, [off, hv_idx * bt], ainv_out)

    # ---- WY 表示 ----
    u_bf, w_bf = pypto_wy(tinv_bf, vc, kc, bf32, expg)

    # ---- 输出 + state 递推 ----
    pypto.set_vec_tile_shapes(_VEC_TR, _VEC_TC)
    h_bf = pypto.cast(state, pypto.DT_BF16)                   # 舍入点 5，两处复用
    o_bf, s_next = pypto_out_and_state(
        qc, kc, u_bf, w_bf, h_bf, state, dec_in, expg, gcum, dk, bt, scale_val)

    # [:] 是 MOVE 语义：s_next 此后不可再读。
    state[:] = s_next
    return o_bf, gcum


# =============================================================================
# Layer I — kernel 实现（所有 pypto.loop 都在这里；无 @pypto.frontend.jit）
# =============================================================================


def _gdr_one_head_step(
        q, k, v, beta, gate, cank, tril_incl, strict_low, eye_stack,
        o_out, state_out, gcum_out, ainv_out, qhat_out, khat_out,
        qrstd_out, krstd_out,
        off, act, hv_idx, h_idx, sidx,
        dk, bt, scale_val, res_flag, use_l2norm, is_tail):
    """Build one head's DAG for one chunk; ``is_tail`` is a trace-time bool."""
    pypto.set_vec_tile_shapes(_VEC_TR, _VEC_TC)
    q2 = pypto.view(q, [bt, dk], [off, h_idx * dk], valid_shape=[act, dk])
    k2 = pypto.view(k, [bt, dk], [off, h_idx * dk], valid_shape=[act, dk])
    v2 = pypto.view(v, [bt, dk], [off, hv_idx * dk], valid_shape=[act, dk])
    b2 = pypto.view(beta, [bt, 1], [off, hv_idx], valid_shape=[act, 1])
    g2 = pypto.view(gate, [bt, 1], [off, hv_idx], valid_shape=[act, 1])

    if is_tail:
        qp = pypto.fillpad(q2, "constant", 0.0)
        kp = pypto.fillpad(k2, "constant", 0.0)
        vp = pypto.fillpad(v2, "constant", 0.0)
        bp = pypto.fillpad(b2, "constant", 0.0)
        gp = pypto.fillpad(g2, "constant", 0.0)

        if use_l2norm:
            qp, qrstd = pypto_l2norm(qp, _VEC_TR, _VEC_TC)
            kp, krstd = pypto_l2norm(kp, _VEC_TR, _VEC_TC)

        o_bf, gcum = pypto_chunk_core(
            qp, kp, vp, bp, gp, cank, tril_incl, strict_low,
            eye_stack, ainv_out, off, hv_idx, act,
            dk, bt, scale_val, res_flag, True)

        pypto.set_vec_tile_shapes(_VEC_TR, _VEC_TC)
        o_v = pypto.view(o_bf, [bt, dk], [0, 0], valid_shape=[act, dk])
        pypto.assemble(o_v, [off, hv_idx * dk], o_out)

        if res_flag == 1:
            g_v = pypto.view(gcum, [bt, 1], [0, 0], valid_shape=[act, 1])
            pypto.assemble(g_v, [off, hv_idx], gcum_out)
            if use_l2norm:
                qp_v = pypto.view(qp, [bt, dk], [0, 0], valid_shape=[act, dk])
                kp_v = pypto.view(kp, [bt, dk], [0, 0], valid_shape=[act, dk])
                pypto.assemble(qp_v, [off, h_idx * dk], qhat_out)
                pypto.assemble(kp_v, [off, h_idx * dk], khat_out)
                qr_v = pypto.view(qrstd, [bt, 1], [0, 0], valid_shape=[act, 1])
                kr_v = pypto.view(krstd, [bt, 1], [0, 0], valid_shape=[act, 1])
                pypto.assemble(qr_v, [off, h_idx], qrstd_out)
                pypto.assemble(kr_v, [off, h_idx], krstd_out)

        pypto.assemble(cank, [sidx * dk, 0], state_out)
        return

    if use_l2norm:
        q2, qrstd = pypto_l2norm(q2, _VEC_TR, _VEC_TC)
        k2, krstd = pypto_l2norm(k2, _VEC_TR, _VEC_TC)

    o_bf, gcum = pypto_chunk_core(
        q2, k2, v2, b2, g2, cank, tril_incl, strict_low,
        eye_stack, ainv_out, off, hv_idx, act,
        dk, bt, scale_val, res_flag, False)

    pypto.set_vec_tile_shapes(_VEC_TR, _VEC_TC)
    pypto.assemble(o_bf, [off, hv_idx * dk], o_out)

    if res_flag == 1:
        pypto.assemble(gcum, [off, hv_idx], gcum_out)
        if use_l2norm:
            pypto.assemble(q2, [off, h_idx * dk], qhat_out)
            pypto.assemble(k2, [off, h_idx * dk], khat_out)
            pypto.assemble(qrstd, [off, h_idx], qrstd_out)
            pypto.assemble(krstd, [off, h_idx], krstd_out)


def _gdr_fwd_body(q, k, v, beta, gate, states, tril_incl, strict_low, eye_stack,
                  seqlens, o_out, state_out, gcum_out, ainv_out, qhat_out, khat_out,
                  qrstd_out, krstd_out,
                  hv, grp, dk, bt, scale_val, res_flag, use_l2norm):
    """flat 时间轴上的 chunk-parallel 前向扫描。

    外层把 ``(序列 n, value 头组)`` 合并成一层 loop；组内 head 用 Python 静态展开，
    共享同一个 chunk loop，使独立 head DAG 在一次 root 提交中并行可见。

    ``use_l2norm=True`` 时，q/k 的 L2 归一化内联到 chunk 循环内逐 chunk 完成。
    ``res_flag``（Python int，编译期特化）为 1 时额外写回反向残差。
    """
    pypto.experimental.set_operation_options(combine_axis=True)

    n_seq = seqlens.shape[0] - 1

    for nh in pypto.loop(n_seq * hv, name="seq_head", idx_name="nh"):
        n_idx = nh // hv
        hv_idx = nh - n_idx * hv
        s0 = seqlens[n_idx]
        slen = seqlens[n_idx + 1] - s0
        sidx = n_idx * hv + hv_idx
        h_idx = hv_idx // grp

        pypto.set_vec_tile_shapes(_VEC_TR, _VEC_TC)
        cank = pypto.view(states, [dk, dk], [sidx * dk, 0])

        for c in pypto.loop(0, slen, bt, name="chunk", idx_name="c", unroll_list=[8, 4, 2]):
            off = s0 + c
            act = (slen - c).min(bt)

            if pypto.cond(pypto.is_loop_end(c)):
                _gdr_one_head_step(
                    q, k, v, beta, gate, cank,
                    tril_incl, strict_low, eye_stack,
                    o_out, state_out, gcum_out, ainv_out, qhat_out, khat_out,
                    qrstd_out, krstd_out,
                    off, act, hv_idx, h_idx, sidx,
                    dk, bt, scale_val, res_flag, use_l2norm, True)
            else:
                _gdr_one_head_step(
                    q, k, v, beta, gate, cank,
                    tril_incl, strict_low, eye_stack,
                    o_out, state_out, gcum_out, ainv_out, qhat_out, khat_out,
                    qrstd_out, krstd_out,
                    off, act, hv_idx, h_idx, sidx,
                    dk, bt, scale_val, res_flag, use_l2norm, False)


# =============================================================================
# Layer J — 唯一的 @pypto.frontend.jit 入口（纯类型签名 + 一行委托）
#
# 全模块只此一个 kernel：L2 归一化曾因 RAW 依赖问题被拆成独立 kernel，现已并回
# （见 _gdr_fwd_body），省掉一次 ~370us 的 launch 固定开销。
#
# H / HV / K / V / BT 走**非张量参数**：它们要出现在 pypto.view 的 shape（只接受
# Python int）与 Python range() 里。PyPTO 按值特化 trace。张量注解对应轴写 DYNAMIC。
# =============================================================================

@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 256,
        "device_sched_mode": 1,
        "launch_sched_aicpu_num": 3,
    },
    pass_options={
        "vec_nbuffer_setting": {-2: 1, -1: 32},
        "cube_l1_reuse_setting": {-1: 32},
        "cube_nbuffer_setting": {-1: 8},
    },
)


def _gdr_fwd_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),                   # [TT,H*K] 已归一化
    k: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),                   # [TT,H*K] 已归一化
    v: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),                   # [TT,HV*V]
    beta: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),                # [TT,HV]
    gate: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),                # [TT,HV] log 域
    states: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),              # [N*HV*K,V]
    tril_incl: pypto.Tensor([128, 128], pypto.DT_FP32),                               # [BT,BT] 含对角（静态：BT=128）
    strict_low: pypto.Tensor([128, 128], pypto.DT_FP32),                              # [BT,BT] 严格（静态：BT=128）
    eye_stack: pypto.Tensor([_KDA_MIN, 128], pypto.DT_FP32),                          # block-diag I
    seqlens: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),                           # [N+1]
    o_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),               # [TT,HV*V]
    state_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),           # [N*HV*K,V]
    gcum_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),            # [TT,HV] 反向残差
    ainv_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),            # [TT,HV*BT] A_inv 反向残差
    qhat_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),            # [TT,H*K] 反向残差
    khat_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),            # [TT,H*K] 反向残差
    qrstd_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),           # [TT,H]  反向残差
    krstd_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),           # [TT,H]  反向残差
    hv: int, grp: int, dk: int, bt: int,                                              # 编译期特化
    scale_val: float, res_flag: int,
    use_l2norm: bool = False,
):
    """gdr_fwd JIT 入口。``use_l2norm=True`` 时 q/k 的 L2 归一化在 chunk 循环内逐 chunk
    内联完成（参考 chunk_kda_impl.py），不再走前置 pass。
    """
    _gdr_fwd_body(q, k, v, beta, gate, states, tril_incl, strict_low, eye_stack,
                  seqlens, o_out, state_out, gcum_out, ainv_out, qhat_out, khat_out,
                  qrstd_out, krstd_out,
                  hv, grp, dk, bt, scale_val, res_flag, use_l2norm)


# =============================================================================
# Layer G — host 侧常量矩阵（纯 torch，仅 alloc / 填 0-1，无算子数学）
# =============================================================================


def _build_masks(bt, device):
    """构造 kernel 需要的 3 个常量矩阵。

    ⚠️ 结果按 ``(bt, device)`` 缓存（``_MASK_CACHE``）：这些矩阵只依赖 chunk_size 与
    设备，与输入数据无关，但原实现**每次 wrapper 调用都重建一遍**，实测 ~148us/次，
    占 E2E 的 3%。

    Returns:
        ``(tril_incl[BT,BT], strict_low[BT,BT]（值为 -1，负号已折入）, eye_stack[16,128])``
        均为 fp32，每个由单条 ``torch.*`` 一次性建成（无 Python 循环）。

        ``eye_stack[16,128] = torch.eye(16).repeat(1,8)`` —— BT=128 求逆叶子用的 block-diag
        单位阵。常量只依赖 BT=128，故 host 侧一次性建好并缓存（P0-A），替代原先每 chunk
        在 kernel 内从 ``eye_bt`` 的 8 个对角块拼 ``eye_stack`` 的 8 view + 8 ``+0.0`` + 1 concat。
    """
    key = (bt, str(device))
    hit = _MASK_CACHE.get(key)
    if hit is not None:
        return hit

    ones = torch.ones(bt, bt, dtype=torch.float32, device=device)
    tril_incl = torch.tril(ones, diagonal=0).contiguous()        # t >= s
    # 取负：把 a_neg 的 -1 折进常量，省掉 kernel 里每 chunk 一次 mul(·,-1.0)。
    strict_low = -torch.tril(ones, diagonal=-1).contiguous()     # t > s，值为 -1
    eye_stack = (torch.eye(_KDA_MIN, dtype=torch.float32, device=device)
                 .repeat(1, bt // _KDA_MIN).contiguous())        # [16,128] block-diag I

    out = (tril_incl, strict_low, eye_stack)
    _MASK_CACHE[key] = out
    return out


# =============================================================================
# Layer K — host wrapper（本模块**唯一**对外入口，SPEC §3）
#
# 参数与 B2 (custom/gated_delta_rule/chunk.py) 的 chunk_gated_delta_rule 同序。
# 单层设计：契约校验与 host 侧数据准备合并在同一个函数里，不再套转发壳。
#
# 职责（顺序固定，**校验必须全部先于任何 device 分配 / JIT 调用**，
# 否则非法输入会先吃掉一块显存再报错）：
#   ① 不支持能力检查 + 形状校验  ② layout 展平 + 段表 + state / 掩码准备
#   ③ 预分配输出                ④ 单次 JIT 调用            ⑤ reshape 还原
# =============================================================================


def chunk_gated_delta_rule_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor = None,
    cu_seqlens_cpu: torch.LongTensor = None,
    cp_context=None,
    *,
    return_bwd_residuals: bool = False,
    **kwargs,
) -> tuple:
    """Gated Delta Rule 前向（chunk-parallel），仅前向。

    Args:
        q: ``[B, T, H, K]`` bf16。``use_qk_l2norm_in_kernel=False`` 时要求上游已 L2 归一化。
        k: ``[B, T, H, K]`` bf16。
        v: ``[B, T, HV, V]`` bf16。``HV > H`` 触发 GVA，要求 ``HV % H == 0``。
        g: ``[B, T, HV]`` fp32，**log 空间**遗忘门（kernel 内只做 chunk-local cumsum）。
        beta: ``[B, T, HV]`` bf16，已在 post-sigmoid 空间。
        scale: q 的缩放；``None`` → ``K ** -0.5``。
        initial_state: ``[N, HV, K, V]`` fp32，**K 在 V 前**（``state_v_first=False``）。
        output_final_state: 是否返回 ``final_state``。
        use_qk_l2norm_in_kernel: 内部对 q/k 做 L2 归一化（eps=1e-6，加性）。
        use_beta_sigmoid_in_kernel: **不支持** → ``NotImplementedError``。
        allow_neg_eigval: **不支持** → ``NotImplementedError``。
        state_v_first: 仅接受 ``False``；``True`` → ``NotImplementedError``。
        cu_seqlens: ``[N+1]`` int64，varlen 累积长度；提供时要求 ``B == 1``。
        cu_seqlens_cpu: 兼容位，未使用。
        cp_context: **不支持** → ``NotImplementedError``。
        return_bwd_residuals: 仅关键字。``True`` 时额外返回反向所需的残差字典。
            g_cum/A_inv 由 PyPTO kernel 写回；同时启用 L2 norm 时，q_hat/k_hat/rstd
            由 NPU 融合 RMSNorm 生成并作为主 kernel 的归一化 q/k 输入。
            ``False``（默认）时 kernel 图与开启该参数前逐条指令相同，零开销。
        **kwargs: ``chunk_size``（默认 128，仅接受 128）；``use_gate_in_kernel`` /
            ``A_log`` / ``dt_bias`` **不支持** → ``NotImplementedError``。

    Returns:
        ``return_bwd_residuals=False``（默认）→ ``(o, final_state)``：
        ``o`` 形如 ``[B, T, HV, V]``，dtype 跟随 ``q``；
        ``final_state`` 形如 ``[N, HV, K, V]`` fp32（``output_final_state=False`` 时为 ``None``）。

        ``return_bwd_residuals=True`` → ``(o, final_state, residuals)``，
        ``residuals`` 为 dict：

        * ``"q_hat"``  ``[B,T,H,K]``，dtype 同 ``q`` —— kernel 实际用的 L2 归一化后的 q。
        * ``"k_hat"``  ``[B,T,H,K]``，dtype 同 ``k`` —— 同上，对 k。
        * ``"q_rstd"`` ``[B,T,H]`` fp32 —— ``rsqrt(Σq² + 1e-6)``。
        * ``"k_rstd"`` ``[B,T,H]`` fp32 —— 同上，对 k。
        * ``"g_cum"``  ``[B,T,HV]`` fp32 —— g 的 **chunk-local inclusive** 前缀和，
          **自然对数**域（不含 RCP_LN2）。恒有值。

        前 4 项在 ``use_qk_l2norm_in_kernel=False`` 时为 ``None``（此时 kernel 不做
        归一化，也就不存在 ``q_hat``；且不会为它们分配任何显存）。同时请求 residual
        与 L2 norm 时，host 侧用融合 RMSNorm 产生这 4 项，并把已归一化 q/k 传入 PyPTO。

    Raises:
        ValueError: 头数不匹配 / ``chunk_size`` 非法 / varlen 下 ``B != 1`` /
            ``initial_state`` 条数与序列数不符 / ``K != V``。
        NotImplementedError: 请求了本前向精简版不支持的能力。

    Note:
        host 侧把 varlen 与非 varlen 统一成「flat 时间轴 ``[TT, ...]`` + ``seqlens`` 段表」，
        并把头维**折叠进列**（``[TT, H*K]`` / ``[TT, HV*V]``），使 kernel 内每个 chunk 切片
        都是纯 2D view；随后**只调一次** ``_gdr_fwd_kernel``，再 reshape 回用户 layout。

    Example:
        >>> o, ht = chunk_gated_delta_rule_wrapper(q, k, v, g, beta, output_final_state=True)
    """
    # ---- 不支持的能力（SPEC §8）----
    _unsupported = []
    if use_beta_sigmoid_in_kernel:
        _unsupported.append("use_beta_sigmoid_in_kernel")
    if allow_neg_eigval:
        _unsupported.append("allow_neg_eigval")
    if state_v_first:
        _unsupported.append("state_v_first=True")
    if cp_context is not None:
        _unsupported.append("cp_context")
    if kwargs.get("use_gate_in_kernel", False):
        _unsupported.append("use_gate_in_kernel")
    if kwargs.get("A_log", None) is not None:
        _unsupported.append("A_log")
    if kwargs.get("dt_bias", None) is not None:
        _unsupported.append("dt_bias")
    if _unsupported:
        raise NotImplementedError(
            f"gdr_fwd 是前向精简版，不支持：{', '.join(_unsupported)}。"
            f"如需完整能力请使用 B2 (custom/gated_delta_rule/chunk.py)。"
        )

    if "head_first" in kwargs:
        warnings.warn(
            "head_first is deprecated: gdr_fwd only accepts the [B, T, H, K] layout, this parameter will be ignored.",
            DeprecationWarning,
            stacklevel=2,
        )

    # ---- 形状与取值校验（对齐 B2 chunk.py:517-556）----
    if q.shape[2] != k.shape[2]:
        raise ValueError(
            f"q and k must have the same number of heads, "
            f"but got q.shape[2]={q.shape[2]} and k.shape[2]={k.shape[2]}"
        )
    h, hv = q.shape[2], v.shape[2]
    if hv % h != 0:
        raise ValueError(
            f"For GVA, num_v_heads (HV={hv}) must be evenly divisible by "
            f"num_heads (H={h}), but got HV % H = {hv % h}"
        )
    chunk_size = int(kwargs.get("chunk_size", 128))
    if chunk_size != 128:
        raise ValueError(
            f"`chunk_size` must be 128 for Gated Delta Rule (gdr_fwd), "
            f"got {chunk_size}."
        )
    if q.shape[-1] != v.shape[-1]:
        raise ValueError(
            f"gdr_fwd 当前仅支持 K == V，got K={q.shape[-1]}, V={v.shape[-1]}"
        )
    # 头维（head dim K==V）恒为 128：kernel 的 [128,128] 分块求逆按此特化。
    # 用传入 state 的末维判断（无 state 时回退 v 的末维）。
    _head_dim = initial_state.shape[-1] if initial_state is not None else v.shape[-1]
    if _head_dim != 128:
        raise ValueError(
            f"gdr_fwd 当前仅支持 head dim (K==V) == 128，got {_head_dim}"
        )
    if g.shape[2] != hv or beta.shape[2] != hv:
        raise ValueError(
            f"g/beta 的头数必须等于 HV={hv}，收到 g={g.shape[2]}, beta={beta.shape[2]}"
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} "
                f"when using `cu_seqlens`. Please flatten variable-length inputs "
                f"before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number "
                f"of input sequences, i.e., {len(cu_seqlens) - 1} rather than "
                f"{initial_state.shape[0]}."
            )

    # ==== 以上全部为校验；以下才允许分配显存 / 调 kernel ====================
    out_dtype = q.dtype
    device = q.device
    b, t, k_dim = q.shape[0], q.shape[1], q.shape[3]
    v_dim = v.shape[3]
    bt = chunk_size                                  # 已在上方校验并转 int（恒为 128）
    grp = hv // h

    # res+l2 专用路径：RMSNorm(x, gamma=1/sqrt(K), eps=EPS/K) 与
    # x / sqrt(sum(x²)+EPS) 数学等价。FP32 融合输出回原 dtype；RMSNorm 返回的
    # reverse-rms 再除 sqrt(K)，即 PyPTO L2Norm 所需的 rstd。
    # 其余路径不进入此分支，传给 kernel 的 q/k 与 use_l2norm 标量保持原样。
    _host_l2_res = return_bwd_residuals and use_qk_l2norm_in_kernel
    _kernel_use_l2norm = use_qk_l2norm_in_kernel
    _q_kernel, _k_kernel = q, k
    _qhat_host = _khat_host = _qrstd_host = _krstd_host = None
    if _host_l2_res:
        _gk = (k_dim, str(device))
        _gamma = _RMS_GAMMA_CACHE.get(_gk)
        if _gamma is None:
            _gamma = torch.full(
                (k_dim,), float(k_dim ** -0.5), dtype=torch.float32, device=device)
            _RMS_GAMMA_CACHE[_gk] = _gamma
        _eps_rms = float(_EPS_L2 / k_dim)
        _qhat_host, _qrstd_host = torch_npu.npu_rms_norm(
            q, _gamma, epsilon=_eps_rms)
        _khat_host, _krstd_host = torch_npu.npu_rms_norm(
            k, _gamma, epsilon=_eps_rms)
        _qhat_host = _qhat_host.to(q.dtype)
        _khat_host = _khat_host.to(k.dtype)
        _qrstd_host = (_qrstd_host / float(k_dim ** 0.5)).reshape(b, t, h)
        _krstd_host = (_krstd_host / float(k_dim ** 0.5)).reshape(b, t, h)
        _q_kernel, _k_kernel = _qhat_host, _khat_host
        _kernel_use_l2norm = False

    # ---- 1. layout：varlen 与非 varlen 统一成 flat 时间轴 + 段表 ----
    # 头维**折叠进列**（[TT, H*K] / [TT, HV*V]），使 kernel 内每个 chunk 切片都是
    # 纯 2D view（列偏移 h*K），彻底避免改变 rank 的 reshape 破坏 vec tile 元数据。
    q2 = _q_kernel.reshape(b * t, h * k_dim).to(torch.bfloat16).contiguous()     # [TT,H*K]
    k2 = _k_kernel.reshape(b * t, h * k_dim).to(torch.bfloat16).contiguous()
    v2 = v.reshape(b * t, hv * v_dim).to(torch.bfloat16).contiguous()    # [TT,HV*V]
    beta2 = beta.reshape(b * t, hv).to(torch.bfloat16).contiguous()
    gate2 = g.reshape(b * t, hv).to(torch.float32).contiguous()          # log 域，保持 fp32

    if cu_seqlens is None:
        # 非 varlen 的段表只依赖 (b, t, device)，缓存掉：每次重建一个 device 张量
        # 要付一次 kernel launch，而 host 侧总开销已占 E2E 的 ~17%。
        _sk = (b, t, str(device))
        seqlens = _SEQLEN_CACHE.get(_sk)
        if seqlens is None:
            seqlens = torch.arange(0, b * t + 1, t, dtype=torch.int32,
                                   device=device).contiguous()
            _SEQLEN_CACHE[_sk] = seqlens
        n_seq = b
    else:
        seqlens = cu_seqlens.to(device=device, dtype=torch.int32).contiguous()
        n_seq = seqlens.numel() - 1

    # ---- 2. initial_state → [N*HV*K, V] fp32（None ⇒ 零状态）----
    if initial_state is None:
        states = torch.zeros(n_seq * hv * k_dim, v_dim, dtype=torch.float32, device=device)
    else:
        states = (initial_state.to(device=device, dtype=torch.float32)
                  .contiguous().reshape(n_seq * hv * k_dim, v_dim))

    tril_incl, strict_low, eye_stack = _build_masks(bt, device)
    scale_val = float(scale) if scale is not None else float(k_dim ** -0.5)

    # ---- 3. 预分配输出（torch.*，显式 dtype / device）----
    o_out = torch.empty(b * t, hv * v_dim, dtype=torch.bfloat16, device=device)
    state_out = torch.zeros(n_seq * hv * k_dim, v_dim, dtype=torch.float32, device=device)

    # ---- 3b. 反向残差缓冲 ----
    # res_flag 是编译期特化的 Python int：关闭时下面的 1×1 哑张量既不会被
    # kernel 访问，对应的 store 也根本不会被 trace 出来。哑张量只为满足 JIT 的
    # 张量签名（张量参数必须先于标量参数），代价是 4 个 float 的显存。
    # 常规 L2 归一化内联到 chunk 循环；res+l2 专用路径已由融合 RMSNorm 生成
    # qhat/khat/rstd，因此 PyPTO 侧对应输出使用 dummy，不分配也不 assemble。
    res_flag = 1 if return_bwd_residuals else 0
    _want_kernel_l2_res = res_flag == 1 and _kernel_use_l2norm

    def _buf(rows, cols, dtype, alive):
        if not alive:                                # 哑张量：形状/内容都无意义，全局复用
            key = (str(dtype), str(device))          # 免掉每次调用 5 次 1×1 device 分配
            hit = _DUMMY_CACHE.get(key)
            if hit is None:
                hit = torch.zeros(1, 1, dtype=dtype, device=device)
                _DUMMY_CACHE[key] = hit
            return hit
        return torch.empty(rows, cols, dtype=dtype, device=device)

    qhat_out = _buf(b * t, h * k_dim, torch.bfloat16, _want_kernel_l2_res)
    khat_out = _buf(b * t, h * k_dim, torch.bfloat16, _want_kernel_l2_res)
    qrstd_out = _buf(b * t, h, torch.float32, _want_kernel_l2_res)
    krstd_out = _buf(b * t, h, torch.float32, _want_kernel_l2_res)
    gcum_out = _buf(b * t, hv, torch.float32, res_flag == 1)
    ainv_out = _buf(b * t, hv * bt, torch.float32, res_flag == 1)

    # ---- 4. 唯一的 kernel 调用 ----
    _gdr_fwd_kernel(
        q2, k2, v2, beta2, gate2, states,
        tril_incl, strict_low, eye_stack, seqlens,
        o_out, state_out, gcum_out, ainv_out, qhat_out, khat_out, qrstd_out, krstd_out,
        hv, grp, k_dim, bt, scale_val, res_flag, _kernel_use_l2norm
    )

    # ---- 5. reshape 还原到用户 layout ----
    # varlen（cu_seqlens 非空）下 host 侧强制 B == 1，故 [TT, ...] → [1, T, ...]
    # 与非 varlen 走完全相同的 reshape，无需分支——与 o 的处理方式一致。
    o = o_out.reshape(b, t, hv, v_dim)
    final_state = state_out.reshape(n_seq, hv, k_dim, v_dim)
    o = o.to(out_dtype)
    fs = final_state if output_final_state else None

    if res_flag == 0:
        return o, fs

    residuals = {
        "q_hat": _qhat_host if _host_l2_res else (
            qhat_out.reshape(b, t, h, k_dim).to(q.dtype)
            if _want_kernel_l2_res else None),
        "k_hat": _khat_host if _host_l2_res else (
            khat_out.reshape(b, t, h, k_dim).to(k.dtype)
            if _want_kernel_l2_res else None),
        "q_rstd": _qrstd_host if _host_l2_res else (
            qrstd_out.reshape(b, t, h) if _want_kernel_l2_res else None),
        "k_rstd": _krstd_host if _host_l2_res else (
            krstd_out.reshape(b, t, h) if _want_kernel_l2_res else None),
        "g_cum": gcum_out.reshape(b, t, hv),
        "A_inv": ainv_out.reshape(b, t, hv, bt),
    }
    return o, fs, residuals
