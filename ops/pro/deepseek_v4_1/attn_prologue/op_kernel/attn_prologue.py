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

"""attn_prologue：手写 PyPTO-Pro（``@pl.jit``）融合 MIX 算子。

一个 MIX kernel 完成 attn_prologue 的全部计算，两条流（``section_cube`` AIC /
``section_vector`` AIV）并发，用两次 ``pl.system.sync_all(MIX)`` 全局屏障衔接：

* **A1（Cube）**：``qa = X @ Wqa^T``、``kv = X @ Wkv^T``（MXFP8 ``matmul_mx``，
  FP32 累加，K=H=5120），逐 128 列块写 GM workspace ``ws_qa[T,R]`` / ``ws_kv[T,D]``。
* **barrier1**：``sync_all(MIX)``，投影结果对 Vector 可见。
* **A2（Vector）**：逐行 ``ws_qa`` → RMSNorm(gamma_qr) → 动态 MXFP8 量化 → ``qr``(e4m3)
  + ``descale_qr``(e8m0)；逐行 ``ws_kv`` → RMSNorm(gamma_kv) → 末尾 Dr TailRoPE
  (GPTJ_INTERLEAVED) → BF16 → kv_compress FP8 group-32（payload + BF16 scale）
  → 按 ``cache_index`` 散写 544B ``kv_cache`` 记录。
* **barrier2**：``sync_all(MIX)``，``qr`` 对 Cube 可见（GM store→reload 一致性）。
* **B（Cube）**：逐 (head, col-block) ``q_proj = qr @ Wqb^T``（MXFP8，K=R=1280），
  acc ``[BASE_M,BN_B]`` 经 ``DualModeSplitM`` 搬到 ``cv_ub``，逐 beat ``set_cross_core``。
* **B-epi（Vector）**：逐 beat ``wait_cross_core`` → 每 head 最后一个 col-block 的末尾
  Dr 做 TailRoPE(GPTJ_INTERLEAVED) → BF16 → 写 ``q[T,N,D]``。

冻结几何：``H=5120, R=1280, N=64, D=512, Dr=64``，``T`` 是唯一动态轴，按 ``BASE_M=128``
行切 tile；权重 L2 策略由 tiling key 隔离（``Profile=0``：``T<=128`` 用
``NotAllocKeep``；``Profile=1``：``T>128`` 用默认策略），工作分配仍按运行期
``n_tiles`` 在 ``SPLIT_T_TILES`` 处二选一。

kernel 为 MTE2（GM→L1 载入）带宽受限（``aic_mte2_ratio`` 0.8–0.9，MAC 重叠在载入之下），设计围绕最小化与隐藏载入流量展开（详见 ``DESIGN.md``）：
* **工作分配**：A1 按 ``(tile, col-block)`` 铺开；B 在 ``n_tiles>=32`` 走 split-T（一核独占整
  tile、``qr`` 每 tile 只驻留一次），否则按 ``(tile, head, col-block)`` 单元以 ``num_cores``
  跨步铺开——相邻核同一时刻读相邻 128 行 Wqb 带，NZ k-frac 页内密度最大化，使 42 MB
  权重流以 L2→L1 聚合流带宽满速运行。
* **A1 X 复用**：prefill 用 G=2 分组（2 个输出块共享一次 X K-窗口载入，X 流量减半），decode 退回逐块最大化并行。
* **A1 split-K（decode，T<=128）**：门控 ``n_tiles*NA1*2 <= num_cores`` 时 K=H 切成 2 个分区，
  28 个 ``(blk,ks)`` 单元铺满核（ua=ks*NA1+blk 保证带内局部性），部分和写 ``ws`` 的第
  ``ks*n_tiles*128`` 行带；A2 双臂各载两带一次 ``pl.add`` 合并（2 带时与 Kahan 等价），wrapper
  的 ws 行数门控与 kernel 内门控逐字节一致。
* **载入效率**：``K0=256`` 增大 MTE2 突发；L1 权重窗口 depth-4 预取；``sw_k/sa_l1/sq_k``
  等小 scale 缓冲 depth-2 轮转消除逐 beat WAR；distributed-B 的 ``qr/scale`` 冗余载入按 tile-change 提升。
* **A2 并行化（decode）**：``n_tiles<=2`` 时行循环拆成独立 QR/KV 任务（``2*t_live`` 个任务铺满
  64 AIV）；prefill 保持行串行循环。
  ``gamma_qr/gamma_kv`` 为 kernel 输入、不依赖 A1，其载入提升到 barrier1 之前与屏障等待重叠。

数值规则与独立 golden（``tests/assets/attn_prologue_golden_cpu.py``）严格一致：QB 消费量化后的
``qr``；KV 先 norm+rope、转 BF16，再按 kv_compress 规则量化；RoPE 是 GPTJ_INTERLEAVED，符号由
sin/cos 表承载；MXFP8 共享指数 ``e=max(floor(log2(amax))-8,-127)``，零组 payload=0 且
scale 字节=0，倒数 ``2^-e`` 由 FP32 指数字段构造。

建议阅读顺序：公开入口 ``attn_prologue`` → 主 kernel ``attn_prologue_kernel`` → QB 阶段 helper
``_qb_beat`` / ``_q_epi_beat`` → 四个 VF helper ``qr_row_vf`` / ``kv_row_vf`` / ``q_bf16_vf`` /
``q_rope_bf16_vf``。
"""

from __future__ import annotations

from typing import NamedTuple

import dataclasses
import torch

import torch_npu  # noqa: F401

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info
from pypto_pro.runtime.tilingkey import TilingKeyField

# ---------------------------------------------------------------------------
# Frozen geometry / tiling constants
# ---------------------------------------------------------------------------
H = 5120            # hidden size (QA/KV reduction)
R = 1280            # rank size (QA output width / QB reduction)
N = 64              # head count
D = 512             # head size (KV output width / per-head Q width)
DR = 64             # RoPE width (tail)
DR_HALF = DR // 2   # 32

BASE_M = 128        # T-tile height (GEMM M)
ROWS_VEC = BASE_M // 2   # 64 rows per AIV half (DualModeSplitM)
VL = 64             # FP32 lanes per vector register

K0 = 256            # L0 K step (multiple of 64, mandatory for matmul_mx).
                    # 128->256 halves the K-loop count (NK_A 40->20, NK_B 10->5) so
                    # every GM->L1 weight/X burst doubles (16KB->32KB) — fewer, larger
                    # MTE2 transactions for this MTE2-bound kernel.  L0A/L0B depth-2
                    # is then EXACTLY full (2x32KB=64KB); L0_MX 2x2KB=4KB exactly.
SK0 = K0 // 32      # 8 MX scale groups per K step
NK_A = H // K0      # 20  (QA/KV K steps)
NK_B = R // K0      # 5   (QB K steps)
SQ_ALL_A = H // 32  # 160 (A-scale groups over full H)
SQ_ALL_B = R // 32  # 40  (A-scale groups over full R)

BN_A = 128                 # QA/KV output column block
BN_B = 128                 # QB output column block
QA_BLOCKS = R // BN_A      # 10
KV_BLOCKS = D // BN_A      # 4
HEAD_BLOCKS = D // BN_B    # 4 col-blocks per head
BEATS_PER_TILE = N * HEAD_BLOCKS   # 256
NA1 = QA_BLOCKS + KV_BLOCKS        # 14 A1 output col-blocks per tile (QA then KV)
# A1 groups 2 output col-blocks per unit so ONE X K-window load
# feeds both blocks' MMA -> X L2->L1 traffic halves (14 reloads/tile -> 7).  Needs
# 2 L0C accumulators (a1_acc depth-2) and 2 weight-scale slots (sb_l1 depth-2).
# QA_BLOCKS(10) and KV_BLOCKS(4) are both even, so groups are uniform (no rag).
A1_GRP = 2
QA_GROUPS = QA_BLOCKS // A1_GRP    # 5
KV_GROUPS = KV_BLOCKS // A1_GRP    # 2
NA1G = QA_GROUPS + KV_GROUPS       # 7 A1 groups per tile
# Work-distribution threshold: with at least this many T-tiles, split-T (one core
# owns a whole tile, qr staged once per tile) is best; below it, spread the QB
# (tile,head) units and the A1 (tile,block) units across all cores so decode/mid-T
# engage every core instead of idling 30 of 32.  Mirrors indexer_prologue_qw.
SPLIT_T_TILES = 32

MX_GROUP = 32
E8M0_BIAS = 127
QUANT_EXP_HEADROOM = 8
FP32_EXP_SHIFT = 23
E4M3_MAX = 448.0
INV_RANK = 1.0 / R        # hoisted: scalar float division is illegal inside a VF
INV_HEAD = 1.0 / D
QR_GROUPS = R // MX_GROUP       # 40
QR_PAD = ((R + VL - 1) // VL + 1) * VL   # 1344: pad qa/gamma so the last 32-group's
#                                          64-lane load stays in-tile
KV_VEC = D // VL                # 8 full 64-lane vectors per KV row
KV_QUANT_GROUPS = D // MX_GROUP  # 16 group-32 scales in the KV cache record
KV_RECORD_BYTES = D + 2 * KV_QUANT_GROUPS  # 544: E4M3 payload + BF16 scales
INV_E4M3_MAX = 0.002232142857142857        # 1/448, same literal as kv_compress

FRACTAL_NZ = 29
T_MAX = 262144


class AttnPrologueTilingKey:
    """Weight L2 policy + optional qr_norm emission (host tiling keys).

    Profile 0 (T <= BASE_M, one tile): stream weights with NotAllocKeep so dirty
    L2 lines are not evicted.  Profile 1 (T > BASE_M, multiple tiles): keep the
    default policy so the weights stay resident across tiles.
    QrNorm 0: the optional qr_norm output is absent; QrNorm 1: it is emitted.
    """
    Profile = TilingKeyField(bits=1, values=[0, 1])
    QrNorm = TilingKeyField(bits=1, values=[0, 1])


@dataclasses.dataclass
class AttnPrologueTiling:
    t: int
    cache_rows: int
    block_size: int
    block_stride: int
    row_stride: int
    ws_rows: int
    norm_eps: float
    emit_qr_norm: int

# cross-core event ids: 0,1 = Cube->Vec "acc ready"; 2,3 = Vec->Cube "slot free".
# sync_all(MIX) steals 11,12,13 (AIC) / 12,13 (AIV) -> never use them by hand.
EV_READY = 0
EV_FREE = 2
_AIC_RESERVED = (11, 12, 13)

# ---------------------------------------------------------------------------
# On-chip address maps (byte offsets).  There is no allocator and overlap is not
# checked, so every map ends in an explicit fit assertion (see _check_constants).
# ---------------------------------------------------------------------------
# L1 (Mat, 512 KB)  — mapped for K0=256.  a_l1 (X K-slice) and b_nz (W
# K-slice) double in bytes; the depth-4 weight rotation is kept contiguous.  All
# region starts are 8 KB (0x2000) aligned as before; total 0x6C000=432 KB<=512 KB.
L1_A_L1 = 0x00000        # X K-slice [BASE_M,K0=256] e4m3 NZ, depth2 (0x00000,0x08000)=64KB
L1_SA_L1 = 0x10000       # descale_x [BASE_M,SQ_ALL_A] e8m0 ZZ (20KB)
L1_B_NZ = 0x16000        # W (wqa/wkv/wqb) K-slice [BN,K0=256] e4m3 NZ, depth4 slots 0,1
L1_SB_L1 = 0x36000       # descale_w [SQ_ALL_A,BN_A] e8m0 NN, depth2 (40KB)
L1_QR_L1 = 0x40000       # qr resident [BASE_M,R] e4m3 NZ (QB, 160KB)
L1_SQ_K = 0x68000        # descale_qr [BASE_M,SQ_ALL_B] e8m0 ZZ (QB)
L1_SW_K = 0x6A000        # descale_wqb [SQ_ALL_B,BN_B] e8m0 NN (QB), depth2 slot0
# sw_k rotates depth-2 (slot1 at 0x6C000): a single slot would make every beat's
# descale_wqb load WAR-stall on the previous beat's L1->L0B_MX scale moves; the
# rotation lets beat k+1's load issue while beat k's reads retire (same mechanism
# class as the b_nz rotation and the qr-load tile-change hoist).
L1_SW_K2 = 0x6C000       # descale_wqb depth2 slot1
# The two remaining scale buffers also rotate depth-2 (parked in the free L1
# tail) so a unit's load never WAR-stalls on the previous unit's last
# L1->L0_MX moves (same mechanism as the sw_k rotation).
L1_SA_L1_2 = 0x6E000     # descale_x depth2 slot1 (20KB)
L1_SQ_K2 = 0x74000       # descale_qr depth2 slot1 (5KB)
L1_B_NZ2 = 0x26000       # W K-slice depth4 slots 2,3 (0x26000,0x2E000)
L1_END = 0x7C000
L1_BYTES = 512 * 1024
# B 2-tile pair (pair-major QR hoist): A1 and B are separated by barrier2, so
# the B-only buffers are overlaid on the A1-dead region + the B tail.
# Layout during B:
#   QR tile0 [BASE_M,R]      0x40000 (existing L1_QR_L1, 160 KB)
#   QR tile1 [BASE_M,R]      0x00000 (overlay A1 X/scale/weight, 160 KB)
#   Wqb 5 x [BN_B,K0]        0x28000,0x30000,0x38000,0x68000,0x70000 (160 KB)
#   descale_qr tile0/tile1   0x78000,0x79400 (5 KB each)
#   descale_wqb full         0x7A800 (5 KB)
L1_QR1 = 0x00000         # QR tile1 [BASE_M,R] e4m3 NZ (160 KB)
L1_WBLK = 0x28000        # first of five Wqb K-slice slots
L1_SQK0 = 0x78000        # QR tile0 scale [BASE_M,SQ_ALL_B] e8m0 ZZ
L1_SQK1 = 0x79400        # QR tile1 scale
L1_SWBLK = 0x7A800       # descale_wqb [SQ_ALL_B,BN_B] e8m0 NN

# L0A (Left, 64 KB) / L0B (Right, 64 KB): [BASE_M,K0] / [K0,BN] e4m3, depth2
# At K0=256 each slot is 32 KB so depth-2 EXACTLY fills the 64 KB L0A/L0B.
L0A_BASE = 0x0000        # slots 0x0000, 0x8000
L0B_BASE = 0x0000        # slots 0x0000, 0x8000
L0_SLOT = BASE_M * K0    # 32 KB
L0_BYTES = 64 * 1024
# L0A_MX / L0B_MX (4 KB each): scale slot address == data slot address >> 4
L0MX_BASE = 0x0000       # slots 0x0000, 0x0800  (== 0x0000>>4, 0x8000>>4)
L0MX_BYTES = 4 * 1024

# L0C (Acc, 256 KB)
L0C_A1_ACC = 0x00000     # QA/KV acc [BASE_M,BN_A] FP32, depth2 (0x00000,0x10000)
L0C_Q_ACC = 0x20000      # QB acc [BASE_M,BN_B] FP32, depth2 (0x20000,0x30000)
L0C_END = 0x40000
L0C_BYTES = 256 * 1024

# UB (Vec, usable 216 KB)
UB_CV0 = 0x00000         # cv_ub slot0 [ROWS_VEC,BN_B] FP32 (32 KB)
UB_CV1 = 0x08000         # cv_ub slot1
UB_QA_ROW = 0x10000      # A2: qa row [1,QR_PAD] FP32
UB_GAMMA_QR = 0x11600    # A2: gamma_qr [1,QR_PAD] FP32
UB_QR_PAYLOAD = 0x12C00  # A2: qr payload [1,R] UINT8
UB_EXP = 0x13200         # A2: exponent scratch [1,VL] INT32
UB_SCALE_U8 = 0x13400    # A2: packed scale bytes [1,VL] UINT8
UB_KV_ROW = 0x13600      # A2: kv row [1,D] FP32
UB_GAMMA_KV = 0x13E00    # A2: gamma_kv [1,D] FP32
UB_KVB = 0x14600         # A2: kv record [1,544] UINT8 (payload + BF16 scales)
UB_COS_ROW = 0x14A00     # A2: cos row [1,DR] FP32
UB_SIN_ROW = 0x14C00     # A2: sin row [1,DR] FP32
UB_Q_BF16_0 = 0x15000    # B-epi: q bf16 out [ROWS_VEC,BN_B] depth2 (0x15000,0x19000)
UB_Q_BF16_1 = 0x19000
UB_COS_UB = 0x1D000      # B-epi: cos [ROWS_VEC,DR] FP32
UB_SIN_UB = 0x21000      # B-epi: sin [ROWS_VEC,DR] FP32
# L1 split-K (decode-only, gate n_tiles*NA1*2 <= num_cores): second ws band row
# staged for the A2 merge (fl(band0+band1) == 2-band Kahan, comp starts at 0).
UB_QA_ROW2 = 0x25000     # A2 merge: qa band-1 row [1,QR_PAD] FP32
UB_KV_ROW2 = 0x26600     # A2 merge: kv band-1 row [1,D] FP32
# Barrier-window preload: stage A2's first KV_PRE cos/sin rows BEFORE barrier1
# (the vector side is idle during A1) so the task-split KV arm skips those GM
# latencies inside the exposed A2 window.
# KV_PRE=4 covers EVERY KV task in the gated domain (n_tiles==2 => t_live<=256,
# stripe 64 => <=4 KV tasks per AIV); slots are [1,DR] FP32 at 512B stride.
KV_PRE = 4
UB_COS_PRE = 0x26E00     # preloaded cos rows, KV_PRE slots (256B used / 512B stride)
UB_SIN_PRE = 0x27600     # preloaded sin rows, KV_PRE slots
UB_QR_NORM = 0x27E00     # A2: normalized QR BF16 [1,R]
UB_COS_UB2 = 0x28800     # B-epi: tile1 cos [ROWS_VEC,DR] FP32
UB_SIN_UB2 = 0x2C800     # B-epi: tile1 sin [ROWS_VEC,DR] FP32
UB_END = 0x30800
UB_BYTES = 216 * 1024
UB_PRE_STRIDE = 512
# Preload mutex reservation: section_vector ids 0..17 are taken by the A2 /
# B-epi / split-K tiles (see make_tile_group calls); cos_pre/sin_pre take 18..25.
COS_PRE_MUTEX = (18, 19, 20, 21)
SIN_PRE_MUTEX = (22, 23, 24, 25)
_VEC_MUTEX_TAKEN = tuple(range(0, 18))


def _check_constant(cond, msg):
    """Import-time hard check (module ``assert`` is stripped under ``python -O``)."""
    if not cond:
        raise RuntimeError(msg)


def _check_memory_map():
    # memory-map fit
    _check_constant(L1_END <= L1_BYTES, f"L1 overflow: {L1_END} > {L1_BYTES}")
    _check_constant(L1_QR1 + BASE_M * R <= L1_WBLK,
                    "B QR tile1 overlay overflows into Wqb slots")
    _check_constant(L1_WBLK + 3 * L0_SLOT <= L1_QR_L1,
                    "B Wqb 3-slot overlay overflows into QR tile0")
    _check_constant(0x70000 + L0_SLOT <= L1_SQK0,
                    "B Wqb 5th slot overflows into QR scale0")
    _check_constant(L1_SQK0 + BASE_M * SQ_ALL_B <= L1_SQK1,
                    "B QR scale0 overflows into QR scale1")
    _check_constant(L1_SQK1 + BASE_M * SQ_ALL_B <= L1_SWBLK,
                    "B QR scale1 overflows into Wqb scale")
    _check_constant(L1_SWBLK + SQ_ALL_B * BN_B <= L1_END,
                    "B Wqb scale overflows L1_END")
    _check_constant(L0C_END <= L0C_BYTES, f"L0C overflow: {L0C_END} > {L0C_BYTES}")
    # A1 depth-2 region fits (no overlap into the next region)
    _check_constant(L1_SB_L1 + 2 * SQ_ALL_A * BN_A <= L1_QR_L1,
                    "sb_l1 depth-2 overflows into qr_l1")
    _check_constant(L1_SW_K + SQ_ALL_B * BN_B <= L1_SW_K2
                    and L1_SW_K2 + SQ_ALL_B * BN_B <= L1_END,
                    "sw_k depth-2 slots overlap or overflow L1")
    _check_constant(L1_SA_L1_2 + BASE_M * SQ_ALL_A <= L1_SQ_K2
                    and L1_SQ_K2 + BASE_M * SQ_ALL_B <= L1_END
                    and L1_SW_K2 + SQ_ALL_B * BN_B <= L1_SA_L1_2,
                    "sa_l1/sq_k depth-2 tail slots overlap or overflow L1")
    _check_constant(L0C_A1_ACC + 2 * BASE_M * BN_A * 4 <= L0C_Q_ACC,
                    "a1_acc depth-2 overflows into q_acc")
    _check_constant(QA_BLOCKS % A1_GRP == 0 and KV_BLOCKS % A1_GRP == 0,
                    "A1 grouping needs QA_BLOCKS/KV_BLOCKS divisible by A1_GRP")
    _check_constant(2 * L0_SLOT <= L0_BYTES, "L0A/L0B depth-2 overflow")
    _check_constant(UB_KVB + KV_RECORD_BYTES <= UB_COS_ROW,
                    "kv record buffer overflows into cos_row")
    _check_constant(KV_RECORD_BYTES == D + 2 * (D // MX_GROUP),
                    "KV record must be payload + one BF16 scale per 32 values")
    _check_constant(UB_QR_NORM + R * 2 <= UB_END, "qr_norm tile does not fit before UB_END")
    _check_constant(UB_END <= UB_BYTES, f"UB overflow: {UB_END} > {UB_BYTES}")


def _check_constants():
    # event-id reservations
    ids = {EV_READY, EV_READY + 1, EV_FREE, EV_FREE + 1}
    _check_constant(not (ids & set(_AIC_RESERVED)),
                    f"cross-core event ids {sorted(ids)} hit sync_all(MIX) reserved {_AIC_RESERVED}")
    _check_constant(all(0 <= i <= 15 for i in ids), "event id out of 0..15")
    # MX scale address == data address >> 4
    _check_constant(L0MX_BASE == (L0A_BASE >> 4), "L0A_MX base != L0A base >> 4")
    _check_constant((L0A_BASE + L0_SLOT) >> 4 == L0MX_BASE + (L0_SLOT >> 4),
                    "rotating L0A slot does not rotate its MX scale slot at >>4")
    _check_constant(L0MX_BASE + 2 * (L0_SLOT >> 4) <= L0MX_BYTES, "L0_MX depth-2 overflow")
    _check_memory_map()
    # preload slot fit + vec mutex-id reservation (no collision with 0..17)
    _check_constant(UB_COS_PRE + KV_PRE * UB_PRE_STRIDE <= UB_SIN_PRE
                    and UB_SIN_PRE + KV_PRE * UB_PRE_STRIDE <= UB_END,
                    "cos_pre/sin_pre slots overlap or overflow UB")
    _check_constant(DR * 4 <= UB_PRE_STRIDE, "preload slot stride too small for [1,DR] FP32")
    _check_constant(len(COS_PRE_MUTEX) == KV_PRE and len(SIN_PRE_MUTEX) == KV_PRE,
                    "preload groups need one mutex id per slot")
    _pre_ids = set(COS_PRE_MUTEX) | set(SIN_PRE_MUTEX)
    _check_constant(not (_pre_ids & set(_VEC_MUTEX_TAKEN)),
                    f"preload mutex ids {sorted(_pre_ids)} collide with vec ids {_VEC_MUTEX_TAKEN}")
    _check_constant(len(_pre_ids) == 2 * KV_PRE, "preload mutex ids must be distinct")
    # Between-barrier prefetch uses b_nz slots only; it must not exceed the
    # rotation depth (else a prefetched window would be overwritten pre-consume)
    # and must stay congruent 0 (mod depth) so b_zn alignment is preserved.
    _check_constant(NK_B - 1 == 4, "prefetch count must equal b_nz depth (4)")
    # geometry sanity
    _check_constant(H % K0 == 0 and R % K0 == 0, "H/R must be multiples of K0")
    _check_constant(NK_A % 2 == 0, "L1 split-K SK=2 needs an even NK_A")
    _check_constant(K0 % 64 == 0, "matmul_mx requires K multiple of 64")
    _check_constant(R % MX_GROUP == 0 and D % MX_GROUP == 0, "R/D must be multiples of MX_GROUP")
    _check_constant(D % BN_B == 0 and HEAD_BLOCKS >= 1, "D must be a multiple of BN_B")
    _check_constant(QR_PAD % VL == 0 and QR_PAD >= R + VL, "QR_PAD must cover the last group over-read")


_check_constants()


# ---------------------------------------------------------------------------
# Vector helpers (raw VF).  NOTE: a @pl.vector_function cannot call a plain
# Python helper nor return a register, so the RMSNorm reduction and the RoPE
# gather-index construction are inlined into each VF that needs them.
# ---------------------------------------------------------------------------
@pl.vector_function
def qr_row_vf(qa_tile, gamma_tile, qr_u8_tile, qr_norm_tile,
              enable_qr_norm: pl.DT_INT32, exp_tile, scale_tile, eps: pl.DT_FP32):
    """One QR row: RMSNorm(R) -> dynamic MXFP8 quant -> payload bytes + scale bytes.

    Matches golden exactly: ``normed = (qa*rstd)*gamma``; per 32-group shared
    exponent ``e=max((bitcast_i32(amax)>>23)-135,-127)``; zero group -> scale byte 0
    and payload 0; ``payload=E4M3(clamp(normed*2^-e,±448))`` (RNE, saturate); scale
    byte ``e+127``.  Exponents are packed to bytes with PACK4 into ``scale_tile``.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    gmask = vf.update_mask(MX_GROUP, dtype=pl.DT_FP32)      # 32 active lanes
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)

    # ---- RMSNorm over R: rstd = 1/sqrt(mean(v^2)+eps) (inline) ----
    acc = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for j in pl.range(R // VL):
        xj = vf.load_align(qa_tile, j * VL)
        sq = vf.mul(xj, xj, preg)
        s = vf.reduce_sum(sq, preg)
        acc = vf.add(acc, s, preg)
    ss = vf.full(acc, preg, pos=pl.DuplicatePos.LOWEST)     # broadcast lane0 -> all
    ms = vf.muls(ss, INV_RANK, preg)
    ms = vf.adds(ms, eps, preg)
    root = vf.sqrt(ms, preg)
    rstd = vf.div(one, root, preg)

    zero_i = vf.full(0, preg, dtype=pl.DT_INT32)
    for g in pl.range(QR_GROUPS):
        off = g * MX_GROUP
        v = vf.load_align(qa_tile, off)
        v = vf.mul(v, rstd, gmask)
        gam = vf.load_align(gamma_tile, off)
        v = vf.mul(v, gam, gmask)                            # normed = (qa*rstd)*gamma
        if enable_qr_norm:
            qb = vf.astype(v, gmask, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                           round_mode=pl.VFRoundMode.CAST_RINT)
            qbu = vf.bit_cast(qb, dtype=pl.DT_UINT32)
            vf.store_align(qr_norm_tile, qbu, gmask, off, dist=pl.StoreDist.PACK)
        absv = vf.abs(v, gmask)
        m = vf.reduce_max(absv, gmask)
        amax = vf.full(m, gmask, pos=pl.DuplicatePos.LOWEST)
        # shared exponent
        ab = vf.bit_cast(amax, dtype=pl.DT_INT32)
        exp_bits = vf.shift_right(ab, FP32_EXP_SHIFT, gmask)
        e = vf.adds(exp_bits, -(E8M0_BIAS + QUANT_EXP_HEADROOM), gmask)
        e = vf.maxs(e, -E8M0_BIAS, gmask)
        nonzero = vf.ne(ab, zero_i, gmask)
        # reciprocal 2^-e via the FP32 exponent field (no divide)
        neg_e = vf.neg(e, gmask)
        field = vf.adds(neg_e, E8M0_BIAS, gmask)
        field = vf.shift_left(field, FP32_EXP_SHIFT, gmask)
        recip = vf.bit_cast(field, dtype=pl.DT_FP32)
        # payload = E4M3(clamp(normed*recip, ±448))
        scaled = vf.mul(v, recip, gmask)
        scaled = vf.maxs(scaled, -E4M3_MAX, gmask)
        scaled = vf.mins(scaled, E4M3_MAX, gmask)
        payload = vf.astype(scaled, gmask, dtype=pl.DT_FP8E4M3FN,
                            layout=pl.CastLayout.ZERO,
                            round_mode=pl.VFRoundMode.CAST_RINT,
                            saturate=pl.SaturateMode.ON)
        vf.store_align(qr_u8_tile, payload, gmask, off, dist=pl.StoreDist.PACK4)
        # scale byte = e+127 (zero group -> 0)
        e_byte = vf.adds(e, E8M0_BIAS, gmask)
        e_byte = vf.select(e_byte, zero_i, nonzero)
        vf.store_align(exp_tile, e_byte, gmask, g, dist=pl.StoreDist.FIRST_ELEMENT)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    pack_mask = vf.update_mask(QR_GROUPS, dtype=pl.DT_FP32)  # 40 lanes
    exp_reg = vf.load_align(exp_tile, 0)
    vf.store_align(scale_tile, exp_reg, pack_mask, 0, dist=pl.StoreDist.PACK4)


@pl.vector_function
def kv_row_vf(kv_tile, gamma_tile, cos_tile, sin_tile, kv_bf16, eps: pl.DT_FP32):
    """One KV row: RMSNorm(D) -> TailRoPE(last Dr) -> BF16.

    Non-tail vectors are ``(kv*rstd)*gamma`` -> BF16.  The tail is de-interleaved
    (gather), normalized per half, RoPE'd with GPTJ_INTERLEAVED, then BF16.  The
    BF16 row is the input of ``kv_compress_row_vf``; the FP32 values are rounded
    once here, matching the standalone kv_compress input dtype.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    m_half = vf.update_mask(DR_HALF, dtype=pl.DT_FP32)       # 32 active lanes
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    # ---- GPTJ_INTERLEAVED gather indices (clamped to 0..63), inline ----
    lane = vf.arange(0, dtype=pl.DT_INT32)
    max63 = vf.full(63, preg, dtype=pl.DT_INT32)
    two = vf.shift_left(lane, 1, preg)
    even_idx = vf.min(two, max63, preg)
    even_u = vf.bit_cast(even_idx, dtype=pl.DT_UINT32)
    odd_raw = vf.adds(two, 1, preg)
    odd_idx = vf.min(odd_raw, max63, preg)
    odd_u = vf.bit_cast(odd_idx, dtype=pl.DT_UINT32)

    # ---- RMSNorm over D (inline) ----
    acc = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for j in pl.range(D // VL):
        xj = vf.load_align(kv_tile, j * VL)
        sq = vf.mul(xj, xj, preg)
        s = vf.reduce_sum(sq, preg)
        acc = vf.add(acc, s, preg)
    ss = vf.full(acc, preg, pos=pl.DuplicatePos.LOWEST)
    ms = vf.muls(ss, INV_HEAD, preg)
    ms = vf.adds(ms, eps, preg)
    root = vf.sqrt(ms, preg)
    rstd = vf.div(one, root, preg)

    # non-tail vectors: columns [0, D-DR)
    for j in pl.range(KV_VEC - 1):
        off = j * VL
        v = vf.load_align(kv_tile, off)
        v = vf.mul(v, rstd, preg)
        gam = vf.load_align(gamma_tile, off)
        v = vf.mul(v, gam, preg)
        b = vf.astype(v, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                      round_mode=pl.VFRoundMode.CAST_RINT)
        bu = vf.bit_cast(b, dtype=pl.DT_UINT32)
        vf.store_align(kv_bf16, bu, preg, off, dist=pl.StoreDist.PACK)

    # tail: columns [D-DR, D) with GPTJ_INTERLEAVED RoPE
    tail_off = D - DR
    t = vf.load_align(kv_tile, tail_off)                     # 64 lanes = tail[0..63]
    gt = vf.load_align(gamma_tile, tail_off)
    even = vf.gather(t, even_u)
    odd = vf.gather(t, odd_u)
    even_g = vf.gather(gt, even_u)
    odd_g = vf.gather(gt, odd_u)
    even = vf.mul(even, rstd, m_half)
    even = vf.mul(even, even_g, m_half)
    odd = vf.mul(odd, rstd, m_half)
    odd = vf.mul(odd, odd_g, m_half)
    cos_row = vf.load_align(cos_tile, 0)                     # cos[0..63]
    sin_row = vf.load_align(sin_tile, 0)                     # sin[0..63]
    ce = vf.gather(cos_row, even_u)
    se = vf.gather(sin_row, even_u)
    co = vf.gather(cos_row, odd_u)
    so = vf.gather(sin_row, odd_u)
    even_cos = vf.mul(even, ce, m_half)
    odd_sin = vf.mul(odd, se, m_half)
    first = vf.sub(even_cos, odd_sin, m_half)
    even_sin = vf.mul(even, so, m_half)
    odd_cos = vf.mul(odd, co, m_half)
    second = vf.add(even_sin, odd_cos, m_half)
    il0, il1 = vf.interleave(first, second)
    fb = vf.astype(il0, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                   round_mode=pl.VFRoundMode.CAST_RINT)
    fbu = vf.bit_cast(fb, dtype=pl.DT_UINT32)
    vf.store_align(kv_bf16, fbu, preg, tail_off, dist=pl.StoreDist.PACK)


@pl.vector_function
def kv_compress_row_vf(kv_bf16, kvb_tile, kvb_scale):
    """One BF16 KV row -> kv_compress FP8 E4M3 group-32 record.

    Per group: ``s = round_up_pow2(max(amax,1e-4)/448)`` in FP32;
    ``payload = E4M3(clamp(x/s, ±448))``; scale stored as BF16.  The payload and
    the 16 BF16 scales fill the 544-byte cache record.
    """
    pm32 = vf.update_mask(MX_GROUP, dtype=pl.DT_FP32)
    pm1 = vf.update_mask(1, dtype=pl.DT_FP32)
    emask = vf.full(0x7F800000, pm1, dtype=pl.DT_UINT32)
    mmask = vf.full(0x007FFFFF, pm1, dtype=pl.DT_UINT32)
    increment = vf.full(0x00800000, pm1, dtype=pl.DT_UINT32)
    zero32 = vf.full(0, pm1, dtype=pl.DT_UINT32)
    for g in pl.range(KV_QUANT_GROUPS):
        low = vf.load_align(kv_bf16, g * MX_GROUP, dist=pl.LoadDist.UNPK)
        x32 = vf.astype(low, pm32, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        ax = vf.abs(x32, pm32)
        amax = vf.reduce_max(ax, pm32)
        amax = vf.full(amax, pm32, pos=pl.DuplicatePos.LOWEST)
        floor_max = vf.maxs(amax, 1.0e-4, pm1)
        unrounded = vf.muls(floor_max, INV_E4M3_MAX, pm1)
        raw = vf.bit_cast(unrounded, dtype=pl.DT_UINT32)
        exp = vf.and_(raw, emask, pm1)
        mantissa = vf.and_(raw, mmask, pm1)
        nz = vf.ne(mantissa, 0, pm1)
        inc = vf.select(increment, zero32, nz)
        rounded_exp = vf.add(exp, inc, pm1)
        scale32 = vf.bit_cast(rounded_exp, dtype=pl.DT_FP32)
        scale_broadcast = vf.full(scale32, pm32)
        divided = vf.div(x32, scale_broadcast, pm32)
        bounded = vf.maxs(divided, -E4M3_MAX, pm32)
        bounded = vf.mins(bounded, E4M3_MAX, pm32)
        quantized = vf.astype(bounded, pm32, dtype=pl.DT_FP8E4M3FN,
                              layout=pl.CastLayout.ZERO,
                              round_mode=pl.VFRoundMode.CAST_RINT,
                              saturate=pl.SaturateMode.ON)
        qbytes = vf.bit_cast(quantized, dtype=pl.DT_UINT8)
        vf.store_align(kvb_tile, qbytes, pm32, g * MX_GROUP,
                       dist=pl.StoreDist.PACK4_B32)
        sb16 = vf.astype(scale32, pm1, dtype=pl.DT_BF16,
                         layout=pl.CastLayout.ZERO,
                         round_mode=pl.VFRoundMode.CAST_RINT,
                         saturate=pl.SaturateMode.OFF)
        vf.store_align(kvb_scale, sb16, pm1, g,
                       dist=pl.StoreDist.FIRST_ELEMENT_B16)


@pl.vector_function
def q_bf16_vf(cv_tile, q_tile, rows: pl.DT_INT64):
    """Non-tail Q col-block: BF16 cast all BN_B columns for ``rows`` rows."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(rows):
        base = r * BN_B
        x0 = vf.load_align(cv_tile, base)
        b0 = vf.astype(x0, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                       round_mode=pl.VFRoundMode.CAST_RINT)
        b0u = vf.bit_cast(b0, dtype=pl.DT_UINT32)
        vf.store_align(q_tile, b0u, preg, base, dist=pl.StoreDist.PACK)
        x1 = vf.load_align(cv_tile, base + VL)
        b1 = vf.astype(x1, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                       round_mode=pl.VFRoundMode.CAST_RINT)
        b1u = vf.bit_cast(b1, dtype=pl.DT_UINT32)
        vf.store_align(q_tile, b1u, preg, base + VL, dist=pl.StoreDist.PACK)


@pl.vector_function
def q_rope_bf16_vf(cv_tile, q_tile, cos_tile, sin_tile, rows: pl.DT_INT64):
    """Tail Q col-block (last of a head): BF16 prefix + TailRoPE(GPTJ_INTERLEAVED).

    Columns [0,BN_B-DR) are plain BF16; the last DR columns are de-interleaved,
    RoPE'd (no gamma/rstd for Q), cast to BF16 and written as two halves.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    m_half = vf.update_mask(DR_HALF, dtype=pl.DT_FP32)
    # ---- GPTJ_INTERLEAVED gather indices (clamped to 0..63), inline ----
    lane = vf.arange(0, dtype=pl.DT_INT32)
    max63 = vf.full(63, preg, dtype=pl.DT_INT32)
    two = vf.shift_left(lane, 1, preg)
    even_idx = vf.min(two, max63, preg)
    even_u = vf.bit_cast(even_idx, dtype=pl.DT_UINT32)
    odd_raw = vf.adds(two, 1, preg)
    odd_idx = vf.min(odd_raw, max63, preg)
    odd_u = vf.bit_cast(odd_idx, dtype=pl.DT_UINT32)
    tail_col = BN_B - DR          # 64
    for r in pl.range(rows):
        base = r * BN_B
        # prefix columns [0, BN_B-DR) = [0,64)
        x0 = vf.load_align(cv_tile, base)
        b0 = vf.astype(x0, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                       round_mode=pl.VFRoundMode.CAST_RINT)
        b0u = vf.bit_cast(b0, dtype=pl.DT_UINT32)
        vf.store_align(q_tile, b0u, preg, base, dist=pl.StoreDist.PACK)
        # tail columns [BN_B-DR, BN_B) = [64,128)
        t = vf.load_align(cv_tile, base + tail_col)
        even = vf.gather(t, even_u)
        odd = vf.gather(t, odd_u)
        cos_row = vf.load_align(cos_tile, r * DR)
        sin_row = vf.load_align(sin_tile, r * DR)
        ce = vf.gather(cos_row, even_u)
        se = vf.gather(sin_row, even_u)
        co = vf.gather(cos_row, odd_u)
        so = vf.gather(sin_row, odd_u)
        even_cos = vf.mul(even, ce, m_half)
        odd_sin = vf.mul(odd, se, m_half)
        first = vf.sub(even_cos, odd_sin, m_half)
        even_sin = vf.mul(even, so, m_half)
        odd_cos = vf.mul(odd, co, m_half)
        second = vf.add(even_sin, odd_cos, m_half)
        il0, il1 = vf.interleave(first, second)
        fb = vf.astype(il0, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                       round_mode=pl.VFRoundMode.CAST_RINT)
        fbu = vf.bit_cast(fb, dtype=pl.DT_UINT32)
        vf.store_align(q_tile, fbu, preg, base + tail_col, dist=pl.StoreDist.PACK)


# ---------------------------------------------------------------------------
# QB stage helpers (plain Python -> traced into the enclosing section).  Shared
# by the split-T (per-tile) and distributed (per (tile,head) unit) B paths so the
# inner MXFP8 K-loop + CV ping-pong is byte-identical between them.
# ---------------------------------------------------------------------------
def _qb_beat(qr_l1, sq_k, b_nz, b_zn, sw_k, l0a, l0b, scale_a_l0, scale_b_l0,
             q_acc, cv_ub, descale_wqb, wqb, col0, beat, pre, rows_full):
    """One QB (head,col-block) output: MXFP8 K=R reduction -> cv_ub handoff.

    ``pre`` (distributed path only): this beat's descale_wqb slot and Wqb
    windows kk=0..NK_B-2 were already loaded into exactly the slots this beat's
    rotations land on (prefetched between the two barriers); skip those GM loads
    and consume the prefetched data.  Window NK_B-1 is never prefetched (b_nz
    depth-4 < NK_B=5) and always loads.  All .next() calls stay unconditional so
    the rotation counters are identical on both paths.
    """
    # MMA and FIX must use the same packed L0C row stride. Round to 32 so
    # DualModeSplitM gives each AIV an aligned 16-row multiple; the vector
    # consumer uses this same split instead of a fixed 64-row boundary.
    rows_r = (rows_full + 31) // 32 * 32
    swk = sw_k.next()          # rotate per beat (kills the single-slot WAR)
    wqn = b_nz.next()
    if pre:
        pl.set_validshape(swk, [SQ_ALL_B, BN_B])
        pl.set_validshape(wqn, [BN_B, K0])
    else:
        pl.set_validshape(swk, [SQ_ALL_B, BN_B])
        pl.load(swk, descale_wqb, [col0, 0, 0], order=[1, 0])
        pl.set_validshape(wqn, [BN_B, K0])
        if Profile == 0:  # noqa: F821
            pl.load(wqn, wqb, [col0, 0], l2_cache_mode=pl.TLoadL2Hint.NotAllocKeep)
        else:
            pl.load(wqn, wqb, [col0, 0])
    la = l0a.next()
    pl.set_validshape(la, [rows_r, K0])
    pl.move(la, qr_l1.current(), offset=[0, 0])
    lb = l0b.next()
    pl.move(lb, b_zn.next())
    sa = scale_a_l0.next()
    pl.move(sa, sq_k.current(), offset=[0, 0])
    sb = scale_b_l0.next()
    pl.move(sb, swk, offset=[0, 0])
    pl.matmul_mx(q_acc[beat % 2], la, lb, sa, sb)
    for kk in pl.range(1, NK_B):
        wqn = b_nz.next()
        if kk == NK_B - 1:
            pl.set_validshape(wqn, [BN_B, K0])
            if Profile == 0:  # noqa: F821
                pl.load(wqn, wqb, [col0, kk * K0], l2_cache_mode=pl.TLoadL2Hint.NotAllocKeep)
            else:
                pl.load(wqn, wqb, [col0, kk * K0])
        elif pre:
            pl.set_validshape(wqn, [BN_B, K0])   # consume prefetched window
        else:
            pl.set_validshape(wqn, [BN_B, K0])
            if Profile == 0:  # noqa: F821
                pl.load(wqn, wqb, [col0, kk * K0], l2_cache_mode=pl.TLoadL2Hint.NotAllocKeep)
            else:
                pl.load(wqn, wqb, [col0, kk * K0])
        la = l0a.next()
        pl.set_validshape(la, [rows_r, K0])
        pl.move(la, qr_l1.current(), offset=[0, kk * K0])
        lb = l0b.next()
        pl.move(lb, b_zn.next())
        sa = scale_a_l0.next()
        pl.move(sa, sq_k.current(), offset=[0, kk * SK0])
        sb = scale_b_l0.next()
        pl.move(sb, swk, offset=[kk * SK0, 0])
        pl.matmul_mx_acc(q_acc[beat % 2], q_acc[beat % 2], la, lb, sa, sb)
    pl.set_validshape(q_acc[beat % 2], [rows_r, BN_B])
    if beat >= 2:
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=EV_FREE + beat % 2)
    slot = cv_ub[beat % 2]
    pl.set_validshape(slot, [rows_r // 2, BN_B])
    pl.move(slot, q_acc[beat % 2], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=EV_READY + beat % 2)


def _qb_beat_resident5(qr_tile, sqk, wb5_zn, swblk, l0a, l0b,
                       scale_a_l0, scale_b_l0, q_acc, cv_ub, beat, rows_full):
    """One QB beat using five L1-resident Wqb K-slices (2-tile pair reuse)."""
    rows_r = (rows_full + 31) // 32 * 32
    for kk in pl.range(NK_B):
        la = l0a.next()
        pl.set_validshape(la, [rows_r, K0])
        pl.move(la, qr_tile, offset=[0, kk * K0])
        lb = l0b.next()
        pl.move(lb, wb5_zn.next())
        sa = scale_a_l0.next()
        pl.move(sa, sqk, offset=[0, kk * SK0])
        sb = scale_b_l0.next()
        pl.move(sb, swblk.current(), offset=[kk * SK0, 0])
        if kk == 0:
            pl.matmul_mx(q_acc[beat % 2], la, lb, sa, sb)
        else:
            pl.matmul_mx_acc(q_acc[beat % 2], q_acc[beat % 2], la, lb, sa, sb)
    pl.set_validshape(q_acc[beat % 2], [rows_r, BN_B])
    if beat >= 2:
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=EV_FREE + beat % 2)
    slot = cv_ub[beat % 2]
    pl.set_validshape(slot, [rows_r // 2, BN_B])
    pl.move(slot, q_acc[beat % 2], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=EV_READY + beat % 2)


def _qb_beat_pair5(qr0, qr1, sqk0, sqk1, wb5_zn, swblk, l0a, l0b,
                   scale_a_l0, scale_b_l0, q_acc, cv_ub, beat, rows0, rows1):
    """One interleaved K loop for a 2-tile pair: one L0B weight move serves both MMAs."""
    rows_r0 = (rows0 + 31) // 32 * 32
    rows_r1 = (rows1 + 31) // 32 * 32
    for kk in pl.range(NK_B):
        lb = l0b.next()
        pl.move(lb, wb5_zn.next())
        sb = scale_b_l0.next()
        pl.move(sb, swblk.current(), offset=[kk * SK0, 0])
        la0 = l0a.next()
        pl.set_validshape(la0, [rows_r0, K0])
        pl.move(la0, qr0, offset=[0, kk * K0])
        la1 = l0a.next()
        pl.set_validshape(la1, [rows_r1, K0])
        pl.move(la1, qr1, offset=[0, kk * K0])
        sa0 = scale_a_l0.next()
        pl.move(sa0, sqk0, offset=[0, kk * SK0])
        sa1 = scale_a_l0.next()
        pl.move(sa1, sqk1, offset=[0, kk * SK0])
        if kk == 0:
            pl.matmul_mx(q_acc[0], la0, lb, sa0, sb)
            pl.matmul_mx(q_acc[1], la1, lb, sa1, sb)
        else:
            pl.matmul_mx_acc(q_acc[0], q_acc[0], la0, lb, sa0, sb)
            pl.matmul_mx_acc(q_acc[1], q_acc[1], la1, lb, sa1, sb)
    pl.set_validshape(q_acc[0], [rows_r0, BN_B])
    if beat >= 2:
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=EV_FREE + beat % 2)
    slot0 = cv_ub[beat % 2]
    pl.set_validshape(slot0, [rows_r0 // 2, BN_B])
    pl.move(slot0, q_acc[0], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=EV_READY + beat % 2)
    pl.set_validshape(q_acc[1], [rows_r1, BN_B])
    if beat + 1 >= 2:
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=EV_FREE + (beat + 1) % 2)
    slot1 = cv_ub[(beat + 1) % 2]
    pl.set_validshape(slot1, [rows_r1 // 2, BN_B])
    pl.move(slot1, q_acc[1], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=EV_READY + (beat + 1) % 2)


def _q_epi_beat(cv_ub, q_bf16, q2d, cos_ub, sin_ub, ro, store_rows, rows_vec, col0, beat, is_tail):
    """One B-epi beat: consume cv_ub -> (RoPE if tail) BF16 -> store q col-block."""
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=EV_READY + beat % 2)
    slot = cv_ub[beat % 2]
    pl.set_validshape(slot, [rows_vec, BN_B])
    qt = q_bf16[beat % 2]
    pl.set_validshape(qt, [ROWS_VEC, BN_B])
    if is_tail:
        q_rope_bf16_vf(slot, qt, cos_ub.current(), sin_ub.current(), store_rows)
    else:
        q_bf16_vf(slot, qt, store_rows)
    if store_rows > 0:
        pl.set_validshape(qt, [store_rows, BN_B])
        pl.store(q2d, qt, [ro, col0])
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=EV_FREE + beat % 2)


def _a1_group(x, wgt, dsw, ws_out, sa_l1, sb_l1, a_l1, b_nz, b_zn, l0a, l0b,
              scale_a_l0, scale_b_l0, a1_acc, tile_row, rows_full, c0, c1):
    """One A1 group: TWO output col-blocks (row offsets c0,c1) sharing ONE X
    K-window load per kk (halves X L2->L1 traffic).

    ``wgt``/``dsw``/``ws_out`` are the QA triple (wqa/descale_wqa/ws_qa) or the KV
    triple (wkv/descale_wkv/ws_kv).  Block j reduces the full K=H into a1_acc[j],
    initialised on kk==0 (matmul_mx) and accumulated after (matmul_mx_acc), exactly
    as the ungrouped V1 did per block; only the X load is shared.  sb_l1 rotates
    depth-2 so both blocks' weight scales stay resident across the kk loop.
    """
    # Tail tile: the NZ L1 X window is loaded with rows_full rows, so the L0A
    # move must be narrowed to match (rounded up to the 16-row NZ fractal, as
    # indexer_prologue_qw does).  Moving a full [BASE_M,K0] window instead
    # corrupts the WHOLE tail tile, not just the unloaded rows: measured on
    # DAV_3510, only output columns [0,16) of each BN_A block stayed correct
    # and the rest became NaN/~FLT_MAX whenever roundup16(rows_full) < BASE_M.
    rows_r = (rows_full + 15) // 16 * 16
    sb0 = sb_l1.next()
    pl.set_validshape(sb0, [SQ_ALL_A, BN_A])
    pl.load(sb0, dsw, [c0, 0, 0], order=[1, 0])
    sb1 = sb_l1.next()
    pl.set_validshape(sb1, [SQ_ALL_A, BN_A])
    pl.load(sb1, dsw, [c1, 0, 0], order=[1, 0])
    for kk in pl.range(NK_A):
        xa = a_l1.next()                       # ONE X K-window load, reused by both blocks
        pl.set_validshape(xa, [rows_full, K0])
        pl.load(xa, x, [tile_row, kk * K0])
        # ---- block 0 -> a1_acc[0] ----
        wqn = b_nz.next()
        pl.set_validshape(wqn, [BN_A, K0])
        if Profile == 0:  # noqa: F821
            pl.load(wqn, wgt, [c0, kk * K0], l2_cache_mode=pl.TLoadL2Hint.NotAllocKeep)
        else:
            pl.load(wqn, wgt, [c0, kk * K0])
        la = l0a.next()
        pl.set_validshape(la, [rows_r, K0])
        pl.move(la, xa)
        lb = l0b.next()
        pl.move(lb, b_zn.next())
        sa = scale_a_l0.next()
        pl.move(sa, sa_l1.current(), offset=[0, kk * SK0])
        sb = scale_b_l0.next()
        pl.move(sb, sb0, offset=[kk * SK0, 0])
        if kk == 0:
            pl.matmul_mx(a1_acc[0], la, lb, sa, sb)
        else:
            pl.matmul_mx_acc(a1_acc[0], a1_acc[0], la, lb, sa, sb)
        # ---- block 1 -> a1_acc[1] (reuses the SAME xa) ----
        wqn = b_nz.next()
        pl.set_validshape(wqn, [BN_A, K0])
        if Profile == 0:  # noqa: F821
            pl.load(wqn, wgt, [c1, kk * K0], l2_cache_mode=pl.TLoadL2Hint.NotAllocKeep)
        else:
            pl.load(wqn, wgt, [c1, kk * K0])
        la = l0a.next()
        pl.set_validshape(la, [rows_r, K0])
        pl.move(la, xa)
        lb = l0b.next()
        pl.move(lb, b_zn.next())
        sa = scale_a_l0.next()
        pl.move(sa, sa_l1.current(), offset=[0, kk * SK0])
        sb = scale_b_l0.next()
        pl.move(sb, sb1, offset=[kk * SK0, 0])
        if kk == 0:
            pl.matmul_mx(a1_acc[1], la, lb, sa, sb)
        else:
            pl.matmul_mx_acc(a1_acc[1], a1_acc[1], la, lb, sa, sb)
    pl.set_validshape(a1_acc[0], [rows_full, BN_A])
    pl.store(ws_out, a1_acc[0], [tile_row, c0])
    pl.set_validshape(a1_acc[1], [rows_full, BN_A])
    pl.store(ws_out, a1_acc[1], [tile_row, c1])


def _a1_block(x, wgt, dsw, ws_out, sa_l1, sb_l1, a_l1, b_nz, b_zn, l0a, l0b,
              scale_a_l0, scale_b_l0, a1_acc, tile_row, rows_full, c0):
    """One A1 output col-block (UNGROUPED, decode path): full K=H reduction into
    a1_acc[0].  Used when there are too few T-tiles for A1_GRP grouping to fill
    the cores (grouping would halve A1 parallelism); 14 blocks/tile engage up to 14
    cores vs 7 for groups, and at decode A1 is weight-bandwidth-bound so more cores
    issuing MTE2 finishes sooner.  sb_l1[0] is a fixed slot (one block at a time)."""
    # Tail tile: the NZ L1 X window is loaded with rows_full rows, so the L0A
    # move must be narrowed to match (rounded up to the 16-row NZ fractal, as
    # indexer_prologue_qw does).  Moving a full [BASE_M,K0] window instead
    # corrupts the WHOLE tail tile, not just the unloaded rows: measured on
    # DAV_3510, only output columns [0,16) of each BN_A block stayed correct
    # and the rest became NaN/~FLT_MAX whenever roundup16(rows_full) < BASE_M.
    rows_r = (rows_full + 15) // 16 * 16
    sb = sb_l1[0]
    pl.set_validshape(sb, [SQ_ALL_A, BN_A])
    pl.load(sb, dsw, [c0, 0, 0], order=[1, 0])
    for kk in pl.range(NK_A):
        xa = a_l1.next()
        pl.set_validshape(xa, [rows_full, K0])
        pl.load(xa, x, [tile_row, kk * K0])
        wqn = b_nz.next()
        pl.set_validshape(wqn, [BN_A, K0])
        if Profile == 0:  # noqa: F821
            pl.load(wqn, wgt, [c0, kk * K0], l2_cache_mode=pl.TLoadL2Hint.NotAllocKeep)
        else:
            pl.load(wqn, wgt, [c0, kk * K0])
        la = l0a.next()
        pl.set_validshape(la, [rows_r, K0])
        pl.move(la, xa)
        lb = l0b.next()
        pl.move(lb, b_zn.next())
        sa = scale_a_l0.next()
        pl.move(sa, sa_l1.current(), offset=[0, kk * SK0])
        sbm = scale_b_l0.next()
        pl.move(sbm, sb, offset=[kk * SK0, 0])
        if kk == 0:
            pl.matmul_mx(a1_acc[0], la, lb, sa, sbm)
        else:
            pl.matmul_mx_acc(a1_acc[0], a1_acc[0], la, lb, sa, sbm)
    pl.set_validshape(a1_acc[0], [rows_full, BN_A])
    pl.store(ws_out, a1_acc[0], [tile_row, c0])


def _a1_block_sk(x, wgt, dsw, ws_out, sa_l1, sb_l1, a_l1, b_nz, b_zn, l0a, l0b,
                 scale_a_l0, scale_b_l0, a1_acc, ws_row, rows_full, c0, kk_lo, kk_hi):
    """One A1 output col-block for ONE K-half (L1 split-K, decode-only path).

    Same as ``_a1_block`` except the reduction runs over ``[kk_lo, kk_hi)`` and the
    FP32 partial lands in the split-K band row ``ws_row`` (the wrapper doubles the
    ws rows under a byte-identical gate).  Valid only for ``n_tiles == 1`` (the gate
    ``n_tiles*NA1*2 <= num_cores`` admits no other value on a 32-core grid), hence
    the fixed X/store tile row 0.  Numerics: A2 merges the two bands with one FP32
    add -- for exactly 2 partials a Kahan compensated merge degenerates to the
    same ``fl(band0 + band1)`` (compensation starts at 0), keeping the summation
    order inside the FP64-adjudicated byte gate.
    """
    # Tail tile: the NZ L1 X window is loaded with rows_full rows, so the L0A
    # move must be narrowed to match (rounded up to the 16-row NZ fractal, as
    # indexer_prologue_qw does).  Moving a full [BASE_M,K0] window instead
    # corrupts the WHOLE tail tile, not just the unloaded rows: measured on
    # DAV_3510, only output columns [0,16) of each BN_A block stayed correct
    # and the rest became NaN/~FLT_MAX whenever roundup16(rows_full) < BASE_M.
    rows_r = (rows_full + 15) // 16 * 16
    sb = sb_l1[0]
    pl.set_validshape(sb, [SQ_ALL_A, BN_A])
    pl.load(sb, dsw, [c0, 0, 0], order=[1, 0])
    for kk in pl.range(kk_lo, kk_hi):
        xa = a_l1.next()
        pl.set_validshape(xa, [rows_full, K0])
        pl.load(xa, x, [0, kk * K0])
        wqn = b_nz.next()
        pl.set_validshape(wqn, [BN_A, K0])
        if Profile == 0:  # noqa: F821
            pl.load(wqn, wgt, [c0, kk * K0], l2_cache_mode=pl.TLoadL2Hint.NotAllocKeep)
        else:
            pl.load(wqn, wgt, [c0, kk * K0])
        la = l0a.next()
        pl.set_validshape(la, [rows_r, K0])
        pl.move(la, xa)
        lb = l0b.next()
        pl.move(lb, b_zn.next())
        sa = scale_a_l0.next()
        pl.move(sa, sa_l1.current(), offset=[0, kk * SK0])
        sbm = scale_b_l0.next()
        pl.move(sbm, sb, offset=[kk * SK0, 0])
        if kk == kk_lo:
            pl.matmul_mx(a1_acc[0], la, lb, sa, sbm)
        else:
            pl.matmul_mx_acc(a1_acc[0], a1_acc[0], la, lb, sa, sbm)
    pl.set_validshape(a1_acc[0], [rows_full, BN_A])
    pl.store(ws_out, a1_acc[0], [ws_row, c0])


# ---------------------------------------------------------------------------
# Fused MIX kernel: A1 projections -> A2 norm/quant/scatter -> B QB projection
# -> B-epi RoPE/BF16 (stage details and numerical rules: module docstring).
# ---------------------------------------------------------------------------
@pl.jit(arch="3510", auto_mutex=True, compile_timeout=300, tiling_key=AttnPrologueTilingKey)
def attn_prologue_kernel(
    x: pl.Tensor[[pl.DYNAMIC, H], pl.DT_FP8E4M3FN],
    wqa: pl.Tensor[[R, H], pl.DT_FP8E4M3FN, pl.NZ],
    wqb: pl.Tensor[[N * D, R], pl.DT_FP8E4M3FN, pl.NZ],
    wkv: pl.Tensor[[D, H], pl.DT_FP8E4M3FN, pl.NZ],
    descale_x: pl.Tensor[[pl.DYNAMIC, H // 64, 2], pl.DT_FP8E8M0],
    descale_wqa: pl.Tensor[[R, H // 64, 2], pl.DT_FP8E8M0],
    descale_wqb: pl.Tensor[[N * D, R // 64, 2], pl.DT_FP8E8M0],
    descale_wkv: pl.Tensor[[D, H // 64, 2], pl.DT_FP8E8M0],
    norm_weight_qr: pl.Tensor[[R], pl.DT_FP32],
    norm_weight_kv: pl.Tensor[[D], pl.DT_FP32],
    rope_sin: pl.Tensor[[pl.DYNAMIC, DR], pl.DT_FP32],
    rope_cos: pl.Tensor[[pl.DYNAMIC, DR], pl.DT_FP32],
    cache_index: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    kv_cache: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 1, KV_RECORD_BYTES], pl.DT_UINT8],
    qr_norm: pl.Tensor[[pl.DYNAMIC, R], pl.DT_BF16],
    q: pl.Tensor[[pl.DYNAMIC, N, D], pl.DT_BF16],
    qr: pl.Tensor[[pl.DYNAMIC, R], pl.DT_FP8E4M3FN],
    descale_qr: pl.Tensor[[pl.DYNAMIC, R // 64, 2], pl.DT_FP8E8M0],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: AttnPrologueTiling,
):
    """Fused attn_prologue MIX kernel; see the module docstring for the contract."""
    t_live = tiling.t
    n_tiles = (t_live + BASE_M - 1) // BASE_M
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    # Weight L2 policy is a tiling-key constant (see AttnPrologueTilingKey):
    # single-tile decode streams weights with NotAllocKeep so they do not evict
    # dirty L2 lines; multi-tile prefill keeps the default policy so the shared
    # weights stay resident across tiles.
    # Split-T relies on one core owning whole tiles. On a 28-core device a
    # 32-tile workload is badly imbalanced; use distributed-B until the tile
    # count is a multiple of the grid or large enough to amortize the tail.
    split_t = (n_tiles >= SPLIT_T_TILES) and (
        n_tiles % num_cores == 0 or n_tiles >= 4 * num_cores)
    cache_rows = tiling.cache_rows
    block_size = tiling.block_size
    block_stride = tiling.block_stride
    row_stride = tiling.row_stride
    ws_qa = pl.make_tensor(workspace, [tiling.ws_rows, R], dtype=pl.DT_FP32)
    ws_kv = pl.make_tensor(
        pl.addptr(workspace, tiling.ws_rows * R * 4), [tiling.ws_rows, D],
        dtype=pl.DT_FP32)

    # 2-D views of higher-rank GM tensors (no data moved)
    q2d = pl.make_tensor(q, [t_live, N * D])
    qr_u8 = pl.make_tensor(qr, [t_live, R], dtype=pl.DT_UINT8)
    qr_norm_gm = pl.make_tensor(qr_norm, [t_live, R], dtype=pl.DT_BF16)
    dqr_u8 = pl.make_tensor(descale_qr, [t_live, QR_GROUPS], dtype=pl.DT_UINT8)
    kv_cache_pages = pl.make_tensor(
        kv_cache, [cache_rows // block_size, block_stride], dtype=pl.DT_UINT8)
    gamma_qr_gm = pl.make_tensor(norm_weight_qr, [1, R])
    gamma_kv_gm = pl.make_tensor(norm_weight_kv, [1, D])

    # cross-section handoff buffer (touched by BOTH sections) -> declared outside
    cv_ub = pl.make_tile_group(
        type=pl.TileType(shape=[ROWS_VEC, BN_B], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[UB_CV0, UB_CV1], mutex_ids=[14, 15],
    )

    # =================== CUBE SECTION ===================
    with pl.section_cube():
        # ---- L1 tiles ----
        a_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, K0], dtype=pl.DT_FP8E4M3FN,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                             valid_shape=[-1, -1], compact=1),
            addrs=[L1_A_L1, L1_A_L1 + L0_SLOT], mutex_ids=[0, 1])
        sa_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, SQ_ALL_A], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZZ,
                             valid_shape=[-1, -1]),
            addrs=[L1_SA_L1, L1_SA_L1_2], mutex_ids=[2, 27])
        b_nz = pl.make_tile_group(
            type=pl.TileType(shape=[BN_A, K0], dtype=pl.DT_FP8E4M3FN,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                             valid_shape=[-1, -1], compact=1),
            addrs=[L1_B_NZ, L1_B_NZ + L0_SLOT, L1_B_NZ2, L1_B_NZ2 + L0_SLOT],
            mutex_ids=[3, 4, 22, 23])
        b_zn = pl.make_tile_group(
            type=pl.TileType(shape=[K0, BN_A], dtype=pl.DT_FP8E4M3FN,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                             valid_shape=[-1, -1], compact=1),
            addrs=[L1_B_NZ, L1_B_NZ + L0_SLOT, L1_B_NZ2, L1_B_NZ2 + L0_SLOT],
            mutex_ids=[3, 4, 22, 23])
        sb_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[SQ_ALL_A, BN_A], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NN,
                             valid_shape=[-1, -1]),
            addrs=[L1_SB_L1, L1_SB_L1 + SQ_ALL_A * BN_A], mutex_ids=[5, 25])
        qr_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, R], dtype=pl.DT_FP8E4M3FN,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                             valid_shape=[-1, -1]),
            addrs=L1_QR_L1, mutex_ids=[6])
        sq_k = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, SQ_ALL_B], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZZ,
                             valid_shape=[-1, -1]),
            addrs=[L1_SQ_K, L1_SQ_K2], mutex_ids=[7, 28])
        sw_k = pl.make_tile_group(
            type=pl.TileType(shape=[SQ_ALL_B, BN_B], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NN,
                             valid_shape=[-1, -1]),
            addrs=[L1_SW_K, L1_SW_K2], mutex_ids=[8, 26])
        # B 2-tile pair: full (head,block) Wqb block resident in L1.  Overlaid on
        # the A1 L1 region; A1 is complete before barrier2, B starts after it.
        qr_pair = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, R], dtype=pl.DT_FP8E4M3FN,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                             valid_shape=[-1, -1]),
            addrs=[L1_QR_L1, L1_QR1], mutex_ids=[6, 31])
        sqk0 = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, SQ_ALL_B], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZZ,
                             valid_shape=[-1, -1]),
            addrs=L1_SQK0, mutex_ids=[7])
        sqk1 = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, SQ_ALL_B], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZZ,
                             valid_shape=[-1, -1]),
            addrs=L1_SQK1, mutex_ids=[28])
        wb5 = pl.make_tile_group(
            type=pl.TileType(shape=[BN_B, K0], dtype=pl.DT_FP8E4M3FN,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                             valid_shape=[-1, -1]),
            addrs=[L1_WBLK, L1_WBLK + L0_SLOT, L1_WBLK + 2 * L0_SLOT, 0x68000, 0x70000],
            mutex_ids=[3, 4, 22, 23, 29])
        wb5_zn = pl.make_tile_group(
            type=pl.TileType(shape=[K0, BN_B], dtype=pl.DT_FP8E4M3FN,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                             valid_shape=[-1, -1]),
            addrs=[L1_WBLK, L1_WBLK + L0_SLOT, L1_WBLK + 2 * L0_SLOT, 0x68000, 0x70000],
            mutex_ids=[3, 4, 22, 23, 29])
        swblk = pl.make_tile_group(
            type=pl.TileType(shape=[SQ_ALL_B, BN_B], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NN,
                             valid_shape=[-1, -1]),
            addrs=L1_SWBLK, mutex_ids=[8])
        # ---- L0 tiles (shared by A1 and B: identical shapes) ----
        l0a = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, K0], dtype=pl.DT_FP8E4M3FN,
                             target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=L0A_BASE, mutex_ids=[9, 10])
        l0b = pl.make_tile_group(
            type=pl.TileType(shape=[K0, BN_A], dtype=pl.DT_FP8E4M3FN,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=L0B_BASE, mutex_ids=[11, 12])
        scale_a_l0 = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, SK0], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.ScaleLeft, layout=pl.ZZ),
            addrs=[L0MX_BASE, L0MX_BASE + (L0_SLOT >> 4)], mutex_ids=[13, 16])
        scale_b_l0 = pl.make_tile_group(
            type=pl.TileType(shape=[SK0, BN_A], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.ScaleRight, layout=pl.NN),
            addrs=[L0MX_BASE, L0MX_BASE + (L0_SLOT >> 4)], mutex_ids=[17, 18])
        # ---- L0C accumulators ----
        a1_acc = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, BN_A], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
            addrs=[L0C_A1_ACC, L0C_A1_ACC + BASE_M * BN_A * 4], mutex_ids=[19, 24])
        q_acc = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, BN_B], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
            addrs=[L0C_Q_ACC, L0C_Q_ACC + BASE_M * BN_B * 4], mutex_ids=[20, 21])

        # ---------------- Stage A1: QA + KV projections -> GM workspace --------
        # ADAPTIVE A1 distribution (mirrors the B split-T/distributed gate).
        #  * grouped (enough tiles to fill cores, n_tiles*NA1G >= num_cores): the
        #    (tile, GROUP) units, A1_GRP=2 col-blocks SHARE one X K-window load per
        #    kk -> X L2->L1 traffic halves.  Best for prefill (X-reload dominated).
        #  * ungrouped (few tiles = decode): (tile, block) units, 14 blocks/tile
        #    engage up to 14 cores (vs 7 for groups); A1 is weight-bandwidth-bound
        #    at decode so more cores issuing MTE2 finishes sooner.  Protects decode.
        # Both branches are traced (only one runs); the rotation counters merge like
        # the proven B split-T/distributed gate.
        if n_tiles * NA1G >= num_cores:
            for ua in pl.range(core_id, n_tiles * NA1G, num_cores):
                tile = ua // NA1G
                grp = ua % NA1G
                tile_row = tile * BASE_M
                rows_full = pl.min(BASE_M, t_live - tile_row)
                # stage X's A-scale (descale_x) for this tile (shared by both blocks)
                sal = sa_l1.next()       # depth-2 rotation, kills the unit-edge WAR
                pl.set_validshape(sal, [rows_full, SQ_ALL_A])
                pl.load(sal, descale_x, [tile_row, 0, 0], order=[0, 1])
                if grp < QA_GROUPS:
                    _a1_group(x, wqa, descale_wqa, ws_qa, sa_l1, sb_l1, a_l1, b_nz, b_zn,
                              l0a, l0b, scale_a_l0, scale_b_l0, a1_acc, tile_row, rows_full,
                              (2 * grp) * BN_A, (2 * grp + 1) * BN_A)
                else:
                    kgrp = grp - QA_GROUPS
                    _a1_group(x, wkv, descale_wkv, ws_kv, sa_l1, sb_l1, a_l1, b_nz, b_zn,
                              l0a, l0b, scale_a_l0, scale_b_l0, a1_acc, tile_row, rows_full,
                              (2 * kgrp) * BN_A, (2 * kgrp + 1) * BN_A)
        else:
            if n_tiles * NA1 * 2 <= num_cores:
                # ---- L1 split-K (decode-only): SK=2 K-halves per col-block ----
                # 28 (blk,ks) units engage 28 cores instead of 14; A1 is per-core
                # MTE2-bound (~85 GB/s/core, below the ~1.5 TB/s aggregate stream
                # floor), so doubling the issuers shortens the A1 wall.  Unit
                # order ua = ks*NA1 + blk keeps adjacent cores on adjacent
                # 128-col bands within each K-half (band locality).  Gate is
                # BYTE-IDENTICAL to the
                # wrapper's ws-band allocation gate and to the A2 merge gate below
                # (fires only at n_tiles==1 on a 32-core grid).
                for ua in pl.range(core_id, n_tiles * NA1 * 2, num_cores):
                    blk = ua % NA1
                    ks = ua // NA1
                    rows_full = pl.min(BASE_M, t_live)
                    sal = sa_l1.next()
                    pl.set_validshape(sal, [rows_full, SQ_ALL_A])
                    pl.load(sal, descale_x, [0, 0, 0], order=[0, 1])
                    kk_lo = ks * (NK_A // 2)
                    if blk < QA_BLOCKS:
                        _a1_block_sk(x, wqa, descale_wqa, ws_qa, sa_l1, sb_l1, a_l1,
                                     b_nz, b_zn, l0a, l0b, scale_a_l0, scale_b_l0,
                                     a1_acc, ks * n_tiles * BASE_M, rows_full,
                                     blk * BN_A, kk_lo, kk_lo + NK_A // 2)
                    else:
                        bn = blk - QA_BLOCKS
                        _a1_block_sk(x, wkv, descale_wkv, ws_kv, sa_l1, sb_l1, a_l1,
                                     b_nz, b_zn, l0a, l0b, scale_a_l0, scale_b_l0,
                                     a1_acc, ks * n_tiles * BASE_M, rows_full,
                                     bn * BN_A, kk_lo, kk_lo + NK_A // 2)
            else:
                for ua in pl.range(core_id, n_tiles * NA1, num_cores):
                    tile = ua // NA1
                    blk = ua % NA1
                    tile_row = tile * BASE_M
                    rows_full = pl.min(BASE_M, t_live - tile_row)
                    sal = sa_l1.next()       # depth-2 rotation, kills the unit-edge WAR
                    pl.set_validshape(sal, [rows_full, SQ_ALL_A])
                    pl.load(sal, descale_x, [tile_row, 0, 0], order=[0, 1])
                    if blk < QA_BLOCKS:
                        _a1_block(x, wqa, descale_wqa, ws_qa, sa_l1, sb_l1, a_l1, b_nz, b_zn,
                                  l0a, l0b, scale_a_l0, scale_b_l0, a1_acc, tile_row, rows_full,
                                  blk * BN_A)
                    else:
                        bn = blk - QA_BLOCKS
                        _a1_block(x, wkv, descale_wkv, ws_kv, sa_l1, sb_l1, a_l1, b_nz, b_zn,
                                  l0a, l0b, scale_a_l0, scale_b_l0, a1_acc, tile_row, rows_full,
                                  bn * BN_A)

        # ---------------- barrier1: A1 results visible to Vector ----------------
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        # P2: Cube is idle between the barriers while Vector runs A2 (~4us at
        # T=192).  Prefetch the first distributed-B unit's descale_wqb + Wqb
        # windows kk=0..NK_B-2 into the EXACT slots the unit's first beat rotates
        # onto; that beat consumes them (pre=True) and skips those GM loads.
        # Slot bookkeeping (rotation counters stay deterministic):
        #  * sw_k: next()+load puts the descale in slot (q+1)%2; the bare next()
        #    restores parity so the first beat's next() re-hits the same slot.
        #  * b_nz: exactly depth(4) advances => counter stays congruent (mod 4)
        #    with b_zn (A1 pairs them 1:1) and the first beat's kk=0..3 next()
        #    land on the kk=0..3 prefetched windows in order; kk=NK_B-1 loads as
        #    usual into the slot whose kk=0 window the beat already consumed
        #    (mutex-ordered).  Both advances are 0 mod depth, so the split-T
        #    path's slot sequence is unchanged even though this branch traces.
        # Round H showed the NON-consuming warmup was a no-op; this variant
        # removes the first beat's 4x32KB + 5KB of load latency from B's exposed
        # critical path.  Gate == the distributed B branch below.
        if not split_t:
            if n_tiles != 2:
                hb0 = core_id % (N * HEAD_BLOCKS)
                col00 = (hb0 // HEAD_BLOCKS) * D + (hb0 % HEAD_BLOCKS) * BN_B
                swk = sw_k.next()
                pl.set_validshape(swk, [SQ_ALL_B, BN_B])
                pl.load(swk, descale_wqb, [col00, 0, 0], order=[1, 0])
                sw_k.next()        # parity restore (no load): first beat re-hits swk
                for kk in pl.range(NK_B - 1):
                    wqn = b_nz.next()
                    pl.set_validshape(wqn, [BN_B, K0])
                    if Profile == 0:  # noqa: F821
                        pl.load(wqn, wqb, [col00, kk * K0], l2_cache_mode=pl.TLoadL2Hint.NotAllocKeep)
                    else:
                        pl.load(wqn, wqb, [col00, kk * K0])
        # ---------------- barrier2: qr visible to Cube (A2 done) ----------------
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

        # ---------------- Stage B: QB projection -> cv_ub (CV ping-pong) --------
        # Two work distributions sharing the same _qb_beat inner body:
        #  * split-T (n_tiles >= SPLIT_T_TILES): one core owns whole tiles, qr is
        #    staged once per tile and reused across all N*HEAD_BLOCKS beats.
        #  * distributed: (tile,head) units are spread over all cores so decode/mid-T
        #    engage every core; qr is staged per unit.
        if split_t:
            for tile in pl.range(core_id, n_tiles, num_cores):
                tile_row = tile * BASE_M
                rows_full = pl.min(BASE_M, t_live - tile_row)
                beat0 = (tile // num_cores) * BEATS_PER_TILE
                pl.set_validshape(qr_l1.current(), [rows_full, R])
                pl.load(qr_l1.current(), qr, [tile_row, 0])
                sqk = sq_k.next()        # depth-2 rotation, kills the tile-edge WAR
                pl.set_validshape(sqk, [rows_full, SQ_ALL_B])
                pl.load(sqk, descale_qr, [tile_row, 0, 0], order=[0, 1])
                for head in pl.range(N):
                    for block in pl.range(HEAD_BLOCKS):
                        _qb_beat(qr_l1, sq_k, b_nz, b_zn, sw_k, l0a, l0b,
                                 scale_a_l0, scale_b_l0, q_acc, cv_ub,
                                 descale_wqb, wqb, head * D + block * BN_B,
                                 beat0 + head * HEAD_BLOCKS + block, False, rows_full)
        else:
            if n_tiles == 1:
                # Distribute at (tile,head,BLOCK) unit granularity: 32 cores stream
                # Wqb with adjacent cores on adjacent 128-row bands, maximizing NZ
                # k-frac page density so the 41.9 MB stream runs at the full aggregate
                # L2->L1 rate (a (tile,head)-major order would space cores 512 rows
                # apart and cost ~25% of the stream rate).  Per-beat loads/matmul are
                # byte-identical under either order; beats per core = n_tiles*256/32.
                # qr/descale_qr reloads are hoisted to tile changes (tile = wu // 256;
                # at T<=128 one load per core).
                for wu in pl.range(core_id, n_tiles * N * HEAD_BLOCKS, num_cores):
                    klocal = (wu - core_id) // num_cores
                    tile = wu // (N * HEAD_BLOCKS)
                    hb = wu % (N * HEAD_BLOCKS)
                    head = hb // HEAD_BLOCKS
                    blk = hb % HEAD_BLOCKS
                    tile_row = tile * BASE_M
                    rows_full = pl.min(BASE_M, t_live - tile_row)
                    prev_tile = (wu - num_cores) // (N * HEAD_BLOCKS)
                    if klocal == 0 or tile != prev_tile:
                        pl.set_validshape(qr_l1.current(), [rows_full, R])
                        pl.load(qr_l1.current(), qr, [tile_row, 0])
                        sqk = sq_k.next()    # depth-2 rotation (only advances on reload;
                        pl.set_validshape(sqk, [rows_full, SQ_ALL_B])   # reads via .current()
                        pl.load(sqk, descale_qr, [tile_row, 0, 0], order=[0, 1])  # see this slot)
                    _qb_beat(qr_l1, sq_k, b_nz, b_zn, sw_k, l0a, l0b,
                             scale_a_l0, scale_b_l0, q_acc, cv_ub,
                             descale_wqb, wqb, head * D + blk * BN_B, klocal,
                             klocal == 0, rows_full)
            elif n_tiles == 2:
                n_pairs = (n_tiles + 1) // 2
                max_units = (N * HEAD_BLOCKS + num_cores - 1) // num_cores
                for pair in pl.range(0, n_pairs):
                    tile0 = pair * 2
                    tile_row0 = tile0 * BASE_M
                    rows0 = pl.min(BASE_M, t_live - tile_row0)
                    tile1 = tile0 + 1
                    tile_row1 = tile1 * BASE_M
                    rows1 = pl.min(BASE_M, t_live - tile_row1)
                    pl.set_validshape(qr_pair[0], [rows0, R])
                    pl.load(qr_pair[0], qr, [tile_row0, 0])
                    pl.set_validshape(sqk0.current(), [rows0, SQ_ALL_B])
                    pl.load(sqk0.current(), descale_qr, [tile_row0, 0, 0], order=[0, 1])
                    if tile1 < n_tiles:
                        pl.set_validshape(qr_pair[1], [rows1, R])
                        pl.load(qr_pair[1], qr, [tile_row1, 0])
                        pl.set_validshape(sqk1.current(), [rows1, SQ_ALL_B])
                        pl.load(sqk1.current(), descale_qr, [tile_row1, 0, 0], order=[0, 1])
                    for hb in pl.range(core_id, N * HEAD_BLOCKS, num_cores):
                        head = hb // HEAD_BLOCKS
                        blk = hb % HEAD_BLOCKS
                        col0 = head * D + blk * BN_B
                        hb_local = (hb - core_id) // num_cores
                        beat = 2 * (pair * max_units + hb_local)
                        for kk in pl.range(NK_B):
                            wq = wb5.next()
                            pl.set_validshape(wq, [BN_B, K0])
                            if Profile == 0:  # noqa: F821
                                pl.load(wq, wqb, [col0, kk * K0], l2_cache_mode=pl.TLoadL2Hint.NotAllocKeep)
                            else:
                                pl.load(wq, wqb, [col0, kk * K0])
                        pl.set_validshape(swblk.current(), [SQ_ALL_B, BN_B])
                        pl.load(swblk.current(), descale_wqb, [col0, 0, 0], order=[1, 0])
                        if tile1 < n_tiles:
                            _qb_beat_pair5(qr_pair[0], qr_pair[1], sqk0.current(), sqk1.current(),
                                           wb5_zn, swblk, l0a, l0b, scale_a_l0, scale_b_l0,
                                           q_acc, cv_ub, beat, rows0, rows1)
                        else:
                            _qb_beat_resident5(qr_pair[0], sqk0.current(), wb5_zn, swblk,
                                               l0a, l0b, scale_a_l0, scale_b_l0,
                                               q_acc, cv_ub, beat, rows0)

            else:
                # Distribute at (tile,head,BLOCK) unit granularity: 32 cores stream
                # Wqb with adjacent cores on adjacent 128-row bands, maximizing NZ
                # k-frac page density so the 41.9 MB stream runs at the full aggregate
                # L2->L1 rate (a (tile,head)-major order would space cores 512 rows
                # apart and cost ~25% of the stream rate).  Per-beat loads/matmul are
                # byte-identical under either order; beats per core = n_tiles*256/32.
                # qr/descale_qr reloads are hoisted to tile changes (tile = wu // 256;
                # at T<=128 one load per core).
                for wu in pl.range(core_id, n_tiles * N * HEAD_BLOCKS, num_cores):
                    klocal = (wu - core_id) // num_cores
                    tile = wu // (N * HEAD_BLOCKS)
                    hb = wu % (N * HEAD_BLOCKS)
                    head = hb // HEAD_BLOCKS
                    blk = hb % HEAD_BLOCKS
                    tile_row = tile * BASE_M
                    rows_full = pl.min(BASE_M, t_live - tile_row)
                    prev_tile = (wu - num_cores) // (N * HEAD_BLOCKS)
                    if klocal == 0 or tile != prev_tile:
                        pl.set_validshape(qr_l1.current(), [rows_full, R])
                        pl.load(qr_l1.current(), qr, [tile_row, 0])
                        sqk = sq_k.next()    # depth-2 rotation (only advances on reload;
                        pl.set_validshape(sqk, [rows_full, SQ_ALL_B])   # reads via .current()
                        pl.load(sqk, descale_qr, [tile_row, 0, 0], order=[0, 1])  # see this slot)
                    _qb_beat(qr_l1, sq_k, b_nz, b_zn, sw_k, l0a, l0b,
                             scale_a_l0, scale_b_l0, q_acc, cv_ub,
                             descale_wqb, wqb, head * D + blk * BN_B, klocal,
                             klocal == 0, rows_full)
    # =================== VECTOR SECTION ===================
    with pl.section_vector():
        sub = pl.get_subblock_idx()
        # ---- A2 UB tiles ----
        qa_row = pl.make_tile_group(
            type=pl.TileType(shape=[1, QR_PAD], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_QA_ROW, mutex_ids=[0])
        gamma_qr = pl.make_tile_group(
            type=pl.TileType(shape=[1, QR_PAD], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_GAMMA_QR, mutex_ids=[1])
        qr_payload = pl.make_tile_group(
            type=pl.TileType(shape=[1, R], dtype=pl.DT_UINT8,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_QR_PAYLOAD, mutex_ids=[2])
        qr_norm_tile = pl.make_tile_group(
            type=pl.TileType(shape=[1, R], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_QR_NORM, mutex_ids=[18])
        exp_tile = pl.make_tile_group(
            type=pl.TileType(shape=[1, VL], dtype=pl.DT_INT32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_EXP, mutex_ids=[3])
        scale_u8 = pl.make_tile_group(
            type=pl.TileType(shape=[1, VL], dtype=pl.DT_UINT8,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_SCALE_U8, mutex_ids=[4])
        kv_row = pl.make_tile_group(
            type=pl.TileType(shape=[1, D], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_KV_ROW, mutex_ids=[5])
        kv_bf16 = pl.make_tile_group(
            type=pl.TileType(shape=[1, D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_KV_ROW, mutex_ids=[5])
        gamma_kv = pl.make_tile_group(
            type=pl.TileType(shape=[1, D], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_GAMMA_KV, mutex_ids=[6])
        kvb = pl.make_tile_group(
            type=pl.TileType(shape=[1, KV_RECORD_BYTES], dtype=pl.DT_UINT8,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_KVB, mutex_ids=[7])
        kvb_scale = pl.make_tile_group(
            type=pl.TileType(shape=[1, KV_QUANT_GROUPS], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_KVB + D, mutex_ids=[7])
        cos_row = pl.make_tile_group(
            type=pl.TileType(shape=[1, DR], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_COS_ROW, mutex_ids=[8])
        sin_row = pl.make_tile_group(
            type=pl.TileType(shape=[1, DR], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_SIN_ROW, mutex_ids=[9])
        # ---- B-epi UB tiles ----
        q_bf16 = pl.make_tile_group(
            type=pl.TileType(shape=[ROWS_VEC, BN_B], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[UB_Q_BF16_0, UB_Q_BF16_1], mutex_ids=[10, 11])
        cos_ub = pl.make_tile_group(
            type=pl.TileType(shape=[ROWS_VEC, DR], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_COS_UB, mutex_ids=[12])
        sin_ub = pl.make_tile_group(
            type=pl.TileType(shape=[ROWS_VEC, DR], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_SIN_UB, mutex_ids=[13])
        cos_ub2 = pl.make_tile_group(
            type=pl.TileType(shape=[ROWS_VEC, DR], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_COS_UB2, mutex_ids=[14])
        sin_ub2 = pl.make_tile_group(
            type=pl.TileType(shape=[ROWS_VEC, DR], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_SIN_UB2, mutex_ids=[15])
        # ---- L1 split-K merge tiles (decode-only gate) ----
        qa_row2 = pl.make_tile_group(
            type=pl.TileType(shape=[1, QR_PAD], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_QA_ROW2, mutex_ids=[16])
        kv_row2 = pl.make_tile_group(
            type=pl.TileType(shape=[1, D], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_KV_ROW2, mutex_ids=[17])
        kv_bf16_2 = pl.make_tile_group(
            type=pl.TileType(shape=[1, D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=UB_KV_ROW2, mutex_ids=[17])
        # ---- P1: preloaded A2 cos/sin rows (decode task-split domain only) ----
        cos_pre = pl.make_tile_group(
            type=pl.TileType(shape=[1, DR], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[UB_COS_PRE + k * UB_PRE_STRIDE for k in range(KV_PRE)],
            mutex_ids=list(COS_PRE_MUTEX))
        sin_pre = pl.make_tile_group(
            type=pl.TileType(shape=[1, DR], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[UB_SIN_PRE + k * UB_PRE_STRIDE for k in range(KV_PRE)],
            mutex_ids=list(SIN_PRE_MUTEX))

        # rK (L2): gamma_qr/gamma_kv are kernel INPUTS (no A1 dependency) — load
        # them BEFORE barrier1 so the GM reads overlap the barrier wait instead of
        # sitting on A2's critical path.
        pl.set_validshape(gamma_qr.current(), [1, R])
        pl.load(gamma_qr.current(), gamma_qr_gm, [0, 0])
        pl.set_validshape(gamma_kv.current(), [1, D])
        pl.load(gamma_kv.current(), gamma_kv_gm, [0, 0])

        worker = core_id * pl.get_subblock_num() + sub
        n_aiv = num_cores * pl.get_subblock_num()
        # P1: first KV task id of this AIV's stripe (tasks ascend; QR == task <
        # t_live, so every stripe is [QR..., KV...] and kv0 is the first KV hit).
        kv0 = worker + ((t_live - worker + n_aiv - 1) // n_aiv) * n_aiv

        # P1: preload the cos/sin rows of the first KV_PRE KV tasks while Cube
        # still runs A1 (Vector idle window before barrier1).  Gate == the A2
        # task-split arm below (SK gate OFF and n_tiles<=2), so SK/prefill are
        # untouched.  Slot k holds row kv0 + k*n_aiv - t_live; the consumer's
        # `task < 2*t_live` guard matches the `prow < t_live` load guard, so an
        # unloaded slot is never read.
        if n_tiles * NA1 * 2 > num_cores and n_tiles <= 2:
            for k in pl.range(KV_PRE):
                prow = kv0 + k * n_aiv - t_live
                if prow < t_live:
                    pl.set_validshape(cos_pre[k], [1, DR])
                    pl.load(cos_pre[k], rope_cos, [prow, 0])
                    pl.set_validshape(sin_pre[k], [1, DR])
                    pl.load(sin_pre[k], rope_sin, [prow, 0])

        # ---------------- barrier1 ----------------
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

        # ---------------- Stage A2: RMSNorm / quant / scatter ----------------
        if n_tiles * NA1 * 2 <= num_cores:
            # ---- L1 split-K merge (SK=2): band1 lives at row n_tiles*BASE_M+row ----
            # One FP32 add per element: for exactly 2 partials this is byte-identical
            # to a Kahan merge (compensation starts at 0 => corrected == part,
            # total == fl(b0+b1)); KV uses plain summation.  Gate is byte-identical
            # to the A1 SK branch and the wrapper's ws allocation; qr/descale_qr/kv
            # outputs are byte-identical to the unsplit path on every decode case.
            for task in pl.range(worker, 2 * t_live, n_aiv):
                if task < t_live:
                    # ---- QR (merged) ----
                    pl.set_validshape(qa_row.current(), [1, R])
                    pl.load(qa_row.current(), ws_qa, [task, 0])
                    pl.set_validshape(qa_row2.current(), [1, R])
                    pl.load(qa_row2.current(), ws_qa, [n_tiles * BASE_M + task, 0])
                    pl.add(qa_row2.current(), qa_row.current(), qa_row2.current())
                    pl.set_validshape(qr_payload.current(), [1, R])
                    qr_row_vf(qa_row2.current(), gamma_qr.current(), qr_payload.current(),
                              qr_norm_tile.current(), QrNorm,
                              exp_tile.current(), scale_u8.current(), tiling.norm_eps)
                    pl.set_validshape(qr_payload.current(), [1, R])
                    pl.store(qr_u8, qr_payload.current(), [task, 0])
                    if QrNorm == 1:
                        pl.set_validshape(qr_norm_tile.current(), [1, R])
                        pl.store(qr_norm_gm, qr_norm_tile.current(), [task, 0])
                    pl.set_validshape(scale_u8.current(), [1, QR_GROUPS])
                    pl.store(dqr_u8, scale_u8.current(), [task, 0])
                else:
                    # ---- KV (merged) ----
                    row = task - t_live
                    pl.set_validshape(kv_row.current(), [1, D])
                    pl.load(kv_row.current(), ws_kv, [row, 0])
                    pl.set_validshape(kv_row2.current(), [1, D])
                    pl.load(kv_row2.current(), ws_kv, [n_tiles * BASE_M + row, 0])
                    pl.add(kv_row2.current(), kv_row.current(), kv_row2.current())
                    pl.set_validshape(cos_row.current(), [1, DR])
                    pl.load(cos_row.current(), rope_cos, [row, 0])
                    pl.set_validshape(sin_row.current(), [1, DR])
                    pl.load(sin_row.current(), rope_sin, [row, 0])
                    kv_row_vf(kv_row2.current(), gamma_kv.current(), cos_row.current(),
                              sin_row.current(), kv_bf16_2.current(), tiling.norm_eps)
                    kv_compress_row_vf(kv_bf16_2.current(), kvb.current(),
                                       kvb_scale.current())
                    slot = pl.getval(cache_index, row)
                    if slot >= 0 and slot < cache_rows:
                        pl.set_validshape(kvb.current(), [1, KV_RECORD_BYTES])
                        page = slot // block_size
                        row_offset = (slot % block_size) * row_stride
                        pl.store(kv_cache_pages, kvb.current(), [page, row_offset])
        elif n_tiles <= 2:
            # rK (decode): split the striped row loop into independent QR / KV
            # tasks (task < t_live -> QR of row `task`; else KV of row
            # `task - t_live`).  Per-row math byte-identical (same VFs, same
            # single-writer stores); only the worker->work mapping changes.  At
            # T<=32 the old loop left half the AIVs idle and serialized QR+KV on
            # one AIV per row; the split runs them on distinct AIVs (exposed A2
            # wall ~halved; measured -0.5..-1.1us at T=32/72).
            # P1: task ids ascend and the QR/KV threshold is fixed, so each
            # stripe is [QR tasks..., KV tasks...] — splitting the one striped
            # loop into a QR loop then KV loops preserves the per-AIV task set
            # AND order exactly.  The first KV_PRE KV tasks consume the cos/sin
            # rows preloaded before barrier1 (slot k <-> task kv0+k*n_aiv); the
            # tail loop streams any remainder from GM (<=4 KV tasks/AIV in this
            # domain, so it is a zero-trip safety net at T<=256).
            for task in pl.range(worker, t_live, n_aiv):
                # ---- QR ----
                pl.set_validshape(qa_row.current(), [1, R])
                pl.load(qa_row.current(), ws_qa, [task, 0])
                pl.set_validshape(qr_payload.current(), [1, R])
                qr_row_vf(qa_row.current(), gamma_qr.current(), qr_payload.current(),
                          qr_norm_tile.current(), QrNorm,
                          exp_tile.current(), scale_u8.current(), tiling.norm_eps)
                pl.set_validshape(qr_payload.current(), [1, R])
                pl.store(qr_u8, qr_payload.current(), [task, 0])
                if QrNorm == 1:
                    pl.set_validshape(qr_norm_tile.current(), [1, R])
                    pl.store(qr_norm_gm, qr_norm_tile.current(), [task, 0])
                pl.set_validshape(scale_u8.current(), [1, QR_GROUPS])
                pl.store(dqr_u8, scale_u8.current(), [task, 0])
            for k in pl.range(KV_PRE):
                task = kv0 + k * n_aiv
                if task < 2 * t_live:
                    # ---- KV (preloaded cos/sin) ----
                    row = task - t_live
                    pl.set_validshape(kv_row.current(), [1, D])
                    pl.load(kv_row.current(), ws_kv, [row, 0])
                    kv_row_vf(kv_row.current(), gamma_kv.current(), cos_pre[k],
                              sin_pre[k], kv_bf16.current(), tiling.norm_eps)
                    kv_compress_row_vf(kv_bf16.current(), kvb.current(),
                                       kvb_scale.current())
                    slot = pl.getval(cache_index, row)
                    if slot >= 0 and slot < cache_rows:
                        pl.set_validshape(kvb.current(), [1, KV_RECORD_BYTES])
                        page = slot // block_size
                        row_offset = (slot % block_size) * row_stride
                        pl.store(kv_cache_pages, kvb.current(), [page, row_offset])
            for task in pl.range(kv0 + KV_PRE * n_aiv, 2 * t_live, n_aiv):
                # ---- KV (streamed cos/sin) ----
                row = task - t_live
                pl.set_validshape(kv_row.current(), [1, D])
                pl.load(kv_row.current(), ws_kv, [row, 0])
                pl.set_validshape(cos_row.current(), [1, DR])
                pl.load(cos_row.current(), rope_cos, [row, 0])
                pl.set_validshape(sin_row.current(), [1, DR])
                pl.load(sin_row.current(), rope_sin, [row, 0])
                kv_row_vf(kv_row.current(), gamma_kv.current(), cos_row.current(),
                          sin_row.current(), kv_bf16.current(), tiling.norm_eps)
                kv_compress_row_vf(kv_bf16.current(), kvb.current(),
                                   kvb_scale.current())
                slot = pl.getval(cache_index, row)
                if slot >= 0 and slot < cache_rows:
                    pl.set_validshape(kvb.current(), [1, KV_RECORD_BYTES])
                    page = slot // block_size
                    row_offset = (slot % block_size) * row_stride
                    pl.store(kv_cache_pages, kvb.current(), [page, row_offset])
        else:
            # rK-gate: prefill (n_tiles>=3) keeps the ORIGINAL row-serial loop —
            # the task split measurably cost +3% at T=1024/4096 (ungated rK),
            # so big-T keeps the byte-identical old path.
            for row in pl.range(worker, t_live, n_aiv):
                # ---- QR ----
                pl.set_validshape(qa_row.current(), [1, R])
                pl.load(qa_row.current(), ws_qa, [row, 0])
                pl.set_validshape(qr_payload.current(), [1, R])
                qr_row_vf(qa_row.current(), gamma_qr.current(), qr_payload.current(),
                          qr_norm_tile.current(), QrNorm,
                          exp_tile.current(), scale_u8.current(), tiling.norm_eps)
                pl.set_validshape(qr_payload.current(), [1, R])
                pl.store(qr_u8, qr_payload.current(), [row, 0])
                if QrNorm == 1:
                    pl.set_validshape(qr_norm_tile.current(), [1, R])
                    pl.store(qr_norm_gm, qr_norm_tile.current(), [row, 0])
                pl.set_validshape(scale_u8.current(), [1, QR_GROUPS])
                pl.store(dqr_u8, scale_u8.current(), [row, 0])
                # ---- KV ----
                pl.set_validshape(kv_row.current(), [1, D])
                pl.load(kv_row.current(), ws_kv, [row, 0])
                pl.set_validshape(cos_row.current(), [1, DR])
                pl.load(cos_row.current(), rope_cos, [row, 0])
                pl.set_validshape(sin_row.current(), [1, DR])
                pl.load(sin_row.current(), rope_sin, [row, 0])
                kv_row_vf(kv_row.current(), gamma_kv.current(), cos_row.current(),
                          sin_row.current(), kv_bf16.current(), tiling.norm_eps)
                kv_compress_row_vf(kv_bf16.current(), kvb.current(),
                                   kvb_scale.current())
                slot = pl.getval(cache_index, row)
                if slot >= 0 and slot < cache_rows:
                    pl.set_validshape(kvb.current(), [1, KV_RECORD_BYTES])
                    page = slot // block_size
                    row_offset = (slot % block_size) * row_stride
                    pl.store(kv_cache_pages, kvb.current(), [page, row_offset])

        # ---------------- barrier2 ----------------
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

        # ---------------- Stage B-epi: RoPE / BF16 -> q ----------------
        # Mirrors the Cube B distribution exactly so per-core beats line up.
        if split_t:
            for tile in pl.range(core_id, n_tiles, num_cores):
                tile_row = tile * BASE_M
                rows_full = pl.min(BASE_M, t_live - tile_row)
                rows_vec = ((rows_full + 31) // 32 * 32) // 2
                ro = tile_row + sub * rows_vec
                store_rows = pl.min(rows_vec, pl.max(t_live - ro, 0))
                beat0 = (tile // num_cores) * BEATS_PER_TILE
                pl.set_validshape(cos_ub.current(), [store_rows, DR])
                pl.set_validshape(sin_ub.current(), [store_rows, DR])
                pl.load(cos_ub.current(), rope_cos, [ro, 0])
                pl.load(sin_ub.current(), rope_sin, [ro, 0])
                for head in pl.range(N):
                    for block in pl.range(HEAD_BLOCKS - 1):
                        _q_epi_beat(cv_ub, q_bf16, q2d, cos_ub, sin_ub, ro, store_rows, rows_vec,
                                    head * D + block * BN_B,
                                    beat0 + head * HEAD_BLOCKS + block, False)
                    _q_epi_beat(cv_ub, q_bf16, q2d, cos_ub, sin_ub, ro, store_rows, rows_vec,
                                head * D + (HEAD_BLOCKS - 1) * BN_B,
                                beat0 + head * HEAD_BLOCKS + (HEAD_BLOCKS - 1), True)
        else:
            if n_tiles == 1:
                # Mirror of the cube-side (tile,head,block) unit-stride order —
                # one _q_epi_beat per unit, is_tail = last block of the head (runtime
                # branch; VF calls under a runtime if follow the FA-sample precedent).
                # cos/sin RoPE tables depend only on the row offset ro (tile, sub):
                # reload only on this core's first unit or on a tile change (vector
                # mirror of the cube-side qr hoist).  Numerically inert; frees vector
                # MTE2 bytes and removes a per-unit WAR stall on single-slot cos/sin_ub.
                for wu in pl.range(core_id, n_tiles * N * HEAD_BLOCKS, num_cores):
                    klocal = (wu - core_id) // num_cores
                    tile = wu // (N * HEAD_BLOCKS)
                    hb = wu % (N * HEAD_BLOCKS)
                    head = hb // HEAD_BLOCKS
                    blk = hb % HEAD_BLOCKS
                    tile_row = tile * BASE_M
                    rows_full = pl.min(BASE_M, t_live - tile_row)
                    rows_vec = ((rows_full + 31) // 32 * 32) // 2
                    ro = tile_row + sub * rows_vec
                    store_rows = pl.min(rows_vec, pl.max(t_live - ro, 0))
                    prev_tile = (wu - num_cores) // (N * HEAD_BLOCKS)
                    if klocal == 0 or tile != prev_tile:
                        pl.set_validshape(cos_ub.current(), [store_rows, DR])
                        pl.set_validshape(sin_ub.current(), [store_rows, DR])
                        pl.load(cos_ub.current(), rope_cos, [ro, 0])
                        pl.load(sin_ub.current(), rope_sin, [ro, 0])
                    _q_epi_beat(cv_ub, q_bf16, q2d, cos_ub, sin_ub, ro, store_rows, rows_vec,
                                head * D + blk * BN_B, klocal, blk == HEAD_BLOCKS - 1)
            elif n_tiles == 2:
                n_pairs = (n_tiles + 1) // 2
                max_units = (N * HEAD_BLOCKS + num_cores - 1) // num_cores
                for pair in pl.range(0, n_pairs):
                    tile0 = pair * 2
                    tile_row0 = tile0 * BASE_M
                    rows_full0 = pl.min(BASE_M, t_live - tile_row0)
                    rows_vec0 = ((rows_full0 + 31) // 32 * 32) // 2
                    ro0 = tile_row0 + sub * rows_vec0
                    store_rows0 = pl.min(rows_vec0, pl.max(t_live - ro0, 0))
                    pl.set_validshape(cos_ub.current(), [store_rows0, DR])
                    pl.set_validshape(sin_ub.current(), [store_rows0, DR])
                    pl.load(cos_ub.current(), rope_cos, [ro0, 0])
                    pl.load(sin_ub.current(), rope_sin, [ro0, 0])
                    tile1 = tile0 + 1
                    tile_row1 = tile1 * BASE_M
                    rows_full1 = pl.min(BASE_M, t_live - tile_row1)
                    rows_vec1 = ((rows_full1 + 31) // 32 * 32) // 2
                    ro1 = tile_row1 + sub * rows_vec1
                    store_rows1 = pl.min(rows_vec1, pl.max(t_live - ro1, 0))
                    if tile1 < n_tiles:
                        pl.set_validshape(cos_ub2.current(), [store_rows1, DR])
                        pl.set_validshape(sin_ub2.current(), [store_rows1, DR])
                        pl.load(cos_ub2.current(), rope_cos, [ro1, 0])
                        pl.load(sin_ub2.current(), rope_sin, [ro1, 0])
                    for hb in pl.range(core_id, N * HEAD_BLOCKS, num_cores):
                        head = hb // HEAD_BLOCKS
                        blk = hb % HEAD_BLOCKS
                        col0 = head * D + blk * BN_B
                        hb_local = (hb - core_id) // num_cores
                        beat = 2 * (pair * max_units + hb_local)
                        _q_epi_beat(cv_ub, q_bf16, q2d, cos_ub, sin_ub, ro0, store_rows0, rows_vec0,
                                    col0, beat, blk == HEAD_BLOCKS - 1)
                        if tile1 < n_tiles:
                            _q_epi_beat(cv_ub, q_bf16, q2d, cos_ub2, sin_ub2, ro1, store_rows1, rows_vec1,
                                        col0, beat + 1, blk == HEAD_BLOCKS - 1)

            else:
                # Mirror of the cube-side (tile,head,block) unit-stride order —
                # one _q_epi_beat per unit, is_tail = last block of the head (runtime
                # branch; VF calls under a runtime if follow the FA-sample precedent).
                # cos/sin RoPE tables depend only on the row offset ro (tile, sub):
                # reload only on this core's first unit or on a tile change (vector
                # mirror of the cube-side qr hoist).  Numerically inert; frees vector
                # MTE2 bytes and removes a per-unit WAR stall on single-slot cos/sin_ub.
                for wu in pl.range(core_id, n_tiles * N * HEAD_BLOCKS, num_cores):
                    klocal = (wu - core_id) // num_cores
                    tile = wu // (N * HEAD_BLOCKS)
                    hb = wu % (N * HEAD_BLOCKS)
                    head = hb // HEAD_BLOCKS
                    blk = hb % HEAD_BLOCKS
                    tile_row = tile * BASE_M
                    rows_full = pl.min(BASE_M, t_live - tile_row)
                    rows_vec = ((rows_full + 31) // 32 * 32) // 2
                    ro = tile_row + sub * rows_vec
                    store_rows = pl.min(rows_vec, pl.max(t_live - ro, 0))
                    prev_tile = (wu - num_cores) // (N * HEAD_BLOCKS)
                    if klocal == 0 or tile != prev_tile:
                        pl.set_validshape(cos_ub.current(), [store_rows, DR])
                        pl.set_validshape(sin_ub.current(), [store_rows, DR])
                        pl.load(cos_ub.current(), rope_cos, [ro, 0])
                        pl.load(sin_ub.current(), rope_sin, [ro, 0])
                    _q_epi_beat(cv_ub, q_bf16, q2d, cos_ub, sin_ub, ro, store_rows, rows_vec,
                                head * D + blk * BN_B, klocal, blk == HEAD_BLOCKS - 1)
# ---------------------------------------------------------------------------
# Public wrapper: validate + allocate + exactly one launch
# ---------------------------------------------------------------------------


def to_nz(weight: torch.Tensor) -> torch.Tensor:
    """Convert one ND device weight to FRACTAL_NZ (call once at weight-load time)."""
    if weight.device.type != "npu":
        raise ValueError("to_nz requires an NPU tensor")
    return torch_npu.npu_format_cast(weight.contiguous(), FRACTAL_NZ)


_E4M3_DTYPES = (torch.uint8, torch.int8, torch.float8_e4m3fn)
_E8M0_DTYPES = (torch.uint8, torch.int8, torch.float8_e8m0fnu)


class _Geometry(NamedTuple):
    tokens: int
    block_size: int
    cache_rows: int
    page_stride: int
    row_stride: int
    storage_bytes: int


def _cache_geometry(t, kv_cache):
    if (kv_cache.dim() != 4 or kv_cache.shape[2] != 1
            or kv_cache.shape[3] < KV_RECORD_BYTES):
        raise ValueError(
            f"kv_cache must be [P,BS,1,C>= {KV_RECORD_BYTES}] uint8, "
            f"got {tuple(kv_cache.shape)}")
    if kv_cache.dtype != torch.uint8:
        raise ValueError(f"kv_cache must be uint8, got {kv_cache.dtype}")
    # Plain 4-D storage may be reported as NCHW. Keep the caller's page/row strides.
    if torch_npu.get_npu_format(kv_cache) not in (0, 2):
        raise ValueError("kv_cache must use ND/NCHW storage")
    if int(kv_cache.stride(3)) != 1:
        raise ValueError(f"kv_cache last stride must be 1, got {kv_cache.stride(3)}")
    pages, block_size = int(kv_cache.shape[0]), int(kv_cache.shape[1])
    cache_rows = pages * block_size
    if block_size <= 0 or cache_rows <= 0:
        raise ValueError("kv_cache P/BS must be positive")
    row_stride = int(kv_cache.stride(1))
    page_stride = int(kv_cache.stride(0))
    if row_stride < KV_RECORD_BYTES or row_stride % 32:
        raise ValueError(
            f"row_stride must be >= {KV_RECORD_BYTES} and 32B aligned, got {row_stride}")
    if page_stride < block_size * row_stride or page_stride % 32:
        raise ValueError(
            f"page_stride must be >= BS*row_stride ({block_size * row_stride}) "
            f"and 32B aligned, got {page_stride}")
    needed = (pages - 1) * page_stride + (block_size - 1) * row_stride + KV_RECORD_BYTES
    storage_bytes = (kv_cache.untyped_storage().nbytes()
                     - kv_cache.storage_offset() * kv_cache.element_size())
    if storage_bytes < needed:
        raise ValueError(
            f"kv_cache backing storage {storage_bytes} < physical write range {needed}")
    return _Geometry(t, block_size, cache_rows, page_stride, row_stride, storage_bytes)


def _check_dense_nd(name, tensor):
    if not tensor.is_contiguous() or torch_npu.get_npu_format(tensor) != 2:
        raise ValueError(f"{name} must be contiguous with ND storage")


def _validate_metadata(x, wqa, wqb, wkv, descale_x, descale_wqa, descale_wqb,
                       descale_wkv, norm_weight_qr, norm_weight_kv, rope_sin,
                       rope_cos, cache_index, kv_cache):
    """Cheap host assertions; returns the runtime T and cache geometry."""
    if x.dim() != 2 or x.shape[1] != H or x.dtype not in _E4M3_DTYPES:
        raise ValueError(f"x must be [T,{H}] E4M3, got {tuple(x.shape)} {x.dtype}")
    t = x.shape[0]
    if not 1 <= t <= T_MAX:
        raise ValueError(f"T={t} outside supported range [1,{T_MAX}]")
    if tuple(wqa.shape) != (R, H) or wqa.dtype not in _E4M3_DTYPES:
        raise ValueError(f"wqa must be [{R},{H}] E4M3, got {tuple(wqa.shape)} {wqa.dtype}")
    if tuple(wqb.shape) != (N * D, R) or wqb.dtype not in _E4M3_DTYPES:
        raise ValueError(f"wqb must be [{N*D},{R}] E4M3, got {tuple(wqb.shape)} {wqb.dtype}")
    if tuple(wkv.shape) != (D, H) or wkv.dtype not in _E4M3_DTYPES:
        raise ValueError(f"wkv must be [{D},{H}] E4M3, got {tuple(wkv.shape)} {wkv.dtype}")
    if tuple(descale_x.shape) != (t, H // 64, 2) or descale_x.dtype not in _E8M0_DTYPES:
        raise ValueError(f"descale_x must be [{t},{H//64},2] E8M0")
    if tuple(descale_wqa.shape) != (R, H // 64, 2) or descale_wqa.dtype not in _E8M0_DTYPES:
        raise ValueError(f"descale_wqa must be [{R},{H//64},2] E8M0")
    if tuple(descale_wqb.shape) != (N * D, R // 64, 2) or descale_wqb.dtype not in _E8M0_DTYPES:
        raise ValueError(f"descale_wqb must be [{N*D},{R//64},2] E8M0")
    if tuple(descale_wkv.shape) != (D, H // 64, 2) or descale_wkv.dtype not in _E8M0_DTYPES:
        raise ValueError(f"descale_wkv must be [{D},{H//64},2] E8M0")
    if tuple(norm_weight_qr.shape) != (R,) or norm_weight_qr.dtype != torch.float32:
        raise ValueError(f"norm_weight_qr must be [{R}] float32")
    if tuple(norm_weight_kv.shape) != (D,) or norm_weight_kv.dtype != torch.float32:
        raise ValueError(f"norm_weight_kv must be [{D}] float32")
    if tuple(rope_sin.shape) != (t, DR) or rope_sin.dtype != torch.float32:
        raise ValueError(f"rope_sin must be [{t},{DR}] float32")
    if tuple(rope_cos.shape) != (t, DR) or rope_cos.dtype != torch.float32:
        raise ValueError(f"rope_cos must be [{t},{DR}] float32")
    if tuple(cache_index.shape) != (t,) or cache_index.dtype != torch.int64:
        raise ValueError(f"cache_index must be [{t}] int64")
    for name, tensor in (("x", x), ("descale_x", descale_x), ("descale_wqa", descale_wqa),
                         ("descale_wqb", descale_wqb), ("descale_wkv", descale_wkv),
                         ("norm_weight_qr", norm_weight_qr), ("norm_weight_kv", norm_weight_kv),
                         ("rope_sin", rope_sin), ("rope_cos", rope_cos), ("cache_index", cache_index)):
        _check_dense_nd(name, tensor)
    geometry = _cache_geometry(t, kv_cache)
    # NZ gate: catch a forgotten to_nz() before the kernel silently reads ND as NZ
    for name, w in (("wqa", wqa), ("wqb", wqb), ("wkv", wkv)):
        if torch_npu.get_npu_format(w) != FRACTAL_NZ:
            raise ValueError(f"{name} must be FRACTAL_NZ (call to_nz() once at load time)")
    return geometry


def _fused_grid():
    """MIX kernel grid: each block = 1 AIC + 2 AIV, so blocks <= min(cube, vector//2)."""
    try:
        info = get_platform_info()
        cube = int(info.cube_core_num)
        vector = int(info.vector_core_num)
    except Exception:  # noqa: BLE001
        cube, vector = 32, 64
    if cube <= 0 or vector < cube:
        cube, vector = 32, 64
    return max(1, min(cube, vector // 2))


def attn_prologue(
    x, wqa, wqb, wkv,
    descale_x, descale_wqa, descale_wqb, descale_wkv,
    norm_weight_qr, norm_weight_kv,
    rope_sin, rope_cos, cache_index, kv_cache,
    qr_norm=None, norm_eps=1e-6,
):
    """Public entry: validate metadata, allocate outputs+workspace, launch once.

    Returns ``(q bf16[T,N,D], qr e4m3[T,R], descale_qr int8[T,R/64,2])``;
    ``kv_cache`` is updated in place.  The wrapper does no device compute.
    """
    (t, block_size, cache_rows, page_stride, row_stride,
     _storage_bytes) = _validate_metadata(
        x, wqa, wqb, wkv, descale_x, descale_wqa, descale_wqb,
        descale_wkv, norm_weight_qr, norm_weight_kv, rope_sin,
        rope_cos, cache_index, kv_cache)
    dev = x.device
    if qr_norm is None:
        qr_norm = torch.empty((1, R), dtype=torch.bfloat16, device=dev)
        emit_qr_norm = 0
    else:
        if tuple(qr_norm.shape) != (t, R) or qr_norm.dtype != torch.bfloat16 or qr_norm.device != dev:
            raise ValueError(f"qr_norm must be bf16 [{t},{R}] on {dev}")
        _check_dense_nd("qr_norm", qr_norm)
        emit_qr_norm = 1
    grid = _fused_grid()
    n_tiles = (t + BASE_M - 1) // BASE_M
    # L1 split-K ws bands: BYTE-IDENTICAL to the in-kernel A1/A2 gate
    # (n_tiles*NA1*2 <= num_cores, num_cores == the grid launched below).
    # When it fires (n_tiles==1 on a 32-core grid) ws carries 2 K-half bands
    # of n_tiles*BASE_M rows each; otherwise ws stays exactly t rows.
    sk_bands = 2 if n_tiles * NA1 * 2 <= grid else 1
    ws_rows = sk_bands * n_tiles * BASE_M if sk_bands == 2 else t
    q = torch.empty((t, N, D), dtype=torch.bfloat16, device=dev)
    qr = torch.empty((t, R), dtype=torch.float8_e4m3fn, device=dev)
    descale_qr = torch.empty((t, R // 64, 2), dtype=torch.float8_e8m0fnu, device=dev)
    workspace = torch.empty((ws_rows * (R + D) * 4,), dtype=torch.uint8, device=dev)
    tiling = AttnPrologueTiling(
        t=t, cache_rows=cache_rows, block_size=block_size,
        block_stride=page_stride, row_stride=row_stride,
        ws_rows=ws_rows, norm_eps=float(norm_eps),
        emit_qr_norm=emit_qr_norm)
    profile = 0 if t <= BASE_M else 1
    attn_prologue_kernel[None, grid, {"Profile": profile, "QrNorm": emit_qr_norm}](
        x, wqa, wqb, wkv,
        descale_x, descale_wqa, descale_wqb, descale_wkv,
        norm_weight_qr, norm_weight_kv,
        rope_sin, rope_cos, cache_index, kv_cache, qr_norm,
        q, qr, descale_qr, workspace, tiling,
    )
    return q, qr, descale_qr.view(torch.int8)
